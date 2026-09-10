#!/usr/bin/env python3
"""Dump one SLat flow-model evaluation as torch reference.

Inputs from dump_e2e_stages.py:
  slat_x0.samt           (8, N)   initial latent noise
  slat_coords.samt       (N, 4)   [batch, x, y, z] active cells (resolution 64)
  slat_cond_tokens.samt  (1024, M) fused condition tokens

The CFG wrapper has strength 0 for the slat generator -> is_cond -> a single
backbone call with the real condition (no uncond branch). Intermediates are
hooked at every stage the C++ graph debugs: input_layer, each io res block
(pre/post downsample), APE, block0 / block23, out blocks, final feats.
"""
import argparse
import json
import os
import struct

import numpy as np
import torch

from gguf_torch_loader import replace_backbone_from_gguf

SAMT_MAGIC = b"SAMT"


def load_samt(path):
    with open(path, "rb") as f:
        f.read(4)
        nd = struct.unpack("<i", f.read(4))[0]
        ne = struct.unpack(f"<{nd}q", f.read(8 * nd))
        dt = struct.unpack("<i", f.read(4))[0]
        data = np.frombuffer(f.read(), dtype="<f4" if dt == 0 else "<i4")
    return ne, data


def write_samt_f32(path, ne, data):
    data = np.ascontiguousarray(data, dtype="<f4")
    with open(path, "wb") as f:
        f.write(SAMT_MAGIC)
        f.write(struct.pack("<i", len(ne)))
        f.write(struct.pack(f"<{len(ne)}q", *ne))
        f.write(struct.pack("<i", 0))
        f.write(data.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/hf/slat_generator.ckpt")
    ap.add_argument("--yaml", default="checkpoints/hf/slat_generator.yaml")
    ap.add_argument("--e2e-dir", default="cpp_ggml/benchmarks/data/e2e")
    ap.add_argument("--out-dir", default="/tmp/slat_dbg")
    ap.add_argument("--t", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gguf",
                    help=("optional generator GGUF; replaces every reverse_fn.backbone weight "
                          "with its exact GGUF dequantization before the Torch reference"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    import sam3d_objects  # registers the hydra targets
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    yconf = OmegaConf.load(args.yaml)
    bb = yconf["module"]["generator"]["backbone"]
    gen = instantiate(bb)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    gen_sd = {k[len("_base_models.generator."):]: v for k, v in sd.items()
              if k.startswith("_base_models.generator.")}
    del sd
    if args.gguf:
        weight_report = replace_backbone_from_gguf(gen_sd, args.gguf)
        with open(os.path.join(args.out_dir, "reference_weight_scope.json"), "w", encoding="utf-8") as stream:
            json.dump(weight_report, stream, indent=2)
            stream.write("\n")
        print("[ref] same-GGUF backbone loaded: "
              f"{weight_report['backbone_tensors_loaded']} tensors from {weight_report['gguf']}")
    missing, unexpected = gen.load_state_dict(gen_sd, strict=False)
    print(f"[ref] slat generator loaded: missing={len(missing)} unexpected={len(unexpected)}")
    gen = gen.to(args.device).eval()
    # pipeline FlowMatching overrides
    gen.inference_steps = 12
    gen.rescale_t = 3
    gen.reverse_fn.strength = 0.0
    gen.no_shortcut = True

    ne, xa = load_samt(os.path.join(args.e2e_dir, "slat_x0.samt"))
    x = torch.from_numpy(xa.reshape(1, ne[1], ne[0]).copy()).to(args.device)
    ne, ca = load_samt(os.path.join(args.e2e_dir, "slat_coords.samt"))
    coords = torch.from_numpy(ca.reshape(ne[1], ne[0]).copy()).to(args.device)
    ne, ta = load_samt(os.path.join(args.e2e_dir, "slat_cond_tokens.samt"))
    cond = torch.from_numpy(ta.reshape(1, ne[1], ne[0]).copy()).to(args.device)
    print(f"[ref] inputs: x{x.shape} coords{coords.shape} cond{cond.shape}")

    t = torch.tensor([args.t], device=args.device, dtype=torch.float32)

    cap = {}
    wbb = gen.reverse_fn.backbone

    def tok_hook(name):
        def hook(mod, inp, out):
            tt = out[0] if isinstance(out, tuple) else out
            if isinstance(tt, torch.Tensor):
                cap[name] = tt.detach().float().cpu()
            else:
                cap[name] = tt.feats.detach().float().cpu()
        return hook

    wbb.input_layer.register_forward_hook(tok_hook("slat_input_layer"))
    for i, blk in enumerate(wbb.input_blocks):
        blk.conv1.register_forward_hook(tok_hook(f"slat_ib{i}_conv1"))
        blk.conv2.register_forward_hook(tok_hook(f"slat_ib{i}_conv2"))
        if blk.updown is not None:
            blk.updown.register_forward_hook(tok_hook(f"slat_ib{i}_updown"))
    for i in [0, 23]:
        wbb.blocks[i].register_forward_hook(tok_hook(f"slat_block{i}"))
    for i, blk in enumerate(wbb.out_blocks):
        if blk.updown is not None:
            blk.updown.register_forward_hook(tok_hook(f"slat_ob{i}_updown"))
        blk.conv1.register_forward_hook(tok_hook(f"slat_ob{i}_conv1"))
        blk.conv2.register_forward_hook(tok_hook(f"slat_ob{i}_conv2"))
    orig_pos = type(wbb.pos_embedder).forward
    def pos_fwd(self, xx, _f=orig_pos):
        e = _f(self, xx)
        cap["slat_ape"] = e.detach().float().cpu()
        return e
    wbb.pos_embedder.forward = pos_fwd.__get__(wbb.pos_embedder)

    # also capture the downsampled coords (needed by the C++ host tables)
    orig_updown = type(wbb.input_blocks[1].updown).forward
    def ud_fwd(self, st, _f=orig_updown):
        out = _f(self, st)
        cap["slat_down_coords"] = out.coords.detach().cpu()
        return out
    wbb.input_blocks[1].updown.forward = ud_fwd.__get__(wbb.input_blocks[1].updown)

    cap2 = {}
    with torch.no_grad():
        # fine-grained block0 hooks
        def tk(name):
            def hook(m, i, o):
                tt = o[0] if isinstance(o, tuple) else o
                tt = tt.feats if hasattr(tt, "feats") else tt
                cap2[name] = tt.detach().float().cpu()
            return hook
        blk0s = wbb.blocks[0]
        blk0s.adaLN_modulation.register_forward_hook(tk("b0_adaln"))
        blk0s.self_attn.register_forward_hook(tk("b0_attn_out"))
        blk0s.cross_attn.register_forward_hook(tk("b0_cross_out"))
        blk0s.mlp.register_forward_hook(tk("b0_mlp_out"))
        blk0s.self_attn.to_qkv.register_forward_hook(tk("b0_qkv"))
        blk0s.self_attn.q_rms_norm.register_forward_hook(tk("b0_qrms"))
        def qpre(m, args):
            tt = args[0]
            cap2["b0_qpre"] = (tt.feats if hasattr(tt, "feats") else tt).detach().float().cpu()
        blk0s.self_attn.q_rms_norm.register_forward_pre_hook(qpre)
        def atn_in(m, i, o):
            tt = i[0]
            cap2["b0_attn_in"] = (tt.feats if hasattr(tt, "feats") else tt).detach().float().cpu()
        blk0s.self_attn.register_forward_hook(tk("b0_attn_inx"))
        # keep the input capture too (pre-modulate from norm1)
        blk0s.norm1.register_forward_hook(atn_in)
        # capture the block0 INPUT (= skip1 + APE) via a forward hook
        b0 = wbb.blocks[0]
        ob = b0._forward
        def b0fwd(xx, mod, ctx2, _f=ob):
            cap2["in"] = xx.feats.detach().float().cpu()
            return _f(xx, mod, ctx2)
        b0._forward = b0fwd
        y = wbb(x, t, cond, coords, cfg=False)
        v = cap2["in"]
        write_samt_f32(os.path.join(args.out_dir, "slat_block_in.samt"),
                       list(v.shape[::-1]), v)
        for k2 in ["b0_adaln", "b0_qkv", "b0_qpre", "b0_qrms", "b0_attn_out", "b0_cross_out", "b0_mlp_out"]:
            if k2 in cap2:
                vv = cap2[k2]
                write_samt_f32(os.path.join(args.out_dir, f"slat_{k2}.samt"),
                               list(vv.shape[::-1]), vv)
                print(f"[ref] {k2}: {tuple(vv.shape)}")

    print(f"[ref] output: {y.shape}")
    for k in sorted(cap):
        v = cap[k]
        write_samt_f32(os.path.join(args.out_dir, f"{k}.samt"),
                       list(v.shape[::-1]), v)
        print(f"[ref] {k}: {tuple(v.shape)}")
    write_samt_f32(os.path.join(args.out_dir, "slat_v.samt"),
                   list(y.shape[::-1]), y[0].float().cpu())
    print(f"[ref] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
