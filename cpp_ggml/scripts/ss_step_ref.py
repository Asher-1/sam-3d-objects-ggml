#!/usr/bin/env python3
"""Dump a single ShortCut/MOT-DiT velocity evaluation as torch reference.

Inputs come from dump_e2e_stages.py artifacts:
  ss_x0_<modality>.samt   initial noise per modality (x_t at step 0)
  ss_cond_tokens.samt     fused condition tokens (1, 7528, 1024)

t = 0, d = 0 (the first Euler step, no shortcut). Dumps the 5 velocity
outputs (dict order: 6drotation_normalized, scale, shape, translation,
translation_scale) plus the CFG uncond branch for the strength-7 blending.
"""
import argparse
import os
import struct

import numpy as np
import torch

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


MODS = ["6drotation_normalized", "scale", "shape", "translation",
        "translation_scale"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/hf/ss_generator.ckpt")
    ap.add_argument("--yaml", default="checkpoints/hf/ss_generator.yaml")
    ap.add_argument("--e2e-dir", default="cpp_ggml/benchmarks/data/e2e")
    ap.add_argument("--out-dir", default="/tmp/ss_dbg")
    ap.add_argument("--t", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
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
    missing, unexpected = gen.load_state_dict(gen_sd, strict=False)
    print(f"[ref] generator loaded: missing={len(missing)} unexpected={len(unexpected)}")
    gen = gen.to(args.device).eval()
    # pipeline overrides (InferencePipeline defaults)
    gen.inference_steps = 25
    gen.rescale_t = 3
    gen.reverse_fn.strength = 7.0
    gen.reverse_fn.interval = [0, 500]
    gen.reverse_fn.unconditional_handling = "add_flag"
    gen.no_shortcut = True

    # inputs
    x_t = {}
    for m in MODS:
        ne, a = load_samt(os.path.join(args.e2e_dir, f"ss_x0_{m}.samt"))
        x_t[m] = torch.from_numpy(a.reshape(ne[::-1]).copy()).to(args.device)
    _, cond = load_samt(os.path.join(args.e2e_dir, "ss_cond_tokens.samt"))
    cond_t = torch.from_numpy(cond.reshape(1, -1, 1024).copy()).to(args.device)

    t = torch.tensor([args.t], device=args.device, dtype=torch.float32)
    d = torch.tensor([0.0], device=args.device, dtype=torch.float32)

    with torch.no_grad():
        # project_input intermediates
        wbb = gen.reverse_fn.backbone
        ld_in = {}
        for name in wbb.input_latent_mappings:
            ld_in[name] = wbb.latent_mapping[name].to_input(x_t[name])
        merged = wbb.merge_latent_share_transformer(ld_in)
        write_samt_f32(os.path.join(args.out_dir, "ss_proj_shape.samt"),
                       [1024, merged["shape"].shape[1]], merged["shape"][0].float().cpu())
        write_samt_f32(os.path.join(args.out_dir, "ss_proj_pose.samt"),
                       [1024, merged["6drotation_normalized"].shape[1]],
                       merged["6drotation_normalized"][0].float().cpu())
        print(f"[ref] proj_in dumped: {merged['shape'].shape} {merged['6drotation_normalized'].shape}")
        # hook block0 outputs + fine-grained internals (bisect the block)
        cap = {}
        blk0 = gen.reverse_fn.backbone.blocks[0]
        orig_fwd = blk0.forward

        def cap_fwd(x, mod, context, _f=orig_fwd):
            out = _f(x, mod, context)
            for k, v in out.items():
                cap[k] = v.detach().float().cpu()
            return out

        blk0.forward = cap_fwd

        def tok_hook(name):
            def hook(mod, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                if isinstance(t, dict):
                    for k, v in t.items():
                        cap[f"{name}_{k}"] = v.detach().float().cpu()
                else:
                    cap[name] = t.detach().float().cpu()
            return hook

        blk0.adaLN_modulation.register_forward_hook(tok_hook("b0_adaln"))
        blk0.self_attn.register_forward_hook(tok_hook("b0_attn"))
        for mn in ["shape", "6drotation_normalized"]:
            blk0.norm2[mn].register_forward_hook(tok_hook(f"b0_n2_{mn}"))
            blk0.cross_attn[mn].register_forward_hook(tok_hook(f"b0_x_{mn}"))
            blk0.mlp[mn].register_forward_hook(tok_hook(f"b0_mlp_{mn}"))
            # cross internals: q (B, L, C); kv (B, Lkv, 2C) -> split k/v
            blk0.cross_attn[mn].to_q.register_forward_hook(tok_hook(f"b0_xq_{mn}"))

            def kv_hook(mod, inp, out, _mn=mn):
                cap[f"b0_xk_{_mn}"] = out[..., :out.shape[-1] // 2].detach().float().cpu()
                cap[f"b0_xv_{_mn}"] = out[..., out.shape[-1] // 2:].detach().float().cpu()

            blk0.cross_attn[mn].to_kv.register_forward_hook(kv_hook)
        # attention internals: wrap mm_scale_dot_product_attention to capture
        # post-rms q/k (torch (B, L, H, D) == C++ (Hd, N, H) memory)
        orig_mm = blk0.self_attn.mm_scale_dot_product_attention

        def mm_hook(q, k, v, _f=orig_mm):
            cap["b0_q_shape"] = q["shape"].detach().float().cpu()
            cap["b0_k_shape"] = k["shape"].detach().float().cpu()
            cap["b0_q_pose"] = q["6drotation_normalized"].detach().float().cpu()
            return _f(q, k, v)

        blk0.self_attn.mm_scale_dot_product_attention = mm_hook
        # The CFG wrapper (strength=7 > 0 at t=0) runs the backbone TWICE
        # (cond + uncond via force_zeros_cond) and blends, which would
        # overwrite the hooks with uncond-branch values. Call the wrapper
        # directly instead: one backbone call per invocation, and the hooks
        # then hold exactly the branch just run.
        wbb = gen.reverse_fn.backbone
        y_cond = wbb(x_t, t, cond_t, d=d, cfg=False)
        cap_cond = {k: v.clone() for k, v in cap.items()}
        y_uncond = wbb(x_t, t, cond_t, d=d, cfg=True)
        cap = cap_cond
        # verify the host-side CFG blend reproduces the wrapper's output
        y_ref = gen.reverse_fn(x_t, t, cond_t, d=d, p_unconditional=0.0,
                               cfg=False)
        blend = {m: (1 + 7.0) * y_cond[m] - 7.0 * y_uncond[m] for m in MODS}
        err = max((blend[m] - y_ref[m]).abs().max().item() for m in MODS)
        print(f"[ref] CFG blend check (1+7)*cond-7*uncond vs reverse_fn: max|d|={err:.4g}")
        for m in MODS:
            write_samt_f32(os.path.join(args.out_dir, f"ss_yref_{m}.samt"),
                           list(x_t[m].shape), y_ref[m][0].float().cpu())
        write_samt_f32(os.path.join(args.out_dir, "ss_b0_shape.samt"),
                       [1024, 4096], cap["shape"][0].cpu())
        write_samt_f32(os.path.join(args.out_dir, "ss_b0_pose.samt"),
                       [1024, 4], cap["6drotation_normalized"][0].cpu())
        for k in ["b0_adaln", "b0_attn_shape", "b0_attn_6drotation_normalized",
                  "b0_n2_shape", "b0_n2_6drotation_normalized",
                  "b0_x_shape", "b0_x_6drotation_normalized",
                  "b0_mlp_shape", "b0_mlp_6drotation_normalized",
                  "b0_q_shape", "b0_k_shape", "b0_q_pose",
                  "b0_xq_shape", "b0_xk_shape", "b0_xv_shape",
                  "b0_xq_6drotation_normalized", "b0_xk_6drotation_normalized",
                  "b0_xv_6drotation_normalized"]:
            if k not in cap:
                print(f"[ref] MISSING {k}")
                continue
            v = cap[k]
            # (B, L, C) token-major -> ggml (C, N); (B, L, H, D) -> ggml (D, H, L)
            ne = list(v.shape[1:][::-1])
            write_samt_f32(os.path.join(args.out_dir, f"ss_{k}.samt"), ne, v[0].cpu())
        print(f"[ref] block0 internals dumped")

    for m in MODS:
        write_samt_f32(os.path.join(args.out_dir, f"ss_v_{m}.samt"),
                       list(x_t[m].shape), y_cond[m][0].float().cpu())
        write_samt_f32(os.path.join(args.out_dir, f"ss_vu_{m}.samt"),
                       list(x_t[m].shape), y_uncond[m][0].float().cpu())
        print(f"[ref] {m}: [{y_cond[m].min():.4g}, {y_cond[m].max():.4g}]")
    print(f"[ref] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
