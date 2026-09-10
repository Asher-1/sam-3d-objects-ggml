#!/usr/bin/env python3
"""Dump SLat Gaussian decoder intermediates + representation as torch reference.

Inputs from dump_e2e_stages.py:
  slat_feats_final.samt  (8, N)    denormalized slat latent
  slat_coords.samt       (4, N) I32 active cells

Dumps (out-dir, prefix gs_):
  gs_input_layer   (768, N)   after F32 SparseLinear
  gs_ape           (768, N)   F32 absolute-position embedding
  gs_after_ape     (768, N)   F32 input_layer + APE
  gs_torso_input   (768, N)   after the official F16 torso boundary
  gs_b0            (768, N)   after block 0
  gs_b11           (768, N)   after block 11
  gs_raw           (448, N)   out_layer raw output
  gs_xyz / gs_features_dc / gs_scaling / gs_rotation / gs_opacity
                   representation values (to_representation)
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
        if dt in (26, 30, 1):
            data = np.frombuffer(f.read(), dtype="<i4")
        else:
            data = np.frombuffer(f.read(), dtype="<f4")
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
    ap.add_argument("--ckpt", default="checkpoints/hf/slat_decoder_gs.ckpt")
    ap.add_argument("--e2e-dir", default="cpp_ggml/benchmarks/data/e2e")
    ap.add_argument("--out-dir", default="/tmp/gs_ref")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    import sam3d_objects  # registers hydra targets
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    _, x = load_samt(f"{args.e2e_dir}/slat_feats_final.samt")
    _, coords = load_samt(f"{args.e2e_dir}/slat_coords.samt")
    N = len(x) // 8
    print(f"[ref] N={N}")

    yconf = OmegaConf.load("checkpoints/hf/slat_decoder_gs.yaml")
    dec = instantiate(yconf)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    dec.load_state_dict(sd, strict=True)
    dec = dec.to("cuda").eval()

    with torch.no_grad():
        xf = torch.from_numpy(x.reshape(N, 8).copy()).cuda()
        cc = torch.from_numpy(coords.reshape(N, 4).copy()).cuda()
        from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
        st = sp.SparseTensor(feats=xf, coords=cc)

        w = dec
        cap = {}

        def tk(name):
            def hook(mod, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                if isinstance(t, sp.SparseTensor):
                    t = t.feats
                cap[name] = t.detach().float().cpu()
            return hook

        def pre_tk(name):
            def hook(mod, inp):
                t = inp[0]
                if isinstance(t, sp.SparseTensor):
                    t = t.feats
                cap[name] = t.detach().float().cpu()
            return hook

        w.input_layer.register_forward_hook(tk("gs_input_layer_raw"))
        w.pos_embedder.register_forward_hook(tk("gs_ape"))
        # block hooks
        blk0 = w.blocks[0]
        blk0.norm1.register_forward_hook(tk("gs_b0_norm1"))
        blk0.attn.to_qkv.register_forward_hook(tk("gs_b0_qkv"))
        # Capture the attention result before ``to_out`` so the native
        # attention kernel can be evaluated independently from its projection.
        blk0.attn.to_out.register_forward_pre_hook(pre_tk("gs_b0_attn_values"))
        blk0.attn.register_forward_hook(tk("gs_b0_attn"))
        blk0.norm2.register_forward_hook(tk("gs_b0_norm2"))
        blk0.mlp.mlp[0].register_forward_hook(tk("gs_b0_mlp0"))
        blk0.mlp.mlp[1].register_forward_hook(tk("gs_b0_gelu"))
        blk0.mlp.mlp[2].register_forward_hook(tk("gs_b0_mlp2"))
        blk0.register_forward_hook(tk("gs_b0"))
        blk0.register_forward_pre_hook(pre_tk("gs_torso_input"))
        w.blocks[-1].register_forward_hook(tk("gs_b11"))
        w.blocks[1].register_forward_hook(tk("gs_b1"))

        rep = dec(st)[0]

        def flat(t):  # (N, C) feats -> SAMT with ne=(C, N): ggml memory has
            # the FIRST ne axis fastest, so a plain contiguous (N, C) tensor
            # (channel fastest) is exactly the (C, N) ggml layout - NO permute
            return t.detach().float().cpu().contiguous()

        # input_layer + ape combined (C++ stage "ape" = input_layer + ape)
        il = cap["gs_input_layer_raw"]
        ape = cap["gs_ape"]
        write_samt_f32(os.path.join(args.out_dir, "gs_input_layer.samt"),
                       [il.shape[-1], il.shape[0]], flat(il))
        write_samt_f32(os.path.join(args.out_dir, "gs_ape.samt"),
                       [ape.shape[-1], ape.shape[0]], flat(ape))
        write_samt_f32(os.path.join(args.out_dir, "gs_after_ape.samt"),
                       [il.shape[-1], il.shape[0]], flat(il + ape))
        torso_input = cap["gs_torso_input"]
        write_samt_f32(os.path.join(args.out_dir, "gs_torso_input.samt"),
                       [torso_input.shape[-1], torso_input.shape[0]], flat(torso_input))
        for k in ["gs_b0_norm1", "gs_b0_qkv", "gs_b0_attn_values", "gs_b0_attn", "gs_b0_norm2",
                  "gs_b0_mlp0", "gs_b0_gelu", "gs_b0_mlp2", "gs_b0", "gs_b1", "gs_b11"]:
            t = cap[k]
            write_samt_f32(os.path.join(args.out_dir, f"{k}.samt"),
                           [t.shape[-1], t.shape[0]], flat(t))

        # out_layer raw: hook the real module output (avoid recomposing)
        w.out_layer.register_forward_hook(tk("gs_rawh"))
        rep = dec(st)[0]  # rerun with the new hook registered
        rawh = cap["gs_rawh"]
        write_samt_f32(os.path.join(args.out_dir, "gs_raw.samt"),
                       [rawh.shape[-1], rawh.shape[0]], flat(rawh))

        # representation (float, same order as to_representation)
        xyz = rep._xyz.detach().float().cpu().reshape(-1, 3)          # (N*32, 3)
        fdc = rep._features_dc.detach().float().cpu().reshape(-1, 3)
        scl = rep._scaling.detach().float().cpu().reshape(-1, 3)
        rot = rep._rotation.detach().float().cpu().reshape(-1, 4)
        op = rep._opacity.detach().float().cpu().reshape(-1)
        write_samt_f32(os.path.join(args.out_dir, "gs_xyz.samt"), [3, len(xyz)], xyz)
        write_samt_f32(os.path.join(args.out_dir, "gs_features_dc.samt"), [3, len(fdc)], fdc)
        write_samt_f32(os.path.join(args.out_dir, "gs_scaling.samt"), [3, len(scl)], scl)
        write_samt_f32(os.path.join(args.out_dir, "gs_rotation.samt"), [4, len(rot)], rot)
        write_samt_f32(os.path.join(args.out_dir, "gs_opacity.samt"), [1, len(op)], op)
        for k in sorted(cap):
            print(f"[ref] hooked {k}")
        print(f"[ref] done -> {args.out_dir} (xyz {tuple(xyz.shape)})")


if __name__ == "__main__":
    main()
