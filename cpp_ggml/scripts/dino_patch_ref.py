#!/usr/bin/env python3
"""Dump DINOv2 patch-embed GEMM references for the cpp_ggml dino debug.

Produces (in --out-dir):
  dino_img.samt          deterministic (H, W, C) input image, F32
  dino_patch_ref.samt    torch conv2d(stride=14) output, (n_patch, C) F32

The C++ side replays with:
  SAM3D_DEBUG_STAGE=post_patch sam3d-cli dino --model ss_generator-f16.gguf \
      --embedder cemb.emb0 --input dino_img.samt --out got.samt
and compare_nodes.py --ref/--got reports the stats.
"""
import argparse
import os
import struct
import sys

import numpy as np
import torch

SAMT_MAGIC = b"SAMT"
GGML_TYPE_F32 = 0


def write_samt_f32(path, ne, data):
    data = np.ascontiguousarray(data, dtype="<f4")
    with open(path, "wb") as f:
        f.write(SAMT_MAGIC)
        f.write(struct.pack("<i", len(ne)))
        f.write(struct.pack(f"<{len(ne)}q", *ne))
        f.write(struct.pack("<i", GGML_TYPE_F32))
        f.write(data.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/hf/ss_generator.ckpt")
    ap.add_argument("--embedder-index", type=int, default=0,
                    help="which DINO embedder of the condition fuser (emb0/emb1)")
    ap.add_argument("--out-dir", default="/tmp/dino_dbg")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    H = W = 518
    P = 14
    img = ((np.arange(H * W * 3, dtype=np.float32) * 7919) % 255) / 255.0
    img = img.reshape(H, W, 3)  # (H, W, C): C++ input ne=[W, H, C]
    write_samt_f32(os.path.join(args.out_dir, "dino_img.samt"), [W, H, 3], img)

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    prefix = f"_base_models.condition_embedder.embedder_list.{args.embedder_index}.0.backbone."
    kw = prefix + "patch_embed.proj.weight"
    kb = prefix + "patch_embed.proj.bias"
    if kw not in sd:
        # fall back: locate by suffix
        cand = [k for k in sd if k.endswith("patch_embed.proj.weight")]
        print("[ref] keys ending in patch_embed.proj.weight:", cand)
        kw = sorted(cand)[args.embedder_index]
        kb = kw.replace("weight", "bias")
    w = sd[kw].float()  # (OC, C, P, P)
    b = sd[kb].float()
    print(f"[ref] {kw}: {tuple(w.shape)}")

    x = torch.from_numpy(img).permute(2, 0, 1)[None]  # (1, 3, 518, 518)
    with torch.no_grad():
        out = torch.nn.functional.conv2d(x, w, b, stride=P)  # (1, OC, 37, 37)
    out = out[0].permute(1, 2, 0).reshape(-1, out.shape[1])  # (n_patch, C)
    ref = out.numpy()
    print(f"[ref] patch conv output: {ref.shape} "
          f"range=[{ref.min():.6f}, {ref.max():.6f}]")
    write_samt_f32(os.path.join(args.out_dir, "dino_patch_ref.samt"),
                   [ref.shape[1], ref.shape[0]], ref)
    print(f"[ref] wrote {args.out_dir}/dino_img.samt and dino_patch_ref.samt")


if __name__ == "__main__":
    main()
