#!/usr/bin/env python3
"""Dump PointPatchEmbed (cemb.emb2) intermediates as torch references.

Input: the real preprocessed pointmap from dump_e2e_stages.py
       (benchmarks/data/e2e/ss_input_pointmap.samt, torch (1, 3, 518, 518)).
Weights: cemb.emb2.* of ss_generator.ckpt (PointPatchEmbed, 256/8, D=512).

Stages dumped (mirror the C++ debug stages):
  pp_resized     (256, 256, 3)  after nearest resize (token-major, HWC)
  pp_post_proj   (65536, 512)   point_proj over remapped points (pre-window)
  pp_post_block  (1024, 512)    per-window CLS tokens after the window block
  pp_final       (1024, 512)    + pos_embed_patch (the embedder output)
"""
import argparse
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F

SAMT_MAGIC = b"SAMT"


def load_samt(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"SAMT", path
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
    ap.add_argument("--ckpt", default="checkpoints/hf/ss_generator.ckpt")
    ap.add_argument("--embedder-index", type=int, default=2)
    ap.add_argument("--pointmap", default="cpp_ggml/benchmarks/data/e2e/ss_input_pointmap.samt")
    ap.add_argument("--out-dir", default="/tmp/pp_dbg")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # input pointmap (1, 3, 518, 518), torch CHW C-order memory (W fastest)
    ne, flat = load_samt(args.pointmap)
    H = W = 518
    pm = torch.from_numpy(flat.reshape(1, 3, H, W).copy())
    print(f"[ref] pointmap {tuple(pm.shape)} [{pm.min():.4g}, {pm.max():.4g}]")

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    cand = sorted(k for k in sd if k.endswith("point_proj.weight"))
    pre = cand[args.embedder_index - 2].rsplit("point_proj.weight", 1)[0] \
        if len(cand) > 2 else cand[0].rsplit("point_proj.weight", 1)[0]
    print(f"[ref] prefix: {pre}")

    def Wn(name):
        return sd[pre + name].float()

    D = Wn("point_proj.weight").shape[0]     # 512
    in_size = 256
    ps = 8
    n_w = in_size // ps                      # 32

    # ---- resize (nearest, align_corners=False: src = floor(s*(d+0.5)-0.5))
    xyz = F.interpolate(pm, size=in_size, mode="nearest")   # (1, 3, 256, 256)
    # per-dst source indices for the record (C++ replicates this table)
    scale = H / in_size
    src = [min(int(np.floor(scale * (d + 0.5) - 0.5)), H - 1) for d in range(in_size)]
    print(f"[ref] nearest src idx [0..5]: {src[:6]}, last: {src[-1]}")

    xyz_t = xyz[0].permute(1, 2, 0).contiguous()            # (256, 256, 3) HWC
    write_samt_f32(os.path.join(args.out_dir, "pp_resized.samt"),
                   [3, in_size, in_size], xyz_t)            # ne=[3, W, H]

    valid = xyz_t.isfinite().all(dim=-1)                    # (256, 256)
    print(f"[ref] valid: {int(valid.sum())}/{valid.numel()}")

    xyz_safe = xyz_t.clone()
    xyz_safe[~valid] = 0.0
    # remap 'linear' = identity
    x = F.linear(xyz_safe, Wn("point_proj.weight"), Wn("point_proj.bias"))
    x[~valid] = Wn("invalid_xyz_token")
    write_samt_f32(os.path.join(args.out_dir, "pp_post_proj.samt"),
                   [D, in_size * in_size], x.reshape(-1, D))
    print(f"[ref] pp_post_proj {tuple(x.reshape(-1, D).shape)}")

    # ---- windows: (1, 32, 8, 32, 8, D) -> (1024, 64, D) + cls + pos_window
    xw = x.view(1, n_w, ps, n_w, ps, D).permute(0, 1, 3, 2, 4, 5).contiguous()
    xw = xw.view(-1, ps * ps, D)                            # (1024, 64, D)
    cls = Wn("cls_token").expand(xw.shape[0], -1, -1)       # (1024, 1, D)
    toks = torch.cat((cls, xw), dim=1)                      # (1024, 65, D)
    toks = toks + Wn("pos_embed_window")                    # (1, 65, D)
    write_samt_f32(os.path.join(args.out_dir, "pp_post_posw.samt"),
                   [D, 1 + ps * ps, n_w * n_w], toks)       # win-major memory

    # timm Block: LN -> attn -> residual -> LN -> MLP(gelu erf) -> residual
    dumped_any = [False]
    def block(t):
        h = F.layer_norm(t, (D,), Wn("blocks.0.norm1.weight"),
                         Wn("blocks.0.norm1.bias"), 1e-6)
        if not dumped_any[0]:
            write_samt_f32(os.path.join(args.out_dir, "pp_norm1.samt"),
                           [D, 65, 1024], h.contiguous())  # (B,N,D) memory
        qkv = F.linear(h, Wn("blocks.0.attn.qkv.weight"),
                       Wn("blocks.0.attn.qkv.bias"))
        q, k, v = qkv.chunk(3, dim=-1)
        nh = Wn("blocks.0.num_heads") if "blocks.0.num_heads" in sd else torch.tensor(16)
        nh = 16
        Hd = D // nh
        q_flat = q.view(*q.shape[:-1], nh, Hd)   # (B, N, H, Hd) contiguous view
        qh = q_flat.transpose(-3, -2)   # (..., nh, 65, Hd)
        kh = k.view(*k.shape[:-1], nh, Hd).transpose(-3, -2)
        vh = v.view(*v.shape[:-1], nh, Hd).transpose(-3, -2)
        if not dumped_any[0]:
            write_samt_f32(os.path.join(args.out_dir, "pp_q.samt"),
                           [Hd, 65, 16, 1024], q_flat.contiguous())  # (B,N,H,D) memory
            sc_dbg = torch.matmul(qh, kh.transpose(-1, -2)) * (Hd ** -0.5)
            write_samt_f32(os.path.join(args.out_dir, "pp_scores.samt"),
                           [65, 65, 16, 1024], sc_dbg.contiguous())
        o = F.scaled_dot_product_attention(qh, kh, vh)
        if not dumped_any[0] or True:
            # (B, H, N, D) memory == C++ manual attention output layout
            if not dumped_any[0]:
                write_samt_f32(os.path.join(args.out_dir, "pp_att.samt"),
                               [Hd, 65, 16, 1024], o.contiguous())
                print(f"[ref] pp_att dumped {tuple(o.shape)}")
                dumped_any[0] = True
        o = o.transpose(-3, -2).reshape(*t.shape)
        o = F.linear(o, Wn("blocks.0.attn.proj.weight"),
                     Wn("blocks.0.attn.proj.bias"))
        t = t + o
        h = F.layer_norm(t, (D,), Wn("blocks.0.norm2.weight"),
                         Wn("blocks.0.norm2.bias"), 1e-6)
        h = F.linear(h, Wn("blocks.0.mlp.fc1.weight"), Wn("blocks.0.mlp.fc1.bias"))
        h = F.gelu(h)
        h = F.linear(h, Wn("blocks.0.mlp.fc2.weight"), Wn("blocks.0.mlp.fc2.bias"))
        return t + h

    out = block(toks)
    cls_out = out[:, 0].reshape(1, n_w * n_w, D)            # (1, 1024, D)
    write_samt_f32(os.path.join(args.out_dir, "pp_post_block.samt"),
                   [D, n_w * n_w], cls_out.reshape(n_w * n_w, D))

    pos = Wn("pos_embed")                                   # (1, D, 32, 32)
    pos = F.interpolate(pos, size=(n_w, n_w), mode="bilinear", align_corners=False)
    pos = pos.permute(0, 2, 3, 1).reshape(1, n_w * n_w, D)
    final = cls_out + pos
    write_samt_f32(os.path.join(args.out_dir, "pp_final.samt"),
                   [D, n_w * n_w], final.reshape(n_w * n_w, D))
    print(f"[ref] pp_final {tuple(final.reshape(-1, D).shape)} "
          f"[{final.min():.4g}, {final.max():.4g}]")
    print(f"[ref] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
