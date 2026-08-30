#!/usr/bin/env python3
"""Dump DINOv2 forward intermediates as torch references for the cpp_ggml
dino graph debug. Mirrors every SAM3D_DEBUG_STAGE of sam3d-cli:

  post_pos post_embed b0_norm1 b0_q b0_scores b0_probs block0 final

Weights come straight from the SAM 3D checkpoint (the condition embedder's
DINO backbone == dinov2_vitl14_reg), no hub download needed.
"""
import argparse
import math
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F

SAMT_MAGIC = b"SAMT"


def write_samt_f32(path, ne, data):
    data = np.ascontiguousarray(data, dtype="<f4")
    with open(path, "wb") as f:
        f.write(SAMT_MAGIC)
        f.write(struct.pack("<i", len(ne)))
        f.write(struct.pack(f"<{len(ne)}q", *ne))
        f.write(struct.pack("<i", 0))
        f.write(data.tobytes())


class Dump:
    def __init__(self, d):
        self.d = d
        os.makedirs(d, exist_ok=True)

    def t(self, name, t):
        a = t.detach().float().cpu().numpy()
        write_samt_f32(os.path.join(self.d, name + ".samt"),
                       list(a.shape[::-1]), a)
        print(f"[ref] {name}: {a.shape} [{a.min():.4g}, {a.max():.4g}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/hf/ss_generator.ckpt")
    ap.add_argument("--embedder-index", type=int, default=0)
    ap.add_argument("--img", default="/tmp/dino_dbg/dino_img.samt")
    ap.add_argument("--out-dir", default="/tmp/dino_dbg/ref")
    ap.add_argument("--depth", type=int, default=2,
                    help="how many blocks to run (block0 only needs 1)")
    args = ap.parse_args()

    dmp = Dump(args.out_dir)

    # deterministic input, same as dino_patch_ref.py
    H = W = 518
    img = ((np.arange(H * W * 3, dtype=np.float32) * 7919) % 255) / 255.0
    img = img.reshape(H, W, 3)

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    cand = sorted(k for k in sd if k.endswith("patch_embed.proj.weight"))
    pre = cand[args.embedder_index].rsplit("patch_embed.proj.weight", 1)[0]
    print(f"[ref] backbone prefix: {pre}")

    def Wn(name):
        return sd[pre + name].float()

    # ---- DINOv2 preprocessing (Dino wrapper): resize + normalize ----
    x = torch.from_numpy(img).permute(2, 0, 1)[None]  # (1, 3, 518, 518)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    dmp.t("prep_img", x[0].permute(1, 2, 0).contiguous())  # HWC = C++ layout

    # ---- patch embed + cls + pos + registers ----
    w = Wn("patch_embed.proj.weight")   # (OC, C, P, P)
    b = Wn("patch_embed.proj.bias")
    patches = F.conv2d(x, w, b, stride=14)          # (1, C, 37, 37)
    P_n = patches.shape[2] * patches.shape[3]
    C = patches.shape[1]
    patches = patches.flatten(2).transpose(1, 2)     # (1, Np, C)
    cls = Wn("cls_token")                            # (1, 1, C)
    xx = torch.cat((cls.expand(1, -1, -1), patches), dim=1)  # (1, 1+Np, C)
    pos = Wn("pos_embed")                            # (1, 1+Np, C)
    xx = xx + pos
    dmp.t("post_pos", xx[0])          # (N, C) == C++ (C,N) ne=[C,N] memory
    reg = Wn("register_tokens")                      # (1, 4, C)
    xx = torch.cat((xx[:, :1], reg.expand(1, -1, -1), xx[:, 1:]), dim=1)
    N = xx.shape[1]
    dmp.t("post_embed", xx[0])        # (N, C) == C++ memory order

    # ---- blocks ----
    n_heads = 16
    D = C // n_heads
    scale = 1.0 / math.sqrt(D)
    for i in range(args.depth):
        blk = f"blocks.{i}."
        h = F.layer_norm(xx, (C,), Wn(blk + "norm1.weight"),
                         Wn(blk + "norm1.bias"), 1e-6)
        if i == 0:
            dmp.t("b0_norm1", h[0])
        qkv = F.linear(h, Wn(blk + "attn.qkv.weight"),
                       Wn(blk + "attn.qkv.bias"))    # (1, N, 3C)
        q, k, v = qkv.chunk(3, dim=2)                # each (1, N, C)
        if i == 0:
            dmp.t("b0_q", q[0])
        qh = q.view(1, N, n_heads, D).transpose(1, 2)   # (1, H, N, D)
        kh = k.view(1, N, n_heads, D).transpose(1, 2)
        vh = v.view(1, N, n_heads, D).transpose(1, 2)
        sc = torch.matmul(qh, kh.transpose(-1, -2)) * scale  # (1, H, N, N)
        if i == 0:
            # C++ out ne=[Nk, Nq, H]: memory key-fastest, H slowest ==
            # torch sc[0] (H, Nq, Nk) C-order
            write_samt_f32(os.path.join(dmp.d, "b0_scores.samt"),
                           [N, N, n_heads], sc[0].contiguous())
            pr = torch.softmax(sc, dim=-1)
            write_samt_f32(os.path.join(dmp.d, "b0_probs.samt"),
                           [N, N, n_heads], pr[0].contiguous())
            print(f"[ref] b0_scores: {(N, N, n_heads)}")
        att = torch.nn.functional.scaled_dot_product_attention(
            qh, kh, vh)  # (1, H, N, D)
        if i == 0:
            write_samt_f32(os.path.join(dmp.d, "b0_att.samt"),
                           [D, N, n_heads], att[0].contiguous())
            print(f"[ref] b0_att: {(D, N, n_heads)}")
        att = att.transpose(1, 2).reshape(1, N, C)
        att = F.linear(att, Wn(blk + "attn.proj.weight"),
                       Wn(blk + "attn.proj.bias"))
        att = att * Wn(blk + "ls1.gamma")
        xx = xx + att
        h = F.layer_norm(xx, (C,), Wn(blk + "norm2.weight"),
                         Wn(blk + "norm2.bias"), 1e-6)
        h = F.linear(h, Wn(blk + "mlp.fc1.weight"), Wn(blk + "mlp.fc1.bias"))
        h = F.gelu(h)  # exact erf gelu (timm default)
        h = F.linear(h, Wn(blk + "mlp.fc2.weight"), Wn(blk + "mlp.fc2.bias"))
        h = h * Wn(blk + "ls2.gamma")
        xx = xx + h
        if i == 0:
            dmp.t("block0", xx[0])

    # final LayerNorm (over all tokens) + drop registers -> (1370, C)
    xfin = F.layer_norm(xx, (C,), Wn("norm.weight"), Wn("norm.bias"), 1e-6)
    xfin = torch.cat((xfin[:, :1], xfin[:, 5:]), dim=1)  # drop 4 registers
    write_samt_f32(os.path.join(dmp.d, "final.samt"), [C, xfin.shape[1]],
                   xfin[0].contiguous())
    print(f"[ref] final: {(xfin.shape[1], C)}")

    print(f"[ref] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
