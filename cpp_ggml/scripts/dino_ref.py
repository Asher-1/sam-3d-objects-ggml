#!/usr/bin/env python3
# DINOv2 reference for cpp_ggml parity checks.
#
#   # generate input + reference from a real image:
#   python3 dino_ref.py generate --image notebook/images/human_object/image.png \
#       --ckpt checkpoints/hf/ss_generator.ckpt --embedder 0 \
#       --out-img /tmp/dino_img.samt --out-ref /tmp/dino_ref.samt
#   # or compare an existing C++ dump:
#   python3 dino_ref.py compare --ref /tmp/dino_ref.samt --got /tmp/dino_got.samt
import argparse
import os
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

DINO_PREFIX = "_base_models.condition_embedder.module_list.{i}.backbone."


def load_samt(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, nd = struct.unpack("<Ii", data[:8])
    ne = struct.unpack(f"<{nd}q", data[8:8 + 8 * nd])
    (typ,) = struct.unpack("<i", data[8 + 8 * nd:12 + 8 * nd])
    payload = data[12 + 8 * nd:]
    n = int(np.prod(ne))
    if typ == 0:
        arr = np.frombuffer(payload, dtype="<f4", count=n)
    else:
        arr = np.frombuffer(payload, dtype="<f2", count=n).astype(np.float32)
    return ne, arr.reshape(tuple(reversed(ne)))


def save_samt(path, t):
    t = t.detach().float().cpu().contiguous()
    ne = [t.shape[-1], t.shape[-2]] + list(t.shape[:-2][::-1])
    with open(path, "wb") as f:
        f.write(struct.pack("<Ii", 0x544D4153, t.dim()))
        f.write(struct.pack(f"<{t.dim()}q", *ne))
        f.write(struct.pack("<i", 0))
        f.write(t.numpy().tobytes())


class Block(nn.Module):
    """DINOv2 transformer block: x += ls1*attn(norm1(x)); x += ls2*mlp(norm2(x))."""

    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.ls1 = nn.Parameter(torch.ones(dim))
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, dim * 4, bias=True)
        self.fc2 = nn.Linear(dim * 4, dim, bias=True)
        self.ls2 = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        B, N, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, H, N, D)
        att = F.scaled_dot_product_attention(q, k, v)  # scale 1/sqrt(D)
        att = att.transpose(1, 2).reshape(B, N, C)
        x = x + self.ls1 * self.proj(att)
        h = self.norm2(x)
        h = self.fc2(F.gelu(self.fc1(h)))
        return x + self.ls2 * h


class DinoVit(nn.Module):
    def __init__(self, sd):
        super().__init__()
        dim = sd["patch_embed.proj.weight"].shape[0]
        n_reg = sd["register_tokens"].shape[1]
        ps = sd["patch_embed.proj.weight"].shape[-1]
        n_tok = sd["pos_embed"].shape[1]
        self.n_patch = n_tok - 1 - n_reg
        self.norm1 = None
        self.cls_token = nn.Parameter(sd["cls_token"].clone())
        self.register_tokens = nn.Parameter(sd["register_tokens"].clone())
        self.pos_embed = nn.Parameter(sd["pos_embed"].clone())
        self.blocks = nn.ModuleList()
        i = 0
        while f"blocks.{i}.norm1.weight" in sd:
            b = Block(dim, 16)
            b.load_state_dict({
                "norm1.weight": sd[f"blocks.{i}.norm1.weight"],
                "norm1.bias": sd[f"blocks.{i}.norm1.bias"],
                "qkv.weight": sd[f"blocks.{i}.attn.qkv.weight"],
                "qkv.bias": sd[f"blocks.{i}.attn.qkv.bias"],
                "proj.weight": sd[f"blocks.{i}.attn.proj.weight"],
                "proj.bias": sd[f"blocks.{i}.attn.proj.bias"],
                "ls1": sd[f"blocks.{i}.ls1.gamma"],
                "norm2.weight": sd[f"blocks.{i}.norm2.weight"],
                "norm2.bias": sd[f"blocks.{i}.norm2.bias"],
                "fc1.weight": sd[f"blocks.{i}.mlp.fc1.weight"],
                "fc1.bias": sd[f"blocks.{i}.mlp.fc1.bias"],
                "fc2.weight": sd[f"blocks.{i}.mlp.fc2.weight"],
                "fc2.bias": sd[f"blocks.{i}.mlp.fc2.bias"],
                "ls2": sd[f"blocks.{i}.ls2.gamma"],
            })
            self.blocks.append(b)
            i += 1
        self.depth = i
        self.patch_proj = nn.Parameter(sd["patch_embed.proj.weight"].clone())
        self.patch_bias = nn.Parameter(sd["patch_embed.proj.bias"].clone())
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.norm.weight = nn.Parameter(sd["norm.weight"].clone())
        self.norm.bias = nn.Parameter(sd["norm.bias"].clone())

    def forward(self, x, stages=None):  # x (B, 3, H, W) preprocessed
        B = x.shape[0]
        p = self.patch_proj.shape[-1]
        patches = F.conv2d(x, self.patch_proj, self.patch_bias, stride=p)
        patches = patches.flatten(2).transpose(1, 2)  # (B, n_patch, C)
        # DINOv2-reg: pos_embed covers [cls, patches] only; registers are
        # inserted AFTER the positional encoding (no pos of their own).
        x = torch.cat([self.cls_token.expand(B, -1, -1), patches], dim=1)
        x = x + self.pos_embed
        x = torch.cat([x[:, :1],
                       self.register_tokens.expand(B, -1, -1), x[:, 1:]], dim=1)
        if stages is not None:
            stages["post_embed"] = x.clone()
        for i, b in enumerate(self.blocks):
            x = b(x)
            if stages is not None:
                stages[f"block{i}"] = x.clone()
        x = self.norm(x)
        cls, patches = x[:, :1], x[:, 1 + self.register_tokens.shape[1]:]
        return torch.cat([cls, patches], dim=1)  # (B, 1+n_patch, C)


def get_sd(ckpt, embedder):
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    pref = DINO_PREFIX.format(i=embedder)
    out = {}
    for k, v in sd.items():
        if k.startswith(pref):
            out[k[len(pref):]] = v
    assert out, f"no keys with prefix {pref}"
    return out


def preprocess(img_path, size=518):
    img = Image.open(img_path).convert("RGB").resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (x - mean) / std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "compare"])
    ap.add_argument("--image")
    ap.add_argument("--ckpt", default="checkpoints/hf/ss_generator.ckpt")
    ap.add_argument("--embedder", type=int, default=0)
    ap.add_argument("--out-img", default="/tmp/dino_img.samt")
    ap.add_argument("--out-ref", default="/tmp/dino_ref.samt")
    ap.add_argument("--ref")
    ap.add_argument("--got")
    args = ap.parse_args()

    if args.cmd == "compare":
        # ref is torch (N, C) row-major; got is ggml canonical (C, N)
        # (flat = c + C*n). Reshape both to logical [n][c] before diffing.
        ne_r, ref = load_samt(args.ref)
        ne_g, got = load_samt(args.got)
        C = ne_g[0]
        N = got.size // C
        refL = ref.reshape(N, C)
        gotL = got.reshape(C, N).T
        d = np.abs(refL - gotL)
        n_nans = int(np.isnan(d).sum())
        print(f"got ne={ne_g}  max|d|={np.nanmax(d):.6f}  mean|d|={np.nanmean(d):.6f}"
              f"  nan={n_nans}/{d.size}")
        return

    sd = get_sd(args.ckpt, args.embedder)
    vit = DinoVit(sd).eval()
    x = preprocess(args.image)[None]  # (1, 3, 518, 518)
    stages = {}
    with torch.no_grad():
        tokens = vit(x, stages)
    print(f"tokens {tuple(tokens.shape)} min {tokens.min():.4f} max {tokens.max():.4f}")
    os.makedirs("/tmp/dino_ref_stages", exist_ok=True)
    for name, t in stages.items():
        save_samt(f"/tmp/dino_ref_stages/{name}.samt", t[0])
    # C++ input: canonical (W, H, 3) = torch (3, H, W) permute(1,2,0)
    # channel-first torch (3, H, W) is exactly the ggml canonical (W, H, C)
    # flat layout, so it round-trips through SAMT untouched.
    save_samt(args.out_img, x[0])
    save_samt(args.out_ref, tokens[0])  # (N, C) torch = ne {C, N}


if __name__ == "__main__":
    main()
