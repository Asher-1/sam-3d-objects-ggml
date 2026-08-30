#!/usr/bin/env python3
# Dump SparseStructureDecoder intermediate tensors for node-by-node parity
# checks against the C++ engine (SAM3D_DEBUG_NODE=i dumps the same nodes).
#
#   python3 debug_ref.py --ckpt checkpoints/hf --input /tmp/ss_latent.bin \
#       --out-dir /tmp/ssref [--device cuda]
#
# Node order mirrors SsDecoderGraph::build:
#   0 input_layer, 1..2 middle_block.i, 3..10 blocks.i, 11 out_layer
import argparse
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F


def load_samt(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, nd = struct.unpack("<Ii", data[:8])
    assert magic == 0x544D4153, f"bad magic {magic:#x}"
    ne = struct.unpack(f"<{nd}q", data[8:8 + 8 * nd])
    (typ,) = struct.unpack("<i", data[8 + 8 * nd:12 + 8 * nd])
    payload = data[12 + 8 * nd:]
    n = int(np.prod(ne))
    if typ == 0:
        arr = np.frombuffer(payload, dtype="<f4", count=n)
    elif typ == 1:
        arr = np.frombuffer(payload, dtype="<f2", count=n).astype(np.float32)
    else:
        raise RuntimeError(f"unsupported SAMT type {typ}")
    return arr.reshape(tuple(reversed(ne)))


def save_samt(path, t: torch.Tensor):
    t = t.detach().float().cpu().contiguous()
    ne = [t.shape[-1], t.shape[-2]] + list(t.shape[:-2][::-1])
    with open(path, "wb") as f:
        f.write(struct.pack("<Ii", 0x544D4153, t.dim()))
        f.write(struct.pack(f"<{t.dim()}q", *ne))
        f.write(struct.pack("<i", 0))
        f.write(t.numpy().tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/hf")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    sd = torch.load(f"{args.ckpt}/ss_decoder.ckpt", map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]

    latent = load_samt(args.input)
    print(f"input latent {latent.shape}")
    nodes = []

    # This intentionally replays the exact public SparseStructureDecoder
    # algebra rather than importing the package-level model registry.  That
    # registry imports optional Gaussian/mesh dependencies (Kaolin) which are
    # unrelated to this dense Conv3D decoder and made a basic parity probe
    # depend on the complete rendering environment.
    def conv(name, x):
        return F.conv3d(x, sd[f"{name}.weight"].float().to(args.device),
                        sd[f"{name}.bias"].float().to(args.device), padding=1)

    def channel_norm(name, x):
        w = sd[f"{name}.weight"].float().to(args.device)
        b = sd[f"{name}.bias"].float().to(args.device)
        return F.layer_norm(x.permute(0, 2, 3, 4, 1).contiguous(), (x.shape[1],),
                            w, b, 1e-5).permute(0, 4, 1, 2, 3).contiguous()

    def residual(name, x):
        h = F.silu(channel_norm(f"{name}.norm1", x))
        h = conv(f"{name}.conv1", h)
        h = F.silu(channel_norm(f"{name}.norm2", h))
        return conv(f"{name}.conv2", h) + x

    def pixel_shuffle_3d(x):
        # Same reshape/permute ordering as modules/spatial.py.
        batch, channels, d, h, w = x.shape
        x = x.reshape(batch, channels // 8, 2, 2, 2, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
        return x.reshape(batch, channels // 8, d * 2, h * 2, w * 2)

    def upsample(name, x):
        return pixel_shuffle_3d(conv(f"{name}.conv", x))

    with torch.no_grad():
        h = conv("input_layer", torch.from_numpy(latent.copy()).to(args.device))
        nodes.append(("input_layer", h[0]))
        for i in range(2):
            h = residual(f"middle_block.{i}", h)
            nodes.append((f"middle_block.{i}", h[0]))
        for i in range(8):
            name = f"blocks.{i}"
            h = upsample(name, h) if f"{name}.conv.weight" in sd else residual(name, h)
            nodes.append((name, h[0]))
        h = F.silu(channel_norm("out_layer.0", h))
        out = conv("out_layer.2", h)
        nodes.append(("out_layer", out[0]))

    os.makedirs(args.out_dir, exist_ok=True)
    for i, (name, t) in enumerate(nodes):
        p = f"{args.out_dir}/node_{i}.samt"
        save_samt(p, t)
        print(f"node {i:2d} {name:16s} shape {tuple(t.shape)} "
              f"min {t.min().item():.4f} max {t.max().item():.4f} -> {p}")


if __name__ == "__main__":
    main()
