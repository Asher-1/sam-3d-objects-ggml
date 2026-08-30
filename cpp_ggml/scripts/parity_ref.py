#!/usr/bin/env python3
# Generate reference tensors from the official PyTorch models for parity
# checks against the cpp_ggml C++ engine.
#
#   python3 parity_ref.py ss_decoder --ckpt checkpoints/hf \
#       --input /tmp/ss_latent.bin --output /tmp/ss_ref.bin [--device cuda]
#
# The input binary uses the SAMT format written by the C++ side (magic SAMT).
import argparse
import struct
import sys

import numpy as np
import torch
import yaml
from hydra.utils import instantiate


def load_samt(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, nd = struct.unpack("<Ii", data[:8])
    assert magic == 0x544D4153, f"bad magic {magic:#x}"
    ne = struct.unpack(f"<{nd}q", data[8:8 + 8 * nd])
    (typ,) = struct.unpack("<i", data[8 + 8 * nd:12 + 8 * nd])
    payload = data[12 + 8 * nd:]
    n = int(np.prod(ne))
    if typ == 0:  # F32
        arr = np.frombuffer(payload, dtype="<f4", count=n)
    elif typ == 1:  # F16
        arr = np.frombuffer(payload, dtype="<f2", count=n).astype(np.float32)
    else:
        raise RuntimeError(f"unsupported SAMT type {typ}")
    return arr.reshape(tuple(reversed(ne)))  # ggml ne -> torch shape


def save_samt(path, tensor: torch.Tensor):
    t = tensor.detach().float().cpu().contiguous()
    ne = [t.shape[-1], t.shape[-2]] + list(t.shape[:-2][::-1])
    with open(path, "wb") as f:
        f.write(struct.pack("<Ii", 0x544D4153, t.dim()))
        f.write(struct.pack(f"<{t.dim()}q", *ne))
        f.write(struct.pack("<i", 0))
        f.write(t.numpy().tobytes())


def build_ss_decoder(ckpt_dir, device):
    sys.path.insert(0, ".")
    import sam3d_objects  # noqa
    conf = yaml.safe_load(open(f"{ckpt_dir}/ss_decoder.yaml"))
    model = instantiate(conf)
    sd = torch.load(f"{ckpt_dir}/ss_decoder.ckpt", map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["ss_decoder"])
    ap.add_argument("--ckpt", default="checkpoints/hf")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    assert args.model == "ss_decoder"
    model = build_ss_decoder(args.ckpt, args.device)
    latent = load_samt(args.input)
    print(f"input latent {latent.shape} {latent.dtype}")
    with torch.no_grad():
        out = model(torch.from_numpy(latent).to(args.device))
    print(f"output {out.shape} min {out.min().item():.6f} max {out.max().item():.6f}")
    save_samt(args.output, out[0])
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
