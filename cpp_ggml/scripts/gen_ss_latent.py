#!/usr/bin/env python3
"""Generate a synthetic SS-decoder latent input in SAMT format.

The official SAM 3D Objects checkpoints are gated on HuggingFace, so the
parity inputs dumped from the PyTorch pipeline (e.g. /tmp/ss_latent.bin used
by benchmarks/ss_decoder_parity.md) cannot be regenerated here. This script
creates a deterministic stand-in: a fixed-seed normal latent of the same
shape (8 x 16^3, fp32, ne = [16, 16, 16, 8]) that exercises the identical
graph geometry.

SAMT binary layout (see save/load_raw_tensor in cpp_ggml/src/common.cpp):
    magic "SAMT" | int32 ndims | int64 ne[ndims] | int32 type | data

Usage:
    python3 scripts/gen_ss_latent.py [--out PATH] [--seed N] [--scale S]

`--scale` multiplies the latent; larger scales push the decoder logits past
the 0 decision boundary so the occupancy render in plot_benchmarks.py has a
non-empty voxel field (random latents near the data manifold decode to an
all-negative field, matching the -115..-29 range recorded in
benchmarks/ss_decoder_parity.md).
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

SAMT_MAGIC = b"SAMT"
GGML_TYPE_F32 = 0


def write_samt_f32(path: Path, ne: list[int], data: np.ndarray) -> None:
    data = np.ascontiguousarray(data, dtype="<f4")
    with open(path, "wb") as f:
        f.write(SAMT_MAGIC)
        f.write(struct.pack("<i", len(ne)))
        f.write(struct.pack(f"<{len(ne)}q", *ne))  # ne is int64 on disk
        f.write(struct.pack("<i", GGML_TYPE_F32))
        f.write(data.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent /
                                         "benchmarks" / "data" / "ss_latent_synthetic.bin"))
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--scale", type=float, default=3.0)
    args = ap.parse_args()

    # (C, D, H, W) contiguous == SAMT ne [W, H, D, C] with ne[0] fastest
    latent = np.random.default_rng(args.seed).standard_normal((8, 16, 16, 16))
    latent = (latent * args.scale).astype(np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_samt_f32(out, [16, 16, 16, 8], latent)
    print(f"wrote {out}: ne=[16,16,16,8] f32 seed={args.seed} scale={args.scale} "
          f"range=[{latent.min():.3f}, {latent.max():.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
