#!/usr/bin/env python3
"""Compare sam3d-cli dino SAM3D_DEBUG_STAGE dumps against torch references."""
import struct
import sys

import numpy as np


def load_samt(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"SAMT", path
        nd = struct.unpack("<i", f.read(4))[0]
        ne = struct.unpack(f"<{nd}q", f.read(8 * nd))
        dt = struct.unpack("<i", f.read(4))[0]
        dtype = "<f4" if dt == 0 else "<i4"
        data = np.frombuffer(f.read(), dtype=dtype)
    return ne, data


def cmp(name, ref_path, got_path, swap_qk=False):
    ne_r, ref = load_samt(ref_path)
    ne_g, got = load_samt(got_path)
    if ne_r != ne_g:
        print(f"{name:12s} SHAPE MISMATCH ref={ne_r} got={ne_g}")
        return
    if swap_qk:
        N = ne_r[0]
        H = ne_r[2] if len(ne_r) > 2 else 1
        ref = ref.reshape(N, N, H).transpose(0, 1, 2).reshape(-1)
        # C++ (key, query, H) vs torch (query, key, H): swap axes 0/1
        ref3 = ref.reshape(N, N, H)
        ref3 = np.ascontiguousarray(ref3.transpose(1, 0, 2))
        ref = ref3.reshape(-1)
    d = np.abs(ref - got)
    den = max(float(np.abs(ref).max()), 1e-9)
    print(f"{name:12s} ne={ne_r} max|d|={d.max():.6g} rel={d.max()/den:.2e} "
          f"mean|d|={d.mean():.3g} nan={np.isnan(got).sum()}")


def main():
    ref_dir, got_dir = sys.argv[1], sys.argv[2]
    dtype = sys.argv[3] if len(sys.argv) > 3 else "f32"
    for stage in ["post_pos", "post_embed", "b0_norm1", "b0_q"]:
        cmp(stage, f"{ref_dir}/{stage}.samt", f"{got_dir}/{stage}_{dtype}.samt")
    for stage in ["b0_scores", "b0_probs", "b0_att"]:
        cmp(stage, f"{ref_dir}/{stage}.samt", f"{got_dir}/{stage}_{dtype}.samt")
    cmp("block0", f"{ref_dir}/block0.samt", f"{got_dir}/block0_{dtype}.samt")


if __name__ == "__main__":
    main()
