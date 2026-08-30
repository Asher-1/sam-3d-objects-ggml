#!/usr/bin/env python3
"""Compare saved PyTorch and raw GGML SS trajectories step by step."""
import argparse
import json
import os
import struct

import numpy as np

MODS = ["6drotation_normalized", "scale", "shape", "translation",
        "translation_scale"]
OUT_INDEX = {name: index for index, name in enumerate(MODS)}


def read_samt(path):
    with open(path, "rb") as stream:
        if stream.read(4) != b"SAMT":
            raise ValueError(f"{path}: invalid SAMT")
        n_dims, = struct.unpack("<i", stream.read(4))
        shape = struct.unpack(f"<{n_dims}q", stream.read(8 * n_dims))
        dtype, = struct.unpack("<i", stream.read(4))
        if dtype != 0:
            raise ValueError(f"{path}: expected F32 SAMT")
        return shape, np.frombuffer(stream.read(), dtype="<f4").copy()


def stats(reference, actual):
    if reference.size != actual.size:
        raise ValueError(f"element count mismatch: {reference.size} vs {actual.size}")
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    return {"mae": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta * delta))),
            "max_abs": float(np.max(np.abs(delta))),
            "ref_absmean": float(np.mean(np.abs(reference)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-dir", required=True)
    parser.add_argument("--ggml-dir", required=True)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for step in range(1, args.steps + 1):
        ggml_x = os.path.join(args.ggml_dir, f"e2e_ss_state{step}_{{mod}}.samt")
        for mod in MODS:
            torch_path = os.path.join(args.torch_dir, f"ss_torch_x{step:03d}_{mod}.samt")
            if os.path.exists(ggml_x.format(mod=mod)):
                _, ref = read_samt(torch_path)
                _, actual = read_samt(ggml_x.format(mod=mod))
                rows.append({"step": step, "kind": "latent", "modality": mod,
                             **stats(ref, actual)})
            for branch in ("vc", "vu"):
                torch_path = os.path.join(args.torch_dir,
                                          f"ss_torch_{branch}{step:03d}_{mod}.samt")
                ggml_path = os.path.join(args.ggml_dir,
                                         f"e2e_ss_{branch}{step - 1}_{OUT_INDEX[mod]}.samt")
                if os.path.exists(torch_path) and os.path.exists(ggml_path):
                    _, ref = read_samt(torch_path)
                    _, actual = read_samt(ggml_path)
                    rows.append({"step": step, "kind": branch, "modality": mod,
                                 **stats(ref, actual)})
    if not rows:
        raise RuntimeError("no matching trajectory files")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)
        stream.write("\n")
    for row in rows:
        print(f"{row['kind']:6s} step={row['step']:02d} {row['modality']:24s} "
              f"mae={row['mae']:.7g} rmse={row['rmse']:.7g} max={row['max_abs']:.7g}")


if __name__ == "__main__":
    main()
