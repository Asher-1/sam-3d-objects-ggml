#!/usr/bin/env python3
# Compare C++ per-node dumps (SAM3D_DEBUG_NODE=i --out node_i.bin) against the
# torch reference dumps from debug_ref.py. The --tensor mode compares arbitrary
# SAMT files, including sparse-coordinate outputs from a full E2E run.
#
#   python3 compare_nodes.py --ref /tmp/ssref --got /tmp/ssgot [--nodes 12]
import argparse
import struct

import numpy as np


def load_samt_flat(path):
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
    elif typ in (26, 30):
        # Older dump_e2e_stages.py emitted the then-upstream I32 enum (30),
        # while the vendored ggml revision uses 26. Both payloads are I32.
        arr = np.frombuffer(payload, dtype="<i4", count=n)
    else:
        raise RuntimeError(f"unsupported SAMT type {typ}")
    return ne, typ, arr


def compare_tensors(reference, actual):
    ne_ref, typ_ref, ref = load_samt_flat(reference)
    ne_actual, typ_actual, actual = load_samt_flat(actual)
    print(f"reference: ne={list(ne_ref)} type={typ_ref} elements={ref.size}")
    print(f"actual:    ne={list(ne_actual)} type={typ_actual} elements={actual.size}")
    if ref.size == actual.size and ref.size == 0:
        print("both tensors are empty; scalar error metrics are undefined")
    elif ref.size == actual.size:
        delta = np.abs(ref.astype(np.float64) - actual.astype(np.float64))
        print(f"max|d|={delta.max():.9g} mean|d|={delta.mean():.9g} "
              f"rmse={np.sqrt(np.mean(delta * delta)):.9g}")
    else:
        print("element counts differ; scalar tensor statistics are not comparable")

    if typ_ref in (26, 30) and typ_actual in (26, 30) and ne_ref[0] == ne_actual[0] == 4:
        ref_coords = {tuple(row) for row in ref.reshape(-1, 4)}
        actual_coords = {tuple(row) for row in actual.reshape(-1, 4)}
        shared = len(ref_coords & actual_coords)
        union = len(ref_coords | actual_coords)
        jaccard = f"{shared / union:.9f}" if union else "undefined (both empty)"
        print(f"coords: reference={len(ref_coords)} actual={len(actual_coords)} "
              f"shared={shared} only_reference={len(ref_coords - actual_coords)} "
              f"only_actual={len(actual_coords - ref_coords)} "
              f"jaccard={jaccard}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--got", required=True)
    ap.add_argument("--nodes", type=int, default=12)
    ap.add_argument("--tensor", action="store_true",
                    help="compare --ref and --got as individual SAMT files")
    args = ap.parse_args()

    if args.tensor:
        compare_tensors(args.ref, args.got)
        return

    for i in range(args.nodes):
        try:
            ne_r, _, a = load_samt_flat(f"{args.ref}/node_{i}.samt")
            ne_g, _, b = load_samt_flat(f"{args.got}/node_{i}.samt")
        except FileNotFoundError as e:
            print(f"node {i:2d}: missing ({e})")
            continue
        if a.size != b.size:
            print(f"node {i:2d}: SIZE MISMATCH ref ne={ne_r} ({a.size}) "
                  f"got ne={ne_g} ({b.size})")
            continue
        if ne_r != ne_g and list(ne_r[:-1]) != list(ne_g):
            print(f"node {i:2d}: WARN ne ref={ne_r} got={ne_g} (same size)")
        d = np.abs(a - b)
        rel = d.max() / max(1e-9, np.abs(a).max())
        ok = "OK " if rel < 2e-5 else "DIFF"
        print(f"node {i:2d} {ok} ne={list(ne_r)} "
              f"max|d|={d.max():.6f} mean|d|={d.mean():.6f} "
              f"ref[min {a.min():.4f} max {a.max():.4f}] "
              f"got[min {b.min():.4f} max {b.max():.4f}]")


if __name__ == "__main__":
    main()
