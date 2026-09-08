#!/usr/bin/env python3
"""Compare native and official Gaussian observation PNG sequences exactly."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_sequence(directory: Path) -> list[np.ndarray]:
    paths = sorted(directory.glob("view_*.png"))
    if not paths:
        raise FileNotFoundError(f"no view_*.png files under {directory}")
    return [np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for path in paths]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-mae-u8", type=float, default=0.0)
    parser.add_argument("--max-abs-u8", type=int, default=0)
    args = parser.parse_args()

    native = load_sequence(args.native_dir)
    reference = load_sequence(args.reference_dir)
    if len(native) != len(reference):
        raise ValueError(f"view count differs: native={len(native)}, reference={len(reference)}")
    errors = []
    for index, (actual, expected) in enumerate(zip(native, reference)):
        if actual.shape != expected.shape:
            raise ValueError(f"view {index}: shape differs: {actual.shape} vs {expected.shape}")
        difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
        errors.append({
            "view": index,
            "mae_u8": float(difference.mean()),
            "max_abs_u8": int(difference.max()),
            "exact_rgb_pixels": int(np.all(actual == expected, axis=2).sum()),
            "pixel_count": int(actual.shape[0] * actual.shape[1]),
        })
    report = {
        "schema": "sam3d.native-gaussian-renderer.v1",
        "native_dir": str(args.native_dir.resolve()),
        "reference_dir": str(args.reference_dir.resolve()),
        "views": errors,
        "max_mae_u8": max(item["mae_u8"] for item in errors),
        "max_abs_u8": max(item["max_abs_u8"] for item in errors),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["max_mae_u8"] > args.max_mae_u8 or report["max_abs_u8"] > args.max_abs_u8:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
