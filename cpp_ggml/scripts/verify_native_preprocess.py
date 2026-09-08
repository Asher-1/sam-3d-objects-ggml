#!/usr/bin/env python3
"""Compare native SS preprocessing tensors with an official stage fixture.

The native runtime consumes the six ``ss_input_*`` tensors directly.  This
tool treats their source fixture as an external reference and reports every
elementary preprocessing boundary, including NaN support, rather than hiding
preprocessing differences inside the condition embedder output.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


SAMT_MAGIC = b"SAMT"
GGML_TYPE_F32 = 0
FIELDS = (
    "ss_input_image.samt",
    "ss_input_mask.samt",
    "ss_input_pointmap.samt",
    "ss_input_rgb_image.samt",
    "ss_input_rgb_image_mask.samt",
    "ss_input_rgb_pointmap.samt",
    "ss_input_pointmap_scale.samt",
    "ss_input_pointmap_shift.samt",
    "ss_input_rgb_pointmap_scale.samt",
    "ss_input_rgb_pointmap_shift.samt",
)


def read_samt(path: Path) -> tuple[tuple[int, ...], np.ndarray]:
    with path.open("rb") as stream:
        if stream.read(4) != SAMT_MAGIC:
            raise ValueError(f"{path}: invalid SAMT magic")
        rank_data = stream.read(4)
        if len(rank_data) != 4:
            raise ValueError(f"{path}: truncated rank")
        (rank,) = struct.unpack("<i", rank_data)
        if not 1 <= rank <= 8:
            raise ValueError(f"{path}: invalid rank {rank}")
        dimensions = struct.unpack(f"<{rank}q", stream.read(8 * rank))
        type_data = stream.read(4)
        if len(type_data) != 4:
            raise ValueError(f"{path}: truncated type")
        (ggml_type,) = struct.unpack("<i", type_data)
        if ggml_type != GGML_TYPE_F32:
            raise ValueError(f"{path}: expected F32 SAMT, got ggml type {ggml_type}")
        count = int(np.prod(dimensions, dtype=np.int64))
        payload = stream.read()
    expected = count * np.dtype("<f4").itemsize
    if len(payload) != expected:
        raise ValueError(f"{path}: expected {expected} payload bytes, found {len(payload)}")
    return dimensions, np.frombuffer(payload, dtype="<f4")


def compare(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    actual_nan = np.isnan(actual)
    expected_nan = np.isnan(expected)
    same_nan = int(np.logical_and(actual_nan, expected_nan).sum())
    nan_mismatch = int(np.logical_xor(actual_nan, expected_nan).sum())
    finite = np.logical_and(np.isfinite(actual), np.isfinite(expected))
    finite_count = int(finite.sum())
    if finite_count:
        delta = np.abs(actual[finite].astype(np.float64) - expected[finite].astype(np.float64))
        mae = float(delta.mean())
        rmse = float(np.sqrt(np.square(delta).mean()))
        max_abs = float(delta.max())
    else:
        mae = rmse = max_abs = 0.0
    inf_mismatch = int(np.logical_xor(np.isinf(actual), np.isinf(expected)).sum())
    return {
        "elements": int(actual.size),
        "finite_elements": finite_count,
        "same_nan_elements": same_nan,
        "nan_mismatch_elements": nan_mismatch,
        "inf_mismatch_elements": inf_mismatch,
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--max-mae", type=float)
    parser.add_argument("--max-abs", type=float)
    args = parser.parse_args()

    report: dict[str, object] = {
        "schema": "sam3d.native-preprocess.v1",
        "native_dir": str(args.native_dir.resolve()),
        "reference_dir": str(args.reference_dir.resolve()),
        "fields": {},
    }
    passed = True
    for name in FIELDS:
        actual_shape, actual = read_samt(args.native_dir / name)
        expected_shape, expected = read_samt(args.reference_dir / name)
        if actual_shape != expected_shape:
            raise ValueError(f"{name}: shape differs: native={actual_shape}, reference={expected_shape}")
        metrics = compare(actual, expected)
        report["fields"][name] = {"shape": list(actual_shape), **metrics}
        if metrics["nan_mismatch_elements"] or metrics["inf_mismatch_elements"]:
            passed = False
        if args.max_mae is not None and metrics["mae"] > args.max_mae:
            passed = False
        if args.max_abs is not None and metrics["max_abs"] > args.max_abs:
            passed = False
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
