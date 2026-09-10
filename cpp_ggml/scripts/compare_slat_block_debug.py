#!/usr/bin/env python3
"""Compare official and native block-0 SLat debug tensors.

``slat_step_ref.py`` exports official checkpoint observations for the exact
stage names listed below. A matching native capture is produced with one
``SAM3D_DEBUG_STAGE`` invocation of ``sam3d-cli slat-step`` per directory.
The comparison requires both the GGML/SAMT shape and payload layout to match;
equal element counts alone are insufficient evidence of an aligned operator.

The report records whether the reference uses the original checkpoint or the
candidate GGUF's dequantized backbone weights. This prevents a boundary result
from silently attributing weight conversion/quantization error to C++.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


# (native debug stage, official reference basename). These names are public
# command boundaries in slat_flow_graph.cpp and slat_step_ref.py.
STAGES: tuple[tuple[str, str], ...] = (
    ("input_layer", "slat_input_layer"),
    ("block_in", "slat_block_in"),
    ("b0_adaln", "slat_b0_adaln"),
    ("b0_qkv", "slat_b0_qkv"),
    ("b0_qpre", "slat_b0_qpre"),
    ("b0_qrms", "slat_b0_qrms"),
    ("b0_attn_out", "slat_b0_attn_out"),
    ("b0_cross_out", "slat_b0_cross_out"),
    ("b0_mlp_out", "slat_b0_mlp_out"),
    ("block0", "slat_block0"),
)


def load_samt(path: Path) -> tuple[tuple[int, ...], np.ndarray]:
    with path.open("rb") as stream:
        if stream.read(4) != b"SAMT":
            raise ValueError(f"{path}: invalid SAMT magic")
        (rank,) = struct.unpack("<i", stream.read(4))
        if rank < 0:
            raise ValueError(f"{path}: invalid negative rank {rank}")
        shape = struct.unpack(f"<{rank}q", stream.read(8 * rank))
        (value_type,) = struct.unpack("<i", stream.read(4))
        if value_type != 0:
            raise ValueError(f"{path}: expected F32 SAMT, got type {value_type}")
        values = np.frombuffer(stream.read(), dtype="<f4").copy()
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"{path}: payload size {values.size} does not match {shape}")
    return shape, values


def error(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    if reference.size != actual.size:
        raise ValueError(f"tensor size mismatch: {reference.size} vs {actual.size}")
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    return {
        "mae": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.square(delta).mean())),
        "max_abs": float(np.abs(delta).max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-dir", type=Path, required=True)
    parser.add_argument("--ggml-dir", type=Path, required=True,
                        help="directory containing one subdirectory per debug stage")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-stage-mae", type=float,
                        help="optional maximum MAE applied independently to every boundary")
    parser.add_argument("--max-stage-abs", type=float,
                        help="optional maximum absolute error applied independently to every boundary")
    parser.add_argument("--reference-weight-scope",
                        choices=("official-checkpoint", "same-gguf-dequantized"),
                        default="official-checkpoint")
    args = parser.parse_args()
    if args.max_stage_mae is not None and args.max_stage_mae < 0.0:
        parser.error("--max-stage-mae must be non-negative")
    if args.max_stage_abs is not None and args.max_stage_abs < 0.0:
        parser.error("--max-stage-abs must be non-negative")

    rows = []
    for stage, reference_name in STAGES:
        reference_path = args.torch_dir / f"{reference_name}.samt"
        actual_path = args.ggml_dir / stage / f"slat_dbg_{stage}_0.samt"
        if not reference_path.is_file() or not actual_path.is_file():
            raise FileNotFoundError(f"missing debug pair: {reference_path}, {actual_path}")
        reference_shape, reference = load_samt(reference_path)
        actual_shape, actual = load_samt(actual_path)
        if reference_shape != actual_shape:
            raise ValueError(
                f"tensor shape mismatch for {stage}: {reference_shape} vs {actual_shape}")
        rows.append({
            "stage": stage,
            "reference_shape": list(reference_shape),
            "actual_shape": list(actual_shape),
            "elements": int(reference.size),
            **error(reference, actual),
        })

    numeric_gate_configured = args.max_stage_mae is not None or args.max_stage_abs is not None
    failures = []
    for row in rows:
        failed_limits = []
        if args.max_stage_mae is not None and row["mae"] > args.max_stage_mae:
            failed_limits.append("mae")
        if args.max_stage_abs is not None and row["max_abs"] > args.max_stage_abs:
            failed_limits.append("max_abs")
        if failed_limits:
            failures.append({"stage": row["stage"], "limits": failed_limits})

    payload = {
        "schema": "sam3d.slat_block0_debug_comparison.v1",
        "reference_weight_scope": args.reference_weight_scope,
        "candidate_weight_scope": "gguf",
        "max_stage_mae": args.max_stage_mae,
        "max_stage_abs": args.max_stage_abs,
        "numeric_gate_configured": numeric_gate_configured,
        "failed_stages": failures,
        # An unbounded comparison is diagnostic evidence, not an acceptance result.
        "passed": (not failures) if numeric_gate_configured else None,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{row['stage']}: mae={row['mae']:.8g} "
              f"rmse={row['rmse']:.8g} max={row['max_abs']:.8g}")
    return 0 if payload["passed"] is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
