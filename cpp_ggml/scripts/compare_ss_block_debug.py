#!/usr/bin/env python3
"""Compare the official and native block-0 SS debug tensors.

``ss_step_ref.py`` writes the official ShortCut/MOT-DiT boundaries.  A matching
native capture is produced with ``sam3d-cli ss-step`` and one
``SAM3D_DEBUG_STAGE`` per directory.  This utility maps those public stage
names explicitly and reports numerical error without changing either capture.

The report records whether the reference uses official checkpoint weights or
the candidate GGUF's dequantized backbone weights. The latter isolates native
operator error at these boundaries; the former deliberately includes the
weight-conversion/quantization contribution.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from samt_io import read_samt as load_samt


# (native debug stage, output index, official reference basename). ``proj_in``
# and ``block0`` expose both the shape and merged-pose groups.
STAGES: tuple[tuple[str, int, str], ...] = (
    ("proj_in", 0, "ss_proj_shape"),
    ("proj_in", 1, "ss_proj_pose"),
    ("b0_adaln", 0, "ss_b0_adaln"),
    ("b0_attn_in_s", 0, "ss_b0_attn_in_shape"),
    ("b0_attn_in_p", 0, "ss_b0_attn_in_6drotation_normalized"),
    ("b0_qkv_shape", 0, "ss_b0_qkv_shape"),
    ("b0_qkv_6drotation_normalized", 0, "ss_b0_qkv_6drotation_normalized"),
    ("b0_qpre_shape", 0, "ss_b0_qpre_shape"),
    ("b0_qpre_6drotation_normalized", 0, "ss_b0_qpre_6drotation_normalized"),
    ("b0_q_shape", 0, "ss_b0_q_shape"),
    ("b0_q_6drotation_normalized", 0, "ss_b0_q_pose"),
    ("b0_attn_out_s", 0, "ss_b0_attn_shape"),
    ("b0_attn_out_p", 0, "ss_b0_attn_6drotation_normalized"),
    ("b0_cross_in_s", 0, "ss_b0_n2_shape"),
    ("b0_cross_in_p", 0, "ss_b0_n2_6drotation_normalized"),
    ("b0_xq_shape", 0, "ss_b0_xq_shape"),
    ("b0_xkv_shape", 0, "ss_b0_xkv_shape"),
    ("b0_xk_shape", 0, "ss_b0_xk_shape"),
    ("b0_xv_shape", 0, "ss_b0_xv_shape"),
    ("b0_xq_6drotation_normalized", 0, "ss_b0_xq_6drotation_normalized"),
    ("b0_xkv_6drotation_normalized", 0, "ss_b0_xkv_6drotation_normalized"),
    ("b0_xk_6drotation_normalized", 0, "ss_b0_xk_6drotation_normalized"),
    ("b0_xv_6drotation_normalized", 0, "ss_b0_xv_6drotation_normalized"),
    ("b0_cross_out_s", 0, "ss_b0_x_shape"),
    ("b0_cross_out_p", 0, "ss_b0_x_6drotation_normalized"),
    ("b0_mlp_out_s", 0, "ss_b0_mlp_shape"),
    ("b0_mlp_out_p", 0, "ss_b0_mlp_6drotation_normalized"),
    ("block0", 0, "ss_b0_shape"),
    ("block0", 1, "ss_b0_pose"),
)




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
    for stage, index, reference_name in STAGES:
        reference_path = args.torch_dir / f"{reference_name}.samt"
        actual_path = args.ggml_dir / stage / f"ss_dbg_{stage}_{index}.samt"
        if not reference_path.is_file() or not actual_path.is_file():
            raise FileNotFoundError(f"missing debug pair: {reference_path}, {actual_path}")
        reference_shape, reference = load_samt(reference_path)
        actual_shape, actual = load_samt(actual_path)
        if reference_shape != actual_shape:
            raise ValueError(
                f"tensor shape mismatch for {stage}[{index}]: "
                f"{reference_shape} vs {actual_shape}")
        rows.append({"stage": stage, "output_index": index,
                     "reference_shape": list(reference_shape),
                     "actual_shape": list(actual_shape),
                     "elements": int(reference.size), **error(reference, actual)})

    numeric_gate_configured = args.max_stage_mae is not None or args.max_stage_abs is not None
    failures = []
    for row in rows:
        failed_limits = []
        if args.max_stage_mae is not None and row["mae"] > args.max_stage_mae:
            failed_limits.append("mae")
        if args.max_stage_abs is not None and row["max_abs"] > args.max_stage_abs:
            failed_limits.append("max_abs")
        if failed_limits:
            failures.append({"stage": row["stage"], "output_index": row["output_index"],
                             "limits": failed_limits})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "sam3d.ss_block0_debug_comparison.v3",
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
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{row['stage']}[{row['output_index']}]: "
              f"mae={row['mae']:.8g} rmse={row['rmse']:.8g} max={row['max_abs']:.8g}")
    return 0 if payload["passed"] is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
