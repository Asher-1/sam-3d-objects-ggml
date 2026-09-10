#!/usr/bin/env python3
"""Compare official and native block-0 SLat debug tensors.

``slat_step_ref.py`` exports official checkpoint observations for the exact
stage names listed below. A matching native capture is produced with one
``SAM3D_DEBUG_STAGE`` invocation of ``sam3d-cli slat-step`` per directory.
The comparison requires both the GGML/SAMT layout and payload order to match.
Only two explicit serialization differences are accepted: a trailing singleton
dimension, and the known block-0 Q layout where ggml coalesces heads into the
channel dimension. Equal element counts alone are insufficient evidence of an
aligned operator.

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
from samt_io import read_samt as load_samt


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




def error(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    if reference.size != actual.size:
        raise ValueError(f"tensor size mismatch: {reference.size} vs {actual.size}")
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    return {
        "mae": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.square(delta).mean())),
        "max_abs": float(np.abs(delta).max()),
    }


def trailing_singleton_layout_equivalent(
        reference_shape: tuple[int, ...], actual_shape: tuple[int, ...]) -> bool:
    """Accept only rank differences made of terminal dimensions of size one.

    SAMT payloads are flat and written in ggml's native element order. Removing
    a trailing singleton dimension leaves that order unchanged, unlike any
    transpose, reshape, or non-singleton dimension change.
    """
    def without_trailing_singletons(shape: tuple[int, ...]) -> tuple[int, ...]:
        while len(shape) > 1 and shape[-1] == 1:
            shape = shape[:-1]
        return shape

    return (int(np.prod(reference_shape)) == int(np.prod(actual_shape))
            and without_trailing_singletons(reference_shape)
            == without_trailing_singletons(actual_shape))


def align_known_layout(stage: str, reference_shape: tuple[int, ...],
                       actual_shape: tuple[int, ...], actual: np.ndarray) -> tuple[np.ndarray, str]:
    """Return native values in the reference payload layout for known Q tensors.

    PyTorch stores block-0 Q tensors as [token, head, head_dim] and writes
    SAMT dimensions in reverse. ggml stores Q before RMS norm as
    [head_dim * head, token], then after RMS norm as [head_dim, token, head].
    These are the same logical tensor only under the exact shape contracts
    checked below. No generic reshape or transpose is accepted.
    """
    if reference_shape == actual_shape:
        return actual, "exact"
    if trailing_singleton_layout_equivalent(reference_shape, actual_shape):
        return actual, "trailing_singleton"
    if stage == "b0_qpre" and len(reference_shape) == 3 and len(actual_shape) == 2:
        head_dim, heads, tokens = reference_shape
        if actual_shape == (head_dim * heads, tokens):
            return actual, "coalesced_heads"
    if stage == "b0_qrms" and len(reference_shape) == 3 and len(actual_shape) == 3:
        head_dim, heads, tokens = reference_shape
        if actual_shape == (head_dim, tokens, heads):
            native = actual.reshape(heads, tokens, head_dim)
            return native.transpose(1, 0, 2).copy().reshape(-1), "head_token_permute"
    raise ValueError(
        f"tensor shape mismatch for {stage}: {reference_shape} vs {actual_shape}")


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
        actual, layout_mapping = align_known_layout(
            stage, reference_shape, actual_shape, actual)
        rows.append({
            "stage": stage,
            "reference_shape": list(reference_shape),
            "actual_shape": list(actual_shape),
            "layout_mapping": layout_mapping,
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
        "schema": "sam3d.slat_block0_debug_comparison.v2",
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
