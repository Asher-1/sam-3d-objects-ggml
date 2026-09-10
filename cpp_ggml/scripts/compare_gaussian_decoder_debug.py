#!/usr/bin/env python3
"""Compare native Gaussian-decoder graph boundaries with official PyTorch.

``gs_decode_ref.py`` exports the official boundaries from the same frozen SLat
features.  This tool invokes the production ``sam3d-cli gs-decode`` once per
boundary, compares the resulting SAMT tensor, and removes all native debug
dumps when finished.  It is deliberately independent of the image-to-3D
pipeline so the first Gaussian-decoder mismatch remains observable.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


# (native stage, output indices, official reference basename).  The native
# graph exposes Q, K and V separately; restore the official token-major fused
# QKV layout before comparison.
STAGES: tuple[tuple[str, tuple[int, ...], str], ...] = (
    ("input_layer", (0,), "gs_input_layer"),
    ("ape", (0,), "gs_after_ape"),
    ("torso_input", (0,), "gs_torso_input"),
    ("b0_norm1", (0,), "gs_b0_norm1"),
    ("b0_qkv", (0, 1, 2), "gs_b0_qkv"),
    ("b0_attn_values", (0,), "gs_b0_attn_values"),
    ("b0_attn", (0,), "gs_b0_attn"),
    ("b0_norm2", (0,), "gs_b0_norm2"),
    ("b0_mlp0", (0,), "gs_b0_mlp0"),
    ("b0_gelu", (0,), "gs_b0_gelu"),
    ("b0_mlp2", (0,), "gs_b0_mlp2"),
    ("b0", (0,), "gs_b0"),
    ("b1", (0,), "gs_b1"),
    ("b11", (0,), "gs_b11"),
)


def read_samt(path: Path) -> tuple[tuple[int, ...], np.ndarray]:
    with path.open("rb") as stream:
        if stream.read(4) != b"SAMT":
            raise ValueError(f"{path}: invalid SAMT magic")
        (rank,) = struct.unpack("<i", stream.read(4))
        shape = struct.unpack(f"<{rank}q", stream.read(8 * rank))
        (value_type,) = struct.unpack("<i", stream.read(4))
        if value_type != 0:
            raise ValueError(f"{path}: expected F32 SAMT, got type {value_type}")
        values = np.frombuffer(stream.read(), dtype="<f4").copy()
    if values.size != int(np.prod(shape)):
        raise ValueError(f"{path}: payload does not match shape {shape}")
    return shape, values


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float | int]:
    if reference.size != actual.size:
        raise ValueError(f"tensor size mismatch: {reference.size} vs {actual.size}")
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    squared = np.square(delta)
    return {
        "elements": int(delta.size),
        "mae": float(np.abs(delta).mean()),
        "mse": float(squared.mean()),
        "rmse": float(np.sqrt(squared.mean())),
        "max_abs": float(np.abs(delta).max()),
    }


def merge_native_outputs(parts: list[tuple[tuple[int, ...], np.ndarray]]) -> np.ndarray:
    if len(parts) == 1:
        return parts[0][1]
    shapes = [shape for shape, _ in parts]
    if any(len(shape) != 2 for shape in shapes):
        raise ValueError(f"cannot fuse non-matrix debug outputs: {shapes}")
    channels, tokens = shapes[0]
    if any(shape != (channels, tokens) for shape in shapes[1:]):
        raise ValueError(f"incompatible split debug output shapes: {shapes}")
    # SAMT (C, N) is physically token-major [N, C].  The PyTorch fused QKV
    # hook is [N, 3C], so concatenate each token's Q/K/V rows, not the three
    # complete flattened tensors.
    return np.concatenate(
        [values.reshape(tokens, channels) for _, values in parts], axis=1
    ).reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--torch-dir", type=Path, required=True)
    parser.add_argument("--e2e-dir", type=Path, default=root / "benchmarks/data/e2e")
    parser.add_argument("--backend", choices=("cuda", "vulkan"), required=True)
    parser.add_argument("--dtype", default="f16")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-stage-mae", type=float,
                        help="optional independent MAE threshold for every stage")
    parser.add_argument("--portable-attention", action="store_true",
                        help="disable the CUDA compatibility kernel for an explicit-graph A/B")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.max_stage_mae is not None and args.max_stage_mae < 0:
        parser.error("--max-stage-mae must be non-negative")

    executable = (args.build_dir or root / f"build-{args.backend}") / "bin/sam3d-cli"
    model = args.models_dir / f"slat_decoder_gs-{args.dtype}.gguf"
    required = (executable, model, args.e2e_dir / "slat_feats_final.samt",
                args.e2e_dir / "slat_coords.samt",
                *(args.torch_dir / f"{reference}.samt" for _, _, reference in STAGES))
    for path in required:
        if not path.is_file():
            parser.error(f"missing required input: {path}")

    environment = os.environ.copy()
    environment["SAM3D_BACKEND"] = args.backend
    environment["SAM3D_NTHREADS"] = str(args.threads)
    environment.pop("SAM3D_DEBUG_STAGE", None)
    if args.portable_attention:
        environment["SAM3D_GS_PORTABLE_ATTN"] = "1"
    else:
        environment.pop("SAM3D_GS_PORTABLE_ATTN", None)
    library_dir = executable.parents[1] / "lib"
    if library_dir.is_dir():
        environment["LD_LIBRARY_PATH"] = str(library_dir) + os.pathsep + environment.get(
            "LD_LIBRARY_PATH", "")

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sam3d-gs-debug-") as temporary:
        temporary_dir = Path(temporary)
        for stage, indices, reference_name in STAGES:
            stage_dir = temporary_dir / stage
            stage_environment = environment.copy()
            stage_environment["SAM3D_DEBUG_STAGE"] = stage
            command = [str(executable.resolve()), "gs-decode", "--model", str(model.resolve()),
                       str(args.e2e_dir.resolve()), str(stage_dir)]
            print("+", " ".join(command), flush=True)
            subprocess.run(command, env=stage_environment, check=True)
            reference_shape, reference = read_samt(args.torch_dir / f"{reference_name}.samt")
            actual_parts = [read_samt(stage_dir / f"gs_dbg_{stage}_{index}.samt")
                            for index in indices]
            actual = merge_native_outputs(actual_parts)
            rows.append({
                "stage": stage,
                "reference": reference_name,
                "reference_shape": list(reference_shape),
                "native_output_indices": list(indices),
                **metrics(reference, actual),
            })

    failures = [] if args.max_stage_mae is None else [
        row["stage"] for row in rows if float(row["mae"]) > args.max_stage_mae
    ]
    numeric_gate_configured = args.max_stage_mae is not None
    payload = {
        "schema": "sam3d.gaussian_decoder_block_debug.v1",
        "backend": args.backend,
        "dtype": args.dtype,
        "portable_attention": args.portable_attention,
        "max_stage_mae": args.max_stage_mae,
        "numeric_gate_configured": numeric_gate_configured,
        "rows": rows,
        "passed": (not failures) if numeric_gate_configured else None,
        "failed_stages": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{row['stage']}: mae={row['mae']:.9g} rmse={row['rmse']:.9g} "
              f"max={row['max_abs']:.9g}")
    return 0 if not numeric_gate_configured or not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
