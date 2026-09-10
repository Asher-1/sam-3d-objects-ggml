#!/usr/bin/env python3
"""Compare the deployed SLat Euler trajectory with an official PyTorch dump.

The command uses immutable official sparse support, SLat condition tokens and
initial noise.  It therefore diagnoses the SLat GGUF graph itself rather than
an earlier SS occupancy boundary change.  All generated tensors live in a
temporary directory unless ``--keep-debug`` is requested explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from samt_io import read_samt




def error(reference: tuple[tuple[int, ...], np.ndarray],
          actual: tuple[tuple[int, ...], np.ndarray]) -> dict[str, float | int]:
    reference_shape, reference_values = reference
    actual_shape, actual_values = actual
    if reference_shape != actual_shape:
        raise ValueError(f"tensor shape mismatch: {reference_shape} vs {actual_shape}")
    reference = reference_values
    actual = actual_values
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


def compare(torch_dir: Path, debug_dir: Path, steps: int) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for step in range(steps + 1):
        reference_path = torch_dir / f"slat_torch_x{step:03d}.samt"
        actual_path = debug_dir / f"slat_ggml_x{step:03d}.samt"
        if not reference_path.is_file() or not actual_path.is_file():
            raise FileNotFoundError(f"missing trajectory pair: {reference_path}, {actual_path}")
        rows.append({"step": step, **error(read_samt(reference_path), read_samt(actual_path))})
    return rows


def aggregate(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    elements = sum(int(row["elements"]) for row in rows)
    if elements == 0:
        raise ValueError("cannot aggregate an empty trajectory")
    return {
        "elements": elements,
        "mae": sum(float(row["mae"]) * int(row["elements"]) for row in rows) / elements,
        "mse": sum(float(row["mse"]) * int(row["elements"]) for row in rows) / elements,
    }


def verify_same_gguf_weight_scope(torch_dir: Path, model: Path) -> Path:
    """Require the reference dump to prove its candidate-weight provenance."""
    receipt_path = torch_dir / "reference_weight_scope.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "sam3d.gguf_torch_backbone_loader.v1":
        raise ValueError(f"{receipt_path}: unexpected GGUF loader report schema")
    if payload.get("weight_scope") != "same-gguf-dequantized":
        raise ValueError(f"{receipt_path}: unexpected GGUF loader weight scope")
    if Path(payload.get("gguf", "")).resolve() != model.resolve():
        raise ValueError(f"{receipt_path}: reference used a different SLat GGUF")
    if (not isinstance(payload.get("backbone_tensors_loaded"), int)
            or payload["backbone_tensors_loaded"] < 1):
        raise ValueError(f"{receipt_path}: no backbone tensors were loaded")
    return receipt_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--torch-dir", type=Path, required=True,
                        help="independent output from slat_trajectory_ref.py")
    parser.add_argument("--e2e-dir", type=Path, default=root / "benchmarks/data/e2e")
    parser.add_argument("--backend", choices=("cuda", "vulkan"), required=True)
    parser.add_argument("--dtype", default="f16")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-terminal-mae", type=float,
                        help="optional deployment gate for normalized final latent")
    parser.add_argument("--reference-weight-scope",
                        choices=("official-checkpoint", "same-gguf-dequantized"),
                        default="official-checkpoint",
                        help="provenance of --torch-dir; same-GGUF requires its loader receipt")
    parser.add_argument("--keep-debug", action="store_true")
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.max_terminal_mae is not None and args.max_terminal_mae < 0:
        parser.error("--max-terminal-mae must be non-negative")
    executable = (args.build_dir or root / f"build-{args.backend}") / "bin/sam3d-cli"
    required = (
        args.models_dir / f"slat_generator-{args.dtype}.gguf",
        executable,
        args.e2e_dir / "slat_coords.samt",
        args.e2e_dir / "slat_x0.samt",
        args.e2e_dir / "slat_cond_tokens.samt",
        args.torch_dir / f"slat_torch_x{args.steps:03d}.samt",
    )
    for path in required:
        if not path.is_file():
            parser.error(f"missing required input: {path}")
    model = args.models_dir / f"slat_generator-{args.dtype}.gguf"
    receipt_path = None
    if args.reference_weight_scope == "same-gguf-dequantized":
        try:
            receipt_path = verify_same_gguf_weight_scope(args.torch_dir, model)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sam3d-slat-gguf-") as temporary:
        temporary_dir = Path(temporary)
        debug_dir = temporary_dir / "debug"
        debug_dir.mkdir()
        env = os.environ.copy()
        lib_dir = executable.parents[1] / "lib"
        if lib_dir.is_dir():
            env["LD_LIBRARY_PATH"] = str(lib_dir) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        command = [
            str(executable.resolve()), "e2e", "--model", str(args.models_dir.resolve()),
            str(args.e2e_dir.resolve()), "--out", str(temporary_dir / "unused.ply"),
            "--noise-dir", str(args.e2e_dir.resolve()), "--dbg-dir", str(debug_dir),
            "--seed", str(args.seed), "--threads", str(args.threads),
            "--backend", args.backend,
            "--dtype", args.dtype,
            "--stage", "slat",
            "--reference-coords",
            "--slat-cond-path", str((args.e2e_dir / "slat_cond_tokens.samt").resolve()),
            "--dump-slat-steps", "--slat-flow-only",
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, env=env, check=True)
        rows = compare(args.torch_dir, debug_dir, args.steps)
        terminal = rows[-1]
        numeric_gate_configured = args.max_terminal_mae is not None
        passed = (float(terminal["mae"]) <= args.max_terminal_mae
                  if numeric_gate_configured else None)
        payload: dict[str, Any] = {
            "schema": "sam3d.slat_gguf_deployment_trajectory.v2",
            "reference_weight_scope": args.reference_weight_scope,
            "candidate_weight_scope": "gguf",
            "reference_weight_receipt": (str(receipt_path.resolve()) if receipt_path else None),
            "backend": args.backend,
            "dtype": args.dtype,
            "models_dir": str(args.models_dir.resolve()),
            "slat_generator": str((args.models_dir / f"slat_generator-{args.dtype}.gguf").resolve()),
            "torch_dir": str(args.torch_dir.resolve()),
            "e2e_dir": str(args.e2e_dir.resolve()),
            "seed": args.seed,
            "steps": args.steps,
            "aggregate": aggregate(rows),
            "terminal_latent": terminal,
            "max_terminal_mae": args.max_terminal_mae,
            "numeric_gate_configured": numeric_gate_configured,
            "passed": passed,
            "rows": rows,
        }
        if args.keep_debug:
            retained = args.output.with_suffix("").with_name(args.output.stem + "_debug")
            if retained.exists():
                raise FileExistsError(f"refusing to replace existing debug directory: {retained}")
            shutil.copytree(debug_dir, retained)
            payload["debug_dir"] = str(retained.resolve())
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": payload["aggregate"], "terminal_latent": terminal,
                      "passed": payload["passed"]}, indent=2))
    return 0 if passed is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
