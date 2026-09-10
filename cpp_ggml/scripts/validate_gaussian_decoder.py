#!/usr/bin/env python3
"""Gate the native Gaussian decoder with official frozen SLat features.

This isolates the GGUF Gaussian decoder and its representation conversion from
SS/SLat diffusion. It invokes ``sam3d-cli gs-decode`` on the official SLat
latent and compares every public Gaussian field with the matching official E2E
tensor. Temporary decoder outputs are deleted automatically.
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


FIELDS = (
    ("xyz", "decode_gaussian__xyz.samt"),
    ("features_dc", "decode_gaussian__features_dc.samt"),
    ("scaling", "decode_gaussian__scaling.samt"),
    ("rotation", "decode_gaussian__rotation.samt"),
    ("opacity", "decode_gaussian__opacity.samt"),
)


def read_samt(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        if stream.read(4) != b"SAMT":
            raise ValueError(f"{path}: invalid SAMT magic")
        (rank,) = struct.unpack("<i", stream.read(4))
        shape = struct.unpack(f"<{rank}q", stream.read(8 * rank))
        (value_type,) = struct.unpack("<i", stream.read(4))
        if value_type != 0:
            raise ValueError(f"{path}: expected F32 SAMT, got type {value_type}")
        values = np.frombuffer(stream.read(), dtype="<f4").copy()
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} values, got {values.size}")
    return values


def error(reference: np.ndarray, actual: np.ndarray) -> dict[str, float | int]:
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


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    elements = sum(int(row["elements"]) for row in rows)
    if elements == 0:
        raise ValueError("cannot aggregate an empty representation")
    return {
        "elements": elements,
        "mae": sum(float(row["mae"]) * int(row["elements"]) for row in rows) / elements,
        "mse": sum(float(row["mse"]) * int(row["elements"]) for row in rows) / elements,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--e2e-dir", type=Path, default=root / "benchmarks/data/e2e")
    parser.add_argument("--backend", choices=("cuda", "vulkan"), required=True)
    parser.add_argument("--dtype", default="f16")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-field-mae", type=float,
                        help="optional gate applied independently to every field")
    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.max_field_mae is not None and args.max_field_mae < 0:
        parser.error("--max-field-mae must be non-negative")
    executable = (args.build_dir or root / f"build-{args.backend}") / "bin/sam3d-cli"
    model = args.models_dir / f"slat_decoder_gs-{args.dtype}.gguf"
    required = (
        executable,
        model,
        args.e2e_dir / "slat_feats_final.samt",
        args.e2e_dir / "slat_coords.samt",
        *(args.e2e_dir / reference for _, reference in FIELDS),
    )
    for path in required:
        if not path.is_file():
            parser.error(f"missing required input: {path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sam3d-gs-decoder-") as temporary:
        temporary_dir = Path(temporary)
        env = os.environ.copy()
        env["SAM3D_BACKEND"] = args.backend
        env["SAM3D_NTHREADS"] = str(args.threads)
        lib_dir = executable.parents[1] / "lib"
        if lib_dir.is_dir():
            env["LD_LIBRARY_PATH"] = str(lib_dir) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        command = [
            str(executable.resolve()), "gs-decode", "--model", str(model.resolve()),
            str(args.e2e_dir.resolve()), str(temporary_dir),
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, env=env, check=True)
        rows: list[dict[str, Any]] = []
        for field, reference_name in FIELDS:
            rows.append({
                "field": field,
                **error(read_samt(args.e2e_dir / reference_name),
                        read_samt(temporary_dir / f"gs_{field}.samt")),
            })

    failures = []
    if args.max_field_mae is not None:
        failures = [str(row["field"]) for row in rows
                    if float(row["mae"]) > args.max_field_mae]
    numeric_gate_configured = args.max_field_mae is not None
    payload = {
        "schema": "sam3d.gaussian_decoder_deployment.v1",
        "backend": args.backend,
        "dtype": args.dtype,
        "models_dir": str(args.models_dir.resolve()),
        "gaussian_decoder": str(model.resolve()),
        "e2e_dir": str(args.e2e_dir.resolve()),
        "max_field_mae": args.max_field_mae,
        "numeric_gate_configured": numeric_gate_configured,
        "aggregate": aggregate(rows),
        "rows": rows,
        # An unbounded comparison is diagnostic evidence, not a passed gate.
        "passed": (not failures) if numeric_gate_configured else None,
        "failed_fields": failures,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": payload["aggregate"], "rows": rows,
                      "passed": payload["passed"]}, indent=2))
    return 0 if not numeric_gate_configured or not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
