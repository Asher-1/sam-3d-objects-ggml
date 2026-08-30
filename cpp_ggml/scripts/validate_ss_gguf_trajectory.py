#!/usr/bin/env python3
"""Gate an exported SS GGUF against the independent PyTorch Euler trajectory.

This is a deployment gate, not a differentiable training proxy.  It invokes
the built C++ SS stage with the same condition tokens, noise and Euler schedule
as raw E2E inference, then compares every emitted velocity and latent state
with ``ss_trajectory_ref.py`` output.  Keep temporary candidate outputs outside
``benchmarks/`` and ``models/gguf/`` until this gate and raw rendered E2E pass.
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


MODALITIES = ["6drotation_normalized", "scale", "shape", "translation", "translation_scale"]


def read_samt(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        if stream.read(4) != b"SAMT":
            raise ValueError(f"{path}: invalid SAMT magic")
        (dimensions,) = struct.unpack("<i", stream.read(4))
        shape = struct.unpack(f"<{dimensions}q", stream.read(8 * dimensions))
        (kind,) = struct.unpack("<i", stream.read(4))
        if kind != 0:
            raise ValueError(f"{path}: expected F32 SAMT, got type {kind}")
        values = np.frombuffer(stream.read(), dtype="<f4").copy()
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} values, got {values.size}")
    return values


def error(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    if reference.size != actual.size:
        raise ValueError(f"tensor size mismatch: {reference.size} vs {actual.size}")
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    return {
        "elements": int(delta.size),
        "mae": float(np.abs(delta).mean()),
        "mse": float(np.square(delta).mean()),
        "rmse": float(np.sqrt(np.square(delta).mean())),
        "max_abs": float(np.abs(delta).max()),
    }


def compare(torch_dir: Path, debug_dir: Path, steps: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        for index, modality in enumerate(MODALITIES):
            comparisons = [
                ("latent", torch_dir / f"ss_torch_x{step:03d}_{modality}.samt",
                 debug_dir / f"e2e_ss_state{step}_{modality}.samt"),
                ("vc", torch_dir / f"ss_torch_vc{step:03d}_{modality}.samt",
                 debug_dir / f"e2e_ss_vc{step - 1}_{index}.samt"),
            ]
            if step <= 19:  # CFG is active only while the scaled timestep is <= 500.
                comparisons.append(("vu", torch_dir / f"ss_torch_vu{step:03d}_{modality}.samt",
                                    debug_dir / f"e2e_ss_vu{step - 1}_{index}.samt"))
            for kind, reference_path, actual_path in comparisons:
                if not reference_path.is_file() or not actual_path.is_file():
                    raise FileNotFoundError(f"missing trajectory pair: {reference_path}, {actual_path}")
                rows.append({"step": step, "kind": kind, "modality": modality,
                             **error(read_samt(reference_path), read_samt(actual_path))})
    return rows


def terminal_by_modality(rows: list[dict[str, Any]], steps: int) -> dict[str, dict[str, float]]:
    return {str(row["modality"]): {key: float(row[key]) for key in ("mae", "mse", "rmse", "max_abs")}
            for row in rows if row["step"] == steps and row["kind"] == "latent"}


def first_velocity_by_modality(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {f"{row['kind']}:{row['modality']}":
            {key: float(row[key]) for key in ("mae", "mse", "rmse", "max_abs")}
            for row in rows if row["step"] == 1 and row["kind"] in ("vc", "vu")}


def aggregate_mse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return element-weighted MSEs for every sampler step and the full prefix.

    The strict candidate gate below intentionally evaluates every public tensor
    independently.  This complementary summary is not a replacement gate: it
    makes the total deployed error measurable when a candidate improves large
    shape tensors while slightly worsening a small pose tensor.
    """
    def summary(group: list[dict[str, Any]]) -> dict[str, float | int]:
        elements = sum(int(row["elements"]) for row in group)
        if elements == 0:
            raise ValueError("cannot aggregate an empty trajectory row group")
        squared_error = sum(float(row["mse"]) * int(row["elements"]) for row in group)
        absolute_error = sum(float(row["mae"]) * int(row["elements"]) for row in group)
        return {"elements": elements, "mae": absolute_error / elements,
                "mse": squared_error / elements}

    steps = sorted({int(row["step"]) for row in rows})
    return {
        "all_rows": summary(rows),
        "by_step": {str(step): summary([row for row in rows if int(row["step"]) == step])
                    for step in steps},
    }


def compare_baseline(rows: list[dict[str, Any]], baseline_path: Path) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_rows = baseline.get("rows")
    if not isinstance(baseline_rows, list):
        raise ValueError(f"{baseline_path}: no rows array")
    keyed = {(int(row["step"]), str(row["kind"]), str(row["modality"])): row
             for row in baseline_rows}
    failures = []
    for row in rows:
        key = (int(row["step"]), str(row["kind"]), str(row["modality"]))
        reference = keyed.get(key)
        if reference is None:
            failures.append(f"baseline missing {key}")
            continue
        if float(row["mse"]) > float(reference["mse"]):
            failures.append(f"{key}: mse regressed")
    candidate_aggregate = aggregate_mse(rows)
    current_by_key = {(int(row["step"]), str(row["kind"]), str(row["modality"])): row
                      for row in rows}
    # Historical reports predate the explicit ``elements`` field.  Their MSE
    # remains valid, and the matching candidate tensor supplies its immutable
    # element count for a like-for-like weighted prefix comparison.
    baseline_prefix = []
    for row in baseline_rows:
        key = (int(row["step"]), str(row["kind"]), str(row["modality"]))
        current = current_by_key.get(key)
        if current is None:
            continue
        enriched = dict(row)
        enriched.setdefault("elements", current["elements"])
        baseline_prefix.append(enriched)
    if not baseline_prefix:
        raise ValueError(f"{baseline_path}: no rows for the requested prefix")
    baseline_prefix_aggregate = aggregate_mse(baseline_prefix)
    return {
        "baseline": str(baseline_path.resolve()),
        "pareto_non_regressing": not failures,
        "failures": failures,
        "aggregate_mse": {
            "candidate_prefix": candidate_aggregate["all_rows"],
            "baseline_prefix": baseline_prefix_aggregate["all_rows"],
            "ratio": candidate_aggregate["all_rows"]["mse"] /
                     baseline_prefix_aggregate["all_rows"]["mse"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--torch-dir", type=Path, required=True,
                        help="independent output from ss_trajectory_ref.py")
    parser.add_argument("--e2e-dir", type=Path, default=root / "benchmarks/data/e2e")
    parser.add_argument("--backend", choices=("cuda", "vulkan"), required=True)
    parser.add_argument("--dtype", default="q4_0")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--baseline", type=Path,
                        help="require every matching deployment trajectory MSE not to regress")
    parser.add_argument("--keep-debug", action="store_true")
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be positive")
    model = args.models_dir / f"ss_generator-{args.dtype}.gguf"
    executable = (args.build_dir or root / f"build-{args.backend}") / "bin/sam3d-cli"
    cond_tokens = args.e2e_dir / "ss_cond_tokens.samt"
    for path in (model, executable, cond_tokens):
        if not path.is_file():
            parser.error(f"missing required input: {path}")
    for modality in MODALITIES:
        if not (args.torch_dir / f"ss_torch_x{args.steps:03d}_{modality}.samt").is_file():
            parser.error(f"missing PyTorch terminal trajectory for {modality}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sam3d-ss-gguf-") as temporary:
        temporary_dir = Path(temporary)
        debug_dir = temporary_dir / "debug"
        debug_dir.mkdir()
        env = os.environ.copy()
        env.update({
            "SAM3D_BACKEND": args.backend,
            "SAM3D_E2E_DTYPE": args.dtype,
            "SAM3D_E2E_STAGE": "ss",
            "SAM3D_E2E_SS_FLOW_ONLY": "1",
            # Keep the canonical 25-step Euler schedule but stop after the
            # requested prefix. This makes a one-step deployment probe cheap
            # without changing the timestep or delta of that first step.
            "SAM3D_E2E_SS_STEPS": str(args.steps),
            "SAM3D_E2E_SS_COND_PATH": str(cond_tokens.resolve()),
        })
        lib_dir = executable.parents[1] / "lib"
        if lib_dir.is_dir():
            env["LD_LIBRARY_PATH"] = str(lib_dir) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        command = [str(executable.resolve()), "e2e", "--model", str(args.models_dir.resolve()),
                   str(args.e2e_dir.resolve()), "--out", str(temporary_dir / "ss"),
                   "--noise-dir", str(args.e2e_dir.resolve()), "--dbg-dir", str(debug_dir),
                   "--seed", str(args.seed), "--threads", str(args.threads)]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, env=env, check=True)
        rows = compare(args.torch_dir, debug_dir, args.steps)
        payload: dict[str, Any] = {
            "schema": "sam3d.ss_gguf_deployment_trajectory.v1",
            "backend": args.backend,
            "dtype": args.dtype,
            "models_dir": str(args.models_dir.resolve()),
            "ss_generator": str(model.resolve()),
            "torch_dir": str(args.torch_dir.resolve()),
            "e2e_dir": str(args.e2e_dir.resolve()),
            "seed": args.seed,
            "steps": args.steps,
            "terminal_latent": terminal_by_modality(rows, args.steps),
            "first_velocity": first_velocity_by_modality(rows),
            "aggregate_mse": aggregate_mse(rows),
            "rows": rows,
        }
        if args.baseline:
            payload["selection"] = compare_baseline(rows, args.baseline)
        if args.keep_debug:
            retained = args.output.with_suffix("").with_name(args.output.stem + "_debug")
            if retained.exists():
                raise FileExistsError(f"refusing to replace existing debug directory: {retained}")
            shutil.copytree(debug_dir, retained)
            payload["debug_dir"] = str(retained.resolve())
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"terminal_latent": payload["terminal_latent"],
                      "selection": payload.get("selection")}, indent=2))
    return 0 if payload.get("selection", {}).get("pareto_non_regressing", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
