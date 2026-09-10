#!/usr/bin/env python3
"""Generate and compare official SS/SLat block-boundary observations.

This is a diagnostic/localization runner, not a replacement for raw image E2E
acceptance. It consumes a *fresh* ``dump_e2e_stages.py`` directory, produces
SS/SLat Torch observations, captures every exported native block-0 boundary
with the selected GGUF model, then writes one immutable JSON summary.

``official-checkpoint`` reference weights localize end-to-end divergence but
include conversion/quantization error. ``same-gguf-dequantized`` loads every
generator-backbone dependency from the selected GGUF file into the official
Torch architecture, separating that source of error from the native graph.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from compare_slat_block_debug import STAGES as SLAT_STAGES
from compare_ss_block_debug import STAGES as SS_STAGES


REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_ROOT = REPO_ROOT / "cpp_ggml"
SCRIPTS = CPP_ROOT / "scripts"


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def run_step(name: str, command: list[str], steps: list[dict[str, Any]],
             env: dict[str, str] | None = None) -> bool:
    print(f"[{name}] + {command_text(command)}", flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, check=False)
    steps.append({
        "name": name,
        "command": command_text(command),
        "returncode": result.returncode,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "status": "PASS" if result.returncode == 0 else "FAIL",
    })
    return result.returncode == 0


def require_file(parser: argparse.ArgumentParser, label: str, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        parser.error(f"missing {label}: {resolved}")
    return resolved


def require_dir(parser: argparse.ArgumentParser, label: str, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        parser.error(f"missing {label}: {resolved}")
    return resolved


def native_env(binary: Path, backend: str, threads: int, debug_stage: str,
               ss_attention: str | None) -> dict[str, str]:
    env = os.environ.copy()
    # A capture must not inherit an unrelated active stage or attention mode.
    env.pop("SAM3D_DEBUG_STAGE", None)
    env.pop("SAM3D_SS_STRICT_ATTN", None)
    env.update({
        "SAM3D_BACKEND": backend,
        "SAM3D_NTHREADS": str(threads),
        "SAM3D_T": "0",
        "SAM3D_DEBUG_STAGE": debug_stage,
    })
    if ss_attention == "strict":
        env["SAM3D_SS_STRICT_ATTN"] = "1"
    lib_dir = binary.parents[1] / "lib"
    if lib_dir.is_dir():
        env["LD_LIBRARY_PATH"] = str(lib_dir) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def read_comparison(path: Path, reference_weight_scope: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("reference_weight_scope") != reference_weight_scope:
        raise ValueError(f"{path}: unexpected reference weight scope")
    if payload.get("candidate_weight_scope") != "gguf":
        raise ValueError(f"{path}: unexpected candidate weight scope")
    if "passed" not in payload:
        raise ValueError(f"{path}: comparison did not report a pass state")
    return payload


def read_trajectory_comparison(path: Path, reference_weight_scope: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("reference_weight_scope") != reference_weight_scope:
        raise ValueError(f"{path}: unexpected reference weight scope")
    if payload.get("candidate_weight_scope") != "gguf":
        raise ValueError(f"{path}: unexpected candidate weight scope")
    if "passed" not in payload:
        raise ValueError(f"{path}: trajectory comparison did not report a pass state")
    return payload


def verify_same_gguf_weight_scope(path: Path, model: Path) -> None:
    """Require the reference process to leave the loader's factual receipt."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "sam3d.gguf_torch_backbone_loader.v1":
        raise ValueError(f"{path}: unexpected GGUF loader report schema")
    if payload.get("weight_scope") != "same-gguf-dequantized":
        raise ValueError(f"{path}: unexpected GGUF loader weight scope")
    if Path(payload.get("gguf", "")).resolve() != model.resolve():
        raise ValueError(f"{path}: reference used a different GGUF model")
    if not isinstance(payload.get("backbone_tensors_loaded"), int) or \
            payload["backbone_tensors_loaded"] < 1:
        raise ValueError(f"{path}: no backbone tensors were loaded")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-python", type=Path, required=True,
                        help="official SAM 3D Objects Python interpreter")
    parser.add_argument("--binary", type=Path, required=True,
                        help="sam3d-cli compiled for --backend")
    parser.add_argument("--models-dir", type=Path, default=CPP_ROOT / "models/gguf")
    parser.add_argument("--e2e-dir", type=Path, required=True,
                        help="fresh official stage directory from dump_e2e_stages.py")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="new or empty output directory")
    parser.add_argument("--backend", choices=("cuda", "vulkan"), required=True)
    parser.add_argument("--dtype", choices=("f16", "q8_0", "q4_0"), default="f16")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--ss-attention", choices=("normal", "strict"), default="strict")
    parser.add_argument("--reference-device", default="cuda",
                        help="device used by official checkpoint observations")
    parser.add_argument("--reference-weight-scope",
                        choices=("official-checkpoint", "same-gguf-dequantized"),
                        default="same-gguf-dequantized",
                        help=("Torch reference weights: original checkpoint for broad divergence, "
                              "or every generator-backbone tensor dequantized from the candidate GGUF"))
    parser.add_argument("--ss-checkpoint", type=Path,
                        default=REPO_ROOT / "checkpoints/hf/ss_generator.ckpt")
    parser.add_argument("--ss-yaml", type=Path,
                        default=REPO_ROOT / "checkpoints/hf/ss_generator.yaml")
    parser.add_argument("--slat-checkpoint", type=Path,
                        default=REPO_ROOT / "checkpoints/hf/slat_generator.ckpt")
    parser.add_argument("--slat-yaml", type=Path,
                        default=REPO_ROOT / "checkpoints/hf/slat_generator.yaml")
    parser.add_argument("--max-stage-mae", type=float,
                        help="optional per-boundary MAE gate")
    parser.add_argument("--max-stage-abs", type=float,
                        help="optional per-boundary maximum absolute-error gate")
    parser.add_argument("--trajectory", action="store_true",
                        help=("also compare all 25 SS/SLat Euler states against a fresh Torch "
                              "trajectory using the selected reference-weight scope"))
    parser.add_argument("--trajectory-ss-compute-dtype", choices=("f32", "f16"), default="f32",
                        help="Torch SS operator dtype for --trajectory")
    parser.add_argument("--trajectory-slat-compute-dtype", choices=("f32", "f16"), default="f32",
                        help="Torch SLat operator dtype for --trajectory")
    parser.add_argument("--trajectory-ss-baseline", type=Path,
                        help="optional same-scope SS trajectory report that may not regress")
    parser.add_argument("--trajectory-slat-max-terminal-mae", type=float,
                        help="optional same-scope final SLat latent MAE limit")
    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.max_stage_mae is not None and args.max_stage_mae < 0.0:
        parser.error("--max-stage-mae must be non-negative")
    if args.max_stage_abs is not None and args.max_stage_abs < 0.0:
        parser.error("--max-stage-abs must be non-negative")
    if args.trajectory_slat_max_terminal_mae is not None and \
            args.trajectory_slat_max_terminal_mae < 0.0:
        parser.error("--trajectory-slat-max-terminal-mae must be non-negative")

    official_python = require_file(parser, "official Python", args.official_python)
    binary = require_file(parser, "native executable", args.binary)
    models_dir = require_dir(parser, "GGUF models directory", args.models_dir)
    e2e_dir = require_dir(parser, "official stage directory", args.e2e_dir)
    ss_checkpoint = require_file(parser, "SS checkpoint", args.ss_checkpoint)
    ss_yaml = require_file(parser, "SS config", args.ss_yaml)
    slat_checkpoint = require_file(parser, "SLat checkpoint", args.slat_checkpoint)
    slat_yaml = require_file(parser, "SLat config", args.slat_yaml)
    ss_model = require_file(parser, "SS GGUF", models_dir / f"ss_generator-{args.dtype}.gguf")
    slat_model = require_file(parser, "SLat GGUF", models_dir / f"slat_generator-{args.dtype}.gguf")
    for required in ("ss_x0_shape.samt", "ss_cond_tokens.samt", "slat_x0.samt",
                     "slat_coords.samt", "slat_cond_tokens.samt"):
        require_file(parser, "official stage tensor", e2e_dir / required)

    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        parser.error(f"refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch_ss_dir = out_dir / "official_ss"
    torch_slat_dir = out_dir / "official_slat"
    native_ss_dir = out_dir / "native_ss"
    native_slat_dir = out_dir / "native_slat"
    ss_report = out_dir / "ss_block0_comparison.json"
    slat_report = out_dir / "slat_block0_comparison.json"
    trajectory_ss_dir = out_dir / "official_ss_trajectory"
    trajectory_slat_dir = out_dir / "official_slat_trajectory"
    trajectory_ss_report = out_dir / "ss_trajectory_comparison.json"
    trajectory_slat_report = out_dir / "slat_trajectory_comparison.json"
    steps: list[dict[str, Any]] = []
    same_gguf_reference = args.reference_weight_scope == "same-gguf-dequantized"
    ss_gguf_args = ["--gguf", str(ss_model)] if same_gguf_reference else []
    slat_gguf_args = ["--gguf", str(slat_model)] if same_gguf_reference else []

    ss_reference_ok = run_step(
        "official_ss_block0_reference",
        [
            str(official_python), str(SCRIPTS / "ss_step_ref.py"),
            "--ckpt", str(ss_checkpoint), "--yaml", str(ss_yaml),
            "--e2e-dir", str(e2e_dir), "--out-dir", str(torch_ss_dir),
            "--t", "0", "--device", args.reference_device, *ss_gguf_args,
        ],
        steps,
    )
    slat_reference_ok = False
    if ss_reference_ok:
        slat_reference_ok = run_step(
            "official_slat_block0_reference",
            [
                str(official_python), str(SCRIPTS / "slat_step_ref.py"),
                "--ckpt", str(slat_checkpoint), "--yaml", str(slat_yaml),
                "--e2e-dir", str(e2e_dir), "--out-dir", str(torch_slat_dir),
                "--t", "0", "--device", args.reference_device, *slat_gguf_args,
            ],
            steps,
        )
    else:
        steps.append({"name": "official_slat_block0_reference", "status": "SKIPPED",
                      "reason": "official SS reference generation failed"})

    if same_gguf_reference and ss_reference_ok and slat_reference_ok:
        try:
            verify_same_gguf_weight_scope(torch_ss_dir / "reference_weight_scope.json", ss_model)
            verify_same_gguf_weight_scope(torch_slat_dir / "reference_weight_scope.json", slat_model)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            steps.append({"name": "verify_same_gguf_reference_weights", "status": "FAIL",
                          "error": str(error)})
            ss_reference_ok = False
            slat_reference_ok = False
        else:
            steps.append({"name": "verify_same_gguf_reference_weights", "status": "PASS"})

    native_ss_ok = ss_reference_ok and slat_reference_ok
    if native_ss_ok:
        native_ss_dir.mkdir()
        for stage, _, _ in SS_STAGES:
            stage_dir = native_ss_dir / stage
            if stage_dir.exists():
                continue
            stage_dir.mkdir()
            native_ss_ok &= run_step(
                f"native_ss_{stage}",
                [str(binary), "ss-step", "--model", str(ss_model), str(e2e_dir), str(stage_dir)],
                steps, native_env(binary, args.backend, args.threads, stage, args.ss_attention),
            )
            if not native_ss_ok:
                break
    else:
        steps.append({"name": "native_ss_block0_capture", "status": "SKIPPED",
                      "reason": "official block references did not complete"})

    native_slat_ok = native_ss_ok
    if native_slat_ok:
        native_slat_dir.mkdir()
        for stage, _ in SLAT_STAGES:
            stage_dir = native_slat_dir / stage
            stage_dir.mkdir()
            native_slat_ok &= run_step(
                f"native_slat_{stage}",
                [str(binary), "slat-step", "--model", str(slat_model), str(e2e_dir), str(stage_dir)],
                steps, native_env(binary, args.backend, args.threads, stage, None),
            )
            if not native_slat_ok:
                break
    else:
        steps.append({"name": "native_slat_block0_capture", "status": "SKIPPED",
                      "reason": "native SS block capture did not complete"})

    threshold_args: list[str] = []
    if args.max_stage_mae is not None:
        threshold_args.extend(("--max-stage-mae", str(args.max_stage_mae)))
    if args.max_stage_abs is not None:
        threshold_args.extend(("--max-stage-abs", str(args.max_stage_abs)))
    ss_compare_ok = False
    slat_compare_ok = False
    if native_slat_ok:
        ss_compare_ok = run_step(
            "compare_ss_block0_boundaries",
            [sys.executable, str(SCRIPTS / "compare_ss_block_debug.py"),
             "--torch-dir", str(torch_ss_dir), "--ggml-dir", str(native_ss_dir),
             "--output", str(ss_report),
             "--reference-weight-scope", args.reference_weight_scope, *threshold_args],
            steps,
        )
        slat_compare_ok = run_step(
            "compare_slat_block0_boundaries",
            [sys.executable, str(SCRIPTS / "compare_slat_block_debug.py"),
             "--torch-dir", str(torch_slat_dir), "--ggml-dir", str(native_slat_dir),
             "--output", str(slat_report),
             "--reference-weight-scope", args.reference_weight_scope, *threshold_args],
            steps,
        )
    else:
        steps.extend((
            {"name": "compare_ss_block0_boundaries", "status": "SKIPPED",
             "reason": "native captures did not complete"},
            {"name": "compare_slat_block0_boundaries", "status": "SKIPPED",
             "reason": "native captures did not complete"},
        ))

    trajectory_ss_reference_ok = False
    trajectory_slat_reference_ok = False
    trajectory_references_verified = False
    trajectory_ss_compare_ok = False
    trajectory_slat_compare_ok = False
    trajectory_ss_comparison = None
    trajectory_slat_comparison = None
    trajectory_numeric_gate_configured = (
        args.trajectory_ss_baseline is not None
        or args.trajectory_slat_max_terminal_mae is not None)
    trajectory_completed = False
    trajectory_passed = None
    if args.trajectory:
        trajectory_ss_reference_ok = run_step(
            "official_ss_trajectory_reference",
            [
                str(official_python), str(SCRIPTS / "ss_trajectory_ref.py"),
                "--checkpoint", str(ss_checkpoint), "--config", str(ss_yaml),
                "--e2e-dir", str(e2e_dir), "--out-dir", str(trajectory_ss_dir),
                "--device", args.reference_device, "--compute-dtype",
                args.trajectory_ss_compute_dtype, "--steps", "25", *ss_gguf_args,
            ],
            steps,
        )
        if trajectory_ss_reference_ok:
            trajectory_slat_reference_ok = run_step(
                "official_slat_trajectory_reference",
                [
                    str(official_python), str(SCRIPTS / "slat_trajectory_ref.py"),
                    "--checkpoint", str(slat_checkpoint), "--config", str(slat_yaml),
                    "--e2e-dir", str(e2e_dir), "--out-dir", str(trajectory_slat_dir),
                    "--device", args.reference_device, "--compute-dtype",
                    args.trajectory_slat_compute_dtype, "--steps", "25", *slat_gguf_args,
                ],
                steps,
            )
        else:
            steps.append({"name": "official_slat_trajectory_reference", "status": "SKIPPED",
                          "reason": "official SS trajectory generation failed"})

        trajectory_references_verified = trajectory_ss_reference_ok and trajectory_slat_reference_ok
        if same_gguf_reference and trajectory_references_verified:
            try:
                verify_same_gguf_weight_scope(
                    trajectory_ss_dir / "reference_weight_scope.json", ss_model)
                verify_same_gguf_weight_scope(
                    trajectory_slat_dir / "reference_weight_scope.json", slat_model)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                steps.append({"name": "verify_same_gguf_trajectory_reference_weights",
                              "status": "FAIL", "error": str(error)})
                trajectory_references_verified = False
            else:
                steps.append({"name": "verify_same_gguf_trajectory_reference_weights",
                              "status": "PASS"})

        if trajectory_references_verified:
            ss_trajectory_command = [
                sys.executable, str(SCRIPTS / "validate_ss_gguf_trajectory.py"),
                "--models-dir", str(models_dir), "--torch-dir", str(trajectory_ss_dir),
                "--e2e-dir", str(e2e_dir), "--backend", args.backend, "--dtype", args.dtype,
                "--build-dir", str(binary.parents[1]), "--output", str(trajectory_ss_report),
                "--steps", "25", "--threads", str(args.threads), "--ss-attention",
                args.ss_attention, "--reference-weight-scope", args.reference_weight_scope,
            ]
            if args.trajectory_ss_baseline is not None:
                ss_trajectory_command.extend(("--baseline", str(args.trajectory_ss_baseline)))
            trajectory_ss_compare_ok = run_step(
                "compare_ss_euler_trajectory", ss_trajectory_command, steps)

            slat_trajectory_command = [
                sys.executable, str(SCRIPTS / "validate_slat_gguf_trajectory.py"),
                "--models-dir", str(models_dir), "--torch-dir", str(trajectory_slat_dir),
                "--e2e-dir", str(e2e_dir), "--backend", args.backend, "--dtype", args.dtype,
                "--build-dir", str(binary.parents[1]), "--output", str(trajectory_slat_report),
                "--steps", "25", "--threads", str(args.threads),
                "--reference-weight-scope", args.reference_weight_scope,
            ]
            if args.trajectory_slat_max_terminal_mae is not None:
                slat_trajectory_command.extend((
                    "--max-terminal-mae", str(args.trajectory_slat_max_terminal_mae)))
            trajectory_slat_compare_ok = run_step(
                "compare_slat_euler_trajectory", slat_trajectory_command, steps)
        else:
            steps.extend((
                {"name": "compare_ss_euler_trajectory", "status": "SKIPPED",
                 "reason": "official trajectory references did not complete"},
                {"name": "compare_slat_euler_trajectory", "status": "SKIPPED",
                 "reason": "official trajectory references did not complete"},
            ))

        trajectory_ss_comparison = (
            read_trajectory_comparison(trajectory_ss_report, args.reference_weight_scope)
            if trajectory_ss_compare_ok else None)
        trajectory_slat_comparison = (
            read_trajectory_comparison(trajectory_slat_report, args.reference_weight_scope)
            if trajectory_slat_compare_ok else None)
        trajectory_completed = (
            trajectory_ss_comparison is not None and trajectory_slat_comparison is not None)
        if trajectory_numeric_gate_configured and trajectory_completed:
            trajectory_passed = bool(
                trajectory_ss_comparison["passed"] and trajectory_slat_comparison["passed"])

    ss_comparison = read_comparison(ss_report, args.reference_weight_scope) if ss_compare_ok else None
    slat_comparison = read_comparison(slat_report, args.reference_weight_scope) if slat_compare_ok else None
    numeric_gate_configured = args.max_stage_mae is not None or args.max_stage_abs is not None
    comparisons_completed = ss_comparison is not None and slat_comparison is not None
    passed = None
    if numeric_gate_configured and comparisons_completed:
        passed = bool(ss_comparison["passed"] and slat_comparison["passed"])

    report = {
        "schema": "sam3d.operator_boundary_oracle.v2",
        "reference_weight_scope": args.reference_weight_scope,
        "candidate_weight_scope": "gguf",
        "backend": args.backend,
        "dtype": args.dtype,
        "ss_attention": args.ss_attention,
        "reference_device": args.reference_device,
        "e2e_dir": str(e2e_dir),
        "models": {"ss_generator": str(ss_model), "slat_generator": str(slat_model)},
        "thresholds": {"max_stage_mae": args.max_stage_mae,
                       "max_stage_abs": args.max_stage_abs},
        "numeric_gate_configured": numeric_gate_configured,
        "comparisons_completed": comparisons_completed,
        "artifacts": {
            "official_ss": str(torch_ss_dir), "official_slat": str(torch_slat_dir),
            "official_ss_weight_scope": (
                str(torch_ss_dir / "reference_weight_scope.json") if same_gguf_reference else None),
            "official_slat_weight_scope": (
                str(torch_slat_dir / "reference_weight_scope.json") if same_gguf_reference else None),
            "native_ss": str(native_ss_dir), "native_slat": str(native_slat_dir),
            "ss_report": str(ss_report), "slat_report": str(slat_report),
            "official_ss_trajectory": str(trajectory_ss_dir) if args.trajectory else None,
            "official_slat_trajectory": str(trajectory_slat_dir) if args.trajectory else None,
            "ss_trajectory_report": str(trajectory_ss_report) if args.trajectory else None,
            "slat_trajectory_report": str(trajectory_slat_report) if args.trajectory else None,
        },
        "ss": ss_comparison,
        "slat": slat_comparison,
        "trajectory": {
            "enabled": args.trajectory,
            "reference_weight_scope": args.reference_weight_scope,
            "candidate_weight_scope": "gguf",
            "ss_compute_dtype": args.trajectory_ss_compute_dtype,
            "slat_compute_dtype": args.trajectory_slat_compute_dtype,
            "numeric_gate_configured": trajectory_numeric_gate_configured,
            "completed": trajectory_completed,
            "passed": trajectory_passed,
            "ss": trajectory_ss_comparison,
            "slat": trajectory_slat_comparison,
        },
        "steps": steps,
        # No empirical threshold means this is deliberately only a diagnostic run.
        "passed": passed,
    }
    report_path = out_dir / "operator_oracle_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    if not comparisons_completed or (args.trajectory and not trajectory_completed):
        return 1
    return 0 if passed is not False and trajectory_passed is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
