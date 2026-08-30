#!/usr/bin/env python3
"""Measure the official and ggml image-to-3D release matrix in one command.

The matrix is intentionally raw: every ggml entry consumes one immutable,
complete official condition dump, generates a Gaussian PLY, and compares its
official-camera render with the PLY in that same dump.  A fresh official run
provides the latency reference for the identical image, mask, and seed.
Results are written even when an individual release gate fails, but a missing
exclusive-GPU provenance or a failed quality/speed requirement keeps the
command non-zero.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_ROOT = REPO_ROOT / "cpp_ggml"
RUNNER = CPP_ROOT / "scripts/run_image_to_3d.py"
PYTORCH_BENCHMARK = CPP_ROOT / "scripts/bench_e2e_pipeline.py"
VALIDATOR = CPP_ROOT / "scripts/validate_e2e_benchmark.py"
PLOTTER = CPP_ROOT / "scripts/plot_e2e_latency.py"
RELEASE_MAX_E2E_MS = 70000


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    print("+", command_text(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, text=True, check=False)


def read_last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    except json.JSONDecodeError:
        return None
    return rows[-1] if rows and isinstance(rows[-1], dict) else None


def load_summary(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def ggml_variants() -> list[dict[str, str]]:
    """Return deployment candidates, never experimental checkpoint paths."""
    variants: list[dict[str, str]] = []
    for backend in ("cuda", "vulkan"):
        variants.extend((
            {"id": f"{backend}-f16", "backend": backend, "dtype": "f16",
             "runner": f"GGML {backend.upper()} f16 raw",
             "quantization_policy": "F16 GGUF for every generative stage"},
            {"id": f"{backend}-q8", "backend": backend, "dtype": "q8_0",
             "runner": f"GGML {backend.upper()} q8_0 raw",
             "quantization_policy": "Q8_0 GGUF for every generative stage"},
            {"id": f"{backend}-q4", "backend": backend, "dtype": "q4_0",
             "ss_dtype": "q4_k", "ss_decoder_dtype": "f16",
             "runner": f"GGML {backend.upper()} q4_k SS-sensitive-F16 + q4_0 SLat/Gaussian",
             "quantization_policy": "Q4_K SS generator with retained sensitive F16 matrices and F16 "
                                    "SS decoder; Q4_0 SLat and Gaussian decoder"},
        ))
    return variants


def release_contract(variants: list[dict[str, str]]) -> dict[str, Any]:
    required: list[dict[str, Any]] = []
    for variant in variants:
        requirement: dict[str, Any] = {
            "id": variant["id"], "runner": variant["runner"],
            "require_quality": True, "require_exclusive_gpu": True,
            "require_faster_than_reference": True,
        }
        if variant["id"].endswith("-q4"):
            requirement["require_faster_than"] = variant["id"].replace("-q4", "-q8")
        required.append(requirement)
    return {
        "schema": "sam3d.e2e.release-contract.v2",
        "reference_runner": "PyTorch official F16 streamed",
        "require_reference_exclusive_gpu": True,
        "max_render_mae": 0.01,
        "max_latency_ms": RELEASE_MAX_E2E_MS,
        "required": required,
    }


def candidate_command(args: argparse.Namespace, variant: dict[str, str], output: Path,
                      conditions: Path) -> list[str]:
    command = [
        sys.executable, str(RUNNER), "--image", str(args.image.resolve()),
        "--mask-dir", str(args.mask_dir.resolve()), "--mask-index", str(args.mask_index),
        "--backend", variant["backend"], "--dtype", variant["dtype"],
        "--models-dir", str(args.models_dir.resolve()), "--conditions-dir", str(conditions),
        "--python", str(args.official_python.resolve()), "--seed", str(args.seed),
        "--threads", str(args.threads), "--replay-noise",
        "--render-frames", str(args.render_frames),
        "--render-resolution", str(args.render_resolution),
        "--max-render-mae", "0.01", "--max-e2e-ms", str(RELEASE_MAX_E2E_MS),
        "--require-exclusive-gpu", "--out-dir", str(output),
    ]
    if "ss_dtype" in variant:
        command.extend(("--ss-dtype", variant["ss_dtype"]))
    if "ss_decoder_dtype" in variant:
        command.extend(("--ss-decoder-dtype", variant["ss_decoder_dtype"]))
    return command


def candidate_row(variant: dict[str, str], output: Path, command: list[str],
                  returncode: int) -> dict[str, Any]:
    summary_path = output / "run_summary.json"
    summary = load_summary(summary_path)
    row: dict[str, Any] = {
        "runner": variant["runner"], "backend": variant["backend"], "dtype": variant["dtype"],
        "quantization_policy": variant["quantization_policy"],
        "command": command_text(command), "run_summary": str(summary_path),
    }
    if summary is None:
        row["status"] = "failed before generating a run summary"
        row["returncode"] = returncode
        return row
    row.update({
        "latency_ms": summary.get("ggml_e2e_ms"),
        "phases_ms": {
            "condition": summary.get("ggml_condition_ms"),
            "ss": summary.get("ggml_ss_ms"),
            "slat_plus_gaussian": summary.get("ggml_slat_ms"),
        },
        "render_mae": summary.get("render_mae"),
        "render_report": summary.get("render_report"),
        "output": summary.get("ggml_ply"),
        "gpu_exclusivity": summary.get("gpu_exclusivity"),
        "provenance": summary.get("provenance"),
        "strict_reference_coords": summary.get("strict_reference_coords"),
        "status": "measured" if returncode == 0 else "measured; one or more release gates failed",
        "returncode": returncode,
    })
    render_report = summary.get("render_report")
    if render_report:
        metrics = load_summary(Path(str(render_report)))
        if metrics is not None:
            row["render_psnr_db"] = metrics.get("psnr_db")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--mask-index", type=int, default=14)
    parser.add_argument("--official-python", type=Path, required=True,
                        help="Python interpreter with the official SAM 3D Objects dependencies")
    parser.add_argument("--models-dir", type=Path, default=CPP_ROOT / "models/gguf")
    parser.add_argument("--conditions-dir", type=Path,
                        default=CPP_ROOT / "benchmarks/data/e2e",
                        help="immutable complete official stage dump used by every GGML row")
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="new directory for conditions, PLY files, renders, and summaries")
    parser.add_argument("--output-report", type=Path, required=True,
                        help="new E2E JSON matrix report")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--pytorch-warmup", type=int, default=0)
    parser.add_argument("--pytorch-iters", type=int, default=1)
    parser.add_argument("--render-frames", type=int, default=60)
    parser.add_argument("--render-resolution", type=int, default=512)
    args = parser.parse_args()

    for label, path in (("image", args.image), ("mask directory", args.mask_dir),
                        ("official Python", args.official_python), ("models directory", args.models_dir),
                        ("conditions directory", args.conditions_dir)):
        if not path.exists():
            parser.error(f"missing {label}: {path}")
    if not (args.conditions_dir / "manifest.json").is_file():
        parser.error(f"conditions directory has no manifest: {args.conditions_dir}")
    if not (args.conditions_dir / "output_gs.ply").is_file():
        parser.error(f"conditions directory has no PyTorch reference PLY: {args.conditions_dir}")
    if args.output_report.exists():
        parser.error(f"refusing to overwrite report: {args.output_report}")
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        parser.error(f"work directory must be new or empty: {args.work_dir}")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    variants = ggml_variants()
    conditions = args.conditions_dir.resolve()
    pytorch_output = args.work_dir / "pytorch_reference"
    pytorch_jsonl = args.work_dir / "pytorch_latency.jsonl"
    pytorch_command = [
        str(args.official_python.resolve()), str(PYTORCH_BENCHMARK),
        "--image", str(args.image.resolve()), "--mask-dir", str(args.mask_dir.resolve()),
        "--mask-index", str(args.mask_index), "--out-dir", str(pytorch_output),
        "--json", str(pytorch_jsonl), "--warmup", str(args.pytorch_warmup),
        "--iters", str(args.pytorch_iters), "--require-exclusive-gpu",
    ]
    pytorch_result = run(pytorch_command)
    pytorch = read_last_jsonl(pytorch_jsonl)
    rows: list[dict[str, Any]] = []
    if pytorch is None:
        rows.append({
            "runner": "PyTorch official F16 streamed", "status": "failed before recording reference",
            "returncode": pytorch_result.returncode, "command": command_text(pytorch_command),
        })
    else:
        rows.append({
            "runner": "PyTorch official F16 streamed", "backend": "cuda", "dtype": "f16",
            "latency_ms": pytorch.get("e2e_ms_mean"), "warmup": pytorch.get("warmup"),
            "iters": pytorch.get("iters"), "gpu_exclusivity": pytorch.get("gpu_exclusivity"),
            "output": str(conditions / "output_gs.ply"),
            "timed_output": str(pytorch_output / "output_gs.ply"), "status": "measured",
            "command": command_text(pytorch_command), "provenance": pytorch,
        })

    reference_ready = pytorch is not None
    for variant in variants:
        output = args.work_dir / variant["id"]
        command = candidate_command(args, variant, output, conditions)
        if not reference_ready:
            rows.append({
                "runner": variant["runner"], "backend": variant["backend"], "dtype": variant["dtype"],
                "quantization_policy": variant["quantization_policy"],
                "status": "not run because the exclusive-GPU PyTorch reference was unavailable",
                "command": command_text(command),
            })
            continue
        result = run(command)
        rows.append(candidate_row(variant, output, command, result.returncode))

    report = {
        "schema": "sam3d.e2e.matrix.v1",
        "workload": "one image + mask to one Gaussian PLY, then 60-view official-camera render",
        "input": f"{args.image.resolve()}, mask {args.mask_index}, seed {args.seed}",
        "latency_gate_ms": RELEASE_MAX_E2E_MS,
        "release_gate": release_contract(variants),
        "rows": rows,
        "note": "Every timing row requires an empty NVIDIA compute-client list before model load. "
                "All GGML render comparisons use the immutable complete official condition dump. "
                "A missing value is a failed measurement, never a zero-latency placeholder.",
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output_report}")

    latency_plot = args.output_report.with_suffix(".png")
    metrics_plot = args.output_report.with_name(args.output_report.stem + "_metrics.png")
    run([sys.executable, str(PLOTTER), "--input", str(args.output_report),
         "--output", str(latency_plot), "--metrics-output", str(metrics_plot)])
    gate_path = args.output_report.with_name(args.output_report.stem + "_gate.json")
    gate = run([sys.executable, str(VALIDATOR), "--input", str(args.output_report),
                "--output", str(gate_path)])
    return gate.returncode


if __name__ == "__main__":
    raise SystemExit(main())
