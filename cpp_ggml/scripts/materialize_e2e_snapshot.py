#!/usr/bin/env python3
"""Publish a reviewed raw E2E run as compact benchmark evidence.

The tool only accepts six completed native image-input summaries and one
exclusive-GPU PyTorch cold-request row with the same timer contract. It writes
the matrix, plots, machine-readable gate, and six representative side-by-side
images. GGUFs, PLYs, tensors, and full render sequences stay in the caller's
work directory.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


CPP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CPP_ROOT.parent
PLOTTER = CPP_ROOT / "scripts/plot_e2e_latency.py"
VALIDATOR = CPP_ROOT / "scripts/validate_e2e_benchmark.py"
RELEASE_MAX_E2E_MS = 70000.0
MAX_RENDER_MAE = 0.01
REQUIRED_TIMER_CONTRACT = "sam3d.cold-request.image-mask-to-gaussian-ply.v2"
VARIANTS = (
    ("cuda-f16", "cuda", "f16"),
    ("cuda-q8", "cuda", "q8_0"),
    ("cuda-q4", "cuda", "q4_0"),
    ("vulkan-f16", "vulkan", "f16"),
    ("vulkan-q8", "vulkan", "q8_0"),
    ("vulkan-q4", "vulkan", "q4_0"),
)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def read_last_jsonl(path: Path) -> dict[str, Any]:
    try:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSONL: {path}: {exc}") from exc
    if not values or not isinstance(values[-1], dict):
        raise RuntimeError(f"no JSON object in {path}")
    return values[-1]


def finite(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"summary has no finite {name}: {value!r}")
    return float(value)


def timer_contract(summary: dict[str, Any]) -> dict[str, Any]:
    value = summary.get("timer_contract")
    if not isinstance(value, dict) or value.get("id") != REQUIRED_TIMER_CONTRACT:
        raise RuntimeError(
            f"summary must use timer contract {REQUIRED_TIMER_CONTRACT!r}: {value!r}")
    return value


def command_text(args: argparse.Namespace, backend: str, dtype: str) -> str:
    build_dir = args.cuda_build_dir if backend == "cuda" else args.vulkan_build_dir
    return " ".join((
        str(args.official_python), str(CPP_ROOT / "scripts/run_image_to_3d.py"),
        "--image", str(args.image), "--mask-dir", str(args.mask_dir),
        "--mask-index", str(args.mask_index), "--backend", backend, "--dtype", dtype,
        "--build-dir", str(build_dir), "--models-dir", str(args.models_dir),
        "--moge-model", str(args.moge_model), "--native-image-input", "--seed", str(args.seed),
        "--threads", str(args.threads), "--ss-attention", "strict", "--render-frames", "60",
        "--render-resolution", "512", "--max-render-mae", "0.01", "--max-e2e-ms", "70000",
        "--require-exclusive-gpu",
    ))


def copy_render_evidence(summary: dict[str, Any], destination: Path, variant_id: str) -> str:
    metrics = Path(str(summary.get("render_report", "")))
    report = metrics.parent
    if not metrics.is_file() or not (report / "side_by_side.png").is_file():
        raise RuntimeError(f"missing render evidence for {variant_id}: {metrics}")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metrics, destination / f"{variant_id}_render_metrics.json")
    shutil.copy2(report / "side_by_side.png", destination / f"{variant_id}_side_by_side.png")
    return str((destination / f"{variant_id}_render_metrics.json").resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-jsonl", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[], metavar="ID=DIR",
                        help="one run_summary.json directory per required matrix id")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--mask-index", type=int, default=14)
    parser.add_argument("--official-python", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--moge-model", type=Path, required=True)
    parser.add_argument("--cuda-build-dir", type=Path, required=True)
    parser.add_argument("--vulkan-build-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--render-output-dir", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--allow-failed-gate", action="store_true")
    args = parser.parse_args()

    supplied: dict[str, Path] = {}
    for value in args.run:
        key, separator, path = value.partition("=")
        if not separator or key in supplied:
            parser.error(f"invalid or duplicate --run value: {value}")
        supplied[key] = Path(path)
    expected_ids = {item[0] for item in VARIANTS}
    if set(supplied) != expected_ids:
        parser.error(f"--run IDs must be exactly: {', '.join(sorted(expected_ids))}")
    if args.output_report.exists() and not args.replace:
        parser.error(f"refusing to overwrite {args.output_report}; pass --replace")

    reference = read_last_jsonl(args.reference_jsonl)
    reference_latency = finite("e2e_ms_mean", reference.get("e2e_ms_mean"))
    reference_contract = timer_contract(reference)
    exclusive = reference.get("gpu_exclusivity", {})
    if not isinstance(exclusive, dict) or not exclusive.get("checked") or exclusive.get("active_compute_processes"):
        raise RuntimeError("reference does not prove exclusive GPU execution")

    rows: list[dict[str, Any]] = [{
        "runner": "PyTorch official staged mixed cold request", "backend": "cuda",
        "dtype": reference.get("dtype"),
        "latency_ms": reference_latency, "warmup": reference.get("warmup"),
        "iters": reference.get("iters"), "gpu_exclusivity": exclusive,
        "timer_contract": reference_contract, "status": "measured", "provenance": reference,
    }]
    for variant_id, backend, dtype in VARIANTS:
        summary_path = supplied[variant_id] / "run_summary.json"
        summary = read_json(summary_path)
        if (summary.get("backend"), summary.get("dtype")) != (backend, dtype):
            raise RuntimeError(f"{variant_id} does not match {summary_path}")
        if summary.get("execution_mode") != "native-image-input" or summary.get("ss_attention") != "strict":
            raise RuntimeError(f"{variant_id} is not a strict raw image-input measurement")
        if summary.get("sampling_mode") != "native-seed":
            raise RuntimeError(f"{variant_id} is not a native-seed measurement")
        candidate_contract = timer_contract(summary)
        gpu = summary.get("gpu_exclusivity", {})
        if not isinstance(gpu, dict) or not gpu.get("checked") or gpu.get("active_compute_processes"):
            raise RuntimeError(f"{variant_id} does not prove exclusive GPU execution")
        latency = finite("ggml_e2e_ms", summary.get("ggml_e2e_ms"))
        mae = finite("render_mae", summary.get("render_mae"))
        render_report = copy_render_evidence(summary, args.render_output_dir, variant_id)
        label = f"GGML {backend.upper()} {dtype} native image input"
        rows.append({
            "runner": label, "backend": backend, "dtype": dtype,
            "quantization_policy": (
                f"{dtype.upper()} GGUF for every generative stage; fixed MoGe preprocessing "
                "model recorded separately"
            ),
            "command": command_text(args, backend, dtype),
            "source_run_id": variant_id,
            "latency_ms": latency,
            "phases_ms": {"native_image_to_ply": latency},
            "render_mae": mae, "render_report": render_report,
            "gpu_exclusivity": gpu, "provenance": summary.get("provenance"),
            "timer_contract": candidate_contract,
            "strict_reference_coords": summary.get("strict_reference_coords"),
            "execution_mode": summary.get("execution_mode"), "sampling_mode": summary.get("sampling_mode"),
            "ss_attention": "strict",
            "status": "measured" if latency <= RELEASE_MAX_E2E_MS and mae <= MAX_RENDER_MAE
            else "measured; one or more release gates failed",
            "returncode": 0 if latency <= RELEASE_MAX_E2E_MS and mae <= MAX_RENDER_MAE else 1,
        })

    required = []
    for variant_id, backend, dtype in VARIANTS:
        item: dict[str, Any] = {
            "id": variant_id, "runner": f"GGML {backend.upper()} {dtype} native image input",
            "require_quality": True, "require_exclusive_gpu": True,
            "require_faster_than_reference": True,
        }
        if dtype == "q4_0":
            item["require_faster_than"] = f"{backend}-q8"
        required.append(item)
    report = {
        "schema": "sam3d.e2e.matrix.v2",
        "workload": "one cold image + mask request to one Gaussian PLY, then 60-view official-camera render",
        "input": f"{args.image.resolve()}, mask {args.mask_index}, seed {args.seed}",
        "candidate_execution_mode": "native-image-input", "candidate_sampling_mode": "native-seed",
        "ss_attention": "strict",
        "build_dirs": {"cuda": str(args.cuda_build_dir.resolve()), "vulkan": str(args.vulkan_build_dir.resolve())},
        "preprocessing_model": str(args.moge_model.resolve()),
        "quantization_scope": "dtype selects every generative GGUF stage; the same explicitly recorded MoGe preprocessing GGUF is used by every row",
        "latency_gate_ms": RELEASE_MAX_E2E_MS,
        "timer_comparability": "required; mismatched timer-contract IDs fail the release gate",
        "release_gate": {"schema": "sam3d.e2e.release-contract.v2",
                         "reference_runner": "PyTorch official staged mixed cold request",
                         "require_candidate_execution_mode": "native-image-input",
                         "require_candidate_sampling_mode": "native-seed",
                         "require_reference_exclusive_gpu": True, "require_matching_timer_contract": True,
                         "max_render_mae": MAX_RENDER_MAE, "max_latency_ms": RELEASE_MAX_E2E_MS,
                         "required": required},
        "rows": rows,
        "note": "The matching cold-request timer spans each runtime's model construction/loading, image-and-mask decode, inference, and PLY file close. The official PLY is used only afterward for the 60-view render score. Internal CLI generation-stage logs are not the E2E latency source of truth.",
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    metrics_name = args.output_report.stem.replace("latency", "metrics") + ".png"
    subprocess.run([sys.executable, str(PLOTTER), "--input", str(args.output_report),
                    "--output", str(args.output_report.with_suffix(".png")),
                    "--metrics-output", str(args.output_report.with_name(metrics_name))],
                   cwd=REPO_ROOT, check=True)
    gate_path = args.output_report.with_name(args.output_report.stem.replace("latency", "gate") + ".json")
    gate = subprocess.run([sys.executable, str(VALIDATOR), "--input", str(args.output_report),
                           "--output", str(gate_path)], cwd=REPO_ROOT, check=False)
    if gate.returncode and not args.allow_failed_gate:
        return gate.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
