#!/usr/bin/env python3
"""Capture a parity-preserving E2E run with stage and GPU profiling evidence.

Arguments following ``--`` are passed to run_image_to_3d.py. The wrapper adds
the selected backend and a JSONL host profile, then exports profiler summaries
beside the trace. CUDA collection expands graph nodes so quantized GEMM and
attention kernels retain individual names in the GPU kernel summary.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "cpp_ggml/scripts/run_image_to_3d.py"


def run(command: list[str], env: dict[str, str] | None = None,
        log_path: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    if log_path is None:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
        return
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True,
                       stdout=stream, stderr=subprocess.STDOUT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "vulkan"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="directory for the .nsys-rep, CSV summaries, and host JSONL")
    parser.add_argument("--nsys", type=Path,
                        help="Nsight Systems executable; defaults to nsys on PATH")
    parser.add_argument("--vulkan-op-timings", action="store_true",
                        help="enable ggml timestamp queries per Vulkan op; not a latency benchmark mode")
    parser.add_argument("runner_args", nargs=argparse.REMAINDER,
                        help="arguments for run_image_to_3d.py, preceded by --")
    args = parser.parse_args()

    nsys = str(args.nsys) if args.nsys else shutil.which("nsys")
    if not nsys:
        parser.error("Nsight Systems was not found; pass --nsys /path/to/nsys")
    if not args.runner_args:
        parser.error("supply run_image_to_3d.py arguments after --")
    runner_args = args.runner_args[1:] if args.runner_args[0] == "--" else args.runner_args

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_base = out_dir / f"{args.backend}_e2e"
    trace_file = trace_base.with_suffix(".nsys-rep")
    host_profile = out_dir / "backend_profile.jsonl"
    runner = [
        sys.executable, str(RUNNER), "--backend", args.backend,
        "--profile-jsonl", str(host_profile), *runner_args,
    ]
    trace_args = ["--trace=cuda,nvtx,osrt", "--cuda-graph-trace=node"]
    reports = ["gpukernsum", "cudaapisum", "gpumemtimesum"]
    environment = os.environ.copy()
    vulkan_timing_log: Path | None = None
    if args.backend == "vulkan":
        trace_args = ["--trace=vulkan,osrt", "--vulkan-gpu-workload=individual"]
        reports = ["vulkanapisum"]
        environment["GGML_VK_DEBUG_MARKERS"] = "1"
        if args.vulkan_op_timings:
            environment["GGML_VK_PERF_LOGGER"] = "1"
            environment["GGML_VK_PERF_LOGGER_FREQUENCY"] = "1"
            vulkan_timing_log = out_dir / "vulkan_op_timings.log"
    run([
        nsys, "profile", "--force-overwrite=true", "--sample=none", "--stats=true",
        "--output", str(trace_base), *trace_args, *runner,
    ], env=environment, log_path=vulkan_timing_log)
    if not trace_file.is_file():
        raise RuntimeError(f"Nsight Systems did not produce {trace_file}")

    stats_base = out_dir / "nsys_stats"
    run([
        nsys, "stats", "--force-export=true", "--force-overwrite=true",
        "--report", ",".join(reports), "--format", "csv", "--output", str(stats_base),
        str(trace_file),
    ], env=environment)
    manifest = {
        "schema": "sam3d.e2e.gpu_profile.v1",
        "backend": args.backend,
        "trace": str(trace_file),
        "host_profile": str(host_profile),
        "reports": reports,
        "runner": runner,
        "vulkan_op_timings": args.vulkan_op_timings,
    }
    if vulkan_timing_log is not None:
        manifest["vulkan_timing_log"] = str(vulkan_timing_log)
    with (out_dir / "profile_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
