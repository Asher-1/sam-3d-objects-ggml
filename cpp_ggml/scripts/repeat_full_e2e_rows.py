#!/usr/bin/env python3
"""Add independent cold-process repeat samples to a completed full-E2E summary.

Gate v2 (``sam3d.full-glb-gate.v2``) requires several independent cold-process
samples per timed row; the publisher derives ``latency_samples_ms`` from
``steps[name]["runs"]`` and the headline ``latency_ms`` must be their median.

This driver replays the *exact* per-row commands already recorded by
``run_full_e2e_parity.py`` into fresh per-attempt directories instead of
teaching the runner an in-process repeat loop.  Every attempt is therefore a
full cold process with its own GPU exclusivity check, exit code, elapsed time
and, for native rows, its own dtype-contract verification.  Binary, model and
input hashes are re-verified against the summary provenance before anything
runs; a mismatch aborts so one calibration set never spans two binaries.

The original summary is never modified; a derived summary is written beside it
and feeds ``publish_full_e2e.py --summary``.
"""
from __future__ import annotations

import argparse
import json
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from run_full_e2e_parity import REPO_ROOT, command_text, require_exclusive_cuda
from run_image_to_3d import sha256_file

DTYPES = ("f16", "q8_0", "q4_k")

# timing step name -> case directory under the work dir, needs dtype contract,
# quantized variant, and the base run's GPU exclusivity step name
TIMING_ROWS: dict[str, dict[str, Any]] = {
    "official_full_image_to_textured_glb_timing": {
        "case_prefix": "official_full_glb_timing", "needs_contract": False,
        "dtype": None, "exclusive_step": "gpu_exclusive_before_official_full_glb_timing"},
}
for _dtype in DTYPES:
    TIMING_ROWS[f"raw_native_cuda_{_dtype}_image_to_pbr"] = {
        "case_prefix": f"raw_native_cuda_{_dtype}", "needs_contract": True,
        "dtype": _dtype, "exclusive_step": f"gpu_exclusive_before_raw_native_cuda_{_dtype}"}
    TIMING_ROWS[f"raw_vulkan_cuda_pbr_{_dtype}_handoff"] = {
        "case_prefix": f"raw_vulkan_cuda_pbr_{_dtype}", "needs_contract": True,
        "dtype": _dtype, "exclusive_step": f"gpu_exclusive_before_raw_vulkan_cuda_pbr_{_dtype}"}

PROVENANCE_HASH_KEYS = (
    ("binary_sha256", "cuda"), ("binary_sha256", "vulkan"),
    ("binary_sha256", "native_pbr"),
)


def verify_provenance(summary: dict[str, Any], work_dir: Path) -> list[str]:
    """Re-hash the binaries and inputs recorded by the base run."""
    problems: list[str] = []
    provenance = summary.get("provenance", {})
    for section, key in PROVENANCE_HASH_KEYS:
        recorded = provenance.get(section, {}).get(key)
        path = summary.get("binaries", {}).get(
            "cuda" if key == "cuda" else "vulkan" if key == "vulkan" else "native_pbr")
        if not isinstance(recorded, str) or not isinstance(path, str):
            problems.append(f"provenance/binary path missing for {key}")
            continue
        actual = sha256_file(Path(path))
        if actual != recorded:
            problems.append(f"binary {key} changed since the base run: {path}")
    for kind, key in (("image", "image"), ("mask", "mask")):
        recorded = provenance.get("input_sha256", {}).get(kind)
        path = summary.get("input", {}).get(key)
        if isinstance(recorded, str) and isinstance(path, str):
            if sha256_file(Path(path)) != recorded:
                problems.append(f"input {kind} changed since the base run: {path}")
    return problems


def rewrite_command(command_text_value: str, work_dir: Path,
                    repeat_root: Path, case_prefix: str) -> list[str]:
    """Point the recorded command's case-directory arguments at the attempt dir."""
    base_prefix = str(work_dir / case_prefix)
    attempt_prefix = str(repeat_root / case_prefix)
    return [attempt_prefix + token[len(base_prefix):]
            if token.startswith(base_prefix) else token
            for token in shlex.split(command_text_value)]


def base_receipt(step: dict[str, Any], exclusivity_step: dict[str, Any] | None,
                 case_dir: Path) -> dict[str, Any]:
    receipt = {
        "attempt": 1, "run_id": "base", "step": step.get("name"),
        "command": step.get("command"), "returncode": step.get("returncode"),
        "elapsed_ms": step.get("elapsed_ms"), "dir": str(case_dir),
        "source": "original run_full_e2e_parity step",
    }
    if exclusivity_step is not None:
        receipt["gpu_exclusivity"] = exclusivity_step.get("gpu_exclusivity")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True,
                        help="completed full_e2e_summary.json from run_full_e2e_parity.py")
    parser.add_argument("--attempts", type=int, required=True,
                        help="additional independent cold-process attempts per row (>= 1)")
    parser.add_argument("--start-attempt", type=int, default=1,
                        help="first repeat index to create (use 3 to add an acceptance batch "
                             "after repeat_1/2 already exist as calibration)")
    parser.add_argument("--wait-minutes", type=int, default=30,
                        help="when the GPU is busy, wait up to this many minutes for a free "
                             "window before recording an attempt as refused")
    parser.add_argument("--out", type=Path,
                        help="derived summary path (default: <summary dir>/full_e2e_summary_repeated.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and print the rewritten commands without executing anything")
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be >= 1")
    if args.start_attempt < 1:
        parser.error("--start-attempt must be >= 1")

    summary_path = args.summary.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    work_dir = summary_path.parent
    steps_by_name = {step.get("name"): step for step in summary.get("steps", [])}

    missing = [name for name in TIMING_ROWS
               if not isinstance(steps_by_name.get(name), dict)
               or steps_by_name[name].get("returncode") != 0
               or not steps_by_name[name].get("command")]
    if missing:
        parser.error(f"base summary lacks successful timed steps for: {', '.join(missing)}")

    problems = verify_provenance(summary, work_dir)
    if problems:
        for problem in problems:
            print(f"provenance mismatch: {problem}", file=sys.stderr)
        return 2

    # The base dtype-contract verdicts come from the recorded verification steps.
    for name, row in TIMING_ROWS.items():
        if not row["needs_contract"]:
            continue
        verdict_name = (f"verify_raw_native_cuda_{row['dtype']}_dtype_contract"
                        if name.startswith("raw_native")
                        else f"verify_raw_vulkan_cuda_pbr_{row['dtype']}_dtype_contract")
        verdict_step = steps_by_name.get(verdict_name)
        steps_by_name[name]["base_dtype_contract_ok"] = bool(
            verdict_step is not None and verdict_step.get("returncode") == 0)

    out_path = (args.out or summary_path.parent / "full_e2e_summary_repeated.json").resolve()
    repeat_plan: dict[int, dict[str, list[str]]] = {}
    for attempt in range(args.start_attempt, args.start_attempt + args.attempts):
        repeat_root = work_dir / f"repeat_{attempt}"
        if repeat_root.exists():
            parser.error(f"attempt directory already exists: {repeat_root} (use a fresh work dir)")
        repeat_plan[attempt] = {
            name: rewrite_command(steps_by_name[name]["command"], work_dir,
                                  repeat_root, row["case_prefix"])
            for name, row in TIMING_ROWS.items()
        }

    if args.dry_run:
        for attempt, commands in repeat_plan.items():
            for name, command in commands.items():
                print(f"[repeat_{attempt}] + {command_text(command)}")
        print("dry run: nothing executed, no summary written")
        return 0

    for attempt, commands in repeat_plan.items():
        repeat_root = work_dir / f"repeat_{attempt}"
        repeat_root.mkdir(parents=True)
        for name, command in commands.items():
            row = TIMING_ROWS[name]
            exclusivity_steps: list[dict[str, Any]] = []
            deadline = time.monotonic() + args.wait_minutes * 60.0
            while True:
                exclusivity_steps.clear()
                exclusive_ok = require_exclusive_cuda(
                    f"gpu_exclusive_before_{name}_repeat{attempt}", exclusivity_steps)
                if exclusive_ok or time.monotonic() >= deadline:
                    break
                detail = exclusivity_steps[-1].get("error") or "GPU busy"
                print(f"[repeat_{attempt}] {name}: GPU busy, waiting ({detail[:120]})", flush=True)
                time.sleep(30.0)
            receipt: dict[str, Any] = {
                "attempt": attempt + 1,  # attempt 1 is the base run itself
                "run_id": f"repeat_{attempt}",
                "step": name, "command": command_text(command),
                "dir": str(repeat_root / row["case_prefix"]),
                "gpu_exclusive": exclusive_ok,
                "gpu_exclusivity": (exclusivity_steps[-1].get("gpu_exclusivity")
                                    if exclusivity_steps else None),
                "provenance_verified": True,
            }
            if not exclusive_ok:
                refusal_detail = (exclusivity_steps[-1].get("error")
                                  if exclusivity_steps else "GPU busy")
                receipt.update({"returncode": None, "elapsed_ms": None,
                                "dtype_contract_ok": False,
                                "error": f"GPU not exclusive; attempt refused: {refusal_detail}"})
                print(f"[repeat_{attempt}] {name}: REFUSED - {refusal_detail[:140]}", flush=True)
                steps_by_name[name].setdefault("runs", []).append(receipt)
                continue
            print(f"[repeat_{attempt}] + {command_text(command)}", flush=True)
            started = time.perf_counter()
            result = subprocess.run(command, cwd=REPO_ROOT, text=True, check=False)
            receipt["returncode"] = result.returncode
            receipt["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
            if row["needs_contract"]:
                contract_path = (repeat_root / row["case_prefix"] /
                                 ("dtype_contract.json" if name.startswith("raw_native")
                                  else "vulkan_dtype_contract.json"))
                contract_command = [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "verify_dtype_contract.py"),
                    "--contract", str(contract_path),
                    "--expected-weight-type", row["dtype"],
                    "--require-stage", "mesh_decoder",
                    "--json-out", str(repeat_root / row["case_prefix"] / "dtype_contract_report.json"),
                ]
                contract_rc = subprocess.run(contract_command, cwd=REPO_ROOT,
                                             text=True, check=False).returncode
                receipt["dtype_contract_ok"] = contract_rc == 0
            else:
                receipt["dtype_contract_ok"] = None
            steps_by_name[name].setdefault("runs", []).append(receipt)

    # Assemble runs[]: the base attempt first, but only when a previous driver
    # pass has not already assembled runs[] for this step (the base receipt would
    # otherwise be duplicated on every re-run).
    for name, row in TIMING_ROWS.items():
        step = steps_by_name[name]
        case_dir = work_dir / row["case_prefix"]
        exclusivity_step = steps_by_name.get(row["exclusive_step"])
        existing = step.get("runs", [])
        has_base = any(item.get("run_id") == "base" for item in existing)
        if has_base:
            runs = list(existing)
        else:
            runs = [base_receipt(step, exclusivity_step, case_dir)]
            if row["needs_contract"]:
                runs[0]["dtype_contract_ok"] = step.get("base_dtype_contract_ok")
            runs.extend(existing)
        step["runs"] = runs
        successful = [item["elapsed_ms"] for item in runs
                      if item.get("returncode") == 0
                      and item.get("dtype_contract_ok") is not False
                      and isinstance(item.get("elapsed_ms"), (int, float))]
        if len(successful) >= 2:
            step["elapsed_ms_single_run"] = step["elapsed_ms"]
            step["elapsed_ms"] = statistics.median(successful)
        step["repeat_summary"] = {
            "samples": len(runs), "successful": len(successful),
            "median_ms": step["elapsed_ms"],
            "min_ms": min(successful) if successful else None,
            "max_ms": max(successful) if successful else None,
        }
        print(f"{name}: samples={len(runs)} successful={len(successful)} "
              f"median_ms={step['elapsed_ms']:.1f}")

    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
