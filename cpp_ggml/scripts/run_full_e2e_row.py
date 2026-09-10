#!/usr/bin/env python3
"""Run one full-E2E raw row (CUDA or Vulkan+CPBR) with runner-identical steps.

The monolithic ``run_full_e2e_parity.py`` needs an exclusively idle GPU for its
full ~1.5 h duration; bursty external compute clients repeatedly invalidated
its tail (Vulkan) rows.  This tool executes a single row's exact step sequence
- GPU exclusivity, timed cold process, dtype contract, PBR structure, GLB and
PLY render comparisons - using paths and binaries from a completed base summary
and a command template from any recorded row of the same backend.

Each invocation writes a ``row_steps.json`` receipt set in runner step schema;
``merge_full_e2e_rows.py`` stitches these receipts into a composite summary with
explicit provenance so ``publish_full_e2e.py`` can consume them.  Repeat a row
by invoking this tool once per attempt into distinct ``--out-dir`` values.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from run_full_e2e_parity import (
    REPO_ROOT,
    SCRIPTS,
    command_text,
    require_exclusive_cuda,
    valid_vulkan_cuda_handoff,
)
from run_image_to_3d import sha256_file

DTYPES = ("f16", "q8_0", "q4_k")


def load_summaries(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(p.resolve().read_text(encoding="utf-8")) for p in paths]


def steps_by_name(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {s.get("name"): s for s in summary.get("steps", [])}


def find_reference_row(summaries: list[dict[str, Any]], backend: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Return (dtype, summary, timed step) of a recorded row for this backend."""
    suffix = "_image_to_pbr" if backend == "cuda" else "_handoff"
    for summary in summaries:
        steps = steps_by_name(summary)
        for dtype in DTYPES:
            prefix = f"raw_native_cuda_{dtype}" if backend == "cuda" else f"raw_vulkan_cuda_pbr_{dtype}"
            step = steps.get(prefix + suffix)
            if step and step.get("command") and step.get("returncode") == 0:
                return dtype, summary, step
    raise SystemExit(f"no recorded successful {backend} row command in the given summaries")


def official_python_of(summary: dict[str, Any]) -> str:
    """The official interpreter, taken from any recorded official-side step."""
    for step in summary.get("steps", []):
        command = step.get("command", "")
        if any(marker in command for marker in (
                "verify_official_pbr_reference.py", "render_glb_compare.py",
                "render_compare.py", "bench_full_e2e_pipeline.py")):
            return shlex.split(command)[0]
    raise SystemExit("could not determine the official python from the base summary")


def parse_option(command: str, option: str) -> str:
    tokens = shlex.split(command)
    return tokens[tokens.index(option) + 1]


def rewrite_dtype_and_dir(command: str, ref_dtype: str, ref_dir: Path,
                          target_dtype: str, row_dir: Path) -> list[str]:
    tokens = shlex.split(command)
    ref_prefix, tgt_prefix = str(ref_dir), str(row_dir)
    rewritten = [tgt_prefix + t[len(ref_prefix):] if t.startswith(ref_prefix) else t
                 for t in tokens]
    for i, t in enumerate(rewritten):
        if t == "--dtype" and i + 1 < len(rewritten) and rewritten[i + 1] == ref_dtype:
            rewritten[i + 1] = target_dtype
    return rewritten


def run_step(name: str, command: list[str], steps: list[dict[str, Any]],
             timed: bool = False) -> bool:
    print(f"[{name}] + {command_text(command)}", flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, check=False)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    steps.append({"name": name, "command": command_text(command),
                  "returncode": result.returncode, "elapsed_ms": elapsed_ms,
                  "status": "PASS" if result.returncode == 0 else "FAIL",
                  **({"timed_step": True} if timed else {})})
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, action="append",
                        help="base full_e2e_summary.json; repeatable, first entry owns the work dir")
    parser.add_argument("--backend", choices=("cuda", "vulkan"), required=True)
    parser.add_argument("--dtype", choices=DTYPES, required=True)
    parser.add_argument("--ss-attention", choices=("normal", "strict"), default="normal",
                        help="SS attention path for the timed row (delivery default: normal)")
    parser.add_argument("--work-dir", type=Path,
                        help="work dir owning official_stages for the comparisons "
                             "(default: first --summary's directory)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="new directory for this row's assets (the case directory)")
    parser.add_argument("--steps-out", type=Path,
                        help="receipts JSON (default: <out-dir>/row_steps.json)")
    args = parser.parse_args()

    summaries = load_summaries(args.summary)
    work_dir = (args.work_dir or args.summary[0].resolve().parent).resolve()
    stages = work_dir / "official_stages"
    reference_glb = stages / "official_pbr_reference" / "official_pbr.glb"
    if not reference_glb.is_file():
        raise SystemExit(f"missing official reference GLB: {reference_glb}")

    ref_dtype, ref_summary, ref_step = find_reference_row(summaries, args.backend)
    ref_summary_dir = None
    for path, summary in zip(args.summary, summaries):
        if summary is ref_summary:
            ref_summary_dir = path.resolve().parent
    ref_dir = ref_summary_dir / (
        f"raw_native_cuda_{ref_dtype}" if args.backend == "cuda" else f"raw_vulkan_cuda_pbr_{ref_dtype}")
    ref_command = ref_step["command"]

    official_python = official_python_of(ref_summary)
    models_dir = parse_option(ref_command, "--model" if args.backend == "cuda" else "--models-dir")
    moge_model = parse_option(ref_command, "--moge-model")
    image = parse_option(ref_command, "--image")
    mask = parse_option(ref_command, "--mask")
    seed = parse_option(ref_command, "--seed")
    threads = parse_option(ref_command, "--threads")

    vulkan_binary = ref_summary["binaries"]["vulkan"]
    pbr_binary = ref_summary["binaries"]["native_pbr"]
    cuda_binary = ref_summary["binaries"]["cuda"]
    # Pin the exact binaries this row executed (merge/provenance evidence).
    row_binaries = ({"cuda": cuda_binary} if args.backend == "cuda"
                    else {"vulkan": vulkan_binary, "native_pbr": pbr_binary})
    binary_sha256 = {name: sha256_file(Path(path)) for name, path in row_binaries.items()}
    row_dir = args.out_dir.resolve()
    row_dir.mkdir(parents=True)
    dtype = args.dtype

    if args.backend == "cuda":
        case = f"raw_native_cuda_{dtype}"
        timed_command = [
            pbr_binary, "image-to-3d", "--model", models_dir, "--moge-model", moge_model,
            "--image", image, "--mask", mask, "--backend", "cuda", "--dtype", dtype,
            "--seed", seed, "--threads", threads,
            "--out", str(row_dir / "output.ply"), "--pbr-out", str(row_dir / "output.glb"),
            "--pose-out", str(row_dir / "pose.json"),
            "--dtype-contract-out", str(row_dir / "dtype_contract.json"),
            "--ss-attention", args.ss_attention,
        ]
        native_glb, native_ply = row_dir / "output.glb", row_dir / "output.ply"
        contract = row_dir / "dtype_contract.json"
    else:
        case = f"raw_vulkan_cuda_pbr_{dtype}"
        blocks = parse_option(ref_command, "--rng-distribution-blocks")
        timed_command = [
            sys.executable, str(SCRIPTS / "run_vulkan_cuda_pbr.py"),
            "--vulkan-binary", vulkan_binary, "--cuda-pbr-binary", pbr_binary,
            "--models-dir", models_dir, "--moge-model", moge_model,
            "--image", image, "--mask", mask, "--out-dir", str(row_dir),
            "--dtype", dtype, "--seed", seed, "--threads", threads,
            "--ss-attention", args.ss_attention, "--rng-distribution-blocks", blocks,
        ]
        native_glb, native_ply = row_dir / "cuda_pbr.glb", row_dir / "vulkan_output.ply"
        contract = row_dir / "vulkan_dtype_contract.json"

    contract_type = "dtype_contract.json" if args.backend == "cuda" else "vulkan_dtype_contract.json"
    commands = {
        "verify_dtype_contract": [
            sys.executable, str(SCRIPTS / "verify_dtype_contract.py"),
            "--contract", contract, "--expected-weight-type", dtype,
            "--require-stage", "mesh_decoder",
            "--json-out", str(row_dir / "dtype_contract_report.json"),
        ],
        "verify_pbr_structure": [
            sys.executable, str(SCRIPTS / "verify_native_pbr_pipeline.py"),
            "--native-glb", native_glb, "--structural-only",
            "--json-out", str(row_dir / "pbr_structure_report.json"),
        ],
        "render_compare_glb": [
            official_python, str(SCRIPTS / "render_glb_compare.py"),
            "--reference-glb", reference_glb, "--native-glb", native_glb,
            "--out-dir", str(row_dir / "glb_render_compare"),
            "--frames", "60", "--resolution", "512",
        ],
        "render_compare_ply": [
            official_python, str(SCRIPTS / "render_compare.py"),
            "--pytorch-ply", stages / "output_gs.ply", "--ggml-ply", native_ply,
            "--condition-dir", stages, "--out-dir", str(row_dir / "render_compare"),
            "--frames", "60", "--resolution", "512", "--max-mae", "0.01",
        ],
    }
    for name, command in list(commands.items()):
        commands[name] = [str(part) for part in command]

    steps: list[dict[str, Any]] = []
    exclusive_ok = require_exclusive_cuda(f"gpu_exclusive_before_{case}", steps)
    ok = exclusive_ok
    timed_ok = False
    if ok:
        timed_ok = run_step(f"{case}_{'image_to_pbr' if args.backend == 'cuda' else 'handoff'}",
                            timed_command, steps, timed=True)
        ok = timed_ok
    if ok and args.backend == "vulkan":
        manifest_ok = valid_vulkan_cuda_handoff(row_dir / "handoff_manifest.json")
        steps.append({"name": f"verify_{case}_handoff_manifest",
                      "command": f"validate {row_dir / 'handoff_manifest.json'}",
                      "returncode": 0 if manifest_ok else 1, "elapsed_ms": 0.0,
                      "status": "PASS" if manifest_ok else "FAIL"})
        ok = manifest_ok
    if ok:
        ok = run_step(f"verify_{case}_dtype_contract", commands["verify_dtype_contract"], steps)
    if ok:
        # The runner spells the structure step without "_pbr" on Vulkan rows.
        structure_name = (f"verify_{case}_pbr_structure" if args.backend == "cuda"
                          else f"verify_{case}_structure")
        ok = run_step(structure_name, commands["verify_pbr_structure"], steps)
    if ok:
        glb_ok = run_step(f"render_compare_{case}_glb", commands["render_compare_glb"], steps)
    if ok and native_ply.is_file():
        run_step(f"render_compare_{case}", commands["render_compare_ply"], steps)

    verdict = {
        "backend": args.backend, "dtype": dtype, "case": case,
        "row_dir": str(row_dir), "derived_from_dtype": ref_dtype,
        "derived_from_summary": str(ref_summary_dir / "full_e2e_summary.json"),
        "row_passed": bool(ok), "timed_passed": bool(timed_ok),
        "gpu_exclusive": exclusive_ok, "ss_attention": args.ss_attention,
        "binary_sha256": binary_sha256, "steps": steps,
    }
    steps_out = (args.steps_out or row_dir / "row_steps.json").resolve()
    steps_out.parent.mkdir(parents=True, exist_ok=True)
    steps_out.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {steps_out} (row {'PASS' if ok else 'FAIL'})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
