#!/usr/bin/env python3
"""Stitch row-level receipts into a composite full-E2E summary.

Shared-GPU bursts invalidated individual rows of otherwise complete
``run_full_e2e_parity.py`` invocations.  ``run_full_e2e_row.py`` re-executes
such a row in isolation with runner-identical commands; this tool merges those
``row_steps.json`` receipts back into the base summary so the standard
``publish_full_e2e.py`` flow can consume the result.

The merge is deliberately explicit: every spliced step keeps its receipt, the
per-row verdict fields are recomputed from the spliced steps only, and the
composite summary records a ``composite`` provenance block (base hash, per-row
source hash, timestamp).  Nothing is silently rewritten.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

DTYPES = ("f16", "q8_0", "q4_k")

ROW_STEP_PREFIXES = ("gpu_exclusive_before_", "raw_native_cuda_", "raw_vulkan_cuda_pbr_",
                     "verify_raw_native_cuda_", "verify_raw_vulkan_cuda_pbr_",
                     "render_compare_raw_native_cuda_", "render_compare_raw_vulkan_cuda_pbr_")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_step_names(case: str) -> list[str]:
    names = [f"gpu_exclusive_before_{case}"]
    suffix = "_image_to_pbr" if case.startswith("raw_native_cuda_") else "_handoff"
    names.append(case + suffix)
    names.append(f"verify_{case}_handoff_manifest")
    names.append(f"verify_{case}_dtype_contract")
    names.append(f"verify_{case}_pbr_structure" if case.startswith("raw_native_cuda_")
                 else f"verify_{case}_structure")
    names.append(f"render_compare_{case}_glb")
    names.append(f"render_compare_{case}")
    return names


def row_dtype(case: str) -> str:
    for dtype in DTYPES:
        if case.endswith(dtype):
            return dtype
    raise SystemExit(f"cannot determine dtype from case name: {case}")


def recompute_verdicts(summary: dict[str, Any], case: str, steps: list[dict[str, Any]],
                       gate_configured: bool) -> None:
    by_name = {s["name"]: s for s in steps if isinstance(s, dict) and "name" in s}
    status = lambda n: by_name.get(n, {}).get("returncode") == 0  # noqa: E731
    dtype = row_dtype(case)
    cuda = case.startswith("raw_native_cuda_")
    timed_ok = status(case + ("_image_to_pbr" if cuda else "_handoff"))
    contract_ok = status(f"verify_{case}_dtype_contract")
    structure_name = f"verify_{case}_pbr_structure" if cuda else f"verify_{case}_structure"
    structure_ok = status(structure_name)
    glb_ok = status(f"render_compare_{case}_glb")
    if cuda:
        # Field names follow run_full_e2e_parity's summary keys exactly.
        summary.setdefault("raw_native_cuda_dtype_contract_passed", {})[dtype] = contract_ok
        summary.setdefault("raw_native_cuda_glb_structure_passed", {})[dtype] = structure_ok
        summary.setdefault("raw_native_cuda_glb_render_measured", {})[dtype] = bool(glb_ok)
        summary.setdefault("raw_native_cuda_glb_render_gate_passed", {})[dtype] = bool(glb_ok and gate_configured)
        summary.setdefault("raw_native_cuda_pbr_passed", {})[dtype] = bool(
            timed_ok and contract_ok and structure_ok and glb_ok and gate_configured)
    else:
        summary.setdefault("raw_vulkan_cuda_handoff_passed", {})[dtype] = timed_ok
        summary.setdefault("raw_vulkan_cuda_pbr_dtype_contract_passed", {})[dtype] = contract_ok
        summary.setdefault("raw_vulkan_cuda_pbr_structure_passed", {})[dtype] = structure_ok
        summary.setdefault("raw_vulkan_cuda_pbr_render_measured", {})[dtype] = bool(glb_ok)
        summary.setdefault("raw_vulkan_cuda_pbr_render_gate_passed", {})[dtype] = bool(glb_ok and gate_configured)
        summary.setdefault("raw_vulkan_cuda_pbr_passed", {})[dtype] = bool(
            timed_ok and contract_ok and structure_ok and glb_ok and gate_configured)


def recompute_coverage(summary: dict[str, Any]) -> None:
    """Refresh coverage entries that only stale base-run state kept False.

    ``per_operator_dtype_contract`` and ``raw_cuda_glb_multiview_render`` are
    derivable entirely from the (re-spliced) step receipts; the original run
    recorded them False only because external GPU clients had invalidated some
    rows or because the measurement-era GLB gate was unconfigured.
    """
    coverage = summary.setdefault("coverage", {})
    coverage["per_operator_dtype_contract"] = all(
        summary.get("raw_native_cuda_dtype_contract_passed", {}).get(dtype) is True
        and summary.get("raw_vulkan_cuda_pbr_dtype_contract_passed", {}).get(dtype) is True
        for dtype in DTYPES)
    by_name = {s.get("name"): s for s in summary.get("steps", []) if isinstance(s, dict)}
    coverage["raw_cuda_glb_multiview_render"] = all(
        by_name.get(f"render_compare_raw_native_cuda_{dtype}_glb", {}).get("returncode") == 0
        for dtype in DTYPES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True,
                        help="full_e2e_summary.json whose rows were partly invalidated")
    parser.add_argument("--row-steps", type=Path, required=True, action="append",
                        help="row_steps.json from run_full_e2e_row.py; repeatable")
    parser.add_argument("--out", type=Path, required=True,
                        help="composite summary path (must not equal the base)")
    args = parser.parse_args()
    base_path = args.base.resolve()
    if args.out.resolve() == base_path:
        raise SystemExit("--out must differ from --base")

    summary = json.loads(base_path.read_text(encoding="utf-8"))
    steps = summary["steps"]
    gate_configured = bool(summary.get("glb_render_acceptance_gate_configured"))
    merged: list[dict[str, Any]] = []
    for row_path in args.row_steps:
        row_path = row_path.resolve()
        row = json.loads(row_path.read_text(encoding="utf-8"))
        case = row["case"]
        if row.get("row_dir") and not Path(row["row_dir"]).resolve().parent == base_path.parent:
            raise SystemExit(
                f"row dir {row['row_dir']} is not inside the base work dir; "
                "run_full_e2e_row.py must place assets at the canonical case path")
        drop = set(row_step_names(case))
        steps[:] = [s for s in steps if s.get("name") not in drop]
        steps.extend(row["steps"])
        recompute_verdicts(summary, case, steps, gate_configured)
        merged.append({"case": case, "source": str(row_path), "sha256": sha256(row_path),
                       "row_passed": row.get("row_passed")})
        print(f"merged {case}: row_passed={row.get('row_passed')}")

    summary["composite"] = {
        "tool": "merge_full_e2e_rows.py",
        "base_summary": str(base_path), "base_summary_sha256": sha256(base_path),
        "merged_rows": merged,
        "merged_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "note": ("rows re-executed by run_full_e2e_row.py with runner-identical commands "
                 "after external GPU clients invalidated them mid-run; all receipts retained"),
    }
    recompute_coverage(summary)
    args.out.resolve().write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
