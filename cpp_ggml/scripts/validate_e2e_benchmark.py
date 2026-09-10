#!/usr/bin/env python3
"""Validate raw image-to-PLY-to-render release gates from one E2E report.

The report owns its release contract. That prevents a benchmark cleanup from
silently changing the verifier's expected runner labels, while keeping every
quality and latency requirement explicit beside the measured evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric(row: dict[str, Any], name: str, base_dir: Path) -> float | None:
    value = row.get(name)
    if value is None and name == "render_mae":
        value = row.get("mae")
    if value is None and row.get("render_report"):
        path = Path(str(row["render_report"]))
        path = path if path.is_absolute() else base_dir / path
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get(name)
        except (OSError, json.JSONDecodeError):
            value = None
    return finite_number(value)


def exclusive_gpu_ok(row: dict[str, Any], base_dir: Path) -> bool:
    """Read runner provenance without accepting a missing or shared-GPU record."""
    record = row.get("gpu_exclusivity")
    if not isinstance(record, dict) and row.get("run_summary"):
        path = Path(str(row["run_summary"]))
        path = path if path.is_absolute() else base_dir / path
        try:
            record = json.loads(path.read_text(encoding="utf-8")).get("gpu_exclusivity")
        except (OSError, json.JSONDecodeError):
            record = None
    return (isinstance(record, dict) and record.get("required") is True and
            record.get("checked") is True and not record.get("active_compute_processes"))


def find_row(rows: list[dict[str, Any]], requirement: dict[str, Any]) -> dict[str, Any] | None:
    """Prefer immutable runner IDs; permit backend/dtype matching for new reports."""
    runner = requirement.get("runner")
    if runner:
        return next((row for row in rows if row.get("runner") == runner), None)
    backend, dtype = requirement.get("backend"), requirement.get("dtype")
    return next((row for row in rows
                 if row.get("backend") == backend and row.get("dtype") == dtype), None)


def timer_contract_id(row: dict[str, Any]) -> str | None:
    contract = row.get("timer_contract")
    if not isinstance(contract, dict):
        provenance = row.get("provenance")
        if isinstance(provenance, dict):
            contract = provenance.get("timer_contract")
    value = contract.get("id") if isinstance(contract, dict) else None
    return value if isinstance(value, str) else None


def legacy_contract(rows: list[dict[str, Any]], max_mae: float, max_latency_ms: float | None,
                    require_q8_speed: bool) -> dict[str, Any]:
    """Keep older reports evaluable without reintroducing label-specific checks."""
    required: list[dict[str, Any]] = []
    for row in rows:
        runner = str(row.get("runner", ""))
        lower = runner.lower()
        if not lower.startswith("ggml ") or " q8_0 " not in lower:
            continue
        required.append({"runner": runner, "max_render_mae": max_mae,
                         "max_latency_ms": max_latency_ms,
                         "require_faster_than_reference": require_q8_speed})
    return {"reference_runner": "PyTorch official F16 streamed", "required": required}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="E2E JSON report")
    parser.add_argument("--max-render-mae", type=float,
                        help="override every candidate's render-MAE limit")
    parser.add_argument("--max-e2e-ms", type=float,
                        help="override every candidate's end-to-end latency limit")
    parser.add_argument("--require-q8-speed", action="store_true",
                        help="legacy reports: require Q8 to beat PyTorch")
    parser.add_argument("--output", type=Path, help="write machine-readable gate result")
    args = parser.parse_args()

    report_path = args.input.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") == "sam3d.full-glb-matrix.v1":
        from publish_full_e2e import validate_full_report
        if args.max_render_mae is not None or args.max_e2e_ms is not None:
            parser.error("full-GLB reports use the recorded contract; overrides are not supported")
        result = validate_full_report(report, report_path.parent)
        payload = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        print(payload)
        return 0 if result["passed"] else 1
    rows = report.get("rows")
    if not isinstance(rows, list):
        parser.error("report has no rows array")
    rows = [row for row in rows if isinstance(row, dict)]
    contract = report.get("release_gate")
    if not isinstance(contract, dict):
        contract = legacy_contract(rows, args.max_render_mae or 0.01,
                                   args.max_e2e_ms, args.require_q8_speed)

    required_execution_mode = contract.get("require_candidate_execution_mode")
    candidate_execution_mode = report.get("candidate_execution_mode")
    required_sampling_mode = contract.get("require_candidate_sampling_mode")
    candidate_sampling_mode = report.get("candidate_sampling_mode")
    reference_runner = contract.get("reference_runner")
    reference = next((row for row in rows if row.get("runner") == reference_runner), None)
    reference_latency = finite_number(reference.get("latency_ms")) if reference else None
    reference_timer_contract = timer_contract_id(reference) if reference else None
    failures: list[str] = []
    if required_execution_mode is not None and candidate_execution_mode != required_execution_mode:
        failures.append(
            "candidate execution mode must be "
            f"{required_execution_mode!r}, got {candidate_execution_mode!r}"
        )
    if required_sampling_mode is not None and candidate_sampling_mode != required_sampling_mode:
        failures.append(
            "candidate sampling mode must be "
            f"{required_sampling_mode!r}, got {candidate_sampling_mode!r}"
        )
    if reference_latency is None:
        failures.append(f"missing measured reference runner: {reference_runner!r}")
    matching_timer_contract_required = bool(contract.get("require_matching_timer_contract", False))
    if matching_timer_contract_required and reference_timer_contract is None:
        failures.append(f"{reference_runner}: missing timer-contract ID")
    reference_exclusive_required = bool(contract.get("require_reference_exclusive_gpu", False))
    reference_exclusive_ok = (not reference_exclusive_required or
                              (reference is not None and exclusive_gpu_ok(reference, report_path.parent)))
    if not reference_exclusive_ok:
        failures.append(f"{reference_runner}: missing an exclusive-GPU timing provenance record")

    checks: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    required = contract.get("required")
    if not isinstance(required, list) or not required:
        failures.append("release_gate.required is empty")
        required = []
    for requirement in required:
        if not isinstance(requirement, dict):
            failures.append("release_gate.required contains a non-object entry")
            continue
        label = str(requirement.get("runner") or
                    f"{requirement.get('backend', '?')}/{requirement.get('dtype', '?')}")
        row = find_row(rows, requirement)
        if row is None:
            failures.append(f"missing required E2E row: {label}")
            checks.append({"candidate": label, "present": False})
            continue
        key = str(requirement.get("id", label))
        selected[key] = row
        latency = finite_number(row.get("latency_ms"))
        mae = metric(row, "render_mae", report_path.parent)
        max_mae = args.max_render_mae if args.max_render_mae is not None else finite_number(
            requirement.get("max_render_mae", contract.get("max_render_mae", 0.01)))
        max_latency = args.max_e2e_ms if args.max_e2e_ms is not None else finite_number(
            requirement.get("max_latency_ms", contract.get("max_latency_ms")))
        quality_required = bool(requirement.get("require_quality", True))
        measured = latency is not None and (not quality_required or mae is not None)
        quality_ok = not quality_required or (mae is not None and max_mae is not None and mae <= max_mae)
        latency_ok = max_latency is None or (latency is not None and latency <= max_latency)
        faster_required = bool(requirement.get("require_faster_than_reference", False))
        faster_ok = not faster_required or (latency is not None and reference_latency is not None and
                                            latency < reference_latency)
        exclusive_required = bool(requirement.get("require_exclusive_gpu", False))
        exclusive_ok = not exclusive_required or exclusive_gpu_ok(row, report_path.parent)
        candidate_timer_contract = timer_contract_id(row)
        timer_contract_ok = (not matching_timer_contract_required or
                             (candidate_timer_contract is not None and
                              candidate_timer_contract == reference_timer_contract))
        check = {
            "candidate": label, "present": True, "latency_ms": latency,
            "render_mae": mae, "max_render_mae": max_mae,
            "max_latency_ms": max_latency, "quality": quality_ok,
            "measured": measured,
            "latency_gate": latency_ok, "speed_vs_reference": faster_ok,
            "speed_gate_required": faster_required,
            "exclusive_gpu": exclusive_ok,
            "exclusive_gpu_required": exclusive_required,
            "timer_contract": candidate_timer_contract,
            "reference_timer_contract": reference_timer_contract,
            "timer_contract_match": timer_contract_ok,
            "timer_contract_match_required": matching_timer_contract_required,
        }
        checks.append(check)
        if not measured:
            failures.append(f"{label}: missing a measured raw E2E result")
            continue
        if not quality_ok:
            failures.append(f"{label}: render MAE {mae} exceeds {max_mae}")
        if not latency_ok:
            failures.append(f"{label}: E2E latency {latency} ms exceeds {max_latency} ms")
        if not faster_ok:
            failures.append(f"{label}: E2E latency {latency} ms does not beat PyTorch {reference_latency} ms")
        if not exclusive_ok:
            failures.append(f"{label}: missing an exclusive-GPU timing provenance record")
        if not timer_contract_ok:
            failures.append(
                f"{label}: timer contract {candidate_timer_contract!r} does not match "
                f"reference {reference_timer_contract!r}")

        faster_than = requirement.get("require_faster_than")
        if faster_than:
            peer = selected.get(str(faster_than))
            if peer is None:
                peer_requirement = next((candidate for candidate in required
                                         if isinstance(candidate, dict) and candidate.get("id") == faster_than), None)
                peer = find_row(rows, peer_requirement) if peer_requirement else None
            peer_latency = finite_number(peer.get("latency_ms")) if peer else None
            pair_ok = latency is not None and peer_latency is not None and latency < peer_latency
            check["faster_than"] = faster_than
            check["faster_than_gate"] = pair_ok
            if not pair_ok:
                failures.append(f"{label}: must be faster than {faster_than}")

    result = {
        "schema": "sam3d.e2e.gate.v2", "input": str(report_path),
        "candidate_execution_mode": candidate_execution_mode,
        "required_candidate_execution_mode": required_execution_mode,
        "candidate_sampling_mode": candidate_sampling_mode,
        "required_candidate_sampling_mode": required_sampling_mode,
        "reference_runner": reference_runner, "pytorch_latency_ms": reference_latency,
        "reference_timer_contract": reference_timer_contract,
        "timer_contract_match_required": matching_timer_contract_required,
        "reference_exclusive_gpu": reference_exclusive_ok,
        "reference_exclusive_gpu_required": reference_exclusive_required,
        "checks": checks, "passed": not failures, "failures": failures,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
