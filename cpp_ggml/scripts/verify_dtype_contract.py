#!/usr/bin/env python3
"""Validate a native E2E dtype-contract artifact emitted from a ggml graph.

This checks evidence collected from a constructed graph, rather than inferring
the compute types from model filenames.  It deliberately does not claim
PyTorch numerical parity: that still requires the independent trajectory and
render gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_STAGES = (
    "ss_condition_dino",
    "ss_condition_pointpatch",
    "ss_condition_fuser",
    "ss_flow",
    "ss_decoder",
    "slat_condition_dino",
    "slat_condition_fuser",
    "slat_flow",
    "gaussian_decoder",
)


def fail(message: str) -> None:
    raise SystemExit(f"dtype contract verification failed: {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-weight-type",
                        choices=("f32", "f16", "q4_0", "q4_1", "q4_k", "q8_0"),
                        help="require this type as an observed MUL_MAT source")
    parser.add_argument("--require-stage", action="append", default=[],
                        help="additional graph stage that must be present")
    parser.add_argument("--no-default-stages", action="store_true",
                        help="do not require the standard neural E2E stages")
    parser.add_argument("--json-out", type=Path,
                        help="optional normalized validation report")
    args = parser.parse_args()

    try:
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {args.contract}: {error}")
    if contract.get("schema") != "sam3d.e2e.dtype-contract.v1":
        fail(f"unexpected schema {contract.get('schema')!r}")
    if contract.get("complete") is not True:
        fail("recording is incomplete; native execution did not reach its success boundary")
    model_dtypes = contract.get("model_dtypes")
    if not isinstance(model_dtypes, dict):
        fail("model_dtypes must be an object")
    if args.expected_weight_type is not None:
        declared_types = {str(value) for value in model_dtypes.values()}
        if declared_types != {args.expected_weight_type}:
            fail("recorded model dtypes do not match the requested homogeneous run: " +
                 ", ".join(sorted(declared_types)))
    entries = contract.get("entries")
    if not isinstance(entries, list) or not entries:
        fail("entries must be a non-empty list")

    stages: set[str] = set()
    observed_mul_mat_types: set[str] = set()
    total_nodes = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            fail(f"entry {index} is not an object")
        stage = entry.get("stage")
        operation = entry.get("op")
        inputs = entry.get("input_types")
        output = entry.get("output_type")
        nodes = entry.get("node_count")
        max_elements = entry.get("max_output_elements")
        if not isinstance(stage, str) or not stage:
            fail(f"entry {index} has no stage")
        if not isinstance(operation, str) or not operation:
            fail(f"entry {index} has no operation")
        if not isinstance(inputs, list) or not all(isinstance(value, str) for value in inputs):
            fail(f"entry {index} has invalid input_types")
        if not isinstance(output, str) or not output:
            fail(f"entry {index} has no output_type")
        if not isinstance(nodes, int) or nodes <= 0:
            fail(f"entry {index} has invalid node_count")
        if not isinstance(max_elements, int) or max_elements <= 0:
            fail(f"entry {index} has invalid max_output_elements")
        stages.add(stage)
        total_nodes += nodes
        if operation == "MUL_MAT":
            observed_mul_mat_types.update(inputs)

    required_stages = set(args.require_stage)
    if not args.no_default_stages:
        required_stages.update(DEFAULT_STAGES)
    missing_stages = sorted(required_stages - stages)
    if missing_stages:
        fail("missing required stages: " + ", ".join(missing_stages))
    if not observed_mul_mat_types:
        fail("no observed MUL_MAT entries")
    if (args.expected_weight_type is not None and
            args.expected_weight_type not in observed_mul_mat_types):
        fail("expected MUL_MAT source type " + args.expected_weight_type +
             ", observed " + ", ".join(sorted(observed_mul_mat_types)))

    report = {
        "schema": "sam3d.e2e.dtype-contract-validation.v1",
        "contract": str(args.contract.resolve()),
        "backend": contract.get("backend"),
        "model_dtypes": model_dtypes,
        "stage_count": len(stages),
        "stages": sorted(stages),
        "entry_count": len(entries),
        "aggregated_graph_nodes": total_nodes,
        "mul_mat_source_types": sorted(observed_mul_mat_types),
        "expected_weight_type": args.expected_weight_type,
        "passed": True,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
