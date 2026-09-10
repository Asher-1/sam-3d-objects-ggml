#!/usr/bin/env python3
"""Compare the portable Philox sampler with the verified CUDA sampler.

CUDA's ``sincosf``/``logf`` and the host C++ math library need not return the
same final floating-point bit. This verifier freezes the CUDA launch-capacity
contract, then gates the complete state/scatter stream by measured numeric
error rather than falsely requiring byte equality across math libraries.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path

from verify_pytorch_cuda_rng import read_samt, reference_draws


def parse_sizes(text: str) -> list[int]:
    try:
        sizes = [int(value) for value in text.split(",")]
    except ValueError as error:
        raise ValueError("--sizes must be comma-separated positive integers") from error
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError("--sizes must contain at least one positive integer")
    return sizes


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def contract_blocks(path: Path) -> int:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
        blocks = contract["distribution_blocks"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"invalid CUDA RNG contract: {path}") from error
    if not isinstance(blocks, int) or blocks <= 0:
        raise ValueError(f"invalid CUDA RNG distribution_blocks: {blocks!r}")
    return blocks


def compare_payloads(cuda_payload: bytes, portable_payload: bytes) -> dict[str, float | int]:
    if len(cuda_payload) != len(portable_payload) or len(cuda_payload) % 4 != 0:
        raise ValueError("RNG payload lengths differ or are not F32")
    maximum = 0.0
    total = 0.0
    differing_values = 0
    for offset in range(0, len(cuda_payload), 4):
        cuda_bits = cuda_payload[offset:offset + 4]
        portable_bits = portable_payload[offset:offset + 4]
        if cuda_bits != portable_bits:
            differing_values += 1
        cuda_value = struct.unpack_from("<f", cuda_payload, offset)[0]
        portable_value = struct.unpack_from("<f", portable_payload, offset)[0]
        error = abs(cuda_value - portable_value)
        if not math.isfinite(error):
            raise ValueError(f"non-finite RNG comparison error at byte {offset}")
        maximum = max(maximum, error)
        total += error
    count = len(cuda_payload) // 4
    return {
        "count": count,
        "different_float_bits": differing_values,
        "max_abs": maximum,
        "mean_abs": total / count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-binary", type=Path, required=True)
    parser.add_argument("--portable-binary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sizes", default="1,3,4,5,255,256,257,4096,6,3,32768,3,1,210248")
    parser.add_argument("--reference-dir", type=Path,
                        help="derive the complete SS-to-SLat stream from a fresh official stage oracle")
    parser.add_argument("--max-abs", type=float, default=3.0e-6)
    parser.add_argument("--max-mean-abs", type=float, default=4.0e-7)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.max_abs < 0.0 or args.max_mean_abs < 0.0:
        parser.error("RNG error thresholds must be non-negative")
    for label, binary in (("CUDA", args.cuda_binary), ("portable", args.portable_binary)):
        if not binary.is_file():
            parser.error(f"missing {label} binary: {binary}")
    if args.reference_dir is not None:
        sizes, _ = reference_draws(args.reference_dir)
    else:
        sizes = parse_sizes(args.sizes)

    with tempfile.TemporaryDirectory(prefix="sam3d-portable-philox-") as temporary:
        root = Path(temporary)
        cuda_dir = root / "cuda"
        portable_dir = root / "portable"
        sizes_text = ",".join(str(size) for size in sizes)
        run([
            str(args.cuda_binary), "rng-dump", "--implementation", "cuda", "--seed", str(args.seed),
            "--sizes", sizes_text, "--out-dir", str(cuda_dir),
        ])
        blocks = contract_blocks(cuda_dir / "rng_contract.json")
        run([
            str(args.portable_binary), "rng-dump", "--implementation", "portable",
            "--distribution-blocks", str(blocks), "--seed", str(args.seed),
            "--sizes", sizes_text, "--out-dir", str(portable_dir),
        ])

        draws: list[dict[str, float | int]] = []
        total_count = 0
        total_error = 0.0
        maximum_error = 0.0
        different_float_bits = 0
        for index, size in enumerate(sizes):
            cuda_dimensions, cuda_payload = read_samt(cuda_dir / f"rng_{index:02d}.samt")
            portable_dimensions, portable_payload = read_samt(portable_dir / f"rng_{index:02d}.samt")
            if cuda_dimensions != portable_dimensions:
                raise ValueError(f"draw {index}: RNG dimensions differ")
            metrics = compare_payloads(cuda_payload, portable_payload)
            if metrics["count"] != size:
                raise ValueError(f"draw {index}: expected {size} samples, got {metrics['count']}")
            metrics["draw_index"] = index
            draws.append(metrics)
            total_count += int(metrics["count"])
            total_error += float(metrics["mean_abs"]) * int(metrics["count"])
            maximum_error = max(maximum_error, float(metrics["max_abs"]))
            different_float_bits += int(metrics["different_float_bits"])

    mean_error = total_error / total_count
    report = {
        "schema": "sam3d.portable-philox-regression.v1",
        "seed": args.seed,
        "distribution_blocks": blocks,
        "source": "verified CUDA Philox sampler",
        "portable_binary": str(args.portable_binary.resolve()),
        "cuda_binary": str(args.cuda_binary.resolve()),
        "draws": draws,
        "total_count": total_count,
        "different_float_bits": different_float_bits,
        "max_abs": maximum_error,
        "mean_abs": mean_error,
        "thresholds": {"max_abs": args.max_abs, "max_mean_abs": args.max_mean_abs},
        "passed": maximum_error <= args.max_abs and mean_error <= args.max_mean_abs,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
