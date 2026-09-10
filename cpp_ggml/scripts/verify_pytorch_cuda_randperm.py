#!/usr/bin/env python3
"""Check the native Philox randperm port against CUDA PyTorch exactly."""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from verify_pytorch_cuda_rng import reference_draws


def read_i32_samt(path: Path) -> tuple[tuple[int, ...], bytes]:
    with path.open("rb") as stream:
        if stream.read(4) != b"SAMT":
            raise ValueError(f"{path}: invalid SAMT magic")
        (rank,) = struct.unpack("<i", stream.read(4))
        dimensions = struct.unpack(f"<{rank}q", stream.read(8 * rank))
        (ggml_type,) = struct.unpack("<i", stream.read(4))
        # GGML_TYPE_I32 is intentionally checked from the file, rather than
        # relying on the numeric enum value in a Python copy of ggml headers.
        if ggml_type <= 0:
            raise ValueError(f"{path}: expected an integer SAMT tensor")
        payload = stream.read()
    count = 1
    for dimension in dimensions:
        count *= dimension
    if len(payload) != count * 4:
        raise ValueError(f"{path}: expected {count * 4} payload bytes, found {len(payload)}")
    return dimensions, payload


def command(binary: Path, implementation: str, seed: int, sizes: str, permutation_size: int,
            output: Path, blocks: int | None = None) -> list[str]:
    result = [str(binary), "rng-dump", "--implementation", implementation, "--seed", str(seed),
              "--sizes", sizes, "--randperm-size", str(permutation_size),
              "--out-dir", str(output)]
    if blocks is not None:
        result.extend(("--distribution-blocks", str(blocks)))
    return result


def distribution_blocks(contract: Path) -> int:
    payload = json.loads(contract.read_text(encoding="utf-8"))
    blocks = payload.get("distribution_blocks")
    if not isinstance(blocks, int) or blocks <= 0:
        raise ValueError(f"{contract}: missing positive distribution_blocks")
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-binary", type=Path, required=True)
    parser.add_argument("--portable-binary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sizes",
        default="6,3,32768,3,1",
        help="comma-separated normal draws that precede coordinate randperm",
    )
    parser.add_argument("--permutation-sizes", default="42000,100000")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help=("fresh official stage oracle; derives the five SS normal draws that precede "
              "coordinate downsampling"),
    )
    args = parser.parse_args()
    for label, binary in (("CUDA", args.cuda_binary), ("portable", args.portable_binary)):
        if not binary.is_file():
            parser.error(f"missing {label} binary: {binary}")
    try:
        import torch
    except ImportError:
        print("SKIP: PyTorch is unavailable", file=sys.stderr)
        return 77
    if not torch.cuda.is_available():
        print("SKIP: CUDA PyTorch is unavailable", file=sys.stderr)
        return 77

    if args.reference_dir is not None:
        reference_sizes, _ = reference_draws(args.reference_dir)
        if len(reference_sizes) < 2:
            raise ValueError(f"{args.reference_dir}: expected SS and SLat RNG draws")
        # The sixth manifest draw is SLat x0. Coordinate randomization happens
        # immediately after SS decoding and before SLat x0 is sampled.
        sizes = reference_sizes[:-1]
    else:
        sizes = [int(value) for value in args.sizes.split(",")]
    permutation_sizes = [int(value) for value in args.permutation_sizes.split(",")]
    if not sizes or any(value <= 0 for value in sizes):
        parser.error("--sizes must contain positive comma-separated integers")
    if not permutation_sizes or any(value <= 0 for value in permutation_sizes):
        parser.error("--permutation-sizes must contain positive comma-separated integers")

    outcomes: list[dict[str, int]] = []
    with tempfile.TemporaryDirectory(prefix="sam3d-pytorch-cuda-randperm-") as directory:
        root = Path(directory)
        for permutation_size in permutation_sizes:
            cuda_dir = root / f"cuda-{permutation_size}"
            portable_dir = root / f"portable-{permutation_size}"
            cuda_command = command(args.cuda_binary, "cuda", args.seed, args.sizes,
                                   permutation_size, cuda_dir)
            subprocess.run(cuda_command, check=True)
            blocks = distribution_blocks(cuda_dir / "rng_contract.json")
            portable_command = command(args.portable_binary, "portable", args.seed, args.sizes,
                                       permutation_size, portable_dir, blocks)
            subprocess.run(portable_command, check=True)

            torch.manual_seed(args.seed)
            for size in sizes:
                torch.randn(size, device="cuda", dtype=torch.float32)
            expected = torch.randperm(permutation_size, device="cuda", dtype=torch.int32)
            dimensions, cuda_payload = read_i32_samt(cuda_dir / "randperm.samt")
            portable_dimensions, portable_payload = read_i32_samt(portable_dir / "randperm.samt")
            expected_payload = expected.cpu().numpy().tobytes()
            if dimensions != (permutation_size,) or portable_dimensions != dimensions:
                raise AssertionError(f"randperm({permutation_size}) dimensions differ")
            if cuda_payload != expected_payload:
                raise AssertionError(f"CUDA native randperm({permutation_size}) differs from PyTorch")
            if portable_payload != expected_payload:
                raise AssertionError(f"portable randperm({permutation_size}) differs from PyTorch")
            outcomes.append({"count": permutation_size, "distribution_blocks": blocks})
    print(json.dumps({"passed": True, "seed": args.seed, "normal_draw_sizes": sizes,
                      "cases": outcomes}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
