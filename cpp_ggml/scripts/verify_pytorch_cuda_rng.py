#!/usr/bin/env python3
"""Bitwise-check the native Philox normal sampler against torch.randn CUDA."""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from samt_io import read_samt_bytes as read_samt




def reference_draws(reference_dir: Path) -> tuple[list[int], list[bytes]]:
    manifest_path = reference_dir / "manifest.json"
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    provenance = manifest.get("reference_provenance", {})
    if provenance.get("sampling_seed_mode") != "pipeline":
        raise ValueError(
            f"{manifest_path}: expected pipeline sampling_seed_mode, found "
            f"{provenance.get('sampling_seed_mode')!r}"
        )
    names = (
        "ss_x0_6drotation_normalized",
        "ss_x0_scale",
        "ss_x0_shape",
        "ss_x0_translation",
        "ss_x0_translation_scale",
        "slat_x0",
    )
    sizes: list[int] = []
    payloads: list[bytes] = []
    for name in names:
        metadata = manifest.get(name)
        if not isinstance(metadata, dict) or "file" not in metadata:
            raise ValueError(f"{manifest_path}: missing RNG draw {name}")
        dimensions, payload = read_samt(reference_dir / metadata["file"])
        count = 1
        for dimension in dimensions:
            count *= dimension
        sizes.append(count)
        payloads.append(payload)
    return sizes, payloads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sizes", default="1,3,4,5,6,255,256,257,32768,210248")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="official pipeline-mode reference; verifies its complete SS-to-SLat RNG stream",
    )
    args = parser.parse_args()
    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)
    if args.reference_dir is not None:
        sizes, expected = reference_draws(args.reference_dir)
    else:
        try:
            import torch
        except ImportError:
            print("SKIP: PyTorch is unavailable", file=sys.stderr)
            return 77
        if not torch.cuda.is_available():
            print("SKIP: CUDA PyTorch is unavailable", file=sys.stderr)
            return 77
        sizes = [int(value) for value in args.sizes.split(",")]
        if not sizes or any(value <= 0 for value in sizes):
            parser.error("--sizes must contain positive comma-separated integers")
        torch.manual_seed(args.seed)
        expected = [torch.randn(size, device="cuda", dtype=torch.float32).cpu().numpy().tobytes()
                    for size in sizes]
    with tempfile.TemporaryDirectory(prefix="sam3d-pytorch-cuda-rng-") as directory:
        output = Path(directory)
        command = [str(args.binary), "rng-dump", "--seed", str(args.seed),
                   "--sizes", ",".join(str(size) for size in sizes), "--out-dir", str(output)]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        for index, expected_bytes in enumerate(expected):
            _, actual_bytes = read_samt(output / f"rng_{index:02d}.samt")
            if actual_bytes != expected_bytes:
                mismatch = next(position for position, pair in enumerate(zip(actual_bytes, expected_bytes))
                                if pair[0] != pair[1])
                raise AssertionError(
                    f"draw {index} (size={sizes[index]}) differs at byte {mismatch}: "
                    f"native={actual_bytes[mismatch]} torch={expected_bytes[mismatch]}"
                )
    source = "official pipeline reference" if args.reference_dir is not None else "torch.randn"
    print(f"PASS: {len(sizes)} native CUDA Philox draws match {source} bitwise")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
