#!/usr/bin/env python3
"""Verify that a native SAM 3D GGML executable has no cuDNN dependency."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()

    if not args.binary.is_file():
        parser.error(f"missing executable: {args.binary}")
    ldd = shutil.which("ldd")
    if ldd is None:
        parser.error("ldd is required to verify native runtime dependencies")

    result = subprocess.run([ldd, str(args.binary)], text=True, capture_output=True)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        parser.error(f"ldd failed for {args.binary}: {detail}")
    dependencies = result.stdout.lower()
    if "cudnn" in dependencies:
        raise RuntimeError(f"native GGML runtime links cuDNN:\n{result.stdout}")
    print(f"PASS: {args.binary} has no cuDNN dependency")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
