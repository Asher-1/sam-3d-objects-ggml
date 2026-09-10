#!/usr/bin/env python3
"""Verify that a native SAM 3D GGML executable has no cuDNN dependency."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def dependency_errors(dependencies: str) -> list[str]:
    lowered = dependencies.lower()
    errors = []
    if "not found" in lowered:
        errors.append("unresolved shared-library dependency")
    for name in ("cudnn", "libtorch", "libpython"):
        if name in lowered:
            errors.append(f"forbidden native runtime dependency: {name}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
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
    errors = dependency_errors(result.stdout)
    with args.binary.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    report = {"schema": "sam3d.native-runtime-dependencies.v1", "binary": str(args.binary.resolve()),
              "binary_sha256": digest.hexdigest(), "ldd": result.stdout, "passed": not errors,
              "errors": errors, "scope": "linked dependency tree; not a dynamic plugin audit"}
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        print(json.dumps(report, indent=2))
        return 1
    print(f"PASS: {args.binary} resolves its linked libraries without cuDNN, Torch or Python")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
