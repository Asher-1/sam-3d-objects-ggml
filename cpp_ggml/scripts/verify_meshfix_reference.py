#!/usr/bin/env python3
"""Require exact native MeshFix boundary repair against an official fixture.

The fixture must be created by ``generate_official_pbr_reference.py``. It
contains the mesh passed to ``pymeshfix.PyTMesh`` and the arrays returned by
that same official binding before xatlas splits UV seams.
"""
from __future__ import annotations

import argparse
import struct
import subprocess
from pathlib import Path

import numpy as np


def load_samt(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        if stream.read(4) != b"SAMT":
            raise ValueError(f"{path}: invalid SAMT magic")
        (rank,) = struct.unpack("<i", stream.read(4))
        if rank <= 0 or rank > 8:
            raise ValueError(f"{path}: invalid rank {rank}")
        ggml_shape = struct.unpack(f"<{rank}q", stream.read(rank * 8))
        (ggml_type,) = struct.unpack("<i", stream.read(4))
        dtype = {0: np.dtype("<f4"), 26: np.dtype("<i4"), 30: np.dtype("<i4")}.get(ggml_type)
        if dtype is None:
            raise ValueError(f"{path}: unsupported SAMT type {ggml_type}")
        values = np.frombuffer(stream.read(), dtype=dtype)
    expected = int(np.prod(ggml_shape, dtype=np.int64))
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} values, found {values.size}")
    return values.reshape(tuple(reversed(ggml_shape))).copy()


def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty artifact: {path}")
    return path


def compare_exact(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(
            f"{label}: shape/dtype mismatch {actual.shape} {actual.dtype} != "
            f"{expected.shape} {expected.dtype}"
        )
    if not np.array_equal(actual, expected):
        mismatch = np.argwhere(actual != expected)[0].tolist()
        raise ValueError(
            f"{label}: first exact mismatch at {mismatch}: "
            f"native={actual[tuple(mismatch)]} official={expected[tuple(mismatch)]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-boundary-edges", type=int, default=55)
    parser.add_argument("--no-refine", action="store_true")
    args = parser.parse_args()

    binary = require_file(args.binary.resolve())
    stage_dir = args.stage_dir.resolve()
    vertices = require_file(stage_dir / "asset_mincut_vertices_zup.samt")
    faces = require_file(stage_dir / "asset_mincut_faces.samt")
    expected_vertices = load_samt(require_file(stage_dir / "asset_meshfix_vertices_zup.samt"))
    expected_faces = load_samt(require_file(stage_dir / "asset_meshfix_faces.samt"))

    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"--out-dir must be new or empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    native_vertices = out_dir / "vertices.samt"
    native_faces = out_dir / "faces.samt"
    command = [
        str(binary),
        "mesh-repair-boundaries",
        "--vertices", str(vertices),
        "--faces", str(faces),
        "--vertices-out", str(native_vertices),
        "--faces-out", str(native_faces),
        "--max-boundary-edges", str(args.max_boundary_edges),
    ]
    if args.no_refine:
        command.append("--no-refine")
    subprocess.run(command, check=True)

    compare_exact("vertices", load_samt(native_vertices), expected_vertices)
    compare_exact("faces", load_samt(native_faces), expected_faces)
    print("PASS: native MeshFix boundary repair matches the official mesh exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
