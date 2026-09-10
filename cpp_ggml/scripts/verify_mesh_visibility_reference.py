#!/usr/bin/env python3
"""Compare native visibility/mincut output with the captured official boundary.

The reference bundle records the exact mesh passed into MeshFix after the
official visibility rasterization and graph cut. This verifier runs only the
corresponding native stage on the saved post-VTK mesh and requires exact SAMT
arrays. It intentionally stops before MeshFix boundary remeshing.
"""
from __future__ import annotations

import argparse
import struct
import subprocess
from pathlib import Path

import numpy as np
from samt_io import load_samt




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


def compare_visibility(actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(
            f"visibility: shape/dtype mismatch {actual.shape} {actual.dtype} != "
            f"{expected.shape} {expected.dtype}"
        )
    difference = np.abs(actual - expected)
    print(
        "visibility: "
        f"exact={bool(np.array_equal(actual, expected))} "
        f"mae={float(difference.mean()):.9g} "
        f"max_abs={float(difference.max()):.9g} "
        f"mismatched_faces={int(np.count_nonzero(difference))}/{actual.size}"
    )
    compare_exact("visibility", actual, expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--native-rasterizer",
        action="store_true",
        help="compute the 1000-view visibility tensor with native mesh-visibility before mincut",
    )
    parser.add_argument("--visibility-views", type=int, default=1000)
    parser.add_argument("--visibility-resolution", type=int, default=1024)
    args = parser.parse_args()

    binary = require_file(args.binary.resolve())
    stage_dir = args.stage_dir.resolve()
    vertices = require_file(stage_dir / "asset_decimated_vertices_zup.samt")
    faces = require_file(stage_dir / "asset_decimated_faces.samt")
    visibility = require_file(stage_dir / "asset_decimated_visibility.samt")
    expected_vertices = load_samt(require_file(stage_dir / "asset_mincut_vertices_zup.samt"))
    expected_faces = load_samt(require_file(stage_dir / "asset_mincut_faces.samt"))
    expected_candidates = load_samt(require_file(stage_dir / "asset_mincut_candidate_faces.samt"))

    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"--out-dir must be new or empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    native_vertices = out_dir / "vertices.samt"
    native_faces = out_dir / "faces.samt"
    native_candidates = out_dir / "candidates.samt"
    native_visibility = visibility
    if args.native_rasterizer:
        native_visibility = out_dir / "visibility.samt"
        subprocess.run(
            [
                str(binary),
                "mesh-visibility",
                "--vertices", str(vertices),
                "--faces", str(faces),
                "--out", str(native_visibility),
                "--views", str(args.visibility_views),
                "--resolution", str(args.visibility_resolution),
            ],
            check=True,
        )
        compare_visibility(load_samt(native_visibility), load_samt(visibility))
    subprocess.run(
        [
            str(binary),
            "mesh-filter-visibility",
            "--vertices", str(vertices),
            "--faces", str(faces),
            "--visibility", str(native_visibility),
            "--vertices-out", str(native_vertices),
            "--faces-out", str(native_faces),
            "--candidates-out", str(native_candidates),
        ],
        check=True,
    )
    compare_exact("mincut candidate faces", load_samt(native_candidates), expected_candidates)
    compare_exact("vertices", load_samt(native_vertices), expected_vertices)
    compare_exact("faces", load_samt(native_faces), expected_faces)
    print("PASS: native visibility/mincut matches the official pre-MeshFix mesh exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
