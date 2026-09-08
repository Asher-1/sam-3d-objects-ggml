#!/usr/bin/env python3
"""Validate a replayable official PBR reference bundle.

This validates reference-artifact structure only. It intentionally does not
accept a native output: native-vs-reference numerical checks belong to the
future mesh, renderer, texture, and GLB regression stages.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def read_samt_shape(path: Path) -> tuple[int, ...]:
    with path.open("rb") as handle:
        if handle.read(4) != b"SAMT":
            raise ValueError(f"{path}: invalid SAMT magic")
        ndim = struct.unpack("<i", handle.read(4))[0]
        if ndim <= 0 or ndim > 8:
            raise ValueError(f"{path}: invalid SAMT dimension count {ndim}")
        shape = struct.unpack(f"<{ndim}q", handle.read(ndim * 8))
    return tuple(reversed(shape))


def require_file(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"artifact path escapes reference directory: {name}") from error
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty required artifact: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument(
        "--stage-dir",
        type=Path,
        help=("directory containing the SAMT stage artifacts; defaults to the reference "
              "directory's parent for dump_e2e_stages.py compatibility"),
    )
    args = parser.parse_args()
    root = args.reference_dir.resolve()
    manifest_path = require_file(root, "manifest.json")
    manifest = json.loads(manifest_path.read_text())

    if manifest.get("schema") != "sam3d.official-pbr-reference.v1":
        raise ValueError("unexpected reference-bundle schema")
    bake = manifest.get("bake", {})
    if bake.get("mode") != "opt" or bake.get("total_steps") != 2500:
        raise ValueError("reference does not describe the official optimized 2500-step bake")
    if bake.get("views") != 100 or bake.get("resolution") != 1024:
        raise ValueError("reference does not describe the official 100-view 1024px bake")
    schedule = bake.get("selected_views")
    if not isinstance(schedule, list) or len(schedule) != 2500:
        raise ValueError("reference must record all 2500 texture-bake view selections")
    if any(not isinstance(index, int) or index < 0 or index >= 100 for index in schedule):
        raise ValueError("texture-bake view schedule contains an invalid view index")

    glb = require_file(root, "official_pbr.glb")
    if glb.read_bytes()[:4] != b"glTF":
        raise ValueError("official_pbr.glb does not contain a GLB header")
    png = require_file(root, "base_color.png")
    if png.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("base_color.png does not contain a PNG header")
    cameras = np.load(require_file(root, "bake_cameras.npz"))
    if cameras["extrinsics"].shape != (100, 4, 4):
        raise ValueError(f"unexpected extrinsics shape: {cameras['extrinsics'].shape}")
    if cameras["intrinsics"].shape != (100, 3, 3):
        raise ValueError(f"unexpected intrinsics shape: {cameras['intrinsics'].shape}")

    stage_dir = args.stage_dir.resolve() if args.stage_dir is not None else root.parent
    vertices = require_file(stage_dir, "asset_postprocess_vertices.samt")
    faces = require_file(stage_dir, "asset_postprocess_faces.samt")
    uvs = require_file(stage_dir, "asset_uv.samt")
    decimated_vertices = require_file(stage_dir, "asset_decimated_vertices_zup.samt")
    decimated_faces = require_file(stage_dir, "asset_decimated_faces.samt")
    decimated_visibility = require_file(stage_dir, "asset_decimated_visibility.samt")
    mincut_vertices = require_file(stage_dir, "asset_mincut_vertices_zup.samt")
    mincut_faces = require_file(stage_dir, "asset_mincut_faces.samt")
    mincut_candidates = require_file(stage_dir, "asset_mincut_candidate_faces.samt")
    extrinsics_samt = require_file(stage_dir, "bake_extrinsics.samt")
    intrinsics_samt = require_file(stage_dir, "bake_intrinsics.samt")
    if len(read_samt_shape(vertices)) != 2 or read_samt_shape(vertices)[1] != 3:
        raise ValueError("postprocessed vertices must be [V,3]")
    if len(read_samt_shape(faces)) != 2 or read_samt_shape(faces)[1] != 3:
        raise ValueError("postprocessed faces must be [F,3]")
    if len(read_samt_shape(uvs)) != 2 or read_samt_shape(uvs)[1] != 2:
        raise ValueError("UVs must be [V,2]")
    if len(read_samt_shape(decimated_vertices)) != 2 or read_samt_shape(decimated_vertices)[1] != 3:
        raise ValueError("decimated vertices must be [V,3]")
    if len(read_samt_shape(decimated_faces)) != 2 or read_samt_shape(decimated_faces)[1] != 3:
        raise ValueError("decimated faces must be [F,3]")
    if len(read_samt_shape(decimated_visibility)) != 1 or \
            read_samt_shape(decimated_visibility)[0] != read_samt_shape(decimated_faces)[0]:
        raise ValueError("decimated visibility must be F32 [decimated_face_count]")
    if len(read_samt_shape(mincut_vertices)) != 2 or read_samt_shape(mincut_vertices)[1] != 3:
        raise ValueError("pre-MeshFix mincut vertices must be [V,3]")
    if len(read_samt_shape(mincut_faces)) != 2 or read_samt_shape(mincut_faces)[1] != 3:
        raise ValueError("pre-MeshFix mincut faces must be [F,3]")
    if len(read_samt_shape(mincut_candidates)) != 1:
        raise ValueError("mincut candidate faces must be [N]")
    mesh_postprocess = manifest.get("mesh_postprocess", {})
    if mesh_postprocess.get("visibility_frequency_views") != 1000:
        raise ValueError("reference does not capture official 1000-view face visibility")
    if read_samt_shape(extrinsics_samt) != (100, 4, 4):
        raise ValueError("native camera extrinsics must be [100,4,4]")
    if read_samt_shape(intrinsics_samt) != (100, 3, 3):
        raise ValueError("native camera intrinsics must be [100,3,3]")

    hashes = bake.get("observation_sha256")
    if not isinstance(hashes, list) or len(hashes) != 100:
        raise ValueError("reference must contain all 100 observation hashes")
    if bake.get("observations_saved"):
        observations = np.load(require_file(root, "bake_observations.npz"))["rgb"]
        if observations.shape != (100, 1024, 1024, 3):
            raise ValueError(f"unexpected observation shape: {observations.shape}")
        actual_hashes = [
            hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
            for image in observations
        ]
        if actual_hashes != hashes:
            raise ValueError("saved observations do not match their manifest hashes")

    raster_contract = bake.get("raster_contract", {})
    raster_saved = raster_contract.get("saved", False)
    if not isinstance(raster_saved, bool):
        raise ValueError("bake raster-contract saved flag must be boolean")
    if raster_saved:
        if raster_contract.get("views") != 100:
            raise ValueError("saved bake raster contract must contain all 100 observation views")
        artifacts = raster_contract.get("artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != 100:
            raise ValueError("saved bake raster contract must list all 100 artifacts")
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict) or artifact.get("index") != index:
                raise ValueError("bake raster artifacts must be ordered by official observation index")
            relative = artifact.get("file")
            if not isinstance(relative, str):
                raise ValueError("bake raster artifact lacks a relative file path")
            raster_path = require_file(root, relative)
            with np.load(raster_path) as raster:
                if raster["uv"].dtype != np.float32:
                    raise ValueError(f"unexpected UV raster dtype: {raster['uv'].dtype}")
                if raster["uv_dr"].dtype != np.float32:
                    raise ValueError(
                        f"unexpected UV derivative dtype: {raster['uv_dr'].dtype}"
                    )
                if raster["mask"].dtype != np.uint8:
                    raise ValueError(f"unexpected raster coverage dtype: {raster['mask'].dtype}")
                if raster["uv"].shape != (1, 1024, 1024, 2):
                    raise ValueError(f"unexpected UV raster shape: {raster['uv'].shape}")
                if raster["uv_dr"].shape != (1, 1024, 1024, 4):
                    raise ValueError(f"unexpected UV derivative shape: {raster['uv_dr'].shape}")
                if raster["mask"].shape != (1, 1024, 1024):
                    raise ValueError(f"unexpected raster coverage shape: {raster['mask'].shape}")
                for field in ("uv", "uv_dr", "mask"):
                    expected_hash = artifact.get(f"{field}_sha256")
                    if not isinstance(expected_hash, str):
                        raise ValueError(f"bake raster artifact lacks {field} hash")
                    actual_hash = hashlib.sha256(
                        np.ascontiguousarray(raster[field]).tobytes()
                    ).hexdigest()
                    if actual_hash != expected_hash:
                        raise ValueError(f"bake raster {index} {field} does not match its manifest hash")
    elif raster_contract.get("artifacts"):
        raise ValueError("unsaved bake raster contract must not list artifacts")

    print(f"PASS: official PBR reference bundle is complete: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
