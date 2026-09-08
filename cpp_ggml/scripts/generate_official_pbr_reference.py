#!/usr/bin/env python3
"""Generate an official final-PBR reference from frozen decoder artifacts.

The input directory is produced by ``dump_e2e_stages.py``. This program does
not reimplement any official stage: it rebuilds the official Gaussian object
from its dumped hidden tensors and calls the checked-in Python mesh cleanup,
Gaussian renderer, xatlas parameterizer, optimized 2500-step texture bake, and
trimesh GLB export. The result is a deterministic acceptance fixture for the
native C++ path, not a release inference dependency.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import struct
from pathlib import Path

# The public release deliberately omits sam3d_objects.init; the official
# notebook and existing stage dumper both bypass it for standalone tooling.
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import numpy as np
import torch
import trimesh
import trimesh.visual
from PIL import Image

from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian import Gaussian
from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils
from sam3d_objects.model.backbone.tdfy_dit.utils.render_utils import render_multiview


GGML_F32 = 0
GGML_I32_TYPES = {26, 30}


def load_samt(path: Path) -> np.ndarray:
    """Load the project SAMT interchange tensor using its logical NumPy shape."""
    with path.open("rb") as stream:
        if stream.read(4) != b"SAMT":
            raise ValueError(f"{path}: invalid SAMT magic")
        (rank,) = struct.unpack("<i", stream.read(4))
        if rank <= 0 or rank > 8:
            raise ValueError(f"{path}: invalid SAMT rank {rank}")
        ggml_shape = struct.unpack(f"<{rank}q", stream.read(rank * 8))
        (ggml_type,) = struct.unpack("<i", stream.read(4))
        if ggml_type == GGML_F32:
            dtype = np.dtype("<f4")
        elif ggml_type in GGML_I32_TYPES:
            dtype = np.dtype("<i4")
        else:
            raise ValueError(f"{path}: unsupported SAMT ggml type {ggml_type}")
        values = np.frombuffer(stream.read(), dtype=dtype)
    expected = int(np.prod(ggml_shape, dtype=np.int64))
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} values, found {values.size}")
    return values.reshape(tuple(reversed(ggml_shape))).copy()


def save_samt(path: Path, values: np.ndarray) -> None:
    """Write a NumPy [rows, columns] array in the C++ SAMT memory layout."""
    values = np.ascontiguousarray(values)
    if values.dtype == np.float32:
        ggml_type = GGML_F32
    elif values.dtype == np.int32:
        ggml_type = 26
    else:
        raise ValueError(f"{path}: SAMT only accepts F32/I32 here, got {values.dtype}")
    ggml_shape = tuple(reversed(values.shape))
    with path.open("wb") as stream:
        stream.write(b"SAMT")
        stream.write(struct.pack("<i", len(ggml_shape)))
        stream.write(struct.pack(f"<{len(ggml_shape)}q", *ggml_shape))
        stream.write(struct.pack("<i", ggml_type))
        stream.write(values.tobytes())


def require_matrix(root: Path, name: str, columns: int, dtype: np.dtype) -> np.ndarray:
    values = load_samt(root / f"{name}.samt")
    if values.dtype != dtype or values.ndim != 2 or values.shape[1] != columns:
        raise ValueError(f"{name}: expected [N,{columns}] {dtype}, got {values.shape} {values.dtype}")
    return values


def gaussian_from_stage(root: Path, manifest: dict) -> Gaussian:
    raw_init = manifest.get("decode_gaussian_init_params")
    if not isinstance(raw_init, str):
        raise ValueError("manifest lacks decode_gaussian_init_params")
    init_params = ast.literal_eval(raw_init)
    if not isinstance(init_params, dict):
        raise ValueError("decode_gaussian_init_params is not a dictionary")
    gaussian = Gaussian(**init_params)
    fields = {
        "_xyz": require_matrix(root, "decode_gaussian__xyz", 3, np.dtype("float32")),
        "_features_dc": load_samt(root / "decode_gaussian__features_dc.samt"),
        "_opacity": require_matrix(root, "decode_gaussian__opacity", 1, np.dtype("float32")),
        "_scaling": require_matrix(root, "decode_gaussian__scaling", 3, np.dtype("float32")),
        "_rotation": require_matrix(root, "decode_gaussian__rotation", 4, np.dtype("float32")),
    }
    features = fields["_features_dc"]
    if features.dtype != np.float32 or features.ndim != 3 or features.shape[1:] != (1, 3):
        raise ValueError(f"decode_gaussian__features_dc: expected [N,1,3] F32, got {features.shape}")
    count = fields["_xyz"].shape[0]
    if any(value.shape[0] != count for value in fields.values()):
        raise ValueError("Gaussian stage tensors do not share a point count")
    for name, value in fields.items():
        setattr(gaussian, name, torch.from_numpy(value).to(device="cuda", dtype=torch.float32))
    gaussian._features_rest = None
    return gaussian


def mesh_from_stage(root: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = require_matrix(root, "decode_mesh_vertices", 3, np.dtype("float32"))
    raw_faces = load_samt(root / "decode_mesh_faces.samt")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
        raise ValueError(f"decode_mesh_faces: expected [N,3], got {raw_faces.shape}")
    if raw_faces.dtype == np.float32:
        if not np.isfinite(raw_faces).all() or (raw_faces < 0).any() or not np.array_equal(raw_faces, np.floor(raw_faces)):
            raise ValueError("decode_mesh_faces contains invalid F32 indices")
        faces = raw_faces.astype(np.int32)
    elif raw_faces.dtype == np.int32:
        if (raw_faces < 0).any():
            raise ValueError("decode_mesh_faces contains negative indices")
        faces = raw_faces
    else:
        raise ValueError(f"decode_mesh_faces: unsupported dtype {raw_faces.dtype}")
    if faces.size == 0 or int(faces.max()) >= vertices.shape[0]:
        raise ValueError("decode_mesh_faces contains an out-of-range index")
    return vertices, faces


def sha256_arrays(arrays: list[np.ndarray]) -> list[str]:
    return [hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for value in arrays]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-observations", action="store_true")
    parser.add_argument(
        "--save-bake-raster",
        action="store_true",
        help=("save the 100 official per-view UV, UV-derivative, and coverage maps used "
              "by the optimized bake; this is a large external debugging fixture and is "
              "never written under benchmarks/"),
    )
    parser.add_argument(
        "--mesh-cleanup-only",
        action="store_true",
        help=("capture only official VTK/visibility/mincut/MeshFix boundaries; "
              "skip Gaussian observations and the 2500-step texture bake"),
    )
    args = parser.parse_args()

    stage_dir = args.stage_dir.resolve()
    reference_dir = args.reference_dir.resolve()
    manifest_path = stage_dir / "manifest.json"
    if not manifest_path.is_file():
        parser.error(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    reference_dir.mkdir(parents=True, exist_ok=True)

    vertices, faces = mesh_from_stage(stage_dir)

    # These constants are the default `to_glb()` call in the official pipeline.
    simplify = 0.95
    # Instrument the unmodified official call instead of reimplementing its
    # rasterization. The resulting post-VTK mesh and face frequencies let the
    # C++ mincut stage be compared independently from its still-in-progress
    # native visibility rasterizer.
    captured_visibility: dict[str, int] = {}
    captured_meshfix: dict[str, bool] = {}
    captured_graph: dict[str, int] = {}
    official_fill_holes = postprocessing_utils._fill_holes
    official_rasterize = postprocessing_utils.utils3d.torch.rasterize_triangle_faces
    official_meshfix_type = postprocessing_utils._meshfix.PyTMesh
    official_graph_type = postprocessing_utils.igraph.Graph

    class CaptureGraph:
        """Delegate igraph while preserving the actual official mincut side."""

        def __init__(self, *graph_args, **graph_kwargs):
            self._inner = official_graph_type(*graph_args, **graph_kwargs)

        def mincut(self, *mincut_args, **mincut_kwargs):
            cut = self._inner.mincut(*mincut_args, **mincut_kwargs)
            face_count = captured_graph.get("face_count")
            if face_count is None:
                raise RuntimeError("official mincut ran outside the visibility capture")
            candidates = np.sort(np.asarray(
                [vertex for vertex in cut.partition[0] if vertex < face_count], dtype=np.int32
            ))
            save_samt(stage_dir / "asset_mincut_candidate_faces.samt", candidates)
            captured_graph["candidate_count"] = int(candidates.size)
            return cut

        def __getattr__(self, name):
            return getattr(self._inner, name)

    class CaptureMeshFix:
        """Delegate MeshFix while preserving its exact pre-repair input."""

        def __init__(self):
            self._inner = official_meshfix_type()

        def load_array(self, mesh_vertices, mesh_faces):
            save_samt(stage_dir / "asset_mincut_vertices_zup.samt",
                      np.asarray(mesh_vertices, dtype=np.float32))
            save_samt(stage_dir / "asset_mincut_faces.samt",
                      np.asarray(mesh_faces, dtype=np.int32))
            captured_meshfix["pre_repair_saved"] = True
            return self._inner.load_array(mesh_vertices, mesh_faces)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def capture_fill_holes(fill_vertices, fill_faces, *fill_args, **fill_kwargs):
        num_views = int(fill_kwargs.get("num_views", 500))
        captured_graph["face_count"] = int(fill_faces.shape[0])
        visibility = torch.zeros(fill_faces.shape[0], dtype=torch.int32, device=fill_faces.device)
        raster_calls = 0

        def capture_rasterize(*raster_args, **raster_kwargs):
            nonlocal raster_calls
            buffers = official_rasterize(*raster_args, **raster_kwargs)
            face_id = buffers["face_id"][0][buffers["mask"][0] > 0.95] - 1
            face_id = torch.unique(face_id).long()
            if face_id.numel() > 0:
                visibility[face_id] += 1
            raster_calls += 1
            return buffers

        postprocessing_utils.utils3d.torch.rasterize_triangle_faces = capture_rasterize
        try:
            result = official_fill_holes(fill_vertices, fill_faces, *fill_args, **fill_kwargs)
        finally:
            postprocessing_utils.utils3d.torch.rasterize_triangle_faces = official_rasterize
        if raster_calls != num_views:
            raise RuntimeError(
                f"official _fill_holes rendered {raster_calls} views; expected {num_views}"
            )
        save_samt(stage_dir / "asset_decimated_vertices_zup.samt",
                  fill_vertices.detach().cpu().numpy().astype(np.float32))
        save_samt(stage_dir / "asset_decimated_faces.samt",
                  fill_faces.detach().cpu().numpy().astype(np.int32))
        save_samt(stage_dir / "asset_decimated_visibility.samt",
                  (visibility.float() / num_views).detach().cpu().numpy().astype(np.float32))
        captured_visibility["num_views"] = num_views
        return result

    postprocessing_utils._fill_holes = capture_fill_holes
    postprocessing_utils._meshfix.PyTMesh = CaptureMeshFix
    postprocessing_utils.igraph.Graph = CaptureGraph
    try:
        vertices, faces = postprocessing_utils.postprocess_mesh(
            vertices,
            faces,
            simplify=True,
            simplify_ratio=simplify,
            fill_holes=True,
            fill_holes_max_hole_size=0.04,
            fill_holes_max_hole_nbe=int(250 * np.sqrt(1 - simplify)),
            fill_holes_resolution=1024,
            fill_holes_num_views=1000,
            verbose=True,
        )
    finally:
        postprocessing_utils._fill_holes = official_fill_holes
        postprocessing_utils._meshfix.PyTMesh = official_meshfix_type
        postprocessing_utils.igraph.Graph = official_graph_type
    if captured_visibility.get("num_views") != 1000:
        raise RuntimeError("official PBR reference did not capture the required 1000 visibility views")
    if not captured_meshfix.get("pre_repair_saved"):
        raise RuntimeError("official PBR reference did not capture the pre-MeshFix mincut mesh")
    if "candidate_count" not in captured_graph:
        raise RuntimeError("official PBR reference did not capture the mincut partition")

    # `postprocess_mesh()` returns immediately after MeshFix. Freeze that
    # native-language boundary-repair target before xatlas duplicates seams.
    save_samt(stage_dir / "asset_meshfix_vertices_zup.samt", vertices.astype(np.float32))
    save_samt(stage_dir / "asset_meshfix_faces.samt", faces.astype(np.int32))

    if args.mesh_cleanup_only:
        cleanup_manifest = {
            "schema": "sam3d.official-mesh-cleanup-reference.v1",
            "source": "generate_official_pbr_reference.py:official_functions",
            "mesh_postprocess": {
                "simplify_ratio": simplify,
                "fill_holes": True,
                "fill_holes_resolution": 1024,
                "fill_holes_num_views": captured_visibility["num_views"],
                "fill_holes_max_hole_size": 0.04,
                "fill_holes_max_hole_nbe": int(250 * np.sqrt(1 - simplify)),
                "visibility_frequency": "../asset_decimated_visibility.samt",
                "mincut_vertices": "../asset_mincut_vertices_zup.samt",
                "mincut_faces": "../asset_mincut_faces.samt",
                "mincut_candidate_faces": "../asset_mincut_candidate_faces.samt",
                "meshfix_vertices": "../asset_meshfix_vertices_zup.samt",
                "meshfix_faces": "../asset_meshfix_faces.samt",
            },
        }
        (reference_dir / "mesh_cleanup_manifest.json").write_text(
            json.dumps(cleanup_manifest, indent=2) + "\n"
        )
        print(f"PASS: wrote official mesh-cleanup reference {reference_dir}")
        return 0

    gaussian = gaussian_from_stage(stage_dir, manifest)
    vertices, faces, uvs = postprocessing_utils.parametrize_mesh(vertices, faces)
    # Keep both coordinate systems explicit. `parametrize_mesh()` works in the
    # model's z-up coordinates; `to_glb()` rotates those coordinates only at
    # the final export boundary. The pre-rotation fixture is needed to compare
    # the native postprocess/xatlas stages, while asset_postprocess_* is the
    # final GLB geometry emitted by the official pipeline.
    rotated_vertices = vertices @ np.array(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32
    )
    save_samt(stage_dir / "asset_parameterized_vertices_zup.samt", vertices.astype(np.float32))
    save_samt(stage_dir / "asset_parameterized_faces.samt", faces.astype(np.int32))
    save_samt(stage_dir / "asset_postprocess_vertices.samt", rotated_vertices.astype(np.float32))
    save_samt(stage_dir / "asset_postprocess_faces.samt", faces.astype(np.int32))
    save_samt(stage_dir / "asset_uv.samt", uvs.astype(np.float32))

    observations, extrinsics, intrinsics = render_multiview(gaussian, resolution=1024, nviews=100)
    observations = [np.ascontiguousarray(item) for item in observations]
    extrinsics_np = np.stack([item.detach().cpu().numpy() for item in extrinsics]).astype(np.float32)
    intrinsics_np = np.stack([item.detach().cpu().numpy() for item in intrinsics]).astype(np.float32)
    save_samt(stage_dir / "bake_extrinsics.samt", extrinsics_np)
    save_samt(stage_dir / "bake_intrinsics.samt", intrinsics_np)
    np.savez_compressed(reference_dir / "bake_cameras.npz", extrinsics=extrinsics_np, intrinsics=intrinsics_np)

    selected_views: list[int] = []
    original_randint = np.random.randint
    bake_raster_dir = reference_dir / "bake_raster"
    captured_bake_rasters: list[dict[str, object]] = []
    official_bake_rasterize = postprocessing_utils.utils3d.torch.rasterize_triangle_faces

    def capture_randint(*randint_args, **randint_kwargs):
        value = original_randint(*randint_args, **randint_kwargs)
        if np.isscalar(value):
            selected_views.append(int(value))
        return value

    def capture_bake_rasterize(*raster_args, **raster_kwargs):
        """Persist the exact differentiable UV inputs before the official bake consumes them."""
        buffers = official_bake_rasterize(*raster_args, **raster_kwargs)
        if raster_kwargs.get("uv") is None:
            return buffers
        index = len(captured_bake_rasters)
        if index >= 100:
            raise RuntimeError("official optimized bake rasterized more than 100 UV observation views")
        uv = buffers.get("uv")
        uv_dr = buffers.get("uv_dr")
        mask = buffers.get("mask")
        if uv is None or uv_dr is None or mask is None:
            raise RuntimeError("official optimized bake rasterization omitted UV, UV derivatives, or coverage")
        if not args.save_bake_raster:
            captured_bake_rasters.append({"index": index})
            return buffers
        bake_raster_dir.mkdir(parents=True, exist_ok=True)
        uv_np = uv.detach().cpu().numpy().astype(np.float32, copy=False)
        uv_dr_np = uv_dr.detach().cpu().numpy().astype(np.float32, copy=False)
        mask_np = mask.detach().cpu().numpy().astype(np.uint8, copy=False)
        output = bake_raster_dir / f"view_{index:03d}.npz"
        np.savez_compressed(output, uv=uv_np, uv_dr=uv_dr_np, mask=mask_np)
        captured_bake_rasters.append({
            "index": index,
            "file": str(output.relative_to(reference_dir)),
            "uv_sha256": sha256_arrays([uv_np])[0],
            "uv_dr_sha256": sha256_arrays([uv_dr_np])[0],
            "mask_sha256": sha256_arrays([mask_np])[0],
        })
        return buffers

    np.random.seed(args.seed)
    np.random.randint = capture_randint
    postprocessing_utils.utils3d.torch.rasterize_triangle_faces = capture_bake_rasterize
    try:
        texture = postprocessing_utils.bake_texture(
            vertices,
            faces,
            uvs,
            observations,
            [np.any(observation > 0, axis=-1) for observation in observations],
            [item.detach().cpu().numpy() for item in extrinsics],
            [item.detach().cpu().numpy() for item in intrinsics],
            texture_size=1024,
            mode="opt",
            lambda_tv=0.01,
            verbose=True,
            rendering_engine="nvdiffrast",
        )
    finally:
        np.random.randint = original_randint
        postprocessing_utils.utils3d.torch.rasterize_triangle_faces = official_bake_rasterize
    if len(selected_views) != 2500:
        raise RuntimeError(f"official bake selected {len(selected_views)} views, expected 2500")
    if len(captured_bake_rasters) != 100:
        raise RuntimeError(
            f"official optimized bake rasterized {len(captured_bake_rasters)} UV observation views, expected 100"
        )

    texture_path = reference_dir / "base_color.png"
    Image.fromarray(texture).convert("RGBA").save(texture_path)
    material = trimesh.visual.material.PBRMaterial(
        roughnessFactor=1.0,
        baseColorTexture=Image.fromarray(texture),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
    )
    final_mesh = trimesh.Trimesh(
        rotated_vertices,
        faces,
        visual=trimesh.visual.TextureVisuals(uv=uvs, material=material),
    )
    final_mesh.export(reference_dir / "official_pbr.glb")

    if args.save_observations:
        np.savez_compressed(reference_dir / "bake_observations.npz", rgb=np.stack(observations))
    reference_manifest = {
        "schema": "sam3d.official-pbr-reference.v1",
        "source": "generate_official_pbr_reference.py:official_functions",
        "seed": args.seed,
        "mesh_postprocess": {
            "simplify_ratio": simplify,
            "fill_holes": True,
            "fill_holes_resolution": 1024,
            "fill_holes_num_views": 1000,
            "fill_holes_max_hole_size": 0.04,
            "fill_holes_max_hole_nbe": int(250 * np.sqrt(1 - simplify)),
            "visibility_frequency": "../asset_decimated_visibility.samt",
            "visibility_frequency_views": captured_visibility["num_views"],
            "mincut_vertices": "../asset_mincut_vertices_zup.samt",
            "mincut_faces": "../asset_mincut_faces.samt",
            "meshfix_vertices": "../asset_meshfix_vertices_zup.samt",
            "meshfix_faces": "../asset_meshfix_faces.samt",
        },
        "coordinates": {
            "parameterized_mesh": "z-up",
            "asset_postprocess_vertices": "official to_glb z-up to y-up rotation",
            "rotation_matrix_row_vector": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        },
        "uv": {"parameterizer": "xatlas", "texture_size": 1024},
        "bake": {
            "mode": "opt",
            "renderer": "nvdiffrast",
            "views": 100,
            "resolution": 1024,
            "total_steps": 2500,
            "optimizer": "Adam(beta1=0.5,beta2=0.9,lr=1e-2)",
            "lr_schedule": "cosine(1e-2,1e-5)",
            "lambda_tv": 0.01,
            "selected_views": selected_views,
            "observation_sha256": sha256_arrays(observations),
            "observations_saved": args.save_observations,
            "raster_contract": {
                "saved": args.save_bake_raster,
                "views": len(captured_bake_rasters),
                "format": "bake_raster/view_###.npz",
                "arrays": {
                    "uv": "F32 [1,1024,1024,2] before the official dr.texture call",
                    "uv_dr": "F32 [1,1024,1024,4] before the official dr.texture call",
                    "mask": "U8 [1,1024,1024] raster coverage before observation masking",
                },
                "artifacts": captured_bake_rasters if args.save_bake_raster else [],
            },
        },
        "outputs": {
            "glb": "official_pbr.glb",
            "base_color": "base_color.png",
            "cameras": "bake_cameras.npz",
            "extrinsics_samt": "../bake_extrinsics.samt",
            "intrinsics_samt": "../bake_intrinsics.samt",
            "bake_raster_dir": "bake_raster" if args.save_bake_raster else None,
            "decimated_vertices_samt": "../asset_decimated_vertices_zup.samt",
            "decimated_faces_samt": "../asset_decimated_faces.samt",
            "decimated_visibility_samt": "../asset_decimated_visibility.samt",
            "mincut_vertices_samt": "../asset_mincut_vertices_zup.samt",
            "mincut_faces_samt": "../asset_mincut_faces.samt",
            "mincut_candidate_faces_samt": "../asset_mincut_candidate_faces.samt",
            "meshfix_vertices_samt": "../asset_meshfix_vertices_zup.samt",
            "meshfix_faces_samt": "../asset_meshfix_faces.samt",
        },
    }
    (reference_dir / "manifest.json").write_text(json.dumps(reference_manifest, indent=2) + "\n")
    print(f"PASS: wrote official PBR reference {reference_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
