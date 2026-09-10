#!/usr/bin/env python3
"""Render two GLB assets with one deterministic linear-RGB orbit contract.

The reference GLB defines world-space bounds and all camera poses. The native
asset is rendered in that unchanged coordinate system, so a pose, scale, or
translation error contributes to the mask, depth, normal, and RGB metrics.
This is an acceptance renderer, not a replacement glTF viewer: both assets use
the same Lambertian light, linear colour conversion, and rasterizer.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
from PIL import Image

from verify_mesh_export import glb_document


COMPONENT_DTYPES = {
    5120: np.dtype("i1"),
    5121: np.dtype("u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
ACCESSOR_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


@dataclass(frozen=True)
class GlbMesh:
    positions: np.ndarray
    indices: np.ndarray
    texcoords: np.ndarray
    normals: np.ndarray
    texture: np.ndarray


def accessor_array(document: dict, binary: bytes, accessor_index: int) -> np.ndarray:
    accessor = document["accessors"][accessor_index]
    if "sparse" in accessor:
        raise RuntimeError("sparse GLB accessors are not supported by the acceptance renderer")
    component_type = accessor.get("componentType")
    value_type = accessor.get("type")
    count = accessor.get("count")
    if component_type not in COMPONENT_DTYPES or value_type not in ACCESSOR_WIDTHS or not isinstance(count, int):
        raise RuntimeError(f"unsupported GLB accessor: {accessor}")
    view = document["bufferViews"][accessor["bufferView"]]
    dtype = COMPONENT_DTYPES[component_type]
    width = ACCESSOR_WIDTHS[value_type]
    item_bytes = dtype.itemsize * width
    stride = view.get("byteStride", item_bytes)
    if stride < item_bytes:
        raise RuntimeError("GLB accessor byte stride is smaller than its element")
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    required = offset + (count - 1) * stride + item_bytes if count else offset
    if required > len(binary):
        raise RuntimeError("GLB accessor exceeds its binary chunk")
    values = np.ndarray(
        shape=(count, width), buffer=binary, offset=offset, dtype=dtype,
        strides=(stride, dtype.itemsize),
    )
    return values.copy()


def angle_weighted_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    vertices = np.asarray(positions, dtype=np.float64)
    face_positions = vertices[faces]
    left = face_positions[:, 1] - face_positions[:, 0]
    right = face_positions[:, 2] - face_positions[:, 0]
    raw_normals = np.cross(left, right)
    lengths = np.linalg.norm(raw_normals, axis=1)
    valid = np.isfinite(lengths) & (lengths > 0.0)
    normals = np.zeros_like(vertices, dtype=np.float64)
    face_normals = raw_normals[valid] / lengths[valid, None]
    valid_positions = face_positions[valid]
    edge_pairs = (
        (valid_positions[:, 1] - valid_positions[:, 0], valid_positions[:, 2] - valid_positions[:, 0]),
        (valid_positions[:, 2] - valid_positions[:, 1], valid_positions[:, 0] - valid_positions[:, 1]),
        (valid_positions[:, 0] - valid_positions[:, 2], valid_positions[:, 1] - valid_positions[:, 2]),
    )
    for corner, (first, second) in enumerate(edge_pairs):
        angles = np.arctan2(np.linalg.norm(np.cross(first, second), axis=1),
                            np.sum(first * second, axis=1))
        np.add.at(normals, faces[valid, corner], face_normals * angles[:, None])
    lengths = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(lengths) & (lengths > 0.0)
    normals[valid] /= lengths[valid, None]
    return normals.astype(np.float32)


def texture_image(document: dict, binary: bytes, primitive: dict) -> np.ndarray:
    material_index = primitive.get("material", 0)
    material = document["materials"][material_index]
    pbr = material.get("pbrMetallicRoughness", {})
    texture_index = pbr.get("baseColorTexture", {}).get("index")
    if texture_index is None:
        raise RuntimeError("GLB material has no baseColorTexture")
    source_index = document["textures"][texture_index]["source"]
    image = document["images"][source_index]
    if "bufferView" not in image:
        raise RuntimeError("external GLB image URIs are not supported by the acceptance renderer")
    view = document["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    payload = binary[start:start + view["byteLength"]]
    with Image.open(io.BytesIO(payload)) as decoded:
        return np.asarray(decoded.convert("RGBA"), dtype=np.uint8)


def load_glb_mesh(path: Path) -> GlbMesh:
    document, binary = glb_document(path)
    primitive = document["meshes"][0]["primitives"][0]
    attributes = primitive.get("attributes", {})
    for name in ("POSITION", "TEXCOORD_0"):
        if name not in attributes:
            raise RuntimeError(f"{path}: GLB lacks {name}")
    positions = accessor_array(document, binary, attributes["POSITION"])
    texcoords = accessor_array(document, binary, attributes["TEXCOORD_0"])
    indices = accessor_array(document, binary, primitive["indices"]).reshape(-1)
    if positions.dtype != np.float32 or positions.shape[1] != 3:
        raise RuntimeError(f"{path}: POSITION must be F32 VEC3")
    if texcoords.dtype != np.float32 or texcoords.shape != (positions.shape[0], 2):
        raise RuntimeError(f"{path}: TEXCOORD_0 must be F32 VEC2 matching POSITION")
    if indices.size == 0 or indices.size % 3 or int(indices.max()) >= positions.shape[0]:
        raise RuntimeError(f"{path}: invalid triangle index accessor")
    if "NORMAL" in attributes:
        normals = accessor_array(document, binary, attributes["NORMAL"])
        if normals.dtype != np.float32 or normals.shape != positions.shape:
            raise RuntimeError(f"{path}: NORMAL must be F32 VEC3 matching POSITION")
    else:
        normals = angle_weighted_normals(positions, indices)
    return GlbMesh(positions, indices.astype(np.int32, copy=False), texcoords, normals, texture_image(document, binary, primitive))


def normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.maximum(lengths, 1.0e-12)


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = normalize(target - eye)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(forward, up))) > 0.98:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    side = normalize(np.cross(forward, up))
    corrected_up = np.cross(side, forward)
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = side
    view[1, :3] = corrected_up
    view[2, :3] = -forward
    view[0, 3] = -float(np.dot(side, eye))
    view[1, 3] = -float(np.dot(corrected_up, eye))
    view[2, 3] = float(np.dot(forward, eye))
    return view


def perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    focal = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = focal / aspect
    matrix[1, 1] = focal
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def orbit_matrices(reference: GlbMesh, frames: int, fov_deg: float) -> list[np.ndarray]:
    minimum = reference.positions.min(axis=0)
    maximum = reference.positions.max(axis=0)
    center = (minimum + maximum) * 0.5
    radius = max(float(np.linalg.norm(maximum - minimum) * 0.5), 1.0e-3)
    distance = radius * 2.8
    projection = perspective(fov_deg, 1.0, max(radius * 0.01, 1.0e-3), distance + radius * 3.0)
    matrices = []
    elevation = math.radians(20.0)
    for frame in range(frames):
        azimuth = 2.0 * math.pi * frame / frames
        direction = np.array((math.cos(azimuth) * math.cos(elevation), math.sin(elevation),
                              math.sin(azimuth) * math.cos(elevation)), dtype=np.float32)
        eye = center + direction * distance
        matrices.append(projection @ look_at(eye, center))
    return matrices


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32) / 255.0
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    encoded = np.where(values <= 0.0031308, values * 12.92, 1.055 * values ** (1.0 / 2.4) - 0.055)
    return np.round(encoded * 255.0).astype(np.uint8)


def render_mesh(mesh: GlbMesh, matrix: np.ndarray, resolution: int, torch, dr) -> dict[str, np.ndarray]:
    device = torch.device("cuda")
    positions = torch.as_tensor(mesh.positions, dtype=torch.float32, device=device)
    clip = torch.cat((positions, torch.ones((positions.shape[0], 1), device=device)), dim=1)
    clip = clip @ torch.as_tensor(matrix.T, dtype=torch.float32, device=device)
    faces = torch.as_tensor(mesh.indices.reshape(-1, 3), dtype=torch.int32, device=device)
    texcoords = torch.as_tensor(mesh.texcoords, dtype=torch.float32, device=device)
    normals = torch.as_tensor(mesh.normals, dtype=torch.float32, device=device)
    texture = torch.as_tensor(srgb_to_linear(mesh.texture), dtype=torch.float32, device=device)[None]
    context = dr.RasterizeCudaContext(device=device)
    raster, _ = dr.rasterize(context, clip[None], faces, resolution=(resolution, resolution))
    interpolated_uv, _ = dr.interpolate(texcoords[None], raster, faces)
    interpolated_normal, _ = dr.interpolate(normals[None], raster, faces)
    sampled = dr.texture(texture, interpolated_uv, filter_mode="linear", boundary_mode="clamp")[0, ..., :3]
    normal = torch.nn.functional.normalize(interpolated_normal[0], dim=-1, eps=1.0e-12)
    light = torch.tensor((0.45, 0.75, 0.48), dtype=torch.float32, device=device)
    light = torch.nn.functional.normalize(light, dim=0)
    diffuse = torch.clamp(torch.sum(normal * light, dim=-1, keepdim=True), min=0.0)
    linear_rgb = sampled * (0.18 + 0.82 * diffuse)
    mask = raster[0, ..., 3] > 0
    linear_rgb = torch.where(mask[..., None], linear_rgb, torch.zeros_like(linear_rgb))
    depth = raster[0, ..., 2]
    return {
        "linear_rgb": linear_rgb.detach().cpu().numpy(),
        "normal": normal.detach().cpu().numpy(),
        "depth": depth.detach().cpu().numpy(),
        "mask": mask.detach().cpu().numpy(),
    }


def metrics(reference: dict[str, np.ndarray], native: dict[str, np.ndarray]) -> dict[str, float]:
    union = reference["mask"] | native["mask"]
    intersection = reference["mask"] & native["mask"]
    if not np.any(union):
        raise RuntimeError("reference and native GLB renders contain no visible foreground")
    difference = reference["linear_rgb"] - native["linear_rgb"]
    rgb_values = difference[union]
    result = {
        "foreground_union_pixels": int(np.count_nonzero(union)),
        "foreground_intersection_pixels": int(np.count_nonzero(intersection)),
        "mask_iou": float(np.count_nonzero(intersection) / np.count_nonzero(union)),
        "rgb_mae_linear": float(np.mean(np.abs(rgb_values))),
        "rgb_rmse_linear": float(np.sqrt(np.mean(np.square(rgb_values)))),
        "rgb_max_abs_linear": float(np.max(np.abs(rgb_values))),
    }
    if np.any(intersection):
        depth_difference = reference["depth"][intersection] - native["depth"][intersection]
        dot = np.clip(np.sum(reference["normal"][intersection] * native["normal"][intersection], axis=1), -1.0, 1.0)
        angles = np.degrees(np.arccos(dot))
        result.update({
            "depth_mae_ndc": float(np.mean(np.abs(depth_difference))),
            "normal_mean_angle_deg": float(np.mean(angles)),
            "normal_max_angle_deg": float(np.max(angles)),
        })
    return result


def visualization(render: dict[str, np.ndarray]) -> np.ndarray:
    image = np.full((*render["mask"].shape, 3), 0.04, dtype=np.float32)
    image[render["mask"]] = render["linear_rgb"][render["mask"]]
    return linear_to_srgb(image)


def save_first_frame(out_dir: Path, reference: dict[str, np.ndarray], native: dict[str, np.ndarray]) -> None:
    reference_image = visualization(reference)
    native_image = visualization(native)
    difference = np.abs(reference["linear_rgb"] - native["linear_rgb"])
    difference = np.clip(difference * 4.0, 0.0, 1.0)
    imageio.imwrite(out_dir / "reference_glb_view.png", reference_image)
    imageio.imwrite(out_dir / "native_glb_view.png", native_image)
    imageio.imwrite(out_dir / "absolute_difference.png", linear_to_srgb(difference))
    imageio.imwrite(out_dir / "side_by_side.png", np.concatenate((reference_image, native_image), axis=1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-glb", type=Path, required=True)
    parser.add_argument("--native-glb", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--fov-deg", type=float, default=45.0)
    parser.add_argument("--max-rgb-mae-linear", type=float)
    parser.add_argument("--min-mask-iou", type=float)
    parser.add_argument("--max-normal-angle-deg", type=float)
    parser.add_argument("--max-depth-mae-ndc", type=float)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.frames < 1 or args.resolution < 16 or not 1.0 < args.fov_deg < 179.0:
        parser.error("--frames must be positive, --resolution >= 16, and --fov-deg in (1, 179)")
    thresholds = {
        "max_rgb_mae_linear": args.max_rgb_mae_linear,
        "min_mask_iou": args.min_mask_iou,
        "max_normal_angle_deg": args.max_normal_angle_deg,
        "max_depth_mae_ndc": args.max_depth_mae_ndc,
    }
    if any(value is not None and (not math.isfinite(value) or value < 0.0) for value in thresholds.values()):
        parser.error("GLB comparison thresholds must be finite and non-negative")
    if args.min_mask_iou is not None and args.min_mask_iou > 1.0:
        parser.error("--min-mask-iou must be <= 1")

    import nvdiffrast.torch as dr
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GLB multiview comparison requires CUDA nvdiffrast")
    reference = load_glb_mesh(args.reference_glb.resolve())
    native = load_glb_mesh(args.native_glb.resolve())
    matrices = orbit_matrices(reference, args.frames, args.fov_deg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference_frames: list[np.ndarray] = []
    native_frames: list[np.ndarray] = []
    per_frame: list[dict[str, float]] = []
    for index, matrix in enumerate(matrices):
        reference_render = render_mesh(reference, matrix, args.resolution, torch, dr)
        native_render = render_mesh(native, matrix, args.resolution, torch, dr)
        if index == 0:
            save_first_frame(args.out_dir, reference_render, native_render)
        reference_frames.append(visualization(reference_render))
        native_frames.append(visualization(native_render))
        per_frame.append(metrics(reference_render, native_render))
    torch.cuda.synchronize()
    imageio.imwrite(args.out_dir / "reference_glb_orbit.gif", np.stack(reference_frames), loop=0, duration=100)
    imageio.imwrite(args.out_dir / "native_glb_orbit.gif", np.stack(native_frames), loop=0, duration=100)

    aggregate = {
        "frame_count": args.frames,
        "resolution": args.resolution,
        "rgb_mae_linear": float(np.mean([frame["rgb_mae_linear"] for frame in per_frame])),
        "rgb_rmse_linear": float(np.mean([frame["rgb_rmse_linear"] for frame in per_frame])),
        "rgb_max_abs_linear": float(max(frame["rgb_max_abs_linear"] for frame in per_frame)),
        "mask_iou": float(np.mean([frame["mask_iou"] for frame in per_frame])),
        "mask_iou_min": float(min(frame["mask_iou"] for frame in per_frame)),
        "depth_mae_ndc": float(np.mean([frame.get("depth_mae_ndc", float("inf")) for frame in per_frame])),
        "normal_mean_angle_deg": float(np.mean([frame.get("normal_mean_angle_deg", float("inf")) for frame in per_frame])),
        "normal_max_angle_deg": float(max(frame.get("normal_max_angle_deg", float("inf")) for frame in per_frame)),
    }
    passed = (
        all(math.isfinite(value) for value in aggregate.values()) and
        (args.max_rgb_mae_linear is None or aggregate["rgb_mae_linear"] <= args.max_rgb_mae_linear) and
        (args.min_mask_iou is None or aggregate["mask_iou"] >= args.min_mask_iou) and
        (args.max_normal_angle_deg is None or aggregate["normal_max_angle_deg"] <= args.max_normal_angle_deg) and
        (args.max_depth_mae_ndc is None or aggregate["depth_mae_ndc"] <= args.max_depth_mae_ndc)
    )
    report = {
        "schema": "sam3d.glb-render-compare.v1",
        "reference_glb": str(args.reference_glb.resolve()),
        "native_glb": str(args.native_glb.resolve()),
        "camera_contract": "reference-world-bounds fixed orbit, no pose or scale normalization",
        "renderer_contract": "nvdiffrast CUDA, linear RGB, shared Lambertian light",
        "thresholds": {name: value for name, value in thresholds.items() if value is not None},
        "aggregate": aggregate,
        "per_frame": per_frame,
        "passed": passed,
        "quality_gate_configured": all(value is not None for value in thresholds.values()),
        "quality_gate_passed": (passed if all(value is not None for value in thresholds.values()) else None),
    }
    payload = json.dumps(report, indent=2)
    print(payload)
    output = args.json_out or args.out_dir / "glb_render_metrics.json"
    output.write_text(payload + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
