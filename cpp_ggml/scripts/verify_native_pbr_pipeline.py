#!/usr/bin/env python3
"""Validate the structural and image contracts of a native PBR GLB result."""
from __future__ import annotations

import argparse
import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from verify_mesh_export import glb_document, samt_f32, samt_tensor


def accessor_bytes(document: dict, binary: bytes, accessor_index: int) -> bytes:
    accessor = document["accessors"][accessor_index]
    view = document["bufferViews"][accessor["bufferView"]]
    if accessor.get("byteOffset", 0) != 0 or view.get("byteStride") is not None:
        raise RuntimeError("native PBR verifier expects tightly packed GLB accessors")
    start = view.get("byteOffset", 0)
    return binary[start:start + view["byteLength"]]


def accessor_count(document: dict, accessor_index: int) -> int:
    count = document["accessors"][accessor_index].get("count")
    if not isinstance(count, int) or count <= 0:
        raise RuntimeError(f"invalid GLB accessor count: {count}")
    return count


def accessor_f32_vec3(document: dict, binary: bytes, accessor_index: int) -> np.ndarray:
    accessor = document["accessors"][accessor_index]
    if accessor.get("componentType") != 5126 or accessor.get("type") != "VEC3":
        raise RuntimeError("native PBR verifier expects a tightly packed F32 VEC3 accessor")
    values = np.frombuffer(accessor_bytes(document, binary, accessor_index), dtype="<f4")
    if values.size != accessor_count(document, accessor_index) * 3:
        raise RuntimeError("GLB F32 VEC3 accessor byte length does not match its count")
    return values.reshape(-1, 3)


def embedded_base_color(document: dict, binary: bytes) -> np.ndarray:
    image = document["images"][0]
    view = document["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    png = binary[start:start + view["byteLength"]]
    with Image.open(io.BytesIO(png)) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8)


def stage_mesh_paths(stage_dir: Path) -> tuple[Path, Path, bool]:
    """Return the final mesh tensors and whether they still need Y-up rotation.

    Current stage dumps persist ``asset_postprocess_*`` in the exact coordinate
    system exported by the official GLB.  Older fixtures contain only the
    pre-export, Z-up ``asset_parameterized_*`` tensors.
    """
    postprocess_vertices = stage_dir / "asset_postprocess_vertices.samt"
    postprocess_faces = stage_dir / "asset_postprocess_faces.samt"
    if postprocess_vertices.is_file() and postprocess_faces.is_file():
        return postprocess_vertices, postprocess_faces, False

    return (
        stage_dir / "asset_parameterized_vertices_zup.samt",
        stage_dir / "asset_parameterized_faces.samt",
        True,
    )


def load_stage_mesh(stage_dir: Path) -> tuple[np.ndarray, bytes, bytes]:
    vertices_path, faces_path, rotate_zup_to_yup = stage_mesh_paths(stage_dir)
    vertices_shape, vertices_data = samt_f32(vertices_path)
    uv_shape, uv_data = samt_f32(stage_dir / "asset_uv.samt")
    faces_shape, faces_type, faces_data = samt_tensor(faces_path)
    if vertices_shape[0] != 3 or uv_shape != (2, vertices_shape[1]):
        raise RuntimeError(
            f"unexpected parameterized mesh tensors: vertices={vertices_shape}, uv={uv_shape}"
        )
    if faces_shape[0] != 3 or faces_type not in (0, 26):
        raise RuntimeError(
            f"unexpected parameterized face tensor: shape={faces_shape}, type={faces_type}"
        )
    vertices = np.frombuffer(vertices_data, dtype="<f4").reshape(vertices_shape[1], 3)
    if rotate_zup_to_yup:
        vertices = np.stack(
            (vertices[:, 0], vertices[:, 2], -vertices[:, 1]), axis=1
        ).astype("<f4", copy=False)
    if faces_type == 0:
        faces = np.frombuffer(faces_data, dtype="<f4").astype("<u4", copy=False).tobytes()
    else:
        faces = np.frombuffer(faces_data, dtype="<i4").astype("<u4", copy=False).tobytes()
    # The stage tensor precedes trimesh.export_glb, which changes UV origin.
    exported_uv = np.frombuffer(uv_data, dtype="<f4").reshape(-1, 2).copy()
    exported_uv[:, 1] = 1.0 - exported_uv[:, 1]
    return vertices, faces, exported_uv.tobytes()


def image_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    if actual.shape != reference.shape:
        raise RuntimeError(f"texture size differs: native={actual.shape}, official={reference.shape}")
    diff = actual.astype(np.float32) - reference.astype(np.float32)
    return {
        "mae_u8": float(np.mean(np.abs(diff))),
        "rmse_u8": float(np.sqrt(np.mean(np.square(diff)))),
        "max_abs_u8": int(np.max(np.abs(diff))),
        "equal_pixel_ratio": float(np.mean(np.all(actual == reference, axis=2))),
    }


def normal_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    if actual.shape != reference.shape:
        raise RuntimeError(f"normal shape differs: native={actual.shape}, official={reference.shape}")
    actual = actual.astype(np.float64, copy=False)
    reference = reference.astype(np.float64, copy=False)
    actual_length = np.linalg.norm(actual, axis=1)
    reference_length = np.linalg.norm(reference, axis=1)
    valid = (actual_length > 0.0) & (reference_length > 0.0)
    if not np.any(valid):
        return {
            "available": False,
            "valid_vertex_ratio": 0.0,
            "reason": "no vertex has both a finite, non-zero native and reference normal",
        }
    unit_actual = actual[valid] / actual_length[valid, None]
    unit_reference = reference[valid] / reference_length[valid, None]
    dot = np.clip(np.sum(unit_actual * unit_reference, axis=1), -1.0, 1.0)
    angles = np.degrees(np.arccos(dot))
    return {
        "available": True,
        "valid_vertex_ratio": float(np.mean(valid)),
        "mean_angle_deg": float(np.mean(angles)),
        "max_angle_deg": float(np.max(angles)),
        "mae": float(np.mean(np.abs(unit_actual - unit_reference))),
    }


def official_angle_weighted_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Match trimesh's angle-weighted vertex-normal definition in F64."""
    faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    vertices = np.asarray(positions, dtype=np.float64)
    face_positions = vertices[faces]
    edge_a = face_positions[:, 1] - face_positions[:, 0]
    edge_b = face_positions[:, 2] - face_positions[:, 0]
    raw_face_normals = np.cross(edge_a, edge_b)
    face_lengths = np.linalg.norm(raw_face_normals, axis=1)
    valid_faces = np.isfinite(face_lengths) & (face_lengths > 0.0)
    face_normals = raw_face_normals[valid_faces] / face_lengths[valid_faces, None]
    valid_positions = face_positions[valid_faces]
    corner_edges = (
        (valid_positions[:, 1] - valid_positions[:, 0], valid_positions[:, 2] - valid_positions[:, 0]),
        (valid_positions[:, 2] - valid_positions[:, 1], valid_positions[:, 0] - valid_positions[:, 1]),
        (valid_positions[:, 0] - valid_positions[:, 2], valid_positions[:, 1] - valid_positions[:, 2]),
    )
    normals = np.zeros_like(vertices, dtype=np.float64)
    valid_indices = faces[valid_faces]
    for corner, (left, right) in enumerate(corner_edges):
        angles = np.arctan2(np.linalg.norm(np.cross(left, right), axis=1),
                            np.sum(left * right, axis=1))
        np.add.at(normals, valid_indices[:, corner], face_normals * angles[:, None])
    lengths = np.linalg.norm(normals, axis=1)
    nonzero = np.isfinite(lengths) & (lengths > 0.0)
    normals[nonzero] /= lengths[nonzero, None]
    return normals.astype(np.float32)


def threshold_values(args: argparse.Namespace, names: tuple[str, ...]) -> dict[str, float]:
    return {
        name: value for name in names
        if (value := getattr(args, name)) is not None
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-glb", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path,
                        help="official stage directory for exact controlled mesh/UV checks")
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--structural-only", action="store_true",
                        help="validate a raw native GLB without claiming geometry equality to an oracle")
    parser.add_argument("--require-texture-comparison", action="store_true",
                        help="require an official base-color texture measurement")
    parser.add_argument("--require-normal-comparison", action="store_true",
                        help="require an official GLB normal measurement on an exactly matching mesh")
    parser.add_argument("--max-texture-mae-u8", type=float,
                        help="fail when base-color MAE exceeds this 0..255-scale threshold")
    parser.add_argument("--max-texture-rmse-u8", type=float,
                        help="fail when base-color RMSE exceeds this 0..255-scale threshold")
    parser.add_argument("--max-texture-abs-u8", type=float,
                        help="fail when base-color maximum absolute error exceeds this 0..255-scale threshold")
    parser.add_argument("--max-normal-angle-deg", type=float,
                        help="fail when the maximum matching-vertex normal angle exceeds this threshold")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    texture_thresholds = threshold_values(
        args, ("max_texture_mae_u8", "max_texture_rmse_u8", "max_texture_abs_u8"))
    normal_thresholds = threshold_values(args, ("max_normal_angle_deg",))
    if any(not np.isfinite(value) or value < 0.0 for value in (*texture_thresholds.values(), *normal_thresholds.values())):
        parser.error("all comparison thresholds must be finite and non-negative")
    if args.structural_only and args.stage_dir is not None:
        parser.error("--structural-only cannot be combined with --stage-dir")
    if not args.structural_only and args.stage_dir is None:
        parser.error("--stage-dir is required unless --structural-only is selected")
    if args.reference_dir is not None and args.stage_dir is None:
        parser.error("--reference-dir requires --stage-dir for a controlled comparison")
    if (args.require_texture_comparison or args.require_normal_comparison or texture_thresholds or
            normal_thresholds) and args.reference_dir is None:
        parser.error("texture/normal comparison requirements need --reference-dir")

    document, binary = glb_document(args.native_glb)
    primitive = document["meshes"][0]["primitives"][0]
    attributes = set(primitive["attributes"])
    if not {"POSITION", "TEXCOORD_0"}.issubset(attributes) or attributes - {
        "POSITION", "NORMAL", "TEXCOORD_0", "COLOR_0"
    }:
        raise RuntimeError(f"unexpected native PBR attributes: {primitive['attributes']}")
    material = document["materials"][0]["pbrMetallicRoughness"]
    material_matches = (material.get("roughnessFactor", 1.0) == 1.0 and
                        material.get("metallicFactor", 1.0) == 1.0 and
                        material.get("baseColorFactor", [1.0] * 4) == [1.0] * 4 and
                        material.get("baseColorTexture", {}).get("index") == 0)
    if "NORMAL" not in primitive["attributes"]:
        raise RuntimeError("native textured PBR GLB must contain computed vertex normals")
    position_count = accessor_count(document, primitive["attributes"]["POSITION"])
    normal_count = accessor_count(document, primitive["attributes"]["NORMAL"])
    uv_count = accessor_count(document, primitive["attributes"]["TEXCOORD_0"])
    if position_count != normal_count or position_count != uv_count:
        raise RuntimeError(
            "GLB POSITION, NORMAL, and TEXCOORD_0 must have identical vertex counts"
        )
    index_count = accessor_count(document, primitive["indices"])
    if index_count % 3:
        raise RuntimeError("GLB index accessor is not a triangle list")

    actual_positions = accessor_f32_vec3(document, binary, primitive["attributes"]["POSITION"])
    actual_normals = accessor_f32_vec3(document, binary, primitive["attributes"]["NORMAL"])
    index_bytes = accessor_bytes(document, binary, primitive["indices"])

    report: dict[str, object] = {
        "native_glb": str(args.native_glb),
        "vertex_count": int(actual_positions.shape[0]),
        "triangle_count": index_count // 3,
        "structural_contract": "PASS",
        "official_material_contract": "PASS" if material_matches else "FAIL",
        "effective_metallic_factor": material.get("metallicFactor", 1.0),
    }
    passed = material_matches
    controlled_mesh_matches = False
    if args.stage_dir is not None:
        expected_positions, expected_indices, expected_uv = load_stage_mesh(args.stage_dir)
        uv_bytes = accessor_bytes(document, binary, primitive["attributes"]["TEXCOORD_0"])
        positions_equal = actual_positions.shape == expected_positions.shape and np.array_equal(
            actual_positions.view("<u4"), expected_positions.view("<u4")
        )
        uv_equal = uv_bytes == expected_uv
        indices_equal = index_bytes == expected_indices
        report["controlled_mesh_uv"] = {
            "expected_vertex_count": int(expected_positions.shape[0]),
            "actual_vertex_count": int(actual_positions.shape[0]),
            "expected_triangle_count": len(expected_indices) // (3 * 4),
            "actual_triangle_count": index_count // 3,
            "positions_bitwise_equal": positions_equal,
            "uv_bitwise_equal": uv_equal,
            "indices_bitwise_equal": indices_equal,
        }
        controlled_mesh_matches = positions_equal and uv_equal and indices_equal
        if controlled_mesh_matches:
            report["controlled_mesh_uv_contract"] = "PASS"
        else:
            report["controlled_mesh_uv_contract"] = "FAIL"
            passed = False
    if args.reference_dir is not None:
        reference_document, reference_binary = glb_document(args.reference_dir / "official_pbr.glb")
        reference_primitive = reference_document["meshes"][0]["primitives"][0]
        reference_uv = accessor_bytes(reference_document, reference_binary,
                                      reference_primitive["attributes"]["TEXCOORD_0"])
        exported_uv_equal = accessor_bytes(document, binary, primitive["attributes"]["TEXCOORD_0"]) == reference_uv
        report["official_glb_uv_bitwise_equal"] = exported_uv_equal
        actual_uv = np.frombuffer(uv_bytes, dtype="<f4").reshape(-1, 2)
        official_uv = np.frombuffer(reference_uv, dtype="<f4").reshape(-1, 2)
        if actual_uv.shape == official_uv.shape:
            flipped_uv = actual_uv.copy()
            flipped_uv[:, 1] = 1.0 - flipped_uv[:, 1]
            report["uv_export_diagnostic"] = {
                "max_abs_vs_official_glb": float(np.max(np.abs(actual_uv - official_uv))),
                "max_abs_after_v_flip": float(np.max(np.abs(flipped_uv - official_uv))),
            }
        controlled_mesh_matches = controlled_mesh_matches and exported_uv_equal
        passed = passed and exported_uv_equal
        with Image.open(args.reference_dir / "base_color.png") as source:
            reference = np.asarray(source.convert("RGB"), dtype=np.uint8)
        texture = image_metrics(embedded_base_color(document, binary), reference)
        report["texture_vs_official"] = texture

        if controlled_mesh_matches:
            official_normals = official_angle_weighted_normals(
                expected_positions, np.frombuffer(expected_indices, dtype="<u4"))
            report["normals_vs_official"] = normal_metrics(actual_normals, official_normals)
        else:
            report["normals_vs_official"] = {
                "available": False,
                "reason": "controlled mesh/UV/index order does not match the official fixture",
            }

        texture_required = args.require_texture_comparison or bool(texture_thresholds)
        if texture_required:
            texture_passed = True
            for name, threshold in texture_thresholds.items():
                metric = name.removeprefix("max_texture_")
                metric = {"mae_u8": "mae_u8", "rmse_u8": "rmse_u8", "abs_u8": "max_abs_u8"}[metric]
                if texture[metric] > threshold:
                    texture_passed = False
            report["texture_contract"] = {
                "required": True,
                "thresholds": texture_thresholds,
                "passed": texture_passed,
                "quality_gate_configured": len(texture_thresholds) == 3,
                "quality_gate_passed": texture_passed if len(texture_thresholds) == 3 else None,
            }
            passed &= texture_passed

        normal_required = args.require_normal_comparison or bool(normal_thresholds)
        if normal_required:
            normals = report["normals_vs_official"]
            normal_passed = isinstance(normals, dict) and normals.get("available", True) is not False
            if normal_passed and normal_thresholds:
                normal_passed = normals["max_angle_deg"] <= normal_thresholds["max_normal_angle_deg"]
            report["normal_contract"] = {
                "required": True,
                "thresholds": normal_thresholds,
                "passed": normal_passed,
                "quality_gate_configured": bool(normal_thresholds),
                "quality_gate_passed": normal_passed if normal_thresholds else None,
            }
            passed &= normal_passed

    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    if passed:
        print(
            f"PASS: native PBR GLB has {report['vertex_count']} vertices and "
            f"{report['triangle_count']} triangles with structural/material contracts"
        )
        return 0
    print("FAIL: native PBR GLB does not exactly match the official controlled mesh/UV contract")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
