#!/usr/bin/env python3
"""Compare native FlexiCubes output against an official mesh-decoder fixture."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from verify_mesh_decoder_reference import samt_memmap


def f32_tensor(path: Path, channels: int) -> np.ndarray:
    tensor = samt_memmap(path)
    if tensor.dtype != np.dtype("<f4") or tensor.ndim != 2 or tensor.shape[1] != channels:
        raise ValueError(f"{path}: expected F32 [items, {channels}], found {tensor.dtype} {tensor.shape}")
    return np.asarray(tensor)


def index_tensor(path: Path, channels: int) -> np.ndarray:
    tensor = samt_memmap(path)
    if tensor.dtype not in (np.dtype("<f4"), np.dtype("<i4")) or \
            tensor.ndim != 2 or tensor.shape[1] != channels:
        raise ValueError(
            f"{path}: expected F32/I32 [items, {channels}], found {tensor.dtype} {tensor.shape}"
        )
    return np.asarray(tensor)


def metrics(reference: np.ndarray, native: np.ndarray) -> dict[str, float]:
    if reference.shape != native.shape:
        raise ValueError(f"shape mismatch: official {reference.shape}, native {native.shape}")
    delta = native.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(delta)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "max_abs": float(absolute.max()),
    }


def canonical_triangles(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(faces).all() or not np.equal(faces, np.floor(faces)).all():
        raise ValueError("faces must be finite integer-valued F32")
    integer = faces.astype(np.int64)
    canonical = np.sort(integer, axis=1)
    order = np.lexsort((canonical[:, 2], canonical[:, 1], canonical[:, 0]))
    return integer[order], canonical[order]


def cyclic_winding_equal(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(
        np.array_equal(first, second)
        or np.array_equal(first, second[:, [1, 2, 0]])
        or np.array_equal(first, second[:, [2, 0, 1]])
    )


def topology_report(official: np.ndarray, native: np.ndarray) -> dict[str, object]:
    if official.shape != native.shape:
        raise ValueError(f"face shape mismatch: official {official.shape}, native {native.shape}")
    native_ordered, native_canonical = canonical_triangles(native)
    official_ordered, official_canonical = canonical_triangles(official)
    same_triangles = np.array_equal(native_canonical, official_canonical)
    same_winding = same_triangles and cyclic_winding_equal(native_ordered, official_ordered)
    return {
        "same_array_order": bool(np.array_equal(native, official)),
        "same_triangle_set": bool(same_triangles),
        "same_cyclic_winding": bool(same_winding),
        "render_equivalent": bool(same_triangles and same_winding),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument(
        "--reference-prefix",
        default="decode_mesh",
        help="basename prefix for <prefix>_{vertices,faces,vertex_attrs}.samt",
    )
    parser.add_argument(
        "--reference-z-up",
        action="store_true",
        help="rotate official mesh-decoder z-up vertices into final GLB y-up coordinates",
    )
    parser.add_argument("--native-vertices", required=True, type=Path)
    parser.add_argument("--native-faces", required=True, type=Path)
    parser.add_argument("--native-attrs", required=True, type=Path)
    parser.add_argument("--max-position", type=float, default=3e-7)
    parser.add_argument("--max-attributes", type=float, default=5e-7)
    parser.add_argument("--strict-face-order", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    prefix = args.reference_prefix
    official_vertices = f32_tensor(args.reference_dir / f"{prefix}_vertices.samt", 3)
    official_faces = index_tensor(args.reference_dir / f"{prefix}_faces.samt", 3)
    official_attrs = f32_tensor(args.reference_dir / f"{prefix}_vertex_attrs.samt", 6)
    native_vertices = f32_tensor(args.native_vertices, 3)
    native_faces = index_tensor(args.native_faces, 3)
    native_attrs = f32_tensor(args.native_attrs, 6)
    if args.reference_z_up:
        # Match the released ``to_glb()`` row-vector transform used by the
        # native exporter: (x, y, z) -> (x, z, -y).
        official_vertices = np.column_stack(
            (official_vertices[:, 0], official_vertices[:, 2], -official_vertices[:, 1])
        )

    report = {
        "reference_coordinate_system": "z-up -> glb-y-up" if args.reference_z_up else "as-recorded",
        "official": {
            "vertices": int(official_vertices.shape[0]),
            "triangles": int(official_faces.shape[0]),
        },
        "native": {
            "vertices": int(native_vertices.shape[0]),
            "triangles": int(native_faces.shape[0]),
        },
        "positions": metrics(official_vertices, native_vertices),
        "attributes": metrics(official_attrs, native_attrs),
        "topology": topology_report(official_faces, native_faces),
    }
    if report["positions"]["max_abs"] > args.max_position:
        raise ValueError(f"position max abs {report['positions']['max_abs']:.9g} exceeds {args.max_position:.9g}")
    if report["attributes"]["max_abs"] > args.max_attributes:
        raise ValueError(f"attribute max abs {report['attributes']['max_abs']:.9g} exceeds {args.max_attributes:.9g}")
    if not report["topology"]["render_equivalent"]:
        raise ValueError("native triangle set or winding differs from the official mesh")
    if args.strict_face_order and not report["topology"]["same_array_order"]:
        raise ValueError("native face rows are render-equivalent but not array-order identical")

    text = json.dumps(report, indent=2)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
