#!/usr/bin/env python3
"""Validate official sparse mesh-decoder fixtures and compare native outputs.

The official ``dump_e2e_stages.py --dump-mesh-decoder-reference`` produces
five sparse graph boundaries.  This tool first proves that their coordinates
follow the exact two-stage ``SparseSubdivide`` ordering, then optionally
compares a C++ ``sam3d-cli mesh-decode`` output without loading both large raw
feature tensors into RAM at once.
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np

from samt_io import SAMT_MAGIC, GGML_F32 as GGML_TYPE_F32, GGML_I32 as GGML_TYPE_I32

# SAMT fixtures span ggml revisions: this build writes I32 as 26, while the
# canonical E2E capture was written by newer upstream code that used 30. The
# coordinates carry the same signed 32-bit payload in both cases.
GGML_TYPE_I32_LEGACY_UPSTREAM = 30
DTYPES = {
    GGML_TYPE_F32: np.dtype("<f4"),
    GGML_TYPE_I32: np.dtype("<i4"),
    GGML_TYPE_I32_LEGACY_UPSTREAM: np.dtype("<i4"),
}
BASE_STAGES = ("input_layer", "block11", "upsample0", "upsample1", "raw")
BLOCK_STAGES = tuple(f"block{index}" for index in range(12))
ATTENTION_STAGES = tuple(f"block{index}_attention" for index in range(12))
INTERNAL_SUFFIXES = (
    "ape", "norm1", "qkv", "attention_values", "attention_out",
    "norm2", "mlp0", "gelu", "mlp2",
)
INTERNAL_STAGES = tuple(
    f"block{index}_{suffix}" for index in range(12) for suffix in INTERNAL_SUFFIXES
)
UPSAMPLE_INTERNAL_SUFFIXES = (
    "act_norm", "act", "main_sub", "skip_sub", "conv1", "norm", "silu", "conv2", "skip",
)
UPSAMPLE_INTERNAL_STAGES = tuple(
    f"upsample{index}_{suffix}" for index in range(2) for suffix in UPSAMPLE_INTERNAL_SUFFIXES
)
STAGES = BASE_STAGES + BLOCK_STAGES + ATTENTION_STAGES + INTERNAL_STAGES + UPSAMPLE_INTERNAL_STAGES
CHANNELS = {"input_layer": 768, "block11": 768, "upsample0": 192,
            "upsample1": 96, "raw": 101}
CHANNELS.update({stage: 768 for stage in BLOCK_STAGES})
CHANNELS.update({stage: 768 for stage in ATTENTION_STAGES})
CHANNELS.update({stage: 768 for stage in INTERNAL_STAGES})
CHANNELS.update({f"block{index}_qkv": 2304 for index in range(12)})
CHANNELS.update({f"block{index}_mlp0": 3072 for index in range(12)})
CHANNELS.update({f"block{index}_gelu": 3072 for index in range(12)})
for index, (input_channels, output_channels) in enumerate(((768, 192), (192, 96))):
    CHANNELS[f"upsample{index}_act_norm"] = input_channels
    CHANNELS[f"upsample{index}_act"] = input_channels
    CHANNELS[f"upsample{index}_main_sub"] = input_channels
    CHANNELS[f"upsample{index}_skip_sub"] = input_channels
    for suffix in ("conv1", "norm", "silu", "conv2", "skip"):
        CHANNELS[f"upsample{index}_{suffix}"] = output_channels


def samt_header(path: Path):
    with path.open("rb") as handle:
        if handle.read(4) != SAMT_MAGIC:
            raise ValueError(f"{path}: missing SAMT magic")
        ndim_data = handle.read(4)
        if len(ndim_data) != 4:
            raise ValueError(f"{path}: truncated rank")
        ndim = struct.unpack("<i", ndim_data)[0]
        if not 1 <= ndim <= 4:
            raise ValueError(f"{path}: invalid rank {ndim}")
        dimensions = struct.unpack(f"<{ndim}q", handle.read(8 * ndim))
        type_data = handle.read(4)
        if len(type_data) != 4:
            raise ValueError(f"{path}: truncated GGML type")
        ggml_type = struct.unpack("<i", type_data)[0]
        if ggml_type not in DTYPES:
            raise ValueError(f"{path}: unsupported GGML type {ggml_type}")
        offset = handle.tell()
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(f"{path}: non-positive tensor dimension")
    item_count = int(np.prod(dimensions, dtype=np.int64))
    expected_size = offset + item_count * DTYPES[ggml_type].itemsize
    if path.stat().st_size != expected_size:
        raise ValueError(f"{path}: expected {expected_size} bytes, found {path.stat().st_size}")
    # SAMT stores GGML order (last NumPy axis fastest), so restore the original
    # PyTorch shape when presenting or indexing it here.
    return tuple(reversed(dimensions)), DTYPES[ggml_type], offset


def samt_memmap(path: Path):
    shape, dtype, offset = samt_header(path)
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)


def subdivide(coords: np.ndarray) -> np.ndarray:
    """Exact SparseSubdivide coordinate order: parent-major, z fastest."""
    offsets = np.array(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
        dtype=np.int32,
    )
    children = np.repeat(coords, 8, axis=0).astype(np.int32, copy=False)
    children[:, 1:] *= 2
    children[:, 1:] += np.tile(offsets, (len(coords), 1))
    return children


def first_mismatch(actual: np.ndarray, expected: np.ndarray):
    mismatch = np.flatnonzero(np.any(actual != expected, axis=1))
    if mismatch.size == 0:
        return None
    index = int(mismatch[0])
    return {"row": index, "actual": actual[index].tolist(), "expected": expected[index].tolist()}


def validate_reference(reference_dir: Path):
    report = {"reference_dir": str(reference_dir), "stages": {}, "coordinate_checks": {}}
    slat_features = samt_memmap(reference_dir / "slat_feats_final.samt")
    slat_coords = samt_memmap(reference_dir / "slat_coords.samt")
    if slat_features.dtype != np.dtype("<f4") or slat_features.ndim != 2 or slat_features.shape[1] != 8:
        raise ValueError("slat_feats_final.samt must be F32 [tokens, 8]")
    if slat_coords.dtype != np.dtype("<i4") or slat_coords.shape != (slat_features.shape[0], 4):
        raise ValueError("slat_coords.samt must be I32 [tokens, 4] matching slat features")

    base_count = slat_features.shape[0]
    expected_counts = {"input_layer": base_count, "block11": base_count,
                       "upsample0": base_count * 8, "upsample1": base_count * 64,
                       "raw": base_count * 64}
    for index in range(2):
        input_count = base_count * (8 ** index)
        output_count = input_count * 8
        expected_counts[f"upsample{index}_act_norm"] = input_count
        expected_counts[f"upsample{index}_act"] = input_count
        expected_counts[f"upsample{index}_main_sub"] = output_count
        expected_counts[f"upsample{index}_skip_sub"] = output_count
        for suffix in ("conv1", "norm", "silu", "conv2", "skip"):
            expected_counts[f"upsample{index}_{suffix}"] = output_count
    references = {}
    # The five graph boundaries have been part of the fixture contract from
    # its first version.  Per-transformer-block references are additive so an
    # existing fixture remains valid while a new one gains precise bisection.
    stages_to_validate = list(BASE_STAGES)
    for stage in (*BLOCK_STAGES, *ATTENTION_STAGES, *INTERNAL_STAGES, *UPSAMPLE_INTERNAL_STAGES):
        feature_path = reference_dir / f"mesh_decoder_{stage}_features.samt"
        coordinate_path = reference_dir / f"mesh_decoder_{stage}_coords.samt"
        if feature_path.is_file() != coordinate_path.is_file():
            raise ValueError(f"{stage}: feature and coordinate references must be present together")
        if feature_path.is_file() and stage not in stages_to_validate:
            stages_to_validate.append(stage)
    for stage in stages_to_validate:
        feature_path = reference_dir / f"mesh_decoder_{stage}_features.samt"
        coordinate_path = reference_dir / f"mesh_decoder_{stage}_coords.samt"
        if not feature_path.is_file() or not coordinate_path.is_file():
            raise FileNotFoundError(
                f"missing {stage} reference; rerun dump_e2e_stages.py "
                "with --dump-mesh-decoder-reference")
        features = samt_memmap(feature_path)
        coordinates = samt_memmap(coordinate_path)
        expected_count = expected_counts.get(stage, base_count)
        expected_shape = (expected_count, CHANNELS[stage])
        if features.dtype != np.dtype("<f4") or features.shape != expected_shape:
            raise ValueError(f"{feature_path}: expected F32 {expected_shape}, found {features.dtype} {features.shape}")
        if coordinates.dtype != np.dtype("<i4") or coordinates.shape != (expected_count, 4):
            raise ValueError(f"{coordinate_path}: invalid sparse coordinate shape {coordinates.shape}")
        references[stage] = (features, coordinates)
        report["stages"][stage] = {"features": list(features.shape), "coordinates": list(coordinates.shape)}

    expected_coordinates = {
        "input_layer": np.asarray(slat_coords),
        "block11": np.asarray(slat_coords),
    }
    expected_coordinates["upsample0"] = subdivide(expected_coordinates["block11"])
    expected_coordinates["upsample1"] = subdivide(expected_coordinates["upsample0"])
    expected_coordinates["raw"] = expected_coordinates["upsample1"]
    for stage in BLOCK_STAGES:
        if stage in references:
            expected_coordinates[stage] = np.asarray(slat_coords)
    for stage in ATTENTION_STAGES:
        if stage in references:
            expected_coordinates[stage] = np.asarray(slat_coords)
    for stage in INTERNAL_STAGES:
        if stage in references:
            expected_coordinates[stage] = np.asarray(slat_coords)
    for index in range(2):
        input_coordinates = (expected_coordinates["block11"] if index == 0
                             else expected_coordinates["upsample0"])
        output_coordinates = expected_coordinates[f"upsample{index}"]
        for suffix in ("act_norm", "act"):
            stage = f"upsample{index}_{suffix}"
            if stage in references:
                expected_coordinates[stage] = input_coordinates
        for suffix in ("main_sub", "skip_sub", "conv1", "norm", "silu", "conv2", "skip"):
            stage = f"upsample{index}_{suffix}"
            if stage in references:
                expected_coordinates[stage] = output_coordinates
    for stage, expected in expected_coordinates.items():
        mismatch = first_mismatch(np.asarray(references[stage][1]), expected)
        report["coordinate_checks"][stage] = {
            "exact": mismatch is None,
            "first_mismatch": mismatch,
        }
        if mismatch is not None:
            raise ValueError(f"{stage} coordinates do not match SparseSubdivide: {mismatch}")
    report["available_stages"] = stages_to_validate
    return report


def feature_metrics(reference_path: Path, native_path: Path, chunk_values: int):
    reference = samt_memmap(reference_path)
    native = samt_memmap(native_path)
    if reference.dtype != np.dtype("<f4") or native.dtype != np.dtype("<f4"):
        raise ValueError("feature comparison requires F32 SAMT tensors")
    if reference.shape != native.shape:
        raise ValueError(f"feature shape mismatch: reference {reference.shape}, native {native.shape}")
    count = int(reference.size)
    total_abs = 0.0
    total_squared = 0.0
    max_abs = -1.0
    max_flat_index = 0
    for start in range(0, count, chunk_values):
        end = min(count, start + chunk_values)
        delta = np.asarray(native).reshape(-1)[start:end].astype(np.float64) - \
            np.asarray(reference).reshape(-1)[start:end].astype(np.float64)
        if not np.isfinite(delta).all():
            raise ValueError(f"non-finite difference in range [{start}, {end})")
        absolute = np.abs(delta)
        local_index = int(np.argmax(absolute))
        local_max = float(absolute[local_index])
        if local_max > max_abs:
            max_abs = local_max
            max_flat_index = start + local_index
        total_abs += float(absolute.sum())
        total_squared += float(np.square(delta).sum())
    row, channel = divmod(max_flat_index, reference.shape[1])
    return {
        "shape": list(reference.shape),
        "mae": total_abs / count,
        "rmse": (total_squared / count) ** 0.5,
        "max_abs": max_abs,
        "worst_location": {"token": row, "channel": channel},
        "reference": float(reference[row, channel]),
        "native": float(native[row, channel]),
    }


def exact_coordinate_compare(reference_path: Path, native_path: Path):
    reference = samt_memmap(reference_path)
    native = samt_memmap(native_path)
    if reference.dtype != np.dtype("<i4") or native.dtype != np.dtype("<i4"):
        raise ValueError("coordinate comparison requires I32 SAMT tensors")
    if reference.shape != native.shape:
        raise ValueError(f"coordinate shape mismatch: reference {reference.shape}, native {native.shape}")
    mismatch = first_mismatch(np.asarray(native), np.asarray(reference))
    return {"exact": mismatch is None, "first_mismatch": mismatch}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--stage", default="raw")
    parser.add_argument("--native-features", type=Path)
    parser.add_argument("--native-coords", type=Path)
    parser.add_argument("--chunk-values", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--max-mae", type=float,
                        help="fail only when the measured MAE exceeds this explicit threshold")
    parser.add_argument("--max-abs", type=float,
                        help="fail only when the measured maximum absolute error exceeds this threshold")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.chunk_values <= 0:
        parser.error("--chunk-values must be positive")
    if (args.native_features is None) != (args.native_coords is None):
        parser.error("--native-features and --native-coords must be provided together")

    report = validate_reference(args.reference_dir)
    if args.stage not in report["available_stages"]:
        parser.error(
            f"--stage {args.stage!r} is unavailable in this fixture; "
            f"available: {', '.join(report['available_stages'])}"
        )
    if args.native_features is not None:
        feature_path = args.reference_dir / f"mesh_decoder_{args.stage}_features.samt"
        coordinate_path = args.reference_dir / f"mesh_decoder_{args.stage}_coords.samt"
        report["native"] = {
            "stage": args.stage,
            "features": feature_metrics(feature_path, args.native_features, args.chunk_values),
            "coordinates": exact_coordinate_compare(coordinate_path, args.native_coords),
        }
        if not report["native"]["coordinates"]["exact"]:
            raise ValueError("native sparse coordinates differ from the official reference")
        metrics = report["native"]["features"]
        if args.max_mae is not None and metrics["mae"] > args.max_mae:
            raise ValueError(f"MAE {metrics['mae']:.9g} exceeds {args.max_mae:.9g}")
        if args.max_abs is not None and metrics["max_abs"] > args.max_abs:
            raise ValueError(f"max abs {metrics['max_abs']:.9g} exceeds {args.max_abs:.9g}")

    text = json.dumps(report, indent=2)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
