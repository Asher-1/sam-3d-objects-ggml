#!/usr/bin/env python3
"""Require native mesh-camera contracts to match the official PyTorch path.

The official visibility cleanup and texture bake use different camera helpers.
The default visibility contract compares the fixed 1000-view CUDA matrices
bit-for-bit; the bake contract diagnoses the separate OpenCV camera path.
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import utils3d


def load_samt(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        if stream.read(4) != b"SAMT":
            raise ValueError(f"{path}: invalid SAMT magic")
        (rank,) = struct.unpack("<i", stream.read(4))
        shape = struct.unpack(f"<{rank}q", stream.read(rank * 8))
        (ggml_type,) = struct.unpack("<i", stream.read(4))
        if ggml_type != 0:
            raise ValueError(f"{path}: expected F32 SAMT, got ggml type {ggml_type}")
        values = np.frombuffer(stream.read(), dtype=np.dtype("<f4"))
    expected_count = int(np.prod(shape, dtype=np.int64))
    if values.size != expected_count:
        raise ValueError(f"{path}: expected {expected_count} values, found {values.size}")
    return values.reshape(tuple(reversed(shape))).copy()


def save_samt(path: Path, values: np.ndarray) -> None:
    values = np.ascontiguousarray(values, dtype=np.float32)
    ggml_shape = tuple(reversed(values.shape))
    with path.open("wb") as stream:
        stream.write(b"SAMT")
        stream.write(struct.pack("<i", len(ggml_shape)))
        stream.write(struct.pack(f"<{len(ggml_shape)}q", *ggml_shape))
        stream.write(struct.pack("<i", 0))
        stream.write(values.tobytes())


def radical_inverse_base_two(index: int) -> float:
    result = 0.0
    factor = 0.5
    while index:
        result += (index & 1) * factor
        index >>= 1
        factor *= 0.5
    return result


def official_cameras(view_count: int) -> tuple[np.ndarray, np.ndarray]:
    extrinsics = []
    intrinsics = []
    for index in range(view_count):
        # Match sphere_hammersley_sequence() before render_utils converts these
        # NumPy doubles to one CUDA F32 scalar per angle.
        u = index / view_count
        yaw = radical_inverse_base_two(index) * 2.0 * np.pi
        pitch = np.arccos(1.0 - 2.0 * u) - np.pi / 2.0
        yaw_t = torch.tensor(float(yaw), device="cuda")
        pitch_t = torch.tensor(float(pitch), device="cuda")
        eye = torch.stack((
            torch.sin(yaw_t) * torch.cos(pitch_t),
            torch.cos(yaw_t) * torch.cos(pitch_t),
            torch.sin(pitch_t),
        )) * 2.0
        extrinsic = utils3d.torch.extrinsics_look_at(
            eye, torch.zeros(3, device="cuda"), torch.tensor([0.0, 0.0, 1.0], device="cuda")
        )
        fov = torch.deg2rad(torch.tensor(40.0, device="cuda"))
        intrinsic = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extrinsic.cpu().numpy())
        intrinsics.append(intrinsic.cpu().numpy())
    return np.stack(extrinsics), np.stack(intrinsics)


def official_visibility_matrices(view_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce _fill_holes(), whose camera contract differs from texture bake."""
    yaws = torch.tensor(
        [radical_inverse_base_two(index) * 2.0 * np.pi for index in range(view_count)]
    ).cuda()
    pitches = torch.tensor(
        [np.arccos(1.0 - 2.0 * index / view_count) - np.pi / 2.0 for index in range(view_count)]
    ).cuda()
    fov = torch.deg2rad(torch.tensor(40)).cuda()
    projection = utils3d.torch.perspective_from_fov_xy(fov, fov, 1, 3)
    views = []
    for yaw, pitch in zip(yaws, pitches):
        origin = (
            torch.tensor(
                [
                    torch.sin(yaw) * torch.cos(pitch),
                    torch.cos(yaw) * torch.cos(pitch),
                    torch.sin(pitch),
                ]
            )
            .cuda()
            .float()
            * 2.0
        )
        view = utils3d.torch.view_look_at(
            origin,
            torch.tensor([0, 0, 0]).float().cuda(),
            torch.tensor([0, 0, 1]).float().cuda(),
        )
        views.append(view)
    view_values = torch.stack(views, dim=0).cpu().numpy()
    projection_values = projection.cpu().numpy()
    if projection_values.ndim == 2:
        projection_values = np.broadcast_to(projection_values, view_values.shape).copy()
    elif projection_values.shape == (1, 4, 4):
        projection_values = np.broadcast_to(projection_values, view_values.shape).copy()
    if projection_values.shape != view_values.shape:
        raise ValueError(
            f"official visibility projection has unexpected shape {projection_values.shape}"
        )
    return view_values, projection_values


def compare(label: str, native: np.ndarray, official: np.ndarray) -> None:
    if native.shape != official.shape:
        raise ValueError(f"{label}: shape mismatch {native.shape} != {official.shape}")
    difference = np.abs(native - official)
    exact = bool(np.array_equal(native, official))
    mismatch_count = int(np.count_nonzero(native != official))
    print(
        f"{label}: exact={exact} mae={float(difference.mean()):.9g} "
        f"max_abs={float(difference.max()):.9g} mismatches={mismatch_count}/{native.size}"
    )
    if not exact:
        mismatch = np.argwhere(native != official)[0]
        point = tuple(int(value) for value in mismatch)
        raise ValueError(
            f"{label}: first exact mismatch at {point}: "
            f"native={native[point]!r} official={official[point]!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--views", type=int, default=1000)
    parser.add_argument("--contract", choices=("visibility", "bake"), default="visibility")
    parser.add_argument("--official-extrinsics-out", type=Path)
    parser.add_argument("--official-intrinsics-out", type=Path)
    parser.add_argument("--official-views-out", type=Path)
    parser.add_argument("--official-projections-out", type=Path)
    args = parser.parse_args()
    if args.views <= 0:
        raise ValueError("--views must be positive")
    if args.contract == "visibility" and args.views != 1000:
        raise ValueError("the exact visibility contract is the official 1000-view trajectory")
    if bool(args.official_extrinsics_out) != bool(args.official_intrinsics_out):
        raise ValueError("--official-extrinsics-out and --official-intrinsics-out must be supplied together")
    if bool(args.official_views_out) != bool(args.official_projections_out):
        raise ValueError("--official-views-out and --official-projections-out must be supplied together")
    binary = args.binary.resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"missing binary: {binary}")
    with tempfile.TemporaryDirectory(prefix="sam3d-mesh-camera-") as temporary:
        root = Path(temporary)
        native_extrinsics = root / "extrinsics.samt"
        native_intrinsics = root / "intrinsics.samt"
        command = [
            str(binary), "mesh-camera-dump",
            "--extrinsics-out", str(native_extrinsics),
            "--intrinsics-out", str(native_intrinsics),
            "--views", str(args.views),
        ]
        native_views = root / "views.samt"
        native_projections = root / "projections.samt"
        if args.contract == "visibility":
            command.extend((
                "--views-out", str(native_views),
                "--projections-out", str(native_projections),
            ))
        subprocess.run(command, check=True)
        if args.official_extrinsics_out:
            official_extrinsics, official_intrinsics = official_cameras(args.views)
            save_samt(args.official_extrinsics_out, official_extrinsics)
            save_samt(args.official_intrinsics_out, official_intrinsics)
        official_views, official_projections = official_visibility_matrices(args.views)
        if args.official_views_out:
            save_samt(args.official_views_out, official_views)
            save_samt(args.official_projections_out, official_projections)
        if args.contract == "visibility":
            compare("visibility views", load_samt(native_views), official_views)
            compare("visibility projections", load_samt(native_projections), official_projections)
        else:
            official_extrinsics, official_intrinsics = official_cameras(args.views)
            compare("bake extrinsics", load_samt(native_extrinsics), official_extrinsics)
            compare("bake intrinsics", load_samt(native_intrinsics), official_intrinsics)
    print(f"PASS: native {args.contract} camera matrices match the official PyTorch contract exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
