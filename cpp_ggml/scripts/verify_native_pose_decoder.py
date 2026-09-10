#!/usr/bin/env python3
"""Compare the native ScaleShiftInvariant pose decoder with a PyTorch3D oracle.

The oracle consumes exported official SAMT tensors and reconstructs the exact
``pipeline.yaml`` pose path.  It is intentionally independent from
``sam3d_objects`` package imports so a checkpoint's source-tree packaging does
not mask a native pose regression.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path

import torch
from pytorch3d.transforms import Transform3d, matrix_to_quaternion, quaternion_to_matrix


ROTATION_6D_MEAN = torch.tensor(
    [-0.06366084883674913, 0.008438224692279752, 0.00017084786438302483,
     0.0007126610473540038, -0.0030916726538816417, 0.5166093753457688],
    dtype=torch.float32,
)
ROTATION_6D_STD = torch.tensor(
    [0.6656971967514863, 0.6787012271867754, 0.30345010594844524,
     0.4394504420678794, 0.39817973931717104, 0.6176286868761914],
    dtype=torch.float32,
)


def read_f32_samt(path: Path, expected_elements: int) -> torch.Tensor:
    payload = path.read_bytes()
    if len(payload) < 8 or payload[:4] != b"SAMT":
        raise ValueError(f"{path}: invalid SAMT header")
    ndim = struct.unpack_from("<i", payload, 4)[0]
    if ndim < 1 or ndim > 8:
        raise ValueError(f"{path}: invalid SAMT dimension count")
    offset = 8 + 8 * ndim
    if len(payload) < offset + 4:
        raise ValueError(f"{path}: truncated SAMT type field")
    ggml_type = struct.unpack_from("<i", payload, offset)[0]
    # GGML_TYPE_F32 is zero.  Pose inputs are dumped as F32 regardless of the
    # model precision, so accepting another storage type would hide a broken
    # stage-export contract.
    if ggml_type != 0:
        raise ValueError(f"{path}: expected F32 SAMT, got ggml type {ggml_type}")
    offset += 4
    if len(payload) != offset + 4 * expected_elements:
        raise ValueError(f"{path}: expected F32[{expected_elements}] payload")
    values = struct.unpack_from(f"<{expected_elements}f", payload, offset)
    return torch.tensor(values, dtype=torch.float32)


def official_pose(rotation_6d: torch.Tensor, log_scale: torch.Tensor,
                  translation: torch.Tensor, log_translation_scale: torch.Tensor,
                  scene_scale: torch.Tensor, scene_shift: torch.Tensor) -> dict[str, torch.Tensor]:
    rotation_6d = rotation_6d * ROTATION_6D_STD + ROTATION_6D_MEAN
    b1 = torch.nn.functional.normalize(rotation_6d[:3], dim=-1)
    a2 = rotation_6d[3:]
    b2 = torch.nn.functional.normalize(a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1,
                                        dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    decoded_quaternion = matrix_to_quaternion(torch.stack([b1, b2, b3], dim=-1))
    # This mirrors ScaleShiftInvariant.to_instance_pose, including its
    # Transform3d composition/decomposition instead of assuming an isotropic
    # scene scale.  translation_scale is decoded but not used by this convention.
    composed = (
        Transform3d()
        .scale(torch.exp(log_scale)[None])
        .rotate(quaternion_to_matrix(decoded_quaternion)[None])
        .translate(translation[None])
        .compose(Transform3d().scale(scene_scale[None]).translate(scene_shift[None]))
        .get_matrix()[0]
    )
    instance_scale = torch.linalg.vector_norm(composed[:3, :3], dim=-1)
    instance_rotation = matrix_to_quaternion(composed[:3, :3] / instance_scale[:, None])
    return {
        "rotation_wxyz": instance_rotation,
        "translation": composed[3, :3],
        "scale": instance_scale,
        "translation_scale": torch.exp(log_translation_scale)[0],
        "scene_scale": scene_scale,
        "scene_shift": scene_shift,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--rotation-6d", type=Path, required=True)
    parser.add_argument("--log-scale", type=Path, required=True)
    parser.add_argument("--translation", type=Path, required=True)
    parser.add_argument("--log-translation-scale", type=Path, required=True)
    parser.add_argument("--scene-scale", type=Path, required=True)
    parser.add_argument("--scene-shift", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=5.0e-5)
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error(f"missing native binary: {args.binary}")

    inputs = {
        "rotation_6d": read_f32_samt(args.rotation_6d, 6),
        "log_scale": read_f32_samt(args.log_scale, 3),
        "translation": read_f32_samt(args.translation, 3),
        "log_translation_scale": read_f32_samt(args.log_translation_scale, 1),
        "scene_scale": read_f32_samt(args.scene_scale, 3),
        "scene_shift": read_f32_samt(args.scene_shift, 3),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.binary), "pose-decode",
        "--rotation-6d", str(args.rotation_6d),
        "--log-scale", str(args.log_scale),
        "--translation", str(args.translation),
        "--log-translation-scale", str(args.log_translation_scale),
        "--scene-scale", str(args.scene_scale),
        "--scene-shift", str(args.scene_shift),
        "--out", str(args.out),
    ]
    subprocess.run(command, check=True)
    native = json.loads(args.out.read_text(encoding="utf-8"))
    if native.get("schema") != "sam3d.native-pose.v1" or \
            native.get("convention") != "ScaleShiftInvariant":
        raise RuntimeError("native pose output does not declare the expected contract")
    oracle = official_pose(**inputs)
    metrics: dict[str, float] = {}
    for name, expected in oracle.items():
        actual = torch.tensor(native[name], dtype=torch.float32)
        if actual.shape != expected.shape:
            raise RuntimeError(f"{name}: shape mismatch {tuple(actual.shape)} != {tuple(expected.shape)}")
        metrics[name] = float((actual - expected).abs().max())
    result = {
        "schema": "sam3d.native-pose-verify.v1",
        "native": str(args.out.resolve()),
        "atol": args.atol,
        "max_abs_error": metrics,
        "passed": all(value <= args.atol for value in metrics.values()),
    }
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError(f"native pose differs from the PyTorch3D oracle: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
