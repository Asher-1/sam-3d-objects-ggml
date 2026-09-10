#!/usr/bin/env python3
"""Compare the native MoGe ViT-L graph with the official MoGe checkpoint.

This is a regression oracle only.  The C++ runtime neither starts Python nor
loads Torch; the script launches the already-built native CLI and independently
computes the same deterministic point-map reference.  For F16 GGUF weights,
rank-two-or-greater parameters are emulated by a F16-to-F32 round trip, exactly
matching ``convert_tensor``.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from samt_io import read_samt
import torch
from moge.model.v1 import MoGeModel




def deterministic_image(width: int, height: int) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, width, dtype=torch.float32).view(1, 1, 1, width)
    y = torch.linspace(0.0, 1.0, height, dtype=torch.float32).view(1, 1, height, 1)
    x = x.expand(1, 1, height, width)
    y = y.expand(1, 1, height, width)
    return torch.cat((x, y, torch.full_like(x, 0.25)), dim=1)


def official_reference(checkpoint: Path, width: int, height: int,
                       weights: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model = MoGeModel.from_pretrained(checkpoint).eval()
    with torch.no_grad():
        if weights == "f16":
            for parameter in model.parameters():
                if parameter.ndim > 1:
                    parameter.copy_(parameter.to(torch.float16).to(torch.float32))

        model = model.to(device)
        image = deterministic_image(width, height).to(device)
        image = (image - model.image_mean) / model.image_std
        features = model.backbone.get_intermediate_layers(
            image, model.intermediate_layers, return_class_token=True
        )
        points, mask = model.head(features, image)
        xy, z = points[:, :2], points[:, 2:]
        points = torch.cat((xy * z.exp(), z.exp()), dim=1)
    return points.cpu().numpy(), mask.cpu().numpy()


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = actual - reference
    absolute = np.abs(delta)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "max_abs": float(absolute.max()),
        "cosine": float(np.sum(actual * reference) /
                        np.sqrt(np.sum(actual * actual) * np.sum(reference * reference))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), default="cpu")
    parser.add_argument("--reference-device", choices=("cpu", "cuda"), default="cpu",
                        help="device on which the independent PyTorch oracle runs")
    parser.add_argument("--weights", choices=("f16", "f32"), default="f16",
                        help="GGUF tensor precision used by --model")
    parser.add_argument("--width", type=int, default=28)
    parser.add_argument("--height", type=int, default=28)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--strict-attention", action="store_true",
                        help="use F32 accumulation in native flash attention")
    parser.add_argument("--manual-attention", action="store_true",
                        help="diagnose with explicit F32 QK^T/softmax/V attention; not a throughput path")
    parser.add_argument("--max-points-mae", type=float, default=1.0e-4)
    parser.add_argument("--max-points-abs", type=float, default=5.0e-4)
    parser.add_argument("--max-mask-mae", type=float, default=1.0e-4)
    parser.add_argument("--max-mask-abs", type=float, default=5.0e-4)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    for label, path in (("binary", args.binary), ("GGUF model", args.model),
                        ("official checkpoint", args.checkpoint)):
        if not path.is_file():
            parser.error(f"missing {label}: {path}")
    if args.width < 14 or args.height < 14:
        parser.error("--width and --height must be at least 14")
    if args.reference_device == "cuda" and not torch.cuda.is_available():
        parser.error("--reference-device cuda requested but CUDA is unavailable to PyTorch")

    reference_points, reference_mask = official_reference(
        args.checkpoint, args.width, args.height, args.weights,
        torch.device(args.reference_device),
    )
    with tempfile.TemporaryDirectory(prefix="sam3d-moge-parity-") as directory:
        prefix = Path(directory) / "native"
        command = [
            str(args.binary), "moge-smoke", "--model", str(args.model),
            "--backend", args.backend, "--width", str(args.width),
            "--height", str(args.height), "--threads", str(args.threads),
            "--out", str(prefix),
        ]
        print("+", " ".join(command), flush=True)
        environment = None
        if args.strict_attention or args.manual_attention:
            # The former process-wide attention switches (SAM3D_STRICT_ATTN /
            # SAM3D_MANUAL_ATTN) and the TF32 override (SAM3D_E2E_F32_MATMUL)
            # became explicit graph options on 2026-09-11; the MoGe smoke
            # command no longer reads them. Keep the GGML_CUBLAS math-mode
            # override, which is still honored before handle creation and
            # prevents forced TF32 from obscuring a strict-F32 diagnosis.
            environment = dict(__import__("os").environ)
            if args.backend == "cuda":
                environment["GGML_CUDA_STRICT_F32"] = "1"
        subprocess.run(command, check=True, env=environment)
        points_shape, points = read_samt(prefix.with_suffix(".points.samt"))
        mask_shape, mask = read_samt(prefix.with_suffix(".mask_logits.samt"))

    expected_points_shape = (args.width, args.height, 3, 1)
    expected_mask_shape = (args.width, args.height, 1, 1)
    if points_shape != expected_points_shape or mask_shape != expected_mask_shape:
        raise RuntimeError(
            f"native shapes are points={points_shape}, mask={mask_shape}; expected "
            f"{expected_points_shape}, {expected_mask_shape}"
        )
    points_result = metrics(points.reshape(reference_points.shape), reference_points)
    mask_result = metrics(mask.reshape(reference_mask.shape), reference_mask)
    payload = {
        "backend": args.backend,
        "reference_device": args.reference_device,
        "weights": args.weights,
        "strict_attention": args.strict_attention,
        "manual_attention": args.manual_attention,
        "input": {"width": args.width, "height": args.height},
        "points": points_result,
        "mask_logits": mask_result,
        "limits": {
            "points_mae": args.max_points_mae,
            "points_abs": args.max_points_abs,
            "mask_mae": args.max_mask_mae,
            "mask_abs": args.max_mask_abs,
        },
    }
    print(json.dumps(payload, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if (points_result["mae"] > args.max_points_mae or
            points_result["max_abs"] > args.max_points_abs or
            mask_result["mae"] > args.max_mask_mae or
            mask_result["max_abs"] > args.max_mask_abs):
        raise RuntimeError("MoGe native parity gate failed")
    print("PASS: MoGe native point-map parity gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
