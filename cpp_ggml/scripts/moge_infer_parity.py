#!/usr/bin/env python3
"""Compare native raw-image MoGe inference with MoGeModel.infer().

The Python process is an independent regression oracle only. The C++ runtime
loads a GGUF and never imports Python, Torch, SciPy, or cuDNN.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from moge.model.v1 import MoGeModel


def read_samt(path: Path) -> tuple[tuple[int, ...], np.ndarray]:
    with path.open("rb") as stream:
        if stream.read(4) != b"SAMT":
            raise ValueError(f"{path}: invalid SAMT magic")
        (rank,) = struct.unpack("<i", stream.read(4))
        shape = struct.unpack(f"<{rank}q", stream.read(8 * rank))
        (dtype,) = struct.unpack("<i", stream.read(4))
        if dtype != 0:
            raise ValueError(f"{path}: expected F32 SAMT, got ggml type {dtype}")
        values = np.frombuffer(stream.read(), dtype="<f4").copy()
    if values.size != int(np.prod(shape)):
        raise ValueError(f"{path}: truncated tensor")
    return shape, values


def metric(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    if actual.shape != reference.shape:
        raise ValueError(f"shape mismatch: native={actual.shape}, reference={reference.shape}")
    if not np.array_equal(np.isfinite(actual), np.isfinite(reference)):
        raise ValueError("native and reference finite-value masks differ")
    finite = np.isfinite(reference)
    delta = actual[finite].astype(np.float64) - reference[finite].astype(np.float64)
    absolute = np.abs(delta)
    return {
        "mae": float(absolute.mean()) if absolute.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(delta)))) if delta.size else 0.0,
        "max_abs": float(absolute.max()) if absolute.size else 0.0,
    }


def load_rgb(path: Path) -> torch.Tensor:
    # Mirrors the released pipeline: uint8 / 255 promoted by NumPy, then F32.
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    rgb = (rgba[..., :3] / 255).astype(np.float32)
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


def official_reference(checkpoint: Path, image: torch.Tensor, device: torch.device,
                       reference_storage: str, num_tokens: int) -> dict[str, np.ndarray]:
    model = MoGeModel.from_pretrained(checkpoint).eval()
    with torch.no_grad():
        if reference_storage == "f16-all":
            for parameter in model.parameters():
                parameter.copy_(parameter.to(torch.float16).to(torch.float32))
        elif reference_storage == "f16-matrix":
            for parameter in model.parameters():
                if parameter.ndim > 1:
                    parameter.copy_(parameter.to(torch.float16).to(torch.float32))
        model = model.to(device)
        image_on_device = image.to(device)
        height, width = image.shape[1:]
        resize_factor = ((num_tokens * 14 ** 2) / (height * width)) ** 0.5
        resized_width, resized_height = int(width * resize_factor), int(height * resize_factor)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            forward = model.forward(image_on_device.unsqueeze(0), num_tokens=num_tokens)
        result = model.infer(image_on_device, num_tokens=num_tokens, use_fp16=True,
                             force_projection=False, apply_mask=True)
    points = result["points"].float().cpu().numpy()
    pointmap_pytorch3d = points * np.array([-1.0, -1.0, 1.0], dtype=np.float32)
    return {
        "moge_points": points,
        "forward_points": forward["points"].squeeze(0).float().cpu().numpy(),
        "resized_rgb": torch.nn.functional.interpolate(
            image_on_device.unsqueeze(0), (resized_height, resized_width), mode="bicubic",
            align_corners=False, antialias=True,
        ).squeeze(0).permute(1, 2, 0).float().cpu().numpy(),
        "pointmap_raw": pointmap_pytorch3d,
        "mask_logits": forward["mask"].squeeze(0).float().cpu().numpy(),
        "mask": result["mask"].float().cpu().numpy(),
        "depth": result["depth"].float().cpu().numpy(),
        "intrinsics": result["intrinsics"].float().cpu().numpy(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), default="cuda")
    parser.add_argument("--reference-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--reference-storage",
                        choices=("source", "f16-matrix", "f16-all"), default="source",
                        help="checkpoint storage used by the PyTorch oracle before execution")
    parser.add_argument("--num-tokens", type=int, default=2500)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-points-mae", type=float, default=float("inf"))
    parser.add_argument("--max-points-abs", type=float, default=float("inf"))
    parser.add_argument("--max-mask-mae", type=float, default=float("inf"))
    parser.add_argument("--max-mask-abs", type=float, default=float("inf"))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    for label, path in (("binary", args.binary), ("GGUF", args.model),
                        ("checkpoint", args.checkpoint), ("image", args.image)):
        if not path.is_file():
            parser.error(f"missing {label}: {path}")
    if args.reference_device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA PyTorch reference requested but unavailable")

    image = load_rgb(args.image)
    reference = official_reference(args.checkpoint, image, torch.device(args.reference_device),
                                   args.reference_storage, args.num_tokens)
    with tempfile.TemporaryDirectory(prefix="sam3d-moge-infer-parity-") as directory:
        prefix = Path(directory) / "native"
        command = [
            str(args.binary), "moge-infer", "--model", str(args.model), "--input", str(args.image),
            "--out", str(prefix), "--backend", args.backend, "--num-tokens", str(args.num_tokens),
            "--threads", str(args.threads), "--dump-intermediates",
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        height, width = image.shape[1:]
        shapes = {}
        native = {}
        for name in ("moge_points", "forward_points", "pointmap_raw", "mask_logits", "mask",
                     "depth", "intrinsics", "resized_rgb"):
            shape, values = read_samt(prefix.with_name(prefix.name + f".{name}.samt"))
            shapes[name] = list(shape)
            if name in ("moge_points", "forward_points"):
                native[name] = values.reshape(height, width, 3)
            elif name == "pointmap_raw":
                native[name] = values.reshape(3, height, width).transpose(1, 2, 0)
            elif name == "intrinsics":
                native[name] = values.reshape(3, 3)
            elif name == "resized_rgb":
                native[name] = values.reshape(shape[2], shape[1], 3)
            else:
                native[name] = values.reshape(height, width)
        _, focal_shift = read_samt(prefix.with_name(prefix.name + ".focal_shift.samt"))

    payload = {
        "backend": args.backend,
        "reference_device": args.reference_device,
        "reference_storage": args.reference_storage,
        "num_tokens": args.num_tokens,
        "native_samt_shapes": shapes,
        "moge_points": metric(native["moge_points"], reference["moge_points"]),
        "forward_points": metric(native["forward_points"], reference["forward_points"]),
        "resized_rgb": metric(native["resized_rgb"], reference["resized_rgb"]),
        "pointmap_raw": metric(native["pointmap_raw"], reference["pointmap_raw"]),
        "mask_logits": metric(native["mask_logits"], reference["mask_logits"]),
        "mask": metric(native["mask"], reference["mask"]),
        "depth": metric(native["depth"], reference["depth"]),
        "intrinsics": metric(native["intrinsics"], reference["intrinsics"]),
        "native_focal_shift": {"focal": float(focal_shift[0]), "shift": float(focal_shift[1])},
    }
    print(json.dumps(payload, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    points = payload["moge_points"]
    mask = payload["mask_logits"]
    if (points["mae"] > args.max_points_mae or points["max_abs"] > args.max_points_abs or
            mask["mae"] > args.max_mask_mae or mask["max_abs"] > args.max_mask_abs):
        raise RuntimeError("MoGe infer parity gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
