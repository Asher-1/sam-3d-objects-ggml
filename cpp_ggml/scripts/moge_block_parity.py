#!/usr/bin/env python3
"""Locate the first MoGe DINO block that diverges from the official model.

The native CLI is executed once with its block-debug outputs enabled. The
official model is independently evaluated with forward hooks on the same DINO
blocks. This is deliberately a diagnostic, not a runtime dependency.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from samt_io import read_samt


def read_samt_values(path):
    """samt_io.read_samt returns (ne, values); the checks below only need the
    flat numpy values (numpy provides .size and .reshape)."""
    _, values = read_samt(path)
    return values
import torch
from PIL import Image
import torch.nn.functional as F

from moge.model.v1 import MoGeModel




def deterministic_image(width: int, height: int) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, width, dtype=torch.float32).view(1, 1, 1, width)
    y = torch.linspace(0.0, 1.0, height, dtype=torch.float32).view(1, 1, height, 1)
    x = x.expand(1, 1, height, width)
    y = y.expand(1, 1, height, width)
    return torch.cat((x, y, torch.full_like(x, 0.25)), dim=1)


def load_image(path: Path) -> torch.Tensor:
    """Return the native/official common RGB [1, 3, H, W] input in [0, 1]."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)


def moge_input_for_tokens(image: torch.Tensor, num_tokens: int,
                          image_mean: torch.Tensor, image_std: torch.Tensor) -> torch.Tensor:
    """Reproduce MoGeModel.forward's two resize operations before DINO."""
    _, _, original_height, original_width = image.shape
    resize_factor = np.sqrt(num_tokens * 14 * 14 / (original_height * original_width))
    resized_width = int(original_width * resize_factor)
    resized_height = int(original_height * resize_factor)
    image = F.interpolate(
        image,
        size=(resized_height, resized_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    image = (image - image_mean) / image_std
    return F.interpolate(
        image,
        size=(resized_height // 14 * 14, resized_width // 14 * 14),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    absolute = np.abs(actual - reference)
    return {"mae": float(absolute.mean()), "max_abs": float(absolute.max())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--width", type=int, default=28)
    parser.add_argument("--height", type=int, default=28)
    parser.add_argument("--image", type=Path,
                        help="real RGB image; enables the MoGe original-image resize/token path")
    parser.add_argument("--num-tokens", type=int, default=2500,
                        help="MoGe token budget used with --image")
    parser.add_argument("--reference-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--reference-precision", choices=("f32", "cuda-fp16"), default="f32",
                        help="PyTorch execution mode; cuda-fp16 matches MoGeModel.infer(use_fp16=True)")
    parser.add_argument("--reference-tf32", choices=("default", "disabled"), default="default",
                        help="CUDA F32 matrix policy; record it so F32 graph comparisons are reproducible")
    parser.add_argument("--reference-storage", choices=("source", "f16-matrix", "f16-all"),
                        default="source",
                        help="checkpoint storage used by the PyTorch oracle before execution")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.reference_device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA PyTorch reference requested but unavailable")
    if args.reference_precision == "cuda-fp16" and args.reference_device != "cuda":
        parser.error("cuda-fp16 reference precision requires --reference-device cuda")

    device = torch.device(args.reference_device)
    if device.type == "cuda" and args.reference_tf32 == "disabled":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = MoGeModel.from_pretrained(args.checkpoint).eval()
    with torch.no_grad():
        if args.reference_storage == "f16-all":
            for parameter in model.parameters():
                parameter.copy_(parameter.to(torch.float16).to(torch.float32))
        elif args.reference_storage == "f16-matrix":
            for parameter in model.parameters():
                if parameter.ndim > 1:
                    parameter.copy_(parameter.to(torch.float16).to(torch.float32))
    model = model.to(device)
    reference_blocks: list[np.ndarray | None] = [None] * len(model.backbone.blocks)
    reference_attention_projection: list[np.ndarray | None] = [None] * len(model.backbone.blocks)
    reference_attention: list[np.ndarray | None] = [None] * len(model.backbone.blocks)
    reference_mlp_fc1: list[np.ndarray | None] = [None] * len(model.backbone.blocks)
    reference_mlp_gelu: list[np.ndarray | None] = [None] * len(model.backbone.blocks)
    reference_mlp: list[np.ndarray | None] = [None] * len(model.backbone.blocks)
    reference_backbone_image: np.ndarray | None = None
    reference_patch_tokens: np.ndarray | None = None
    reference_position_tokens: np.ndarray | None = None
    reference_backbone_input: np.ndarray | None = None
    reference_attention_context: np.ndarray | None = None
    reference_qkv0: np.ndarray | None = None
    hooks = []
    for index, block in enumerate(model.backbone.blocks):
        if index == 0:
            def capture_backbone_input(_module, inputs):
                nonlocal reference_backbone_input
                reference_backbone_input = inputs[0].detach().float().cpu().numpy()
            hooks.append(block.register_forward_pre_hook(capture_backbone_input))
        if index == 0:
            def capture_qkv(_module, _inputs, output):
                nonlocal reference_qkv0
                reference_qkv0 = output.detach().float().cpu().numpy()
            hooks.append(block.attn.qkv.register_forward_hook(capture_qkv))
            def capture_attention_context(_module, inputs):
                nonlocal reference_attention_context
                reference_attention_context = inputs[0].detach().float().cpu().numpy()
            hooks.append(block.attn.proj.register_forward_pre_hook(capture_attention_context))
        def capture(_module, _inputs, output, index=index):
            reference_blocks[index] = output.detach().float().cpu().numpy()
        hooks.append(block.register_forward_hook(capture))
        def capture_attention(_module, _inputs, output, index=index, block=block):
            # Native dumps are after the attention projection and LayerScale,
            # immediately before the first residual addition in Block.forward.
            reference_attention_projection[index] = output.detach().float().cpu().numpy()
            reference_attention[index] = block.ls1(output).detach().float().cpu().numpy()
        hooks.append(block.attn.register_forward_hook(capture_attention))
        def capture_mlp_fc1(_module, _inputs, output, index=index):
            reference_mlp_fc1[index] = output.detach().float().cpu().numpy()
        hooks.append(block.mlp.fc1.register_forward_hook(capture_mlp_fc1))
        def capture_mlp_gelu(_module, _inputs, output, index=index):
            reference_mlp_gelu[index] = output.detach().float().cpu().numpy()
        hooks.append(block.mlp.act.register_forward_hook(capture_mlp_gelu))
        def capture_mlp(_module, _inputs, output, index=index, block=block):
            # Native dumps are after the MLP and LayerScale, immediately
            # before the second residual addition in Block.forward.
            reference_mlp[index] = block.ls2(output).detach().float().cpu().numpy()
        hooks.append(block.mlp.register_forward_hook(capture_mlp))

    with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=args.reference_precision == "cuda-fp16"):
        if args.image:
            image = moge_input_for_tokens(
                load_image(args.image).to(device), args.num_tokens,
                model.image_mean, model.image_std,
            )
        else:
            image = deterministic_image(args.width, args.height).to(device)
            image = (image - model.image_mean) / model.image_std
        reference_backbone_image = image.permute(0, 2, 3, 1).detach().float().cpu().numpy()
        patch_tokens = model.backbone.patch_embed(image)
        tokens_without_position = torch.cat(
            (model.backbone.cls_token.expand(patch_tokens.shape[0], -1, -1), patch_tokens), dim=1
        )
        position_tokens = model.backbone.interpolate_pos_encoding(
            tokens_without_position, image.shape[-2], image.shape[-1]
        )
        reference_patch_tokens = patch_tokens.detach().float().cpu().numpy()
        reference_position_tokens = position_tokens.detach().float().cpu().numpy()
        model.backbone.get_intermediate_layers(image, model.intermediate_layers,
                                               return_class_token=True)
    for hook in hooks:
        hook.remove()

    with tempfile.TemporaryDirectory(prefix="sam3d-moge-block-parity-") as directory:
        prefix = Path(directory) / "native"
        environment = dict(os.environ)
        if args.image:
            command = [
                str(args.binary), "moge-infer", "--model", str(args.model), "--backend", args.backend,
                "--input", str(args.image), "--num-tokens", str(args.num_tokens),
                "--dump-blocks", "--out", str(prefix),
            ]
        else:
            environment["SAM3D_MOGE_DUMP_BLOCKS"] = "1"
            command = [
                str(args.binary), "moge-smoke", "--model", str(args.model), "--backend", args.backend,
                "--width", str(args.width), "--height", str(args.height), "--out", str(prefix),
            ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True, env=environment)
        if (reference_backbone_image is None or reference_patch_tokens is None or
                reference_position_tokens is None or
                reference_backbone_input is None or reference_attention_context is None or
                reference_qkv0 is None):
            raise RuntimeError("official hooks did not capture DINO input and block 0 QKV")
        native_backbone_image = read_samt_values(prefix.with_suffix(".backbone_image.samt"))
        native_patch_tokens = read_samt_values(prefix.with_suffix(".backbone_patch_tokens.samt"))
        native_position_tokens = read_samt_values(prefix.with_suffix(".backbone_position_tokens.samt"))
        native_backbone_input = read_samt_values(prefix.with_suffix(".backbone_input.samt"))
        native_attention_context = read_samt_values(prefix.with_suffix(".attention_context.samt"))
        if native_backbone_image.size != reference_backbone_image.size:
            raise RuntimeError("native/reference DINO image shape mismatch")
        if native_patch_tokens.size != reference_patch_tokens.size:
            raise RuntimeError("native/reference DINO patch-token shape mismatch")
        if native_position_tokens.size != reference_position_tokens.size:
            raise RuntimeError("native/reference DINO position-token shape mismatch")
        if native_backbone_input.size != reference_backbone_input.size:
            raise RuntimeError("native/reference DINO input shape mismatch")
        if native_attention_context.size != reference_attention_context.size:
            raise RuntimeError("native/reference attention-context shape mismatch")
        backbone_image_metrics = metrics(
            native_backbone_image.reshape(reference_backbone_image.shape), reference_backbone_image
        )
        backbone_patch_metrics = metrics(
            native_patch_tokens.reshape(reference_patch_tokens.shape), reference_patch_tokens
        )
        backbone_position_metrics = metrics(
            native_position_tokens.reshape(reference_position_tokens.shape), reference_position_tokens
        )
        backbone_input_metrics = metrics(
            native_backbone_input.reshape(reference_backbone_input.shape), reference_backbone_input
        )
        attention_context_metrics = metrics(
            native_attention_context.reshape(reference_attention_context.shape), reference_attention_context
        )
        batch, tokens, channels3 = reference_qkv0.shape
        heads = model.backbone.blocks[0].attn.num_heads
        head_dim = channels3 // 3 // heads
        qkv0 = reference_qkv0.reshape(batch, tokens, 3, heads, head_dim).transpose(2, 0, 3, 1, 4)
        qkv_metrics = {}
        for name, reference in zip(("q", "k", "v"), qkv0):
            native = read_samt_values(prefix.with_suffix(f".{name}.samt"))
            if native.size != reference.size:
                raise RuntimeError(f"block 0 {name}: native/reference shape mismatch")
            qkv_metrics[name] = metrics(native.reshape(reference.shape), reference)
        rows = []
        for index, reference in enumerate(reference_blocks):
            if reference is None:
                raise RuntimeError(f"official hook did not capture block {index}")
            native = read_samt_values(prefix.with_suffix(f".block{index}.samt"))
            if native.size != reference.size:
                raise RuntimeError(
                    f"block {index}: native has {native.size} values; expected {reference.size}"
                )
            attention_reference = reference_attention[index]
            attention_projection_reference = reference_attention_projection[index]
            if attention_projection_reference is None:
                raise RuntimeError(f"official hook did not capture attention projection {index}")
            attention_projection = read_samt_values(prefix.with_suffix(f".attention_projection{index}.samt"))
            if attention_projection.size != attention_projection_reference.size:
                raise RuntimeError(
                    f"attention projection {index}: native has {attention_projection.size} values; "
                    f"expected {attention_projection_reference.size}"
                )
            if attention_reference is None:
                raise RuntimeError(f"official hook did not capture attention {index}")
            attention = read_samt_values(prefix.with_suffix(f".attention{index}.samt"))
            if attention.size != attention_reference.size:
                raise RuntimeError(
                    f"attention {index}: native has {attention.size} values; "
                    f"expected {attention_reference.size}"
                )
            fc1_reference = reference_mlp_fc1[index]
            gelu_reference = reference_mlp_gelu[index]
            if fc1_reference is None or gelu_reference is None:
                raise RuntimeError(f"official hook did not capture MLP stages {index}")
            fc1 = read_samt_values(prefix.with_suffix(f".mlp_fc1{index}.samt"))
            gelu = read_samt_values(prefix.with_suffix(f".mlp_gelu{index}.samt"))
            if fc1.size != fc1_reference.size or gelu.size != gelu_reference.size:
                raise RuntimeError(f"MLP stage {index}: native/reference shape mismatch")
            mlp_reference = reference_mlp[index]
            if mlp_reference is None:
                raise RuntimeError(f"official hook did not capture MLP {index}")
            mlp = read_samt_values(prefix.with_suffix(f".mlp{index}.samt"))
            if mlp.size != mlp_reference.size:
                raise RuntimeError(
                    f"MLP {index}: native has {mlp.size} values; expected {mlp_reference.size}"
                )
            rows.append({
                "block": index,
                "attention_projection": metrics(
                    attention_projection.reshape(attention_projection_reference.shape),
                    attention_projection_reference,
                ),
                "attention": metrics(attention.reshape(attention_reference.shape), attention_reference),
                "mlp_fc1": metrics(fc1.reshape(fc1_reference.shape), fc1_reference),
                "mlp_gelu": metrics(gelu.reshape(gelu_reference.shape), gelu_reference),
                "mlp": metrics(mlp.reshape(mlp_reference.shape), mlp_reference),
                "block_output": metrics(native.reshape(reference.shape), reference),
            })

    payload = {
        "backend": args.backend,
        "reference_device": args.reference_device,
        "reference_precision": args.reference_precision,
        "reference_tf32": args.reference_tf32,
        "reference_storage": args.reference_storage,
        "input": str(args.image) if args.image else "synthetic",
        "num_tokens": args.num_tokens if args.image else None,
        "backbone_image": backbone_image_metrics,
        "backbone_patch_tokens": backbone_patch_metrics,
        "backbone_position_tokens": backbone_position_metrics,
        "backbone_input": backbone_input_metrics,
        "block0_attention_context": attention_context_metrics,
        "block0_qkv": qkv_metrics,
        "blocks": rows,
    }
    print(json.dumps(payload, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
