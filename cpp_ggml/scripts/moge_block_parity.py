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
import torch

from moge.model.v1 import MoGeModel


def read_samt(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        if stream.read(4) != b"SAMT":
            raise ValueError(f"{path}: invalid SAMT magic")
        (rank,) = struct.unpack("<i", stream.read(4))
        shape = struct.unpack(f"<{rank}q", stream.read(8 * rank))
        (dtype,) = struct.unpack("<i", stream.read(4))
        if dtype != 0:
            raise ValueError(f"{path}: expected F32 SAMT")
        values = np.frombuffer(stream.read(), dtype="<f4").copy()
    if values.size != int(np.prod(shape)):
        raise ValueError(f"{path}: truncated tensor")
    return values


def deterministic_image(width: int, height: int) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, width, dtype=torch.float32).view(1, 1, 1, width)
    y = torch.linspace(0.0, 1.0, height, dtype=torch.float32).view(1, 1, height, 1)
    x = x.expand(1, 1, height, width)
    y = y.expand(1, 1, height, width)
    return torch.cat((x, y, torch.full_like(x, 0.25)), dim=1)


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
    parser.add_argument("--reference-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.reference_device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA PyTorch reference requested but unavailable")

    device = torch.device(args.reference_device)
    model = MoGeModel.from_pretrained(args.checkpoint).eval()
    with torch.no_grad():
        # Match F16 GGUF storage for rank-two-or-greater parameters.
        for parameter in model.parameters():
            if parameter.ndim > 1:
                parameter.copy_(parameter.to(torch.float16).to(torch.float32))
    model = model.to(device)
    reference_blocks: list[np.ndarray | None] = [None] * len(model.backbone.blocks)
    reference_attention: list[np.ndarray | None] = [None] * len(model.backbone.blocks)
    reference_mlp_fc1: list[np.ndarray | None] = [None] * len(model.backbone.blocks)
    reference_mlp_gelu: list[np.ndarray | None] = [None] * len(model.backbone.blocks)
    reference_mlp: list[np.ndarray | None] = [None] * len(model.backbone.blocks)
    reference_qkv0: np.ndarray | None = None
    hooks = []
    for index, block in enumerate(model.backbone.blocks):
        if index == 0:
            def capture_qkv(_module, _inputs, output):
                nonlocal reference_qkv0
                reference_qkv0 = output.detach().float().cpu().numpy()
            hooks.append(block.attn.qkv.register_forward_hook(capture_qkv))
        def capture(_module, _inputs, output, index=index):
            reference_blocks[index] = output.detach().float().cpu().numpy()
        hooks.append(block.register_forward_hook(capture))
        def capture_attention(_module, _inputs, output, index=index, block=block):
            # Native dumps are after the attention projection and LayerScale,
            # immediately before the first residual addition in Block.forward.
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

    with torch.no_grad():
        image = deterministic_image(args.width, args.height).to(device)
        image = (image - model.image_mean) / model.image_std
        model.backbone.get_intermediate_layers(image, model.intermediate_layers,
                                               return_class_token=True)
    for hook in hooks:
        hook.remove()

    with tempfile.TemporaryDirectory(prefix="sam3d-moge-block-parity-") as directory:
        prefix = Path(directory) / "native"
        environment = dict(os.environ)
        environment["SAM3D_MOGE_DUMP_BLOCKS"] = "1"
        command = [
            str(args.binary), "moge-smoke", "--model", str(args.model), "--backend", args.backend,
            "--width", str(args.width), "--height", str(args.height), "--out", str(prefix),
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True, env=environment)
        if reference_qkv0 is None:
            raise RuntimeError("official hook did not capture block 0 QKV")
        batch, tokens, channels3 = reference_qkv0.shape
        heads = model.backbone.blocks[0].attn.num_heads
        head_dim = channels3 // 3 // heads
        qkv0 = reference_qkv0.reshape(batch, tokens, 3, heads, head_dim).transpose(2, 0, 3, 1, 4)
        qkv_metrics = {}
        for name, reference in zip(("q", "k", "v"), qkv0):
            native = read_samt(prefix.with_suffix(f".{name}.samt"))
            if native.size != reference.size:
                raise RuntimeError(f"block 0 {name}: native/reference shape mismatch")
            qkv_metrics[name] = metrics(native.reshape(reference.shape), reference)
        rows = []
        for index, reference in enumerate(reference_blocks):
            if reference is None:
                raise RuntimeError(f"official hook did not capture block {index}")
            native = read_samt(prefix.with_suffix(f".block{index}.samt"))
            if native.size != reference.size:
                raise RuntimeError(
                    f"block {index}: native has {native.size} values; expected {reference.size}"
                )
            attention_reference = reference_attention[index]
            if attention_reference is None:
                raise RuntimeError(f"official hook did not capture attention {index}")
            attention = read_samt(prefix.with_suffix(f".attention{index}.samt"))
            if attention.size != attention_reference.size:
                raise RuntimeError(
                    f"attention {index}: native has {attention.size} values; "
                    f"expected {attention_reference.size}"
                )
            fc1_reference = reference_mlp_fc1[index]
            gelu_reference = reference_mlp_gelu[index]
            if fc1_reference is None or gelu_reference is None:
                raise RuntimeError(f"official hook did not capture MLP stages {index}")
            fc1 = read_samt(prefix.with_suffix(f".mlp_fc1{index}.samt"))
            gelu = read_samt(prefix.with_suffix(f".mlp_gelu{index}.samt"))
            if fc1.size != fc1_reference.size or gelu.size != gelu_reference.size:
                raise RuntimeError(f"MLP stage {index}: native/reference shape mismatch")
            mlp_reference = reference_mlp[index]
            if mlp_reference is None:
                raise RuntimeError(f"official hook did not capture MLP {index}")
            mlp = read_samt(prefix.with_suffix(f".mlp{index}.samt"))
            if mlp.size != mlp_reference.size:
                raise RuntimeError(
                    f"MLP {index}: native has {mlp.size} values; expected {mlp_reference.size}"
                )
            rows.append({
                "block": index,
                "attention": metrics(attention.reshape(attention_reference.shape), attention_reference),
                "mlp_fc1": metrics(fc1.reshape(fc1_reference.shape), fc1_reference),
                "mlp_gelu": metrics(gelu.reshape(gelu_reference.shape), gelu_reference),
                "mlp": metrics(mlp.reshape(mlp_reference.shape), mlp_reference),
                "block_output": metrics(native.reshape(reference.shape), reference),
            })

    payload = {
        "backend": args.backend,
        "reference_device": args.reference_device,
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
