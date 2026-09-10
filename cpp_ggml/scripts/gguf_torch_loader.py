#!/usr/bin/env python3
"""Load converted generator backbone weights into an official Torch state dict.

This is deliberately a reference-side utility.  It uses the same ``gguf``
reader/dequantizer used by the converter toolchain so an operator-boundary
comparison can distinguish a native graph error from the expected difference
between an official checkpoint and a quantized GGUF model.

Only tensors under ``generator.reverse_fn.backbone`` are replaced.  The
SS/SLat step references consume already-fused condition tensors, so their
remaining generator dependencies are exactly this namespace.  Every such
checkpoint tensor must map to one GGUF tensor and round-trip to its original
Torch shape; partial replacement is rejected.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize

from gguf_schema import rewrite_key


_BACKBONE_PREFIX = "reverse_fn.backbone."
_CHECKPOINT_GENERATOR_PREFIX = "_base_models.generator."
_RESIDUAL_SUFFIXES = (".lora_a", ".lora_b", ".ra", ".rb")


def _is_sparse_convolution(gname: str, shape: tuple[int, ...]) -> bool:
    return (
        len(shape) == 5
        and (gname.startswith("dit.input_blocks.") or gname.startswith("dit.out_blocks."))
        and (gname.endswith(".conv1.weight") or gname.endswith(".conv2.weight")
             or gname.endswith(".conv1.conv.weight") or gname.endswith(".conv2.conv.weight"))
    )


def _restore_checkpoint_layout(gname: str, values: np.ndarray,
                               expected_shape: tuple[int, ...]) -> tuple[np.ndarray, str]:
    """Undo the two intentional converter layouts, then require exact shape."""
    if tuple(values.shape) == expected_shape:
        return values, "identity"

    if gname.endswith(".pos_embed") and len(expected_shape) == 4:
        _, channels, height, width = expected_shape
        storage_shape = (height * width, channels)
        if tuple(values.shape) == storage_shape:
            restored = values.reshape(height, width, channels).transpose(2, 0, 1)[None, ...]
            return np.ascontiguousarray(restored), "pointpatch_pos_embed_inverse"

    if _is_sparse_convolution(gname, expected_shape):
        flattened_shape = (expected_shape[0], int(np.prod(expected_shape[1:], dtype=np.int64)))
        if tuple(values.shape) == flattened_shape:
            return np.ascontiguousarray(values.reshape(expected_shape)), "sparse_conv_inverse"

    raise ValueError(
        f"{gname}: GGUF dequantized shape {tuple(values.shape)} cannot restore "
        f"checkpoint shape {expected_shape}")


def replace_backbone_from_gguf(generator_state: MutableMapping[str, torch.Tensor],
                               gguf_path: str | Path) -> dict[str, Any]:
    """Replace every generator-backbone tensor in-place with GGUF values.

    Required-name and residual validation complete before mutation.  Values
    are then replaced one tensor at a time so this diagnostic does not retain
    a second full generator state in host memory. Optional Q4 low-rank factors
    are currently rejected because the native graph evaluates ``Wq@x +
    B@(A@x)`` as two GEMMs, whereas replacing a Torch linear weight with ``Wq
    + B@A`` changes the accumulation topology. A future Torch wrapper must
    model that topology before residual-bearing GGUF files can use this oracle.
    """
    path = Path(gguf_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GGUF model does not exist: {path}")

    reader = GGUFReader(str(path))
    tensor_map = {tensor.name: tensor for tensor in reader.tensors}
    residuals = sorted(name for name in tensor_map if name.endswith(_RESIDUAL_SUFFIXES))
    if residuals:
        raise ValueError(
            "same-GGUF Torch oracle does not yet support low-rank residual topology; "
            f"found {len(residuals)} factor tensor(s), first: {residuals[0]}")

    required: list[tuple[str, str, torch.Tensor]] = []
    for state_name, value in generator_state.items():
        if not state_name.startswith(_BACKBONE_PREFIX):
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{state_name}: backbone state is not a tensor")
        gname = rewrite_key(_CHECKPOINT_GENERATOR_PREFIX + state_name)
        if gname is None:
            raise ValueError(f"{state_name}: no GGUF name mapping")
        required.append((state_name, gname, value))
    if not required:
        raise ValueError("generator state has no reverse_fn.backbone tensors")

    missing = [gname for _, gname, _ in required if gname not in tensor_map]
    if missing:
        preview = ", ".join(sorted(missing)[:3])
        raise ValueError(f"GGUF is missing {len(missing)} backbone tensor(s): {preview}")

    layout_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    for state_name, gname, original in required:
        source = tensor_map[gname]
        values = np.ascontiguousarray(dequantize(source.data, source.tensor_type), dtype=np.float32)
        restored, layout = _restore_checkpoint_layout(gname, values, tuple(original.shape))
        generator_state[state_name] = torch.from_numpy(restored.copy())
        layout_counts[layout] += 1
        type_counts[str(source.tensor_type)] += 1

    return {
        "schema": "sam3d.gguf_torch_backbone_loader.v1",
        "weight_scope": "same-gguf-dequantized",
        "gguf": str(path),
        "backbone_tensors_loaded": len(required),
        "layout_transforms": dict(sorted(layout_counts.items())),
        "gguf_tensor_types": dict(sorted(type_counts.items())),
        "low_rank_residuals": "absent",
    }


def _load_generator_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise ValueError(f"unexpected checkpoint format: {checkpoint}")
    result = {
        key[len(_CHECKPOINT_GENERATOR_PREFIX):]: value
        for key, value in loaded.items()
        if key.startswith(_CHECKPOINT_GENERATOR_PREFIX) and isinstance(value, torch.Tensor)
    }
    if not result:
        raise ValueError(f"checkpoint has no generator tensors: {checkpoint}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="official generator checkpoint used for the architecture/state namespace")
    parser.add_argument("--gguf", type=Path, required=True,
                        help="converted generator GGUF to dequantize")
    parser.add_argument("--report", type=Path, required=True,
                        help="new JSON report; no model is modified on disk")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"missing checkpoint: {args.checkpoint}")
    if not args.gguf.is_file():
        parser.error(f"missing GGUF: {args.gguf}")
    if args.report.exists():
        parser.error(f"refusing to overwrite report: {args.report}")

    state = _load_generator_state(args.checkpoint)
    report = replace_backbone_from_gguf(state, args.gguf)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"loaded {report['backbone_tensors_loaded']} backbone tensors from {args.gguf}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
