#!/usr/bin/env python3
"""Separate sparse-attention semantics from mesh-decoder projection error.

This is a reference-side regression tool.  It feeds a native block's dumped
QKV tensor and sparse coordinates through the released PyTorch windowed SDPA
implementation, then compares that result with the native attention output.
It therefore answers a narrower question than the full mesh decoder fixture:
given identical QKV values and identical sparse windows, did the native
attention calculation preserve the official semantics?

The C++ runtime never imports this script, Python, or Torch.  The tool expects
the official environment only when a developer explicitly runs the parity
check.  The fixture is produced by ``dump_mesh_decoder_reference.py`` with
``--internals-block`` and the native tensors by ``sam3d-cli mesh-decode``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# These settings must be visible before importing the sparse-attention module.
# The fixture was captured with the released SDPA path; selecting it explicitly
# prevents an installed xFormers/FlashAttention package from changing a result.
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from verify_mesh_decoder_reference import (  # noqa: E402
    exact_coordinate_compare,
    feature_metrics,
    samt_memmap,
)


HEADS = 12
HEAD_DIM = 64
CHANNELS = HEADS * HEAD_DIM


def f32_features(path: Path, channels: int, label: str) -> np.memmap:
    values = samt_memmap(path)
    if values.dtype != np.dtype("<f4") or values.ndim != 2 or values.shape[1] != channels:
        raise ValueError(
            f"{label} must be F32 [tokens, {channels}], found {values.dtype} {values.shape}: {path}"
        )
    return values


def i32_coordinates(path: Path, token_count: int, label: str) -> np.memmap:
    values = samt_memmap(path)
    if values.dtype != np.dtype("<i4") or values.shape != (token_count, 4):
        raise ValueError(
            f"{label} must be I32 [{token_count}, 4], found {values.dtype} {values.shape}: {path}"
        )
    return values


def tensor_metrics(actual: torch.Tensor, expected: np.ndarray, label: str) -> dict[str, object]:
    """Return metrics without copying the full expected tensor through Python."""
    expected_tensor = torch.from_numpy(
        np.array(expected, dtype=np.float32, copy=True, order="C")
    ).to(actual.device)
    if tuple(actual.shape) != tuple(expected_tensor.shape):
        raise ValueError(f"{label}: shape mismatch {tuple(actual.shape)} != {tuple(expected_tensor.shape)}")
    delta = actual.float() - expected_tensor.float()
    if not bool(torch.isfinite(delta).all()):
        raise ValueError(f"{label}: non-finite delta")
    absolute = delta.abs()
    flat_index = int(absolute.reshape(-1).argmax().item())
    row, channel = divmod(flat_index, actual.shape[1])
    result = {
        "shape": list(actual.shape),
        "mae": float(absolute.mean().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "max_abs": float(absolute.reshape(-1)[flat_index].item()),
        "worst_location": {"token": row, "channel": channel},
        "reference": float(expected_tensor[row, channel].item()),
        "native": float(actual[row, channel].item()),
    }
    del expected_tensor, delta, absolute
    return result


def require_exact_coordinates(reference: Path, native: Path, label: str) -> dict[str, object]:
    report = exact_coordinate_compare(reference, native)
    if not report["exact"]:
        raise ValueError(f"{label}: sparse coordinates differ: {report['first_mismatch']}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="official mesh-decoder fixture directory")
    parser.add_argument("--native-qkv", type=Path, required=True,
                        help="native block QKV SAMT tensor in original sparse order")
    parser.add_argument("--native-qkv-coords", type=Path, required=True)
    parser.add_argument("--native-attention", type=Path, required=True,
                        help="native block attention-values SAMT tensor")
    parser.add_argument("--native-attention-coords", type=Path, required=True)
    parser.add_argument("--block", type=int, default=0, choices=range(12))
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--shift", type=int, nargs=3, default=(0, 0, 0),
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-values", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--max-semantic-mae", type=float,
                        help="fail only when same-QKV attention MAE exceeds this explicit threshold")
    parser.add_argument("--max-semantic-abs", type=float,
                        help="fail only when same-QKV attention max abs exceeds this explicit threshold")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.window_size <= 0 or args.chunk_values <= 0:
        parser.error("--window-size and --chunk-values must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("--device cuda was selected but Torch cannot use CUDA")

    reference_dir = args.reference_dir.resolve()
    prefix = f"mesh_decoder_block{args.block}"
    reference_qkv = reference_dir / f"{prefix}_qkv_features.samt"
    reference_qkv_coords = reference_dir / f"{prefix}_qkv_coords.samt"
    reference_attention = reference_dir / f"{prefix}_attention_values_features.samt"
    reference_attention_coords = reference_dir / f"{prefix}_attention_values_coords.samt"
    for path in (reference_qkv, reference_qkv_coords, reference_attention, reference_attention_coords,
                 args.native_qkv, args.native_qkv_coords, args.native_attention,
                 args.native_attention_coords):
        if not path.is_file():
            raise FileNotFoundError(path)

    native_qkv = f32_features(args.native_qkv, CHANNELS * 3, "native QKV")
    native_attention = f32_features(args.native_attention, CHANNELS, "native attention")
    reference_attention_values = f32_features(reference_attention, CHANNELS, "official attention")
    if native_attention.shape[0] != native_qkv.shape[0]:
        raise ValueError("native QKV and attention token counts differ")
    coordinates = i32_coordinates(args.native_qkv_coords, native_qkv.shape[0], "native QKV coords")
    i32_coordinates(args.native_attention_coords, native_attention.shape[0], "native attention coords")

    coordinate_report = {
        "qkv": require_exact_coordinates(reference_qkv_coords, args.native_qkv_coords, "QKV"),
        "attention": require_exact_coordinates(
            reference_attention_coords, args.native_attention_coords, "attention"
        ),
    }
    projection_metrics = feature_metrics(reference_qkv, args.native_qkv, args.chunk_values)

    # Import after the reference files are validated so an incompatible fixture
    # fails before allocating the large block-diagonal SDPA mask on the GPU.
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse import SparseTensor
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.attention.windowed_attn import (
        calc_window_partition,
        sparse_windowed_scaled_dot_product_self_attention,
    )

    # Tensor files are read-only memory maps.  Materialize writable, contiguous
    # host buffers before handing them to PyTorch so the regression is warning-free.
    qkv_tensor = torch.from_numpy(
        np.array(native_qkv, dtype=np.float32, copy=True, order="C")
    ).to(args.device, dtype=torch.float16)
    coords_tensor = torch.from_numpy(
        np.array(coordinates, dtype=np.int32, copy=True, order="C")
    ).to(args.device)
    sparse_qkv = SparseTensor(
        feats=qkv_tensor.reshape(qkv_tensor.shape[0], 3, HEADS, HEAD_DIM),
        coords=coords_tensor,
    )
    forward_indices, _, sequence_lengths, _ = calc_window_partition(
        sparse_qkv, args.window_size, tuple(args.shift)
    )
    if int(forward_indices.numel()) != native_qkv.shape[0] or sum(sequence_lengths) != native_qkv.shape[0]:
        raise RuntimeError("official window partition did not cover every sparse token exactly once")
    with torch.inference_mode():
        official_attention = sparse_windowed_scaled_dot_product_self_attention(
            sparse_qkv, args.window_size, tuple(args.shift)
        ).feats.reshape(native_qkv.shape[0], CHANNELS)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(args.device))

        semantic_metrics = tensor_metrics(official_attention, np.asarray(native_attention),
                                          "native attention vs official same-QKV SDPA")
        upstream_and_attention_metrics = tensor_metrics(
            official_attention, np.asarray(reference_attention_values),
            "official same-QKV SDPA vs fixture"
        )

    native_to_fixture_metrics = feature_metrics(reference_attention, args.native_attention, args.chunk_values)
    report = {
        "schema": "sam3d.mesh-attention-semantics.v1",
        "reference_dir": str(reference_dir),
        "native": {
            "qkv": str(args.native_qkv.resolve()),
            "attention": str(args.native_attention.resolve()),
        },
        "official_contract": {
            "attention_backend": "sdpa",
            "block": args.block,
            "window_size": args.window_size,
            "shift": list(args.shift),
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "window_count": len(sequence_lengths),
            "min_window_tokens": min(sequence_lengths),
            "max_window_tokens": max(sequence_lengths),
            "unique_window_lengths": len(set(sequence_lengths)),
        },
        "coordinates": coordinate_report,
        "projection_to_fixture": projection_metrics,
        "native_attention_to_official_same_qkv": semantic_metrics,
        "official_same_qkv_to_fixture": upstream_and_attention_metrics,
        "native_attention_to_fixture": native_to_fixture_metrics,
    }
    if args.max_semantic_mae is not None and semantic_metrics["mae"] > args.max_semantic_mae:
        raise ValueError(
            f"same-QKV attention MAE {semantic_metrics['mae']:.9g} exceeds "
            f"{args.max_semantic_mae:.9g}"
        )
    if args.max_semantic_abs is not None and semantic_metrics["max_abs"] > args.max_semantic_abs:
        raise ValueError(
            f"same-QKV attention max abs {semantic_metrics['max_abs']:.9g} exceeds "
            f"{args.max_semantic_abs:.9g}"
        )

    text = json.dumps(report, indent=2)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
