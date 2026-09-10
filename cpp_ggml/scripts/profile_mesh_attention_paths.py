#!/usr/bin/env python3
"""Measure numerical effects of the released sparse SDPA execution choices.

This is a reference-side diagnostic.  It never participates in the native
runtime and is deliberately based on the released PyTorch sparse-attention
module.  The report distinguishes the official global masked invocation from
mathematically equivalent equal-length window batches, so a CUDA implementation
is only changed after the numerical source of an error is measured.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("LIDRA_SKIP_INIT", "true")
os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"

import torch
import torch.nn.functional as functional


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from verify_mesh_decoder_reference import feature_metrics, samt_memmap  # noqa: E402


HEADS = 12
HEAD_DIM = 64
CHANNELS = HEADS * HEAD_DIM


def load_features(path: Path, channels: int, name: str) -> np.memmap:
    values = samt_memmap(path)
    if values.dtype != np.dtype("<f4") or values.ndim != 2 or values.shape[1] != channels:
        raise ValueError(f"{name} must be F32 [tokens, {channels}], found {values.dtype} {values.shape}")
    return values


def load_coordinates(path: Path, tokens: int) -> np.memmap:
    values = samt_memmap(path)
    if values.dtype != np.dtype("<i4") or values.shape != (tokens, 4):
        raise ValueError(f"coordinates must be I32 [{tokens}, 4], found {values.dtype} {values.shape}")
    return values


def metric(actual: torch.Tensor, reference: np.ndarray) -> dict[str, object]:
    expected = torch.from_numpy(np.array(reference, dtype=np.float32, copy=True, order="C")).to(actual.device)
    difference = actual.float() - expected
    absolute = difference.abs()
    flat = int(absolute.reshape(-1).argmax().item())
    token, channel = divmod(flat, actual.shape[1])
    result = {
        "shape": list(actual.shape),
        "mae": float(absolute.mean().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "max_abs": float(absolute.reshape(-1)[flat].item()),
        "worst_location": {"token": token, "channel": channel},
        "reference": float(expected[token, channel].item()),
        "actual": float(actual[token, channel].item()),
    }
    del expected, difference, absolute
    return result


def grouped_attention(qkv_sorted: torch.Tensor, sequence_lengths: list[int], mode: str) -> torch.Tensor:
    """Run independent windows in batches grouped by their exact sequence length."""
    if mode not in {
        "sdpa_f16",
        "sdpa_f32",
        "manual_f32",
        "manual_f16",
        "manual_score_f16",
        "manual_weights_f16",
    }:
        raise ValueError(f"unknown grouped attention mode: {mode}")

    spans_by_length: dict[int, list[int]] = defaultdict(list)
    offset = 0
    for length in sequence_lengths:
        spans_by_length[length].append(offset)
        offset += length
    if offset != qkv_sorted.shape[0]:
        raise ValueError("window lengths do not cover sorted QKV")

    output = torch.empty(
        (qkv_sorted.shape[0], HEADS, HEAD_DIM), dtype=torch.float16, device=qkv_sorted.device
    )
    scale = HEAD_DIM ** -0.5
    for length, starts in spans_by_length.items():
        packed = torch.stack([qkv_sorted[start : start + length] for start in starts])
        query, key, value = packed.unbind(dim=2)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        if mode == "sdpa_f16":
            result = functional.scaled_dot_product_attention(query, key, value)
        elif mode == "sdpa_f32":
            result = functional.scaled_dot_product_attention(
                query.float(), key.float(), value.float()
            ).to(torch.float16)
        elif mode == "manual_f32":
            scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
            weights = torch.softmax(scores, dim=-1)
            result = torch.matmul(weights, value.float()).to(torch.float16)
        elif mode == "manual_f16":
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale
            weights = torch.softmax(scores, dim=-1)
            result = torch.matmul(weights, value)
        elif mode == "manual_score_f16":
            scores = (torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale).to(
                torch.float16
            )
            weights = torch.softmax(scores, dim=-1)
            result = torch.matmul(weights, value)
        else:
            scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
            weights = torch.softmax(scores, dim=-1).to(torch.float16)
            result = torch.matmul(weights, value)
        result = result.permute(0, 2, 1, 3)
        for index, start in enumerate(starts):
            output[start : start + length] = result[index]
    return output


def xformers_block_diagonal_attention(
    qkv_sorted: torch.Tensor, sequence_lengths: list[int]
) -> torch.Tensor:
    """Run the public CUTLASS block-diagonal F16 reference, when installed."""
    import xformers.ops as xops

    query, key, value = qkv_sorted.unbind(dim=1)
    mask = xops.fmha.BlockDiagonalMask.from_seqlens(sequence_lengths)
    return xops.memory_efficient_attention(
        query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0), attn_bias=mask
    )[0]


def pitched_mask_attention(
    qkv_sorted: torch.Tensor, sequence_lengths: list[int]
) -> torch.Tensor:
    """Run the official SDPA call with an explicitly CUTLASS-aligned bias pitch.

    The released implementation constructs a logical ``[tokens, tokens]``
    additive mask.  CUDA's memory-efficient path may materialize that logical
    mask with an aligned row stride before launching CUTLASS.  This diagnostic
    keeps the visible shape identical while making that physical pitch explicit,
    so it can distinguish layout conversion from attention arithmetic.
    """
    token_count = qkv_sorted.shape[0]
    padded_key_count = (token_count + 7) // 8 * 8
    q, k, v = qkv_sorted.unbind(dim=1)
    q = q.unsqueeze(0).permute(0, 2, 1, 3)
    k = k.unsqueeze(0).permute(0, 2, 1, 3)
    v = v.unsqueeze(0).permute(0, 2, 1, 3)
    mask_storage = torch.full(
        (token_count, padded_key_count),
        float("-inf"),
        dtype=q.dtype,
        device=q.device,
    )
    start = 0
    for length in sequence_lengths:
        mask_storage[start : start + length, start : start + length] = 0
        start += length
    if start != token_count:
        raise ValueError("sequence lengths do not cover the attention mask")
    mask = mask_storage[:, :token_count].unsqueeze(0).unsqueeze(0)
    result = functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return result.permute(0, 2, 1, 3)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--native-qkv", type=Path, required=True)
    parser.add_argument("--native-qkv-coords", type=Path, required=True)
    parser.add_argument("--native-attention", type=Path)
    parser.add_argument("--block", type=int, default=0, choices=range(12))
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--shift", type=int, nargs=3, default=(0, 0, 0), metavar=("X", "Y", "Z"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("--device cuda was selected but Torch cannot use CUDA")
    if args.window_size <= 0:
        parser.error("--window-size must be positive")

    prefix = f"mesh_decoder_block{args.block}"
    reference_dir = args.reference_dir.resolve()
    reference_attention = reference_dir / f"{prefix}_attention_values_features.samt"
    for path in (reference_attention, args.native_qkv, args.native_qkv_coords):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.native_attention is not None and not args.native_attention.is_file():
        raise FileNotFoundError(args.native_attention)

    native_qkv = load_features(args.native_qkv, CHANNELS * 3, "native QKV")
    coordinates = load_coordinates(args.native_qkv_coords, native_qkv.shape[0])
    reference_values = load_features(reference_attention, CHANNELS, "reference attention")
    qkv = torch.from_numpy(np.array(native_qkv, dtype=np.float32, copy=True, order="C")).to(
        args.device, dtype=torch.float16
    ).reshape(native_qkv.shape[0], 3, HEADS, HEAD_DIM)
    coords = torch.from_numpy(np.array(coordinates, dtype=np.int32, copy=True, order="C")).to(args.device)

    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse import SparseTensor
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.attention.windowed_attn import (
        calc_window_partition,
        sparse_windowed_scaled_dot_product_self_attention,
    )

    sparse_qkv = SparseTensor(feats=qkv, coords=coords)
    forward_indices, backward_indices, sequence_lengths, _ = calc_window_partition(
        sparse_qkv, args.window_size, tuple(args.shift)
    )
    if sum(sequence_lengths) != native_qkv.shape[0]:
        raise RuntimeError("official window partition did not cover all tokens")

    with torch.inference_mode():
        official = sparse_windowed_scaled_dot_product_self_attention(
            sparse_qkv, args.window_size, tuple(args.shift)
        ).feats.reshape(native_qkv.shape[0], CHANNELS)
        grouped_input = qkv[forward_indices]
        grouped_reports = {}
        for mode in (
            "sdpa_f16",
            "sdpa_f32",
            "manual_f32",
            "manual_f16",
            "manual_score_f16",
            "manual_weights_f16",
        ):
            candidate = grouped_attention(grouped_input, sequence_lengths, mode)[backward_indices]
            grouped_reports[mode] = metric(candidate.reshape(native_qkv.shape[0], CHANNELS), official.cpu().numpy())
            del candidate
        try:
            xformers_candidate = xformers_block_diagonal_attention(
                grouped_input, sequence_lengths
            )[backward_indices]
            grouped_reports["xformers_block_diagonal_f16"] = metric(
                xformers_candidate.reshape(native_qkv.shape[0], CHANNELS), official.cpu().numpy()
            )
            del xformers_candidate
        except ImportError:
            grouped_reports["xformers_block_diagonal_f16"] = {"available": False}
        pitched_candidate = pitched_mask_attention(grouped_input, sequence_lengths)[backward_indices]
        grouped_reports["sdpa_f16_explicit_aligned_mask_pitch"] = metric(
            pitched_candidate.reshape(native_qkv.shape[0], CHANNELS), official.cpu().numpy()
        )
        del pitched_candidate
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(args.device))

    report: dict[str, object] = {
        "schema": "sam3d.mesh-attention-path-profile.v1",
        "block": args.block,
        "device": args.device,
        "window_count": len(sequence_lengths),
        "min_window_tokens": min(sequence_lengths),
        "max_window_tokens": max(sequence_lengths),
        "unique_window_lengths": len(set(sequence_lengths)),
        "native_qkv_to_fixture": feature_metrics(
            reference_dir / f"{prefix}_qkv_features.samt", args.native_qkv, 16 * 1024 * 1024
        ),
        "grouped_paths_to_official_same_qkv": grouped_reports,
    }
    if args.native_attention is not None:
        native_attention = load_features(args.native_attention, CHANNELS, "native attention")
        report["native_to_official_same_qkv"] = metric(
            torch.from_numpy(np.array(native_attention, dtype=np.float32, copy=True, order="C")).to(args.device),
            official.cpu().numpy(),
        )

    text = json.dumps(report, indent=2)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
