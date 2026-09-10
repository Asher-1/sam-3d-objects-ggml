#!/usr/bin/env python3
"""Create a reproducible official mesh-decoder fixture from frozen SLat data.

Unlike ``dump_e2e_stages.py``, this tool does not sample the generative
pipeline.  It reads a recorded ``slat_feats_final.samt`` and
``slat_coords.samt``, then runs only the official mesh decoder.  This removes
sampling and multi-model CUDA scheduling from a mesh-decoder parity gate while
retaining the exact released PyTorch module, checkpoint, sparse layout, and
mixed F16 precision contract.

The output format is consumed directly by ``verify_mesh_decoder_reference.py``
and by ``sam3d-cli mesh-decode``.  It is a reference-generation tool only;
the C++ runtime never imports Python or PyTorch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import numpy as np
import torch

from samt_io import GGML_F32, GGML_I32, read_samt_array, write_samt_array as write_samt


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_samt(path: Path, dtype: np.dtype) -> np.ndarray:
    return read_samt_array(path, dtype=dtype)




class FixtureWriter:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest: dict[str, object] = {}

    def features(self, stage: str, sparse) -> None:
        features = sparse.feats.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
        coordinates = sparse.coords.detach().int().cpu().contiguous().numpy().astype("<i4", copy=False)
        if features.ndim != 2 or coordinates.shape != (features.shape[0], 4):
            raise RuntimeError(f"{stage}: invalid sparse output {features.shape} / {coordinates.shape}")
        write_samt(self.root / f"mesh_decoder_{stage}_features.samt", features, GGML_F32)
        write_samt(self.root / f"mesh_decoder_{stage}_coords.samt", coordinates, GGML_I32)
        self.manifest[stage] = {
            "features": list(features.shape),
            "coordinates": list(coordinates.shape),
        }
        print(f"[dump] {stage}: features={list(features.shape)}")

    def save(self, metadata: dict[str, object]) -> None:
        self.manifest["mesh_decoder_reference"] = metadata
        (self.root / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="directory containing slat_feats_final.samt and slat_coords.samt")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO_ROOT / "checkpoints/hf/slat_decoder_mesh.pt")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "checkpoints/hf/slat_decoder_mesh.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dump-blocks", action="store_true",
                        help="record all 12 transformer block boundaries")
    parser.add_argument("--internals-block", type=int, choices=range(12), metavar="BLOCK",
                        help="record layer-normalization, attention and MLP boundaries for one block")
    parser.add_argument("--upsample-internals-block", type=int, choices=range(2), metavar="BLOCK",
                        help="record SparseSubdivideBlock3d module boundaries for one upsample block")
    parser.add_argument("--dump-representation", action="store_true",
                        help="also freeze the official FlexiCubes mesh returned by this forward")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.out_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    config = args.config.resolve()
    for path in (input_dir / "slat_feats_final.samt", input_dir / "slat_coords.samt", checkpoint, config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"--out-dir must be empty: {output_dir}")

    features = read_samt(input_dir / "slat_feats_final.samt", np.dtype("<f4"))
    coordinates = read_samt(input_dir / "slat_coords.samt", np.dtype("<i4"))
    if features.ndim != 2 or features.shape[1] != 8:
        raise ValueError(f"slat_feats_final.samt: expected F32 [tokens, 8], got {features.shape}")
    if coordinates.shape != (features.shape[0], 4):
        raise ValueError("slat coordinates must be I32 [tokens, 4] with the same token count")

    output_dir.mkdir(parents=True, exist_ok=True)
    # A fixture is self-contained so C++ and Python consume exactly the same
    # recorded latent, even after the original e2e directory is rotated.
    shutil.copy2(input_dir / "slat_feats_final.samt", output_dir / "slat_feats_final.samt")
    shutil.copy2(input_dir / "slat_coords.samt", output_dir / "slat_coords.samt")

    # Work both from the editable PyTorch reference environment and directly
    # from a clean checkout. Python otherwise sets sys.path[0] to
    # ``cpp_ggml/scripts`` and cannot resolve the repository package.
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "notebook"))
    import sam3d_objects  # noqa: F401 - registers Hydra targets.
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    decoder = instantiate(OmegaConf.load(config))
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = state_dict.get("state_dict", state_dict)
    decoder.load_state_dict(state_dict, strict=True)
    decoder = decoder.to(args.device).eval()

    writer = FixtureWriter(output_dir)
    captured: set[str] = set()
    hooks = []

    def capture_sparse(stage: str, sparse) -> None:
        if stage in captured:
            raise RuntimeError(f"{stage}: hook fired more than once")
        if not hasattr(sparse, "feats") or not hasattr(sparse, "coords"):
            raise RuntimeError(f"{stage}: expected sparse tensor")
        writer.features(stage, sparse)
        captured.add(stage)

    def sparse_output_hook(stage: str):
        def hook(_module, _inputs, result):
            capture_sparse(stage, result[0] if isinstance(result, tuple) else result)
        return hook

    hooks.append(decoder.input_layer.register_forward_hook(sparse_output_hook("input_layer")))
    block_indices = range(len(decoder.blocks)) if args.dump_blocks else (len(decoder.blocks) - 1,)
    for index in block_indices:
        hooks.append(decoder.blocks[index].register_forward_hook(sparse_output_hook(f"block{index}")))
    hooks.append(decoder.upsample[0].register_forward_hook(sparse_output_hook("upsample0")))
    hooks.append(decoder.upsample[1].register_forward_hook(sparse_output_hook("upsample1")))
    hooks.append(decoder.out_layer.register_forward_hook(sparse_output_hook("raw")))

    if args.internals_block is not None:
        index = args.internals_block
        block = decoder.blocks[index]
        block_input: dict[str, object] = {}

        def capture_block_input(_module, inputs):
            sparse = inputs[0]
            block_input["sparse"] = sparse
            capture_sparse(f"block{index}_ape", sparse)

        def capture_tensor(stage: str, tensor) -> None:
            sparse = block_input.get("sparse")
            if sparse is None or not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"{stage}: missing sparse block input or tensor result")
            capture_sparse(stage, sp.SparseTensor(feats=tensor, coords=sparse.coords))

        def attention_residual(_module, _inputs, result):
            sparse = block_input.get("sparse")
            if sparse is None or not hasattr(result, "feats") or not torch.equal(sparse.coords, result.coords):
                raise RuntimeError("attention residual has incompatible sparse support")
            capture_sparse(
                f"block{index}_attention",
                sp.SparseTensor(feats=sparse.feats + result.feats, coords=sparse.coords),
            )

        hooks.extend((
            block.register_forward_pre_hook(capture_block_input),
            block.norm1.register_forward_hook(
                lambda _module, _inputs, result: capture_tensor(f"block{index}_norm1", result)),
            block.attn.to_qkv.register_forward_hook(
                lambda _module, _inputs, result: capture_tensor(f"block{index}_qkv", result)),
            block.attn.to_out.register_forward_pre_hook(
                lambda _module, inputs: capture_tensor(f"block{index}_attention_values", inputs[0])),
            block.attn.to_out.register_forward_hook(
                lambda _module, _inputs, result: capture_tensor(f"block{index}_attention_out", result)),
            block.attn.register_forward_hook(attention_residual),
            block.norm2.register_forward_hook(
                lambda _module, _inputs, result: capture_tensor(f"block{index}_norm2", result)),
            block.mlp.mlp[0].register_forward_hook(sparse_output_hook(f"block{index}_mlp0")),
            block.mlp.mlp[1].register_forward_hook(sparse_output_hook(f"block{index}_gelu")),
            block.mlp.mlp[2].register_forward_hook(sparse_output_hook(f"block{index}_mlp2")),
        ))

    if args.upsample_internals_block is not None:
        index = args.upsample_internals_block
        block = decoder.upsample[index]
        subdivision_calls = 0

        def capture_subdivision(_module, _inputs, result):
            nonlocal subdivision_calls
            stage = f"upsample{index}_{'main_sub' if subdivision_calls == 0 else 'skip_sub'}"
            capture_sparse(stage, result)
            subdivision_calls += 1

        hooks.extend((
            block.act_layers[0].register_forward_hook(
                sparse_output_hook(f"upsample{index}_act_norm")),
            block.act_layers.register_forward_hook(sparse_output_hook(f"upsample{index}_act")),
            block.sub.register_forward_hook(capture_subdivision),
            block.out_layers[0].register_forward_hook(sparse_output_hook(f"upsample{index}_conv1")),
            block.out_layers[1].register_forward_hook(sparse_output_hook(f"upsample{index}_norm")),
            block.out_layers[2].register_forward_hook(sparse_output_hook(f"upsample{index}_silu")),
            block.out_layers[3].register_forward_hook(sparse_output_hook(f"upsample{index}_conv2")),
            block.skip_connection.register_forward_hook(sparse_output_hook(f"upsample{index}_skip")),
        ))

    representation = None
    try:
        with torch.inference_mode():
            sparse_input = sp.SparseTensor(
                feats=torch.from_numpy(features).to(args.device),
                coords=torch.from_numpy(coordinates).to(args.device),
            )
            representation = decoder(sparse_input)
    finally:
        for hook in hooks:
            hook.remove()

    required = {"input_layer", "block11", "upsample0", "upsample1", "raw"}
    if args.dump_blocks:
        required.update(f"block{index}" for index in range(len(decoder.blocks)))
    if args.internals_block is not None:
        required.update(f"block{args.internals_block}_{suffix}" for suffix in (
            "ape", "norm1", "qkv", "attention_values", "attention_out", "attention",
            "norm2", "mlp0", "gelu", "mlp2",
        ))
    if args.upsample_internals_block is not None:
        required.update(f"upsample{args.upsample_internals_block}_{suffix}" for suffix in (
            "act_norm", "act", "main_sub", "skip_sub", "conv1", "norm", "silu", "conv2", "skip",
        ))
    missing = sorted(required - captured)
    if missing:
        raise RuntimeError(f"official mesh-decoder hooks did not run: {missing}")
    representation_metadata = None
    if args.dump_representation:
        if not isinstance(representation, list) or len(representation) != 1:
            raise RuntimeError("official mesh decoder did not return exactly one mesh representation")
        mesh = representation[0]
        if not mesh.success or mesh.vertex_attrs is None:
            raise RuntimeError("official mesh decoder did not return a populated colored mesh")
        vertices = mesh.vertices.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
        faces = mesh.faces.detach().cpu().contiguous().numpy()
        attributes = mesh.vertex_attrs.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or \
                faces.ndim != 2 or faces.shape[1] != 3 or \
                attributes.shape != (vertices.shape[0], 6):
            raise RuntimeError(
                f"official mesh representation has invalid shapes: {vertices.shape}, {faces.shape}, "
                f"{attributes.shape}"
            )
        if faces.size and (faces.min() < 0 or faces.max() >= vertices.shape[0]):
            raise RuntimeError("official mesh representation has invalid face indices")
        faces = faces.astype("<i4", copy=False)
        write_samt(output_dir / "mesh_representation_vertices.samt", vertices, GGML_F32)
        write_samt(output_dir / "mesh_representation_faces.samt", faces, GGML_I32)
        write_samt(output_dir / "mesh_representation_vertex_attrs.samt", attributes, GGML_F32)
        representation_metadata = {
            "vertices": int(vertices.shape[0]),
            "triangles": int(faces.shape[0]),
            "coordinate_system": "official mesh decoder z-up",
        }

    metadata = {
        "schema": "sam3d.mesh-decoder-reference.v2",
        "source": "official-pytorch-direct-decoder",
        "input_sha256": {
            "slat_feats_final.samt": sha256(output_dir / "slat_feats_final.samt"),
            "slat_coords.samt": sha256(output_dir / "slat_coords.samt"),
        },
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint": str(checkpoint),
        "config": str(config),
        "device": args.device,
        "stages": sorted(captured),
        "subdivision_order": "torch.nonzero(ones(2,2,2)); z-fastest",
    }
    if representation_metadata is not None:
        metadata["representation"] = representation_metadata
    writer.save(metadata)
    print(f"[done] {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
