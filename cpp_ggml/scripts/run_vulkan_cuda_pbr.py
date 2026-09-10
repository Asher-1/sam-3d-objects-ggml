#!/usr/bin/env python3
"""Run Vulkan neural inference and hand its immutable assets to CUDA PBR.

The Vulkan process owns only image/mask-to-Gaussian/mesh inference.  It writes
the Gaussian PLY and raw FlexiCubes mesh before any post-processing.  A
separate CUDA PBR-enabled sam3d-cli then consumes exactly those files through
``pbr-assemble``.  This script never presents the resulting GLB as a Vulkan
renderer output; its manifest records both backend owners and SHA-256 hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def run_step(name: str, command: list[str], environment: dict[str, str],
             steps: list[dict[str, Any]]) -> bool:
    print(f"[{name}] + {command_text(command)}", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
    steps.append({
        "name": name,
        "command": command_text(command),
        "returncode": completed.returncode,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
    })
    return completed.returncode == 0


def require_file(parser: argparse.ArgumentParser, label: str, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        parser.error(f"missing {label}: {resolved}")
    return resolved


def require_dir(parser: argparse.ArgumentParser, label: str, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        parser.error(f"missing {label}: {resolved}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vulkan-binary", type=Path, required=True)
    parser.add_argument("--cuda-pbr-binary", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--moge-model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="new or empty directory for this handoff")
    parser.add_argument("--dtype", choices=("f16", "q8_0", "q4_0"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--ss-attention", choices=("normal", "strict"), default="strict")
    parser.add_argument("--rng-distribution-blocks", type=int,
                        help="CUDA-observed PyTorch Philox distribution block count")
    parser.add_argument("--noise-dir", type=Path,
                        help="explicit official noise replay for diagnostic runs only")
    parser.add_argument("--pbr-views", type=int, default=100)
    parser.add_argument("--pbr-resolution", type=int, default=1024)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--texture-steps", type=int, default=2500)
    args = parser.parse_args()

    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.rng_distribution_blocks is not None and args.rng_distribution_blocks <= 0:
        parser.error("--rng-distribution-blocks must be positive")
    if min(args.pbr_views, args.pbr_resolution, args.texture_size, args.texture_steps) <= 0:
        parser.error("all PBR stage sizes must be positive")

    vulkan_binary = require_file(parser, "Vulkan binary", args.vulkan_binary)
    cuda_pbr_binary = require_file(parser, "CUDA PBR binary", args.cuda_pbr_binary)
    models_dir = require_dir(parser, "GGUF model directory", args.models_dir)
    moge_model = require_file(parser, "MoGe GGUF", args.moge_model)
    image = require_file(parser, "image", args.image)
    mask = require_file(parser, "mask", args.mask)
    noise_dir = None
    if args.noise_dir is not None:
        noise_dir = require_dir(parser, "official noise directory", args.noise_dir)

    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        parser.error(f"refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "ply": out_dir / "vulkan_output.ply",
        "mesh_vertices": out_dir / "vulkan_flexicubes_vertices.samt",
        "mesh_faces": out_dir / "vulkan_flexicubes_faces.samt",
        "pose": out_dir / "vulkan_pose.json",
        "dtype_contract": out_dir / "vulkan_dtype_contract.json",
        "glb": out_dir / "cuda_pbr.glb",
        "texture": out_dir / "cuda_pbr.base_color.png",
    }
    steps: list[dict[str, Any]] = []
    environment = os.environ.copy()
    if args.rng_distribution_blocks is not None:
        environment["SAM3D_PYTORCH_PHILOX_DISTRIBUTION_BLOCKS"] = str(
            args.rng_distribution_blocks)

    vulkan_command = [
        str(vulkan_binary), "image-to-3d", "--model", str(models_dir),
        "--moge-model", str(moge_model), "--image", str(image), "--mask", str(mask),
        "--backend", "vulkan", "--dtype", args.dtype, "--seed", str(args.seed),
        "--threads", str(args.threads), "--ss-attention", args.ss_attention,
        "--out", str(outputs["ply"]), "--mesh-vertices-out", str(outputs["mesh_vertices"]),
        "--mesh-faces-out", str(outputs["mesh_faces"]), "--pose-out", str(outputs["pose"]),
        "--dtype-contract-out", str(outputs["dtype_contract"]),
    ]
    if noise_dir is not None:
        vulkan_command.extend(("--noise-dir", str(noise_dir)))
    vulkan_ok = run_step("vulkan_image_to_gaussian_and_raw_mesh", vulkan_command,
                         environment, steps)
    neural_outputs = {key: path for key, path in outputs.items() if key not in {"glb", "texture"}}
    artifacts_exist = vulkan_ok and all(path.is_file() for path in neural_outputs.values())
    neural_hashes = ({key: file_sha256(path) for key, path in neural_outputs.items()}
                     if artifacts_exist else {})
    if vulkan_ok and not artifacts_exist:
        missing = [str(path) for path in neural_outputs.values() if not path.is_file()]
        steps.append({
            "name": "verify_vulkan_handoff_artifacts",
            "command": "test -f " + " ".join(missing),
            "returncode": 1,
            "elapsed_ms": 0.0,
            "status": "FAIL",
            "error": "Vulkan inference did not produce required handoff artifacts",
            "missing": missing,
        })

    cuda_ok = False
    if artifacts_exist:
        cuda_command = [
            str(cuda_pbr_binary), "pbr-assemble", "--vertices", str(outputs["mesh_vertices"]),
            "--faces", str(outputs["mesh_faces"]), "--ply", str(outputs["ply"]),
            "--out", str(outputs["glb"]), "--views", str(args.pbr_views),
            "--resolution", str(args.pbr_resolution), "--texture-size", str(args.texture_size),
            "--steps", str(args.texture_steps), "--seed", str(args.seed),
        ]
        cuda_ok = run_step("cuda_pbr_assemble_from_vulkan_artifacts", cuda_command,
                           environment, steps)
        if cuda_ok and not all(outputs[key].is_file() for key in ("glb", "texture")):
            cuda_ok = False
            steps.append({
                "name": "verify_cuda_pbr_glb_artifact",
                "command": f"test -f {outputs['glb']}",
                "returncode": 1,
                "elapsed_ms": 0.0,
                "status": "FAIL",
                "error": "CUDA pbr-assemble returned success without GLB and texture artifacts",
            })
    else:
        steps.append({
            "name": "cuda_pbr_assemble_from_vulkan_artifacts",
            "status": "SKIPPED",
            "reason": "required Vulkan PLY or raw mesh artifact is absent",
        })

    artifact_manifest = {
        key: {"path": str(path), "sha256": file_sha256(path)}
        for key, path in outputs.items() if path.is_file()
    }
    unchanged = artifacts_exist and all(
        artifact_manifest.get(key, {}).get("sha256") == digest for key, digest in neural_hashes.items())
    if artifacts_exist and not unchanged:
        steps.append({"name": "verify_immutable_handoff", "status": "FAIL",
                      "error": "a Vulkan input artifact changed during CUDA post-processing"})
    manifest = {
        "schema": "sam3d.vulkan-inference-cuda-pbr-handoff.v1",
        "passed": vulkan_ok and artifacts_exist and cuda_ok and unchanged,
        "input": {"image": str(image), "mask": str(mask), "seed": args.seed},
        "dtype": args.dtype,
        "sampling_mode": "official-noise-replay-diagnostic" if noise_dir else "native-seed",
        "ss_attention": args.ss_attention,
        "rng_distribution_blocks": args.rng_distribution_blocks,
        "neural_backend": "vulkan",
        "pbr_backend": "cuda",
        "pbr_stage": {
            "views": args.pbr_views,
            "resolution": args.pbr_resolution,
            "texture_size": args.texture_size,
            "texture_steps": args.texture_steps,
        },
        "binaries": {"vulkan": str(vulkan_binary), "cuda_pbr": str(cuda_pbr_binary)},
        "binary_sha256": {"vulkan": file_sha256(vulkan_binary), "cuda_pbr": file_sha256(cuda_pbr_binary)},
        "input_sha256": {"image": file_sha256(image), "mask": file_sha256(mask)},
        "neural_artifacts_unchanged": unchanged,
        "artifacts": artifact_manifest,
        "steps": steps,
    }
    manifest_path = out_dir / "handoff_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
