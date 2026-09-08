#!/usr/bin/env python3
"""Measure the official implementation's own final-PBR replay variation.

The optimized texture bake uses CUDA reductions and atomic gradient updates.
The reference implementation can therefore produce slightly different final
atlas bytes across two otherwise identical runs. This tool records that
observed variation before native tolerances are defined; it never changes the
official algorithm or chooses an arbitrary native threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def require_file(root: Path, name: str) -> Path:
    path = root / name
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty artifact: {path}")
    return path


def load_manifest(root: Path) -> dict:
    manifest = json.loads(require_file(root, "manifest.json").read_text())
    if manifest.get("schema") != "sam3d.official-pbr-reference.v1":
        raise ValueError(f"{root}: unexpected reference schema")
    return manifest


def image_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float | int | list[int]]:
    if left.shape != right.shape:
        raise ValueError(f"image shape mismatch: {left.shape} != {right.shape}")
    delta = left.astype(np.float64) - right.astype(np.float64)
    absolute = np.abs(delta)
    return {
        "shape": list(left.shape),
        "mae_u8": float(absolute.mean()),
        "rmse_u8": float(np.sqrt(np.mean(delta * delta))),
        "max_abs_u8": int(absolute.max()),
        "changed_pixels": int(np.any(absolute > 0, axis=2).sum()),
        "total_pixels": int(left.shape[0] * left.shape[1]),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-a", type=Path, required=True)
    parser.add_argument("--reference-b", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    root_a = args.reference_a.resolve()
    root_b = args.reference_b.resolve()
    manifest_a = load_manifest(root_a)
    manifest_b = load_manifest(root_b)
    bake_a = manifest_a["bake"]
    bake_b = manifest_b["bake"]
    if bake_a.get("selected_views") != bake_b.get("selected_views"):
        raise ValueError("the official bake view schedules differ")
    if bake_a.get("observation_sha256") != bake_b.get("observation_sha256"):
        raise ValueError("the official Gaussian observation inputs differ")

    texture_a_path = require_file(root_a, "base_color.png")
    texture_b_path = require_file(root_b, "base_color.png")
    texture_a = np.asarray(Image.open(texture_a_path).convert("RGB"), dtype=np.uint8)
    texture_b = np.asarray(Image.open(texture_b_path).convert("RGB"), dtype=np.uint8)
    result = {
        "schema": "sam3d.official-pbr-repeatability.v1",
        "reference_a": str(root_a),
        "reference_b": str(root_b),
        "input_observations_exact": True,
        "selected_views_exact": True,
        "texture": image_metrics(texture_a, texture_b),
        "artifacts": {
            "base_color_sha256": [sha256(texture_a_path), sha256(texture_b_path)],
            "official_pbr_glb_sha256": [
                sha256(require_file(root_a, "official_pbr.glb")),
                sha256(require_file(root_b, "official_pbr.glb")),
            ],
        },
    }
    rendered = json.dumps(result, indent=2)
    if args.json_out is not None:
        args.json_out.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
