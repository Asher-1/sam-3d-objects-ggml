#!/usr/bin/env python3
"""Render PyTorch and ggml Gaussian PLY files with the official renderer.

Both entities are transformed with the pose dumped by the official pipeline,
normalized independently exactly as notebook/inference.py does, and rendered
with one shared camera trajectory. The report is a visual-output comparison,
not a point-order comparison between two PLY files.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from samt_io import read_samt
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "notebook"))
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = str(Path(sys.executable).resolve().parent.parent)




def load_pose(condition_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = []
    for name in ("pose_rotation", "pose_translation", "pose_scale"):
        _, value = read_samt(condition_dir / f"{name}.samt")
        # The official pipeline's pose tensors are batched (1, 4)/(1, 3)/(1, 3);
        # compose_transform/pytorch3d reject the unbatched dump shape.
        values.append(torch.from_numpy(value).float().cuda().unsqueeze(0))
    return values[0], values[1], values[2]


def load_entity(path: Path):
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian import Gaussian

    entity = Gaussian(
        aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
        sh_degree=0,
        # PLY stores log(actual scale). Loading with zero kernel avoids a
        # sqrt(scale**2 - kernel**2) round-off NaN for tiny serialized scales;
        # the renderer still receives the exact decoded scale.
        mininum_kernel_size=0.0,
        scaling_bias=0.004,
        opacity_bias=0.1,
        scaling_activation="softplus",
        device="cuda",
    )
    entity.load_ply(str(path))
    # Fully opaque points are serialized as +inf by the official exporter.
    # A finite logit of 30 has the same sigmoid value at 8-bit render output.
    entity._opacity = torch.nan_to_num(entity._opacity, nan=0.0,
                                       posinf=30.0, neginf=-30.0)
    tensors = (entity._xyz, entity._features_dc, entity._scaling,
               entity._rotation, entity._opacity)
    if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
        raise ValueError(f"{path}: PLY contains non-finite Gaussian fields")
    return entity


def render_entity(entity, pose, frames: int, resolution: int) -> list[np.ndarray]:
    from inference import make_scene, ready_gaussian_for_video_rendering, render_video

    rotation, translation, scale = pose
    output = {
        "gaussian": [entity],
        "rotation": rotation,
        "translation": translation,
        "scale": scale,
    }
    scene = ready_gaussian_for_video_rendering(make_scene(output))
    return render_video(
        scene,
        r=1,
        fov=60,
        pitch_deg=15,
        yaw_start_deg=-45,
        resolution=resolution,
        num_frames=frames,
        bg_color=(0, 0, 0),
    )["color"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch-ply", type=Path, required=True)
    parser.add_argument("--ggml-ply", type=Path, required=True)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max-mae", type=float, default=None,
                        help="optional release gate on normalized RGB MAE")
    args = parser.parse_args()

    for path in (args.pytorch_ply, args.ggml_ply):
        if not path.is_file():
            parser.error(f"missing PLY: {path}")
    if not torch.cuda.is_available():
        parser.error("the official gsplat renderer requires CUDA")
    try:
        import plyfile  # noqa: F401
        import gsplat  # noqa: F401
    except ImportError as exc:
        parser.error(
            "official rendering dependencies are missing; activate the "
            f"sam3d-objects environment (missing {exc.name})"
        )
    if args.frames < 1 or args.resolution < 16:
        parser.error("--frames must be positive and --resolution must be >= 16")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    pose = load_pose(args.condition_dir)
    with torch.no_grad():
        pytorch_frames = render_entity(
            load_entity(args.pytorch_ply), pose, args.frames, args.resolution
        )
        ggml_frames = render_entity(
            load_entity(args.ggml_ply), pose, args.frames, args.resolution
        )

    pytorch_rgb = np.stack(pytorch_frames).astype(np.float32) / 255.0
    ggml_rgb = np.stack(ggml_frames).astype(np.float32) / 255.0
    difference = np.abs(pytorch_rgb - ggml_rgb)
    mse = float(np.mean(np.square(pytorch_rgb - ggml_rgb)))
    metrics = {
        "frames": args.frames,
        "resolution": args.resolution,
        "mae": float(difference.mean()),
        "rmse": math.sqrt(mse),
        "max_abs": float(difference.max()),
        "psnr_db": None if mse == 0 else -10.0 * math.log10(mse),
        "pytorch_ply": str(args.pytorch_ply.resolve()),
        "ggml_ply": str(args.ggml_ply.resolve()),
    }

    duration = 1000 / 30
    imageio.mimsave(args.out_dir / "pytorch_orbit.gif", pytorch_frames,
                    duration=duration, loop=0)
    imageio.mimsave(args.out_dir / "ggml_orbit.gif", ggml_frames,
                    duration=duration, loop=0)
    middle = args.frames // 2
    imageio.imwrite(args.out_dir / "pytorch_view.png", pytorch_frames[middle])
    imageio.imwrite(args.out_dir / "ggml_view.png", ggml_frames[middle])
    side_by_side = np.concatenate((pytorch_frames[middle], ggml_frames[middle]), axis=1)
    imageio.imwrite(args.out_dir / "side_by_side.png", side_by_side)
    overlay = np.clip(
        0.5 * np.asarray(pytorch_frames[middle], dtype=np.float32)
        + 0.5 * np.asarray(ggml_frames[middle], dtype=np.float32), 0, 255
    ).astype(np.uint8)
    imageio.imwrite(args.out_dir / "overlay.png", overlay)
    heatmap = np.clip(
        difference[middle].mean(axis=2) * 4.0 * 255.0, 0, 255
    ).astype(np.uint8)
    imageio.imwrite(args.out_dir / "absolute_difference.png", heatmap)
    with (args.out_dir / "render_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)

    print(json.dumps(metrics, indent=2))
    if args.max_mae is not None and metrics["mae"] > args.max_mae:
        print(f"render MAE {metrics['mae']:.6f} exceeds {args.max_mae:.6f}",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
