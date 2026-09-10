#!/usr/bin/env python3
"""Multi-object scene reconstruction: the official demo_multi_object flow.

The official SAM 3D Objects "full scene" capability is N independent
single-object reconstructions (one per mask) composed by the official
``make_scene`` pose application, then rendered as a Gaussian-splat orbit.
This script orchestrates exactly that flow for both runners:

* ``--runner python`` calls the official ``Inference`` per mask (the same
  code path as ``notebook/demo_multi_object.ipynb``).
* ``--runner ggml`` calls the native ``sam3d-cli image-to-3d`` per mask
  (without ``--pbr-out``, which skips mesh/FlexiCubes/PBR entirely), reads
  each native Gaussian PLY back through the official ``Gaussian.load_ply``
  activation chain (the native PLY is field-compatible with
  ``gaussian_model.save_ply``), and feeds the official ``make_scene``.

No scene-composition math is reimplemented here: pose application, Gaussian
concatenation, normalization, and rendering all come from the official code.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "notebook"))
os.environ.setdefault("CONDA_PREFIX", sys.prefix)
os.environ.setdefault("CUDA_HOME", os.environ["CONDA_PREFIX"])
os.environ.setdefault("LIDRA_SKIP_INIT", "true")


def move_module(module, device) -> None:
    """Same residency helpers as bench_full_e2e_pipeline.py, kept local so the
    module does not inherit that script's lazy ``global torch`` binding."""
    if module is not None:
        module.to(device)


def move_depth_model(pipeline, device) -> None:
    model = getattr(getattr(pipeline, "depth_model", None), "model", None)
    move_module(model, device)


def move_tensor_fields(value, device) -> None:
    """Move the tensor state of an official representation without copying it."""
    import torch
    for name, field in vars(value).items():
        if torch.is_tensor(field):
            setattr(value, name, field.to(device))
    if hasattr(value, "device"):
        value.device = str(device)

# Official decode-time Gaussian parameters: decoder_gs.py constructs
# Gaussian(sh_degree=0, aabb=[-.5]*3+[1]*3, ...) from
# checkpoints/hf/slat_decoder_gs.yaml representation_config.
GAUSSIAN_PARAMS = {
    "sh_degree": 0,
    "aabb": [-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
    "mininum_kernel_size": 0.0009,  # 3d_filter_kernel_size
    "scaling_bias": 0.004,
    "opacity_bias": 0.1,
    "scaling_activation": "softplus",
}


def enumerate_masks(mask_dir: Path, indices: list[int] | None) -> list[tuple[int, Path]]:
    """Same contract as the official load_masks: masks live at '<idx>.png'."""
    if indices is None:
        index = 0
        indices = []
        while (mask_dir / f"{index}.png").exists():
            indices.append(index)
            index += 1
    masks = []
    for index in indices:
        path = mask_dir / f"{index}.png"
        if not path.is_file():
            raise SystemExit(f"error: missing mask: {path}")
        masks.append((index, path))
    if not masks:
        raise SystemExit(f"error: no masks found in {mask_dir}")
    return masks


def philox_blocks(cli: Path, rng_cli: Path, seed: int, work_dir: Path) -> int:
    """Same CUDA Philox distribution-block contract as the root launcher."""
    out_dir = work_dir / "rng_contract"
    subprocess.run(
        [str(rng_cli), "rng-dump", "--implementation", "cuda", "--seed", str(seed),
         "--sizes", "1", "--out-dir", str(out_dir)],
        check=True,
    )
    contract = json.loads((out_dir / "rng_contract.json").read_text(encoding="utf-8"))
    blocks = contract.get("distribution_blocks")
    if not isinstance(blocks, int) or blocks <= 0:
        raise SystemExit("error: invalid CUDA Philox contract")
    return blocks


def run_ggml_objects(args: argparse.Namespace, masks, work_dir: Path) -> tuple[list[dict], dict]:
    import torch
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import (
        Gaussian,
    )

    rng_cli = args.rng_cli or args.cli
    blocks = philox_blocks(args.cli, rng_cli, args.seed, work_dir) \
        if args.backend == "vulkan" else None

    if not args.ggml_per_process:
        # Session batch: one CLI process serves every mask with the shared
        # model resources (MoGe reload via mmap, one backend for all stages,
        # RGB-only MoGe point map reuse). Outputs land in obj_<ID> trees and
        # the batch manifest carries per-object digests and timings. This is
        # the default; --ggml-per-process keeps the old one-process-per-object
        # behaviour as an explicit A/B baseline.
        list_path = work_dir / "mask_list.txt"
        list_path.write_text(
            "".join(f"{index}\t{mask_path}\n" for index, mask_path in masks),
            encoding="utf-8",
        )
        command = [
            str(args.cli), "image-to-3d",
            "--model", str(args.models_dir), "--moge-model", str(args.moge_model),
            "--image", str(args.image),
            "--mask-list", str(list_path), "--out-dir", str(work_dir),
            "--backend", args.backend, "--dtype", args.dtype,
            "--seed", str(args.seed), "--threads", str(args.threads),
            "--ss-attention", args.ss_attention,
        ]
        if blocks is not None:
            command += ["--philox-blocks", str(blocks)]
        started = time.perf_counter()
        subprocess.run(command, check=True)
        batch_elapsed = time.perf_counter() - started
        manifest_path = work_dir / "batch_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        timings = []
        for obj in manifest["objects"]:
            if obj["status"] != "ok":
                raise SystemExit(
                    f"error: batch object {obj['id']} failed: {obj.get('error', '')}")
            # Re-map the obj_<ID> outputs onto the per-object metadata the
            # scene assembly expects.
            case_dir = work_dir / f"obj_{obj['id']:02d}"
            if not case_dir.exists():
                case_dir = work_dir / f"objects" / f"obj_{obj['id']}"
            timings.append({
                "mask_index": obj["id"],
                "elapsed_s": obj["elapsed_s"],
                "moge_pointmap_reused": obj["moge_pointmap_reused"],
            })
        # The per-mask Gaussian loads below read the batch's PLY files.
        batch_mode = True
    else:
        batch_mode = False

    outputs = []
    for index, mask_path in masks:
        case_dir = work_dir / f"obj_{index:02d}"
        if batch_mode:
            # The batch manifest wrote obj_<ID> (no zero padding) under the
            # objects/ tree; fall back to the legacy padded layout.
            alt_dir = work_dir / "objects" / f"obj_{index}"
            if not (case_dir / "output.ply").exists() and (alt_dir / "output.ply").exists():
                case_dir = alt_dir
        if not batch_mode:
            case_dir.mkdir(parents=True, exist_ok=True)
            command = [
                str(args.cli), "image-to-3d",
                "--model", str(args.models_dir), "--moge-model", str(args.moge_model),
                "--image", str(args.image), "--mask", str(mask_path),
                "--backend", args.backend, "--dtype", args.dtype,
                "--seed", str(args.seed), "--threads", str(args.threads),
                "--ss-attention", args.ss_attention,
                "--out", str(case_dir / "output.ply"),
                "--pose-out", str(case_dir / "pose.json"),
            ]
            if blocks is not None:
                command += ["--philox-blocks", str(blocks)]
            subprocess.run(command, check=True)

        # Official save_ply -> load_ply round trip: the native PLY stores the
        # same fields (xyz, f_dc, opacity logits, log scales, quaternions),
        # so the official activation chain reconstructs the object exactly.
        gs = Gaussian(
            aabb=GAUSSIAN_PARAMS["aabb"],
            sh_degree=GAUSSIAN_PARAMS["sh_degree"],
            mininum_kernel_size=GAUSSIAN_PARAMS["mininum_kernel_size"],
            scaling_bias=GAUSSIAN_PARAMS["scaling_bias"],
            opacity_bias=GAUSSIAN_PARAMS["opacity_bias"],
            scaling_activation=GAUSSIAN_PARAMS["scaling_activation"],
            device="cuda",
        )
        gs.load_ply(str(case_dir / "output.ply"))
        # Compose on CPU like the python runner; the scene moves back to the
        # GPU as a whole right before rendering.
        move_tensor_fields(gs, "cpu")
        pose = json.loads((case_dir / "pose.json").read_text(encoding="utf-8"))
        # pose.json v2 top level is the official receipt schema (batch-nested,
        # scale collapsed to its uniform mean) - exactly what make_scene
        # asserts and consumes.
        outputs.append({
            "mask_index": index,
            "gaussian": [gs],
            "rotation": torch.tensor(pose["rotation"], dtype=torch.float32),
            "translation": torch.tensor(pose["translation"], dtype=torch.float32),
            "scale": torch.tensor(pose["scale"], dtype=torch.float32),
        })
    if batch_mode:
        return outputs, {
            "objects": timings,
            "philox_blocks": blocks,
            "batch_total_elapsed_s": batch_elapsed,
            "batch_mode": True,
        }
    return outputs, {"objects": timings, "philox_blocks": blocks}


def run_python_objects(args: argparse.Namespace, masks, work_dir: Path) -> tuple[list[dict], dict]:
    import torch
    from official_reference_policy import install_staged_mixed_precision_loader

    # The official notebook keeps every module resident, which does not fit a
    # 12 GiB GPU. Reuse the documented staged mixed-precision loader plus the
    # streamed residency choreography already validated by
    # bench_full_e2e_pipeline.py; the numerical policy is identical.
    from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
    InferencePipeline.instantiate_and_load_from_pretrained = (
        install_staged_mixed_precision_loader(
            InferencePipeline.instantiate_and_load_from_pretrained))

    from inference import Inference, load_image, load_mask
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import (
        Gaussian,
    )
    inference = Inference(str(args.config), compile=False)
    pipeline = inference._pipeline
    image = load_image(str(args.image))
    outputs = []
    timings = []
    for index, mask_path in masks:
        case_dir = work_dir / f"obj_{index:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        cache_path = case_dir / "object.pt"
        pose_path = case_dir / "pose.json"
        if args.resume and cache_path.is_file() and pose_path.is_file():
            # Idempotent resume: restore the decoded gaussian state from the
            # tensor cache (no PLY round trip, so no load_ply float32
            # artifacts at the tiny-scale boundary).
            payload = torch.load(cache_path, map_location="cuda", weights_only=True)
            gs = Gaussian(
                aabb=GAUSSIAN_PARAMS["aabb"],
                sh_degree=GAUSSIAN_PARAMS["sh_degree"],
                mininum_kernel_size=GAUSSIAN_PARAMS["mininum_kernel_size"],
                scaling_bias=GAUSSIAN_PARAMS["scaling_bias"],
                opacity_bias=GAUSSIAN_PARAMS["opacity_bias"],
                scaling_activation=GAUSSIAN_PARAMS["scaling_activation"],
                device="cuda",
            )
            for name in ("_xyz", "_features_dc", "_scaling", "_rotation", "_opacity"):
                setattr(gs, name, payload[name].to("cuda"))
            gs.mininum_kernel_size = float(payload["mininum_kernel_size"])
            outputs.append({
                "mask_index": index,
                "gaussian": [gs],
                "rotation": payload["rotation"].to("cuda"),
                "translation": payload["translation"].to("cuda"),
                "scale": payload["scale"].to("cuda"),
            })
            continue
        mask = load_mask(str(mask_path))
        if mask.shape != image.shape[:2]:
            raise SystemExit(f"error: mask {mask_path} dimensions do not match the image")
        rgba_image = inference.merge_mask_to_rgba(image, mask)
        started = time.perf_counter()
        output = run_streamed_scene_object(pipeline, rgba_image, args.seed)
        timings.append({"mask_index": index, "elapsed_s": time.perf_counter() - started})
        gaussian = output["gaussian"][0]
        torch.save({
            "_xyz": gaussian._xyz.cpu(),
            "_features_dc": gaussian._features_dc.cpu(),
            "_scaling": gaussian._scaling.cpu(),
            "_rotation": gaussian._rotation.cpu(),
            "_opacity": gaussian._opacity.cpu(),
            "mininum_kernel_size": float(gaussian.mininum_kernel_size),
            "rotation": output["rotation"].cpu(),
            "translation": output["translation"].cpu(),
            "scale": output["scale"].cpu(),
        }, cache_path)
        # Official pose receipt schema (batch-nested rotation/translation/
        # scale) so every variant exposes the same receipt contract.
        pose_path.write_text(json.dumps({
            "schema": "sam3d.scene-pose.v1",
            "rotation": output["rotation"].cpu().tolist(),
            "translation": output["translation"].cpu().tolist(),
            "scale": output["scale"].cpu().tolist(),
        }, indent=2) + "\n", encoding="utf-8")
        output = dict(output)
        output["mask_index"] = index
        outputs.append(output)
    return outputs, {"objects": timings}


def run_streamed_scene_object(pipeline, rgba_image, seed: int) -> dict:
    """Official streamed stages, stopping after the Gaussian decode.

    Mirrors run_streamed_pbr in bench_full_e2e_pipeline.py up to the Gaussian
    (scene rendering needs no mesh, FlexiCubes, or texture bake). Each
    completed object moves to CPU so N objects fit alongside the next
    object's model residency on a 12 GiB device.
    """
    import torch

    device = pipeline.device
    models = pipeline.models
    conditioners = pipeline.condition_embedders

    for name, module in models.items():
        if name not in {"ss_generator", "ss_decoder"}:
            move_module(module, "cpu")
    move_module(conditioners["slat_condition_embedder"], "cpu")
    move_module(conditioners["ss_condition_embedder"], device)
    move_module(models["ss_generator"], device)
    move_module(models["ss_decoder"], device)
    move_depth_model(pipeline, device)
    torch.cuda.empty_cache()

    image = pipeline.merge_image_and_mask(rgba_image, None)
    pointmap_dict = pipeline.compute_pointmap(image)
    pointmap = pointmap_dict["pointmap"]
    ss_input = pipeline.preprocess_image(image, pipeline.ss_preprocessor, pointmap=pointmap)
    slat_input = pipeline.preprocess_image(image, pipeline.slat_preprocessor)
    torch.manual_seed(seed)
    ss_output = pipeline.sample_sparse_structure(ss_input)
    ss_output.update(pipeline.pose_decoder(
        ss_output,
        scene_scale=ss_input.get("pointmap_scale"),
        scene_shift=ss_input.get("pointmap_shift"),
    ))
    ss_output["scale"] = ss_output["scale"] * ss_output["downsample_factor"]
    coords = ss_output["coords"]
    ss_output.pop("shape", None)

    move_module(models["ss_generator"], "cpu")
    move_module(models["ss_decoder"], "cpu")
    move_module(conditioners["ss_condition_embedder"], "cpu")
    move_depth_model(pipeline, "cpu")
    move_module(models["slat_generator"], device)
    move_module(models["slat_decoder_gs"], device)
    move_module(conditioners["slat_condition_embedder"], device)
    torch.cuda.empty_cache()

    slat = pipeline.sample_slat(slat_input, coords)
    move_module(models["slat_generator"], "cpu")
    move_module(conditioners["slat_condition_embedder"], "cpu")
    torch.cuda.empty_cache()
    decoder_dtype = next(models["slat_decoder_gs"].parameters()).dtype
    slat = slat.to(dtype=decoder_dtype)
    outputs = pipeline.decode_slat(slat, ["gaussian"])
    move_module(models["slat_decoder_gs"], "cpu")
    torch.cuda.empty_cache()

    # make_scene deep-copies and composes on the same device as the stored
    # Gaussian; keep every accumulated object on CPU so N objects fit
    # alongside the next object's model residency on a 12 GiB device.
    move_tensor_fields(outputs["gaussian"][0], "cpu")
    for key in ("rotation", "translation", "scale"):
        if key in ss_output and torch.is_tensor(ss_output[key]):
            ss_output[key] = ss_output[key].cpu()
    return {**ss_output, "gaussian": outputs["gaussian"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("python", "ggml"), required=True)
    parser.add_argument("--ggml-per-process", action="store_true",
                        help="A/B baseline: one CLI process per object instead "
                             "of the default single-process session batch")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-indices", type=str,
                        help="comma-separated mask indices; default: every '<idx>.png' in --mask-dir")
    parser.add_argument("--resume", action="store_true",
                        help="skip objects whose tensor cache and pose receipt already exist")
    # python runner
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "checkpoints/hf/pipeline.yaml")
    # ggml runner
    parser.add_argument("--cli", type=Path, help="sam3d-cli used for image-to-3d")
    parser.add_argument("--rng-cli", type=Path,
                        help="sam3d-cli used for rng-dump (defaults to --cli)")
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "cpp_ggml/models/gguf")
    parser.add_argument("--moge-model", type=Path)
    parser.add_argument("--backend", choices=("cuda", "vulkan"), default="cuda")
    parser.add_argument("--dtype", choices=("f16", "q8_0", "q4_k"), default="q8_0")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--ss-attention", choices=("normal", "strict"), default="normal",
                        help="SS attention path: normal (default) is the F16-KV flash "
                             "delivery path; strict is the F32 score-materialization "
                             "parity diagnostic")
    # rendering (official demo_multi_object defaults)
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"missing image: {args.image}")
    if not args.mask_dir.is_dir():
        parser.error(f"missing mask directory: {args.mask_dir}")
    indices = ([int(part) for part in args.mask_indices.split(",")]
               if args.mask_indices else None)
    masks = enumerate_masks(args.mask_dir, indices)
    if args.runner == "ggml":
        if args.cli is None or not args.cli.is_file():
            parser.error("--runner ggml requires an existing --cli sam3d-cli binary")
        if args.moge_model is None:
            args.moge_model = args.models_dir / "moge_vitl-f16.gguf"
        if not args.moge_model.is_file():
            parser.error(f"missing MoGe GGUF: {args.moge_model}")
    else:
        if not args.config.is_file():
            parser.error(f"missing pipeline config: {args.config}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "objects").mkdir(exist_ok=True)
    work_dir = args.out_dir / "objects"

    import imageio
    import torch
    from inference import make_scene, ready_gaussian_for_video_rendering, render_video

    scene_started = time.perf_counter()
    if args.runner == "python":
        outputs, timings = run_python_objects(args, masks, work_dir)
    else:
        outputs, timings = run_ggml_objects(args, masks, work_dir)

    # Official scene composition: apply each object's pose (rotation,
    # translation, uniform scale) to its Gaussian and concatenate.
    scene_gs = make_scene(*outputs)
    scene_gs.save_ply(str(args.out_dir / "scene_posed.ply"))

    # Record the normalization contract from the POSED scene (before the
    # official normalized_gaussian rescale): the opacity > 0.9 active bounds
    # give the inv_scale and post-scale center that native variants must
    # reuse to render the identical scene frame.
    active = scene_gs.get_opacity.squeeze(-1) > 0.9
    active_xyz = scene_gs.get_xyz[active]
    lower_bounds = active_xyz.min(dim=0).values
    upper_bounds = active_xyz.max(dim=0).values
    inv_scale = float((upper_bounds - lower_bounds).max())
    norm_center = ((lower_bounds + upper_bounds) * 0.5 / inv_scale).tolist()
    (args.out_dir / "normalization.json").write_text(json.dumps({
        "schema": "sam3d.scene-normalization.v1",
        "inv_scale": inv_scale,
        "center": norm_center,
    }, indent=2) + "\n", encoding="utf-8")

    render_gs = ready_gaussian_for_video_rendering(scene_gs)
    # The gsplat renderer requires GPU residency; move the composed scene as
    # a whole after CPU-side composition.
    move_tensor_fields(render_gs, "cuda")
    video = render_video(
        render_gs, r=args.radius, fov=args.fov, resolution=args.resolution,
        num_frames=args.num_frames,
    )["color"]
    frames_dir = args.out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    for frame_index, frame in enumerate(video):
        imageio.imwrite(frames_dir / f"frame_{frame_index:04d}.png", frame)
    imageio.mimsave(args.out_dir / "scene.gif", video, format="GIF", duration=1000 / 30, loop=0)

    manifest = {
        "schema": "sam3d.scene-e2e.v1",
        "runner": args.runner,
        "image": str(args.image.resolve()),
        "mask_dir": str(args.mask_dir.resolve()),
        "mask_indices": [index for index, _ in masks],
        "seed": args.seed,
        "objects": timings,
        "gaussian_counts": [
            {"mask_index": output["mask_index"],
             "gaussians": int(output["gaussian"][0]._xyz.shape[0])}
            for output in outputs
        ],
        "render": {"num_frames": args.num_frames, "radius": args.radius,
                   "fov": args.fov, "resolution": args.resolution},
        "scene_elapsed_s": time.perf_counter() - scene_started,
        "outputs": {
            "posed_ply": str((args.out_dir / "scene_posed.ply").resolve()),
            "gif": str((args.out_dir / "scene.gif").resolve()),
            "frames_dir": str(frames_dir.resolve()),
        },
    }
    if args.runner == "ggml":
        manifest["backend"] = args.backend
        manifest["dtype"] = args.dtype
        manifest["cli"] = str(args.cli.resolve())
    (args.out_dir / "scene_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Scene reconstruction complete: {args.out_dir / 'scene.gif'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
