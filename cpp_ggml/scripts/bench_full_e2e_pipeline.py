#!/usr/bin/env python3
"""Benchmark the official hot-session image/mask -> textured PBR GLB path.

The timer deliberately starts after pipeline construction and input decoding,
then includes streamed weight transfers, neural inference, mesh cleanup, UV,
100-view observation rendering, the official 2500-step texture bake, and the
GLB export/file close. It is therefore comparable only to a native hot-session
timer with the same contract; it is not a cold-process startup benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "notebook"))
os.environ.setdefault("CONDA_PREFIX", sys.prefix)
os.environ.setdefault("CUDA_HOME", os.environ["CONDA_PREFIX"])
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

from gpu_exclusivity import require_exclusive_gpu
from official_reference_policy import (install_staged_mixed_precision_loader,
                                       reference_weight_policy)


def move_module(module, device: str | torch.device) -> None:
    if module is not None:
        module.to(device)


def move_depth_model(pipeline, device: str | torch.device) -> None:
    model = getattr(getattr(pipeline, "depth_model", None), "model", None)
    move_module(model, device)


def move_tensor_fields(value, device: str | torch.device) -> None:
    """Move the tensor state of an official representation without copying it.

    ``Gaussian`` and ``MeshExtractResult`` are lightweight Python containers.
    The official PBR code consumes their tensor fields directly, so changing
    their residency before the next independent stage preserves the values and
    avoids keeping both decoder outputs on a 12 GiB device.
    """
    for name, field in vars(value).items():
        if torch.is_tensor(field):
            setattr(value, name, field.to(device))
    if hasattr(value, "device"):
        value.device = str(device)


def module_dtypes(pipeline) -> dict[str, str]:
    values: dict[str, str] = {}
    for family, modules in (("models", pipeline.models),
                            ("conditioners", pipeline.condition_embedders)):
        for name, module in modules.items():
            if module is None:
                values[f"{family}.{name}"] = "unavailable"
                continue
            parameter = next(module.parameters(), None)
            values[f"{family}.{name}"] = str(parameter.dtype) if parameter is not None else "parameterless"
    depth_model = getattr(getattr(pipeline, "depth_model", None), "model", None)
    if depth_model is not None:
        parameter = next(depth_model.parameters(), None)
        values["depth_model"] = str(parameter.dtype) if parameter is not None else "parameterless"
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_manifest(pipeline, config: Path, stream_weights: bool,
                       fp32_weights: bool) -> dict[str, object]:
    cuda_backend = torch.backends.cuda
    return {
        "schema": "sam3d.reference-execution.v1",
        "config": str(config.resolve()),
        "config_sha256": sha256_file(config),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "default_dtype": str(torch.get_default_dtype()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "flash_sdp_enabled": bool(cuda_backend.flash_sdp_enabled()),
        "mem_efficient_sdp_enabled": bool(cuda_backend.mem_efficient_sdp_enabled()),
        "math_sdp_enabled": bool(cuda_backend.math_sdp_enabled()),
        "weight_residency": "streamed" if stream_weights else "resident",
        "weight_policy": reference_weight_policy(fp32_weights),
        "module_dtypes": module_dtypes(pipeline),
    }


def run_streamed_pbr(pipeline, rgba_image, seed: int):
    """Run the exact official stages while keeping only each active model resident."""
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
    decode_residency = {"before_offload_allocated_bytes": torch.cuda.memory_allocated(device)}
    # Neither decoder consumes the diffusion generator or its conditioner.
    # Retaining them through dense FlexiCubes allocations can exhaust 12 GiB.
    move_module(models["slat_generator"], "cpu")
    move_module(conditioners["slat_condition_embedder"], "cpu")
    torch.cuda.empty_cache()
    decode_residency["after_offload_allocated_bytes"] = torch.cuda.memory_allocated(device)
    decoder_dtype = next(models["slat_decoder_gs"].parameters()).dtype
    slat = slat.to(dtype=decoder_dtype)
    outputs = pipeline.decode_slat(slat, ["gaussian"])

    # The two decoder outputs are independent given ``slat``. Keep the completed
    # Gaussian on CPU while FlexiCubes runs so its tensors and mesh workspaces do
    # not overlap on a 12 GiB device.
    move_tensor_fields(outputs["gaussian"][0], "cpu")
    move_module(models["slat_decoder_gs"], "cpu")
    torch.cuda.empty_cache()
    move_module(models["slat_decoder_mesh"], device)
    outputs.update(pipeline.decode_slat(slat, ["mesh"]))

    # ``to_glb`` first transfers these exact mesh fields to CPU before xatlas.
    # Do that transfer before restoring the Gaussian for the 100-view renderer.
    move_tensor_fields(outputs["mesh"][0], "cpu")

    # No learned module participates in the official postprocess call. Release
    # its weights before the 100-view renderer and 2500-step atlas optimizer.
    move_module(models["slat_generator"], "cpu")
    move_module(models["slat_decoder_gs"], "cpu")
    move_module(models["slat_decoder_mesh"], "cpu")
    move_module(conditioners["slat_condition_embedder"], "cpu")
    torch.cuda.empty_cache()
    move_tensor_fields(outputs["gaussian"][0], device)
    outputs = pipeline.postprocess_slat_output(
        outputs, with_mesh_postprocess=True, with_texture_baking=True,
        use_vertex_color=False,
    )
    if outputs.get("glb") is None:
        raise RuntimeError("official full PBR run did not produce a GLB")
    return {**ss_output, **outputs, "reference_decode_residency": decode_residency}


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": sum(values) / len(values),
        "p50": ordered[len(ordered) // 2],
        "max": ordered[-1],
    }


def main() -> int:
    global torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--mask", type=Path, help="binary mask path; overrides --mask-dir/--mask-index")
    parser.add_argument("--mask-index", type=int, default=0)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "checkpoints/hf/pipeline.yaml")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--stream-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp32-weights", action="store_true",
                        help="retain official F32 modules; default uses the staged mixed-precision policy")
    parser.add_argument("--require-exclusive-gpu", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1:
        parser.error("--warmup must be non-negative and --iters must be positive")
    for label, path in (("image", args.image),
                        ("pipeline config", args.config)):
        if not path.exists():
            parser.error(f"missing {label}: {path}")
    if args.mask is None and args.mask_dir is None:
        parser.error("provide --mask or --mask-dir")
    mask = args.mask or args.mask_dir / f"{args.mask_index}.png"
    if not mask.is_file():
        parser.error(f"missing mask: {mask}")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error(f"refusing to overwrite non-empty output directory: {args.out_dir}")

    # Keep --help and path validation independent of the heavyweight official
    # environment, and fail before creating output when it is absent.
    os.chdir(REPO_ROOT)
    try:
        import torch as official_torch
        from inference import Inference, load_image, load_mask
    except ModuleNotFoundError as error:
        parser.error(
            "official SAM 3D Objects Python dependencies are unavailable "
            f"({error}); invoke this script with the configured SAM3D_PYTHON interpreter"
        )
    torch = official_torch
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gpu_exclusivity = (require_exclusive_gpu("cuda") if args.require_exclusive_gpu else
                       {"required": False, "checked": False, "active_compute_processes": []})

    if not args.fp32_weights:
        from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
        original_loader = InferencePipeline.instantiate_and_load_from_pretrained
        InferencePipeline.instantiate_and_load_from_pretrained = (
            install_staged_mixed_precision_loader(original_loader))

    initialization_started = time.perf_counter()
    inference = Inference(str(args.config), compile=False)
    initialization_ms = (time.perf_counter() - initialization_started) * 1000.0
    image = load_image(str(args.image))
    loaded_mask = load_mask(str(mask))
    if loaded_mask.shape != image.shape[:2]:
        parser.error("mask and image dimensions must match")
    rgba_image = inference.merge_mask_to_rgba(image, loaded_mask)
    pipeline = inference._pipeline
    dtypes = module_dtypes(pipeline)

    durations: list[float] = []
    outputs: list[str] = []
    residency_measurements: list[dict] = []
    for iteration in range(args.warmup + args.iters):
        torch.cuda.synchronize()
        started = time.perf_counter()
        if args.stream_weights:
            output = run_streamed_pbr(pipeline, rgba_image, args.seed)
        else:
            output = pipeline.run(
                rgba_image, None, args.seed, with_mesh_postprocess=True,
                with_texture_baking=True, use_vertex_color=False,
            )
        output_path = args.out_dir / f"official_pbr_{iteration:02d}.glb"
        output["glb"].export(output_path)
        output["glb"].visual.material.baseColorTexture.save(
            output_path.with_suffix(".base_color.png"))
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if iteration >= args.warmup:
            durations.append(elapsed_ms)
            outputs.append(str(output_path.resolve()))
            if "reference_decode_residency" in output:
                residency_measurements.append(output["reference_decode_residency"])
        pose = {key: output[key].detach().cpu().tolist()
                for key in ("rotation", "translation", "scale")
                if key in output and torch.is_tensor(output[key])}
        output_path.with_suffix(".pose.json").write_text(
            json.dumps(pose, indent=2) + "\n", encoding="utf-8")
        del output

    stats = quantiles(durations)
    report = {
        "schema": "sam3d.official-full-e2e.v1",
        "decode_residency_measurements": residency_measurements,
        "timer_contract": {
            "id": "sam3d.hot-session.image-mask-to-textured-pbr-glb.v1",
            "kind": "hot-session",
            "includes": [
                "streamed model transfers", "native official preprocessing", "MoGe", "SS", "SLat",
                "Gaussian and mesh decode", "mesh cleanup", "xatlas UV", "100 Gaussian observations",
                "2500-step Adam/TV bake", "Telea", "GLB export and file close", "base-color PNG export",
            ],
            "excludes": ["pipeline construction", "input image/mask disk decode", "external comparison rendering"],
        },
        "image": str(args.image.resolve()),
        "mask": str(mask.resolve()),
        "seed": args.seed,
        "reference_manifest": reference_manifest(
            pipeline, args.config, args.stream_weights, args.fp32_weights),
        "weight_residency": "streamed" if args.stream_weights else "resident",
        "module_dtypes": dtypes,
        "pipeline_initialization_ms_excluded": initialization_ms,
        "warmup": args.warmup,
        "iters": args.iters,
        "latency_ms": stats,
        "outputs": outputs,
        "gpu_exclusivity": gpu_exclusivity,
    }
    payload = json.dumps(report, indent=2)
    print(payload)
    (args.json_out or args.out_dir / "official_full_e2e.json").write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
