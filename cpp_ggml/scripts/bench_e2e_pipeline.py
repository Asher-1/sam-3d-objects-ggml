#!/usr/bin/env python3
"""PyTorch end-to-end reference benchmark for the ggml integration.

Runs the official SAM 3D Objects pipeline (image + mask -> 3D Gaussians) and
records:
  - end-to-end latency rows (appended to cpp_ggml/benchmarks/e2e_comparison/)
  - the real sparse-structure latent produced by ss_generator, written as a
    SAMT tensor for the C++ ss_decoder parity check (replaces the synthetic
    latent of the earlier reports)
  - the PyTorch ss_decoder occupancy for the same latent
  - the gaussian splat (.ply) and an orbit-render .gif/.png of the result

Usage (repo root):
    python cpp_ggml/scripts/bench_e2e_pipeline.py
"""
import argparse
import json
import os
import struct
import sys
import time

# reduce fragmentation before torch initializes CUDA (12 GB GPU vs the
# official 32 GB fp32 footprint; we run f16 weights, see --fp32-weights)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "notebook"))
os.chdir(REPO_ROOT)
# notebook/inference.py reads CONDA_PREFIX while importing.  Derive it from
# the selected interpreter so this benchmark also works when callers invoke
# the environment's Python directly instead of activating a shell first.
os.environ.setdefault("CONDA_PREFIX", sys.prefix)
os.environ.setdefault("CUDA_HOME", os.environ["CONDA_PREFIX"])
# sam3d_objects.init is not part of the open-source release; the official
# notebook skips it the same way (notebook/inference.py sets this on import).
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

from gpu_exclusivity import require_exclusive_gpu

# This script imports the official pipeline at module load, and that import can
# initialize CUDA or load checkpoints. Reject a contaminated timing environment
# before either operation. main() repeats the check immediately before timing so
# a compute client that appears during import is also detected.
if "--require-exclusive-gpu" in sys.argv:
    require_exclusive_gpu("cuda")

import imageio
import numpy as np
import torch

from inference import (  # noqa: E402  (notebook helpers)
    Inference,
    load_image,
    load_single_mask,
    make_scene,
    ready_gaussian_for_video_rendering,
    render_video,
)

SAMT_MAGIC = b"SAMT"
GGML_TYPE_F32 = 0


def write_samt_f32(path, ne, data):
    """ne in ggml order (ne[0] fastest); data is the C-contiguous buffer."""
    data = np.ascontiguousarray(data, dtype="<f4")
    with open(path, "wb") as f:
        f.write(SAMT_MAGIC)
        f.write(struct.pack("<i", len(ne)))
        f.write(struct.pack(f"<{len(ne)}q", *ne))
        f.write(struct.pack("<i", GGML_TYPE_F32))
        f.write(data.tobytes())


def stats(values):
    s = sorted(values)
    return {"mean": sum(values) / len(values), "min": s[0],
            "p50": s[len(s) // 2], "max": s[-1]}


def move_module(module, device) -> None:
    if module is not None:
        module.to(device)


def run_streamed_gaussian(pipeline, rgba_image, seed: int):
    """Run the official Gaussian path while moving inactive modules to CPU.

    The resident official pipeline requires more than 12 GiB before MoGe's
    working buffers are considered. This function preserves its stage order
    and calls, but keeps only the data-dependent stage on CUDA. Its measured
    latency deliberately includes every transfer.
    """
    device = pipeline.device
    models = pipeline.models
    conditioners = pipeline.condition_embedders

    for name, module in models.items():
        if name not in {"ss_generator", "ss_decoder"}:
            move_module(module, "cpu")
    move_module(conditioners["slat_condition_embedder"], "cpu")
    torch.cuda.empty_cache()

    image = pipeline.merge_image_and_mask(rgba_image, None)
    pointmap_dict = pipeline.compute_pointmap(image)
    pointmap = pointmap_dict["pointmap"]
    ss_input = pipeline.preprocess_image(
        image, pipeline.ss_preprocessor, pointmap=pointmap
    )
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
    move_module(pipeline.depth_model.model, "cpu")
    torch.cuda.empty_cache()

    move_module(models["slat_generator"], device)
    move_module(models["slat_decoder_gs"], device)
    move_module(conditioners["slat_condition_embedder"], device)
    slat = pipeline.sample_slat(slat_input, coords)
    decoder_dtype = next(models["slat_decoder_gs"].parameters()).dtype
    slat = slat.to(dtype=decoder_dtype)
    outputs = pipeline.decode_slat(slat, ["gaussian"])
    outputs = pipeline.postprocess_slat_output(
        outputs, with_mesh_postprocess=False, with_texture_baking=False,
        use_vertex_color=True,
    )
    return {**ss_output, **outputs}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default="notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
    ap.add_argument("--mask-dir", default="notebook/images/shutterstock_stylish_kidsroom_1640806567")
    ap.add_argument("--mask-index", type=int, default=14)
    ap.add_argument("--out-dir", default="cpp_ggml/benchmarks/data/e2e")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--json", default="cpp_ggml/benchmarks/e2e_comparison/pytorch_latency.jsonl")
    ap.add_argument("--config", default="checkpoints/hf/pipeline.yaml")
    ap.add_argument("--stream-weights", action=argparse.BooleanOptionalAction, default=True,
                    help="move inactive official models to CPU between stages; required on 12 GiB GPUs")
    ap.add_argument("--render", action="store_true",
                    help="also render the PyTorch result with gsplat after recording metrics")
    ap.add_argument("--fp32-weights", action="store_true",
                    help="keep official fp32 weights (needs ~24+ GB VRAM); "
                         "default converts all pipeline weights to float16 "
                         "to fit a 12 GB GPU")
    ap.add_argument("--require-exclusive-gpu", action="store_true",
                    help="reject a timing run while another NVIDIA compute client is active")
    args = ap.parse_args()
    gpu_exclusivity = (require_exclusive_gpu("cuda") if args.require_exclusive_gpu else
                       {"required": False, "checked": False, "active_compute_processes": []})

    print(f"[e2e] loading pipeline from {args.config} (compile=False)")
    t0 = time.perf_counter()
    if not args.fp32_weights:
        # The official loader transfers each checkpoint to CUDA before it
        # returns.  Convert while the model is still on CPU, otherwise all
        # F32 checkpoints briefly coexist on a 12 GiB GPU and OOM before a
        # later warmup hook has a chance to cast them.
        from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
        _orig_load = InferencePipeline.instantiate_and_load_from_pretrained

        def _load_f16(self, *load_args, **load_kwargs):
            target_device = load_kwargs.pop("device", "cuda")
            model = _orig_load(self, *load_args, device="cpu", **load_kwargs)
            return model.to(dtype=torch.float16).to(target_device)

        InferencePipeline.instantiate_and_load_from_pretrained = _load_f16

    inference = Inference(args.config, compile=False)
    print(f"[e2e] pipeline loaded in {time.perf_counter() - t0:.1f} s")

    # hook the torch ss_decoder: capture the real ss_generator latent (the
    # input) and the fp32 occupancy logits (the output) plus its wall time
    cap = {}
    ss_decoder = inference._pipeline.models["ss_decoder"]
    orig_forward = ss_decoder.forward

    def hooked_forward(t):
        t0 = time.perf_counter()
        out = orig_forward(t)
        cap["latent"] = t.detach().float().cpu().numpy()   # (1, 8, 16, 16, 16)
        cap["occ"] = out.detach().float().cpu().numpy()    # (1, 1, 64, 64, 64)
        cap["dec_ms"] = (time.perf_counter() - t0) * 1e3
        return out

    ss_decoder.forward = hooked_forward

    image = load_image(args.image)
    mask = load_single_mask(args.mask_dir, index=args.mask_index)
    print(f"[e2e] image {image.shape} mask-index {args.mask_index}")

    times = []
    for i in range(args.warmup + args.iters):
        t0 = time.perf_counter()
        if args.stream_weights:
            rgba_image = inference.merge_mask_to_rgba(image, mask)
            output = run_streamed_gaussian(inference._pipeline, rgba_image, seed=42)
        else:
            output = inference(image, mask, seed=42)
        dt = (time.perf_counter() - t0) * 1e3
        tag = "warmup" if i < args.warmup else "timed"
        print(f"[e2e] iter {i} ({tag}): {dt:.0f} ms (ss_decoder {cap.get('dec_ms', float('nan')):.0f} ms)")
        if i >= args.warmup:
            times.append(dt)

    out_dir = os.path.join(REPO_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # artifacts for the C++ comparison + visual report
    lat = cap["latent"][0]                          # (8, 16, 16, 16) C,D,H,W
    occ = cap["occ"][0]
    if occ.ndim == 4:
        occ = occ[0]                                # (64, 64, 64)
    lat_path = os.path.join(out_dir, "ss_latent_real.samt")
    occ_path = os.path.join(out_dir, "ss_occ_torch.samt")
    write_samt_f32(lat_path, [lat.shape[3], lat.shape[2], lat.shape[1], lat.shape[0]], lat)
    write_samt_f32(occ_path, list(occ.shape[::-1]), occ)
    print(f"[e2e] wrote {lat_path} and {occ_path}")
    print(f"[e2e] torch occupancy range [{occ.min():.2f}, {occ.max():.2f}], "
          f"{int((occ > 0).sum())} voxels > 0")

    ply_path = os.path.join(out_dir, "output_gs.ply")
    output["gs"].save_ply(ply_path)
    print(f"[e2e] wrote {ply_path}")

    s = stats(times)
    row = {
        "component": "pt_e2e_pipeline",
        "model": "sam-3d-objects (official pipeline)",
        "dtype": "float16",
        "backend": "CUDA",
        "device": torch.cuda.get_device_name(0),
        "image": args.image,
        "mask_index": args.mask_index,
        "warmup": args.warmup,
        "iters": len(times),
        "e2e_ms_mean": round(s["mean"], 1),
        "e2e_ms_min": round(s["min"], 1),
        "e2e_ms_p50": round(s["p50"], 1),
        "e2e_ms_max": round(s["max"], 1),
        "ss_decoder_ms_mean": round(cap.get("dec_ms", 0.0), 1),
        "weight_residency": "streamed" if args.stream_weights else "resident",
        "gpu_exclusivity": gpu_exclusivity,
    }
    jsonl_path = os.path.join(REPO_ROOT, args.json)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[e2e] appended {jsonl_path}: {row}")

    if not args.render:
        return

    scene_gs = ready_gaussian_for_video_rendering(make_scene(output))
    video = render_video(scene_gs, r=1, fov=60, pitch_deg=15,
                         yaw_start_deg=-45, resolution=512)["color"]
    gif_path = os.path.join(out_dir, "render_orbit.gif")
    imageio.mimsave(gif_path, video, format="GIF", duration=1000 / 30, loop=0)
    imageio.imwrite(os.path.join(out_dir, "render_view.png"), video[len(video) // 2])
    print(f"[e2e] wrote {gif_path} ({len(video)} frames)")

if __name__ == "__main__":
    main()
