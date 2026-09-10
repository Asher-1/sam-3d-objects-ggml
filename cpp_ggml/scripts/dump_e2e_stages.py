#!/usr/bin/env python3
"""Dump per-stage SAM 3D Objects pipeline tensors as end-to-end references.

Runs the official pipeline in three GPU-memory-bounded stages so the whole
thing fits a 12 GB card (official weights are fp32; generators/embedders are
materialized in bf16, decoders stay fp32 for precise parity):

  stage 0  MoGe pointmap (image+mask -> pointmap, intrinsics)
  stage 1  sparse structure: condition embedders -> ShortCut sampling ->
           ss_decoder occupancy -> coords (+downsample) -> pose decoding
  stage 2  structured latent: slat condition embedder -> flow sampling ->
           slat decoders (gaussian / gaussian_4 / mesh) -> PLY

Every dump is written to <out-dir> as SAMT (.samt) or .npz files plus a
manifest.json describing shapes; the C++ side replays stage by stage and
compares against these references.

Usage (repo root, sam3d-objects env):
    python cpp_ggml/scripts/dump_e2e_stages.py \
        --image notebook/images/.../image.png --mask-dir ... --mask-index 14
"""
import argparse
import contextlib
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
import types

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# The official notebook accesses CONDA_PREFIX directly.  A documented
# interpreter path invocation does not populate it, so derive the active
# environment from that interpreter before importing notebook/inference.py.
if "CONDA_PREFIX" not in os.environ:
    interpreter_dir = os.path.dirname(os.path.abspath(sys.executable))
    os.environ["CONDA_PREFIX"] = os.path.dirname(interpreter_dir)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "notebook"))
os.chdir(REPO_ROOT)
# sam3d_objects.init is not part of the open-source release; the official
# notebook skips it the same way (notebook/inference.py sets this on import).
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import numpy as np  # noqa: E402
import torch  # noqa: E402

SAMT_MAGIC = b"SAMT"
GGML_TYPE_F32 = 0


def sha256_file(path):
    """Return a content digest without retaining a potentially large file."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path):
    """Record the exact file identity used by a reference capture."""
    absolute = os.path.abspath(path)
    if not os.path.isfile(absolute):
        return {"path": absolute, "status": "missing"}
    return {"path": absolute, "sha256": sha256_file(absolute)}


def git_revision():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def cuda_provenance():
    if not torch.cuda.is_available():
        return {"available": False}
    properties = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": properties.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def write_samt_f32(path, ne, data):
    """ne in ggml order (ne[0] fastest); data is the C-contiguous buffer."""
    data = np.ascontiguousarray(data, dtype="<f4")
    with open(path, "wb") as f:
        f.write(SAMT_MAGIC)
        f.write(struct.pack("<i", len(ne)))
        f.write(struct.pack(f"<{len(ne)}q", *ne))
        f.write(struct.pack("<i", GGML_TYPE_F32))
        f.write(data.tobytes())


def write_samt_i32(path, ne, data):
    data = np.ascontiguousarray(data, dtype="<i4")
    with open(path, "wb") as f:
        f.write(SAMT_MAGIC)
        f.write(struct.pack("<i", len(ne)))
        f.write(struct.pack(f"<{len(ne)}q", *ne))
        f.write(struct.pack("<i", 26))  # GGML_TYPE_I32 (vendored ggml)
        f.write(data.tobytes())


class Dumper:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        if os.path.exists(out_dir) and os.listdir(out_dir):
            raise ValueError(
                f"refusing to overwrite non-empty official reference directory: {out_dir}"
            )
        os.makedirs(out_dir, exist_ok=True)
        self.manifest = {}

    def t(self, name, tensor):
        """Dump a torch tensor (any dtype/device) as f32 SAMT."""
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().float().cpu().numpy()
        else:
            arr = np.asarray(tensor, dtype=np.float32)
        ne = list(arr.shape)[::-1]  # ggml order
        path = os.path.join(self.out_dir, f"{name}.samt")
        write_samt_f32(path, ne, arr)
        self.manifest[name] = {"file": f"{name}.samt", "shape": list(arr.shape),
                               "min": float(arr.min()), "max": float(arr.max())}
        print(f"[dump] {name}: shape={list(arr.shape)} "
              f"range=[{arr.min():.4g}, {arr.max():.4g}]")

    def i32(self, name, tensor):
        arr = tensor.detach().int().cpu().numpy()
        ne = list(arr.shape)[::-1]
        path = os.path.join(self.out_dir, f"{name}.samt")
        write_samt_i32(path, ne, arr)
        self.manifest[name] = {"file": f"{name}.samt", "shape": list(arr.shape),
                               "dtype": "i32"}
        print(f"[dump] {name}: shape={list(arr.shape)} (i32)")

    def meta(self, key, value):
        self.manifest[key] = value
        print(f"[meta] {key} = {value}")

    def save(self):
        with open(os.path.join(self.out_dir, "manifest.json"), "w") as f:
            json.dump(self.manifest, f, indent=2)
        print(f"[dump] manifest -> {self.out_dir}/manifest.json")


def basename_key(ckpt_path):
    return os.path.basename(ckpt_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default="notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
    ap.add_argument("--mask-dir", default="notebook/images/shutterstock_stylish_kidsroom_1640806567")
    ap.add_argument("--mask-index", type=int, default=14)
    ap.add_argument(
        "--out-dir",
        default="/tmp/sam3d-official-e2e",
        help="empty output directory; defaults to a disposable /tmp reference",
    )
    ap.add_argument("--config", default="checkpoints/hf/pipeline.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--sampling-seed-mode",
        choices=("pipeline", "stage-isolated"),
        default="pipeline",
        help=("pipeline preserves one CUDA Philox state across SS and SLat, matching "
              "InferencePipeline.run; stage-isolated preserves the historical diagnostic mode"),
    )
    ap.add_argument(
        "--ss-dtype",
        choices=("bf16", "f32"),
        default="bf16",
        help=("materialize the SS generator and SS condition embedder in this dtype; "
              "use f32 when comparing against an F32 GGUF, bf16 for the 12 GB staged oracle"),
    )
    ap.add_argument("--skip-render", action="store_true", default=True)
    ap.add_argument(
        "--dump-mesh-decoder-reference",
        action="store_true",
        help=("save official sparse mesh-decoder stage tensors and coordinates; "
              "the final 101-channel 256^3 feature tensor is large (about 700 MB)"),
    )
    ap.add_argument(
        "--dump-mesh-decoder-blocks",
        action="store_true",
        help=("with --dump-mesh-decoder-reference, also save each of the 12 official "
              "transformer block outputs to locate the first numerical divergence"),
    )
    ap.add_argument(
        "--dump-ss-trajectory",
        action="store_true",
        help=("capture every official SS Euler latent directly from the staged "
              "generator used by this oracle"),
    )
    ap.add_argument(
        "--stop-after-ss",
        action="store_true",
        help="stop after the SS stage; requires --dump-ss-trajectory",
    )
    ap.add_argument(
        "--dump-mesh-decoder-attention-block",
        type=int,
        choices=range(12),
        metavar="BLOCK",
        help=("with --dump-mesh-decoder-reference, save the residual output after "
              "attention in one transformer block; use it to separate attention "
              "error from the following MLP"),
    )
    ap.add_argument(
        "--dump-mesh-decoder-internals-block",
        type=int,
        choices=range(12),
        metavar="BLOCK",
        help=("with --dump-mesh-decoder-reference, save the APE, LayerNorm, QKV, "
              "attention-value, output-projection, and residual boundaries for one "
              "transformer block"),
    )
    ap.add_argument(
        "--dump-final-pbr-reference",
        action="store_true",
        help=("run the official mesh postprocess and 1024px optimized texture bake, "
              "then save the final PBR GLB and replay metadata; this is a reference-only "
              "operation and is never used by the native runtime"),
    )
    ap.add_argument(
        "--dump-bake-observations",
        action="store_true",
        help=("with --dump-final-pbr-reference, additionally save every official 1024px "
              "Gaussian observation; the artifact is large"),
    )
    args = ap.parse_args()
    if args.dump_bake_observations and not args.dump_final_pbr_reference:
        ap.error("--dump-bake-observations requires --dump-final-pbr-reference")
    if (args.dump_mesh_decoder_attention_block is not None and
            args.dump_mesh_decoder_internals_block is not None):
        ap.error("--dump-mesh-decoder-attention-block and "
                 "--dump-mesh-decoder-internals-block are mutually exclusive")
    if (args.dump_mesh_decoder_blocks or args.dump_mesh_decoder_attention_block is not None or
            args.dump_mesh_decoder_internals_block is not None):
        args.dump_mesh_decoder_reference = True
    if args.stop_after_ss and not args.dump_ss_trajectory:
        ap.error("--stop-after-ss requires --dump-ss-trajectory")

    from loguru import logger
    logger.remove()  # quiet the per-step info spam

    from inference import Inference, load_image, load_single_mask  # noqa: E402
    import moge  # noqa: E402
    from sam3d_objects.pipeline import inference_pipeline as inference_pipeline_module  # noqa: E402

    output_dir = os.path.abspath(args.out_dir)
    dmp = Dumper(output_dir)
    selected_mask_path = os.path.join(args.mask_dir, f"{args.mask_index}.png")
    dmp.meta("reference_provenance", {
        "schema": "sam3d.official-e2e-reference.v2",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sampling_seed_mode": args.sampling_seed_mode,
        "ss_dtype": args.ss_dtype,
        "seed": args.seed,
        "repository_git_revision": git_revision(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda": cuda_provenance(),
        "inputs": {
            "image": file_provenance(args.image),
            "selected_mask": file_provenance(selected_mask_path),
            "pipeline_config": file_provenance(args.config),
        },
        "sources": {
            "dump_script": file_provenance(__file__),
            "inference_pipeline": file_provenance(inference_pipeline_module.__file__),
            "moge": file_provenance(moge.__file__),
        },
    })

    print("[e2e] loading pipeline (compile=False, staged loading)...")
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Hook model loading: everything lands in CPU RAM first; generators and
    # condition embedders are converted to bf16, decoders stay fp32. Each
    # stage then moves only the models it needs onto the GPU.
    # ------------------------------------------------------------------
    from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
    orig_load = InferencePipeline.instantiate_and_load_from_pretrained

    def staged_load(self, config, ckpt_path, state_dict_fn=None,
                    state_dict_key="state_dict", device="cuda"):
        model = orig_load(self, config, ckpt_path, state_dict_fn=state_dict_fn,
                          state_dict_key=state_dict_key, device="cpu")
        base = basename_key(ckpt_path)
        # Keep the loader's post-init dtype layout: use_fp16 models internally
        # convert part of their blocks to fp16 and cast activations at the
        # boundary, so any blanket .to(dtype) here breaks the forward pass.
        # Generators/embedders get uniformized only where spconv demands it:
        # slat_generator runs fp16 (spconv rejects bfloat16 and OOMs in fp32
        # on a 12 GB card); the ss-side DINO/PointPatch embedders run bf16.
        if base.startswith("ss_decoder") or base.startswith("slat_decoder"):
            dtxt = "native"
        elif base.startswith("slat_generator"):
            model = model.to(dtype=torch.float16)
            dtxt = "fp16"
        else:
            ss_dtype = torch.float32 if args.ss_dtype == "f32" else torch.bfloat16
            model = model.to(dtype=ss_dtype)
            dtxt = args.ss_dtype
        print(f"[load] {base} -> CPU ({dtxt}), "
              f"params={sum(p.numel() for p in model.parameters())/1e6:.0f}M")
        return model

    InferencePipeline.instantiate_and_load_from_pretrained = staged_load

    # MoGe (fp32 ViT-L, 1.3 GB) is part of the pipeline config; replace it
    # with a placeholder and run it separately in stage 0.
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    moge_cfg = cfg.depth_model
    cfg.depth_model = None
    # must live in the same dir as pipeline.yaml: ckpt paths resolve
    # relative to the config file's directory (workspace_dir)
    patched_cfg = os.path.join(os.path.dirname(args.config), "pipeline_nomoge.yaml")
    OmegaConf.save(cfg, patched_cfg)

    inference = Inference(patched_cfg, compile=False)
    pipeline = inference._pipeline
    print(f"[e2e] pipeline objects loaded in {time.perf_counter()-t0:.1f} s "
          f"(models still on CPU)")

    # image + mask input
    image = load_image(args.image)
    mask = load_single_mask(args.mask_dir, index=args.mask_index)
    print(f"[e2e] image {image.shape} mask-index {args.mask_index}")
    image_np = np.array(image) if not isinstance(image, np.ndarray) else image
    dmp.meta("image_shape", list(image_np.shape))
    rgba = image_np if image_np.dtype == np.uint8 else (np.array(image) if False else None)
    if image_np.dtype != np.uint8:
        rgba = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
    else:
        rgba = image_np
    dmp.t("input_image_rgba", rgba.astype(np.float32))

    merged = inference.merge_mask_to_rgba(np.array(image), np.array(mask))
    # load_single_mask yields {0,1}; merge_mask_to_rgba scales to {0,255} so
    # the alpha-channel-derived mask survives the /255 in image_to_float

    # ==================================================================
    # stage 0: MoGe pointmap
    # ==================================================================
    print("\n================ stage 0: MoGe pointmap ================")
    t0 = time.perf_counter()
    from hydra.utils import instantiate
    depth_model = instantiate(moge_cfg)  # MoGe wrapper handles device itself
    pipeline.depth_model = depth_model

    image_t = torch.from_numpy(pipeline.image_to_float(merged))
    loaded_image = image_t.permute(2, 0, 1).contiguous()[:3]  # (3, H, W)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=pipeline.dtype):
        moge_out = depth_model(loaded_image.to("cuda"))
    pm = moge_out["pointmaps"]  # (H, W, 3) camera space
    # replicate compute_pointmap's pytorch3d camera transform
    from sam3d_objects.pipeline.inference_pipeline_pointmap import camera_to_pytorch3d_camera
    from pytorch3d.transforms import Transform3d
    xform = Transform3d().rotate(camera_to_pytorch3d_camera(device="cuda").rotation).to("cuda")
    pts = xform.transform_points(pm)  # (H, W, 3)
    dmp.t("pointmap_raw", pts.permute(2, 0, 1))  # 3,H,W
    dmp.t("moge_pts_color", loaded_image)
    intr = moge_out.get("intrinsics", None)
    if intr is not None:
        dmp.t("intrinsics", intr)
    dmp.meta("moge_keys", sorted(list(moge_out.keys())))

    pointmap = pts.permute(2, 0, 1)  # 3,H,W as compute_pointmap returns
    del depth_model, moge_out
    pipeline.depth_model = None
    torch.cuda.empty_cache()
    print(f"[stage0] done in {time.perf_counter()-t0:.1f} s")

    # ==================================================================
    # stage 1: sparse structure
    # ==================================================================
    print("\n================ stage 1: sparse structure ================")
    t0 = time.perf_counter()

    # stage hooks on the embedders to capture condition tokens
    cond_tokens_cap = {}

    def hook_embedder(inst, tag):
        orig_fwd = inst.forward

        def wrapped(*a, **k):
            out = orig_fwd(*a, **k)
            cond_tokens_cap[tag] = out.detach().float().cpu()
            return out

        inst.forward = wrapped

    # models live in pipeline.models / pipeline.condition_embedders (CPU);
    # move only the ss-stage ones to GPU
    ss_gen = pipeline.models["ss_generator"].to("cuda")
    ss_cond = pipeline.condition_embedders["ss_condition_embedder"].to("cuda")
    hook_embedder(ss_cond, "ss_cond_tokens")
    ss_dec = pipeline.models["ss_decoder"].to("cuda")

    # hook noise generation + per-step velocities
    noise_cap = {}
    ss_trajectory = {}
    orig_gen_noise = type(ss_gen)._generate_noise

    def gen_noise_cap(self, x_shape, x_device):
        out = orig_gen_noise(self, x_shape, x_device)
        noise_cap["x0"] = {k: v.detach().float().cpu() for k, v in out.items()}
        noise_cap["keys"] = list(out.keys())
        return out

    type(ss_gen)._generate_noise = gen_noise_cap
    if args.dump_ss_trajectory:
        original_generate_iter = ss_gen.generate_iter

        def capture_ss_generate_iter(_self, *generate_args, **generate_kwargs):
            for step, (timestamp, latent, metadata) in enumerate(
                    original_generate_iter(*generate_args, **generate_kwargs), start=1):
                ss_trajectory[step] = {
                    name: value.detach().float().cpu()
                    for name, value in latent.items()
                }
                yield timestamp, latent, metadata

        ss_gen.generate_iter = types.MethodType(capture_ss_generate_iter, ss_gen)

    # hook condition inputs going INTO the generator (via get_condition_input)
    ss_input_dict = pipeline.preprocess_image(merged, pipeline.ss_preprocessor,
                                              pointmap=pointmap)
    for k, v in ss_input_dict.items():
        if isinstance(v, torch.Tensor):
            dmp.t(f"ss_input_{k}", v)
        else:
            dmp.meta(f"ss_input_{k}", str(v))

    slat_input_dict = pipeline.preprocess_image(merged, pipeline.slat_preprocessor)
    # slat inputs dumped later (stage 2 uses the same image); keep in RAM
    # torch.save(slat_input_dict, os.path.join(dmp.out_dir, "slat_input.pt"))

    # InferencePipeline.run seeds once before sparse structure sampling. The
    # default preserves that same Philox stream for the subsequent SLat draw.
    torch.manual_seed(args.seed)
    # ``sample_sparse_structure`` owns the autocast context.  A true F32 oracle
    # therefore has to change the pipeline's stage dtype before entering it;
    # CUDA autocast rejects/ignores float32 and executes the graph in F32.
    if args.ss_dtype == "f32":
        pipeline.shape_model_dtype = torch.float32
    ss_return = pipeline.sample_sparse_structure(ss_input_dict)
    type(ss_gen)._generate_noise = orig_gen_noise
    if args.dump_ss_trajectory:
        del ss_gen.generate_iter

    dmp.meta("ss_noise_keys", noise_cap.get("keys", []))
    for k, v in noise_cap.get("x0", {}).items():
        dmp.t(f"ss_x0_{k}", v)
    for step, latent in sorted(ss_trajectory.items()):
        for name, value in latent.items():
            dmp.t(f"ss_torch_x{step:03d}_{name}", value)
    if args.dump_ss_trajectory:
        dmp.meta("ss_trajectory_steps", sorted(ss_trajectory))

    if "ss_cond_tokens" in cond_tokens_cap:
        dmp.t("ss_cond_tokens", cond_tokens_cap["ss_cond_tokens"])

    shape_latent = ss_return["shape"]
    dmp.t("ss_shape_latent", shape_latent)
    for k in ("6drotation_normalized", "scale", "translation", "translation_scale"):
        if k in ss_return:
            dmp.t(f"ss_pose_latent_{k}", ss_return[k])
    dmp.meta("ss_return_keys", sorted(ss_return.keys()))

    # ss_decoder I/O
    if args.ss_dtype == "f32":
        ss_autocast = contextlib.nullcontext()
    else:
        ss_autocast = torch.autocast(device_type="cuda", dtype=pipeline.shape_model_dtype)
    with torch.no_grad(), ss_autocast:
        lat_in = shape_latent.permute(0, 2, 1).contiguous().view(
            shape_latent.shape[0], 8, 16, 16, 16)
        occ = ss_dec(lat_in)
    dmp.t("ss_latent_dec_input", lat_in)
    dmp.t("ss_occ", occ)

    coords_original = ss_return["coords_original"]
    coords = ss_return["coords"]
    dmp.i32("coords_original", coords_original)
    dmp.i32("coords", coords)
    dmp.meta("downsample_factor", float(ss_return["downsample_factor"]))

    # pose decoding (as run() does)
    pointmap_scale = ss_input_dict.get("pointmap_scale", None)
    pointmap_shift = ss_input_dict.get("pointmap_shift", None)
    ss_return.update(pipeline.pose_decoder(ss_return,
                                           scene_scale=pointmap_scale,
                                           scene_shift=pointmap_shift))
    for k in ("rotation", "translation", "scale"):
        if k in ss_return:
            dmp.t(f"pose_{k}", ss_return[k])

    # keep results for stage 2; offload ss models back to CPU
    coords_gpu = coords
    pipeline.models["ss_generator"].to("cpu")
    pipeline.models["ss_decoder"].to("cpu")
    pipeline.condition_embedders["ss_condition_embedder"].to("cpu")
    del ss_gen, ss_dec, ss_cond
    torch.cuda.empty_cache()
    print(f"[stage1] done in {time.perf_counter()-t0:.1f} s; "
          f"coords {tuple(coords.shape)}")
    if args.stop_after_ss:
        dmp.save()
        print("\n[e2e] SS TRAJECTORY COMPLETE")
        return

    # ==================================================================
    # stage 2: structured latent + decoders
    # ==================================================================
    print("\n================ stage 2: structured latent ================")
    t0 = time.perf_counter()

    slat_gen = pipeline.models["slat_generator"].to("cuda")
    slat_cond = pipeline.condition_embedders["slat_condition_embedder"].to("cuda")
    hook_embedder(slat_cond, "slat_cond_tokens")
    gs_dec = pipeline.models["slat_decoder_gs"].to("cuda")
    gs4_dec = pipeline.models["slat_decoder_gs_4"]  # stays on CPU (not decoded)
    mesh_dec = pipeline.models["slat_decoder_mesh"].to("cuda")

    orig_gen_noise = type(slat_gen)._generate_noise

    def gen_noise_cap2(self, x_shape, x_device):
        out = orig_gen_noise(self, x_shape, x_device)
        noise_cap["slat_x0"] = out.detach().float().cpu()
        return out

    type(slat_gen)._generate_noise = gen_noise_cap2

    if args.sampling_seed_mode == "stage-isolated":
        torch.manual_seed(args.seed)
    slat = pipeline.sample_slat(slat_input_dict, coords_gpu)
    type(slat_gen)._generate_noise = orig_gen_noise

    # the generator + its embedder are done; free ~3.5 GB before decoding
    pipeline.models["slat_generator"].to("cpu")
    pipeline.condition_embedders["slat_condition_embedder"].to("cpu")
    del slat_gen, slat_cond
    torch.cuda.empty_cache()

    if "slat_x0" in noise_cap:
        dmp.t("slat_x0", noise_cap["slat_x0"])
    if "slat_cond_tokens" in cond_tokens_cap:
        dmp.t("slat_cond_tokens", cond_tokens_cap["slat_cond_tokens"])
    dmp.t("slat_feats_final", slat.feats)   # already denormalized (*std+mean)
    dmp.i32("slat_coords", slat.coords)

    # Capture sparse tensors at the C++ graph boundaries.  The hooks retain
    # CPU copies only, so they do not change the official forward path nor its
    # GPU memory lifetime.  Every feature tensor is emitted as F32 SAMT, while
    # its recorded values retain the official module's F16 rounding.
    mesh_decoder_stages = {}
    mesh_decoder_hooks = []
    if args.dump_mesh_decoder_reference:
        def capture_mesh_stage(stage_name):
            def hook(_module, _inputs, result):
                if not hasattr(result, "feats") or not hasattr(result, "coords"):
                    raise RuntimeError(f"mesh decoder {stage_name} hook returned no sparse tensor")
                mesh_decoder_stages[stage_name] = (
                    result.feats.detach().float().cpu(),
                    result.coords.detach().int().cpu(),
                )
            return hook

        diagnostic_block = (args.dump_mesh_decoder_internals_block
                            if args.dump_mesh_decoder_internals_block is not None
                            else args.dump_mesh_decoder_attention_block)
        if diagnostic_block is not None:
            attention_block = diagnostic_block
            attention_input = {}

            def capture_sparse_stage(stage_name, sparse):
                if not hasattr(sparse, "feats") or not hasattr(sparse, "coords"):
                    raise RuntimeError(f"mesh decoder {stage_name} is not sparse")
                mesh_decoder_stages[stage_name] = (
                    sparse.feats.detach().float().cpu(),
                    sparse.coords.detach().int().cpu(),
                )

            def capture_feature_stage(stage_name, features):
                sparse = attention_input.get("sparse")
                if sparse is None or not isinstance(features, torch.Tensor):
                    raise RuntimeError(f"mesh decoder {stage_name} has no tensor features")
                mesh_decoder_stages[stage_name] = (
                    features.detach().float().cpu(),
                    sparse.coords.detach().int().cpu(),
                )

            def capture_attention_block_input(_module, inputs):
                sparse = inputs[0]
                if not hasattr(sparse, "feats") or not hasattr(sparse, "coords"):
                    raise RuntimeError("mesh decoder transformer input is not sparse")
                attention_input["sparse"] = sparse
                if args.dump_mesh_decoder_internals_block is not None:
                    capture_sparse_stage(f"block{attention_block}_ape", sparse)

            def capture_norm1(_module, _inputs, result):
                sparse = attention_input.get("sparse")
                if sparse is None or not isinstance(result, torch.Tensor):
                    raise RuntimeError("mesh decoder norm1 hook has no transformer input")
                mesh_decoder_stages[f"block{attention_block}_norm1"] = (
                    result.detach().float().cpu(),
                    sparse.coords.detach().int().cpu(),
                )

            def capture_qkv(_module, _inputs, result):
                capture_feature_stage(f"block{attention_block}_qkv", result)

            def capture_attention_values(_module, inputs):
                capture_feature_stage(f"block{attention_block}_attention_values", inputs[0])

            def capture_attention_out(_module, _inputs, result):
                capture_feature_stage(f"block{attention_block}_attention_out", result)

            def capture_norm2(_module, _inputs, result):
                capture_feature_stage(f"block{attention_block}_norm2", result)

            def capture_mlp_stage(stage_name):
                def hook(_module, _inputs, result):
                    capture_sparse_stage(stage_name, result)
                return hook

            def capture_attention_residual(_module, _inputs, result):
                sparse = attention_input.get("sparse")
                if sparse is None:
                    raise RuntimeError("mesh decoder attention ran before its block pre-hook")
                if not hasattr(result, "feats") or not hasattr(result, "coords"):
                    raise RuntimeError("mesh decoder attention returned no sparse tensor")
                if not torch.equal(sparse.coords, result.coords):
                    raise RuntimeError("mesh decoder attention changed sparse coordinates")
                mesh_decoder_stages[f"block{attention_block}_attention"] = (
                    (sparse.feats + result.feats).detach().float().cpu(),
                    sparse.coords.detach().int().cpu(),
                )

            mesh_decoder_hooks.append(
                mesh_dec.blocks[attention_block].register_forward_pre_hook(
                    capture_attention_block_input
                )
            )
            mesh_decoder_hooks.append(
                mesh_dec.blocks[attention_block].attn.register_forward_hook(
                    capture_attention_residual
                )
            )
            if args.dump_mesh_decoder_internals_block is not None:
                mesh_decoder_hooks.append(
                    mesh_dec.blocks[attention_block].norm1.register_forward_hook(capture_norm1)
                )
                mesh_decoder_hooks.append(
                    mesh_dec.blocks[attention_block].attn.to_qkv.register_forward_hook(capture_qkv)
                )
                mesh_decoder_hooks.append(
                    mesh_dec.blocks[attention_block].attn.to_out.register_forward_pre_hook(
                        capture_attention_values
                    )
                )
                mesh_decoder_hooks.append(
                    mesh_dec.blocks[attention_block].attn.to_out.register_forward_hook(
                        capture_attention_out
                    )
                )
                mesh_decoder_hooks.append(
                    mesh_dec.blocks[attention_block].norm2.register_forward_hook(capture_norm2)
                )
                mesh_decoder_hooks.append(
                    mesh_dec.blocks[attention_block].mlp.mlp[0].register_forward_hook(
                        capture_mlp_stage(f"block{attention_block}_mlp0")
                    )
                )
                mesh_decoder_hooks.append(
                    mesh_dec.blocks[attention_block].mlp.mlp[1].register_forward_hook(
                        capture_mlp_stage(f"block{attention_block}_gelu")
                    )
                )
                mesh_decoder_hooks.append(
                    mesh_dec.blocks[attention_block].mlp.mlp[2].register_forward_hook(
                        capture_mlp_stage(f"block{attention_block}_mlp2")
                    )
                )

        mesh_stage_modules = [("input_layer", mesh_dec.input_layer)]
        if args.dump_mesh_decoder_blocks:
            mesh_stage_modules.extend(
                (f"block{index}", block) for index, block in enumerate(mesh_dec.blocks)
            )
        else:
            mesh_stage_modules.append(("block11", mesh_dec.blocks[-1]))
        mesh_stage_modules.extend((
            ("upsample0", mesh_dec.upsample[0]),
            ("upsample1", mesh_dec.upsample[1]),
            ("raw", mesh_dec.out_layer),
        ))
        for stage_name, module in mesh_stage_modules:
            mesh_decoder_hooks.append(module.register_forward_hook(capture_mesh_stage(stage_name)))

    try:
        outputs = pipeline.decode_slat(slat, ["gaussian", "mesh"])
    finally:
        for hook in mesh_decoder_hooks:
            hook.remove()

    if args.dump_mesh_decoder_reference:
        required_mesh_stages = ("input_layer", "block11", "upsample0", "upsample1", "raw")
        dumped_mesh_stages = (
            ("input_layer", *[f"block{index}" for index in range(len(mesh_dec.blocks))],
             "upsample0", "upsample1", "raw")
            if args.dump_mesh_decoder_blocks else required_mesh_stages
        )
        diagnostic_stages = ()
        if args.dump_mesh_decoder_internals_block is not None:
            diagnostic_prefix = f"block{args.dump_mesh_decoder_internals_block}"
            diagnostic_stages = (
                f"{diagnostic_prefix}_ape", f"{diagnostic_prefix}_norm1",
                f"{diagnostic_prefix}_qkv", f"{diagnostic_prefix}_attention_values",
                f"{diagnostic_prefix}_attention_out", f"{diagnostic_prefix}_attention",
                f"{diagnostic_prefix}_norm2", f"{diagnostic_prefix}_mlp0",
                f"{diagnostic_prefix}_gelu", f"{diagnostic_prefix}_mlp2",
            )
        elif args.dump_mesh_decoder_attention_block is not None:
            diagnostic_stages = (f"block{args.dump_mesh_decoder_attention_block}_attention",)
        dumped_mesh_stages = (*dumped_mesh_stages, *diagnostic_stages)
        required_mesh_stages = (*required_mesh_stages, *diagnostic_stages)
        missing_mesh_stages = [name for name in required_mesh_stages if name not in mesh_decoder_stages]
        if missing_mesh_stages:
            raise RuntimeError(f"official mesh decoder hooks did not run: {missing_mesh_stages}")
        for stage_name in dumped_mesh_stages:
            features, coordinates = mesh_decoder_stages[stage_name]
            dmp.t(f"mesh_decoder_{stage_name}_features", features)
            dmp.i32(f"mesh_decoder_{stage_name}_coords", coordinates)
        dmp.meta("mesh_decoder_reference", {
            "schema": "sam3d.mesh-decoder-reference.v1",
            "stages": list(dumped_mesh_stages),
            "raw_channels": int(mesh_decoder_stages["raw"][0].shape[1]),
            "subdivision_order": "torch.nonzero(ones(2,2,2)); z-fastest",
        })
    for fmt, out_list in outputs.items():
        if out_list is None:
            continue
        entry = out_list[0]
        dumped_any = False
        if isinstance(entry, dict):
            for k, v in entry.items():
                if isinstance(v, torch.Tensor):
                    dmp.t(f"decode_{fmt}_{k}", v)
                    dumped_any = True
                else:
                    dmp.meta(f"decode_{fmt}_{k}", str(v))
        else:  # custom gaussian / mesh object: sweep public attrs
            for k in sorted(vars(entry).keys()):
                v = getattr(entry, k)
                if isinstance(v, torch.Tensor):
                    dmp.t(f"decode_{fmt}_{k}", v)
                    dumped_any = True
                else:
                    dmp.meta(f"decode_{fmt}_{k}", str(v))
        dmp.meta(f"decode_{fmt}_keys", sorted(list(entry.keys()))
                 if isinstance(entry, dict) else sorted(vars(entry).keys()))
        if not dumped_any:
            print(f"[warn] nothing tensor-like dumped for {fmt}")

    # export gaussian PLY for visual comparison
    try:
        from sam3d_objects.pipeline.inference_pipeline import InferencePipeline as _IP
        gs_obj = outputs["gaussian"][0]
        ply_path = os.path.join(dmp.out_dir, "output_gs.ply")
        gs_obj.save_ply(ply_path)
        dmp.meta("ply", os.path.abspath(ply_path))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] PLY export failed: {e}")

    if args.dump_final_pbr_reference:
        # This deliberately runs the unmodified official postprocess path. It
        # is a reference generator only and is never used by native inference.
        from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils

        asset_dir = os.path.join(dmp.out_dir, "official_pbr_reference")
        os.makedirs(asset_dir, exist_ok=True)
        captured = {}
        official_render_multiview = postprocessing_utils.render_multiview
        official_fill_holes = postprocessing_utils._fill_holes
        official_rasterize = postprocessing_utils.utils3d.torch.rasterize_triangle_faces
        official_meshfix_type = postprocessing_utils._meshfix.PyTMesh
        official_graph_type = postprocessing_utils.igraph.Graph
        official_randint = np.random.randint
        selected_views = []

        def capture_render_multiview(*render_args, **render_kwargs):
            observations, extrinsics, intrinsics = official_render_multiview(
                *render_args, **render_kwargs
            )
            captured["observations"] = observations
            captured["extrinsics"] = extrinsics
            captured["intrinsics"] = intrinsics
            captured["resolution"] = render_kwargs.get(
                "resolution", render_args[1] if len(render_args) > 1 else 512
            )
            captured["nviews"] = render_kwargs.get(
                "nviews", render_args[2] if len(render_args) > 2 else 30
            )
            return observations, extrinsics, intrinsics

        def capture_randint(*randint_args, **randint_kwargs):
            value = official_randint(*randint_args, **randint_kwargs)
            if np.isscalar(value):
                selected_views.append(int(value))
            return value

        class CaptureMeshFix:
            """Delegate MeshFix while freezing the exact pre-repair mincut mesh."""

            def __init__(self):
                self._inner = official_meshfix_type()

            def load_array(self, mesh_vertices, mesh_faces):
                dmp.t("asset_mincut_vertices_zup", mesh_vertices)
                dmp.i32("asset_mincut_faces", torch.from_numpy(mesh_faces))
                captured["pre_repair_saved"] = True
                return self._inner.load_array(mesh_vertices, mesh_faces)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        class CaptureGraph:
            """Delegate igraph while preserving the official mincut side."""

            def __init__(self, *graph_args, **graph_kwargs):
                self._inner = official_graph_type(*graph_args, **graph_kwargs)

            def mincut(self, *mincut_args, **mincut_kwargs):
                cut = self._inner.mincut(*mincut_args, **mincut_kwargs)
                face_count = captured.get("visibility_face_count")
                if face_count is None:
                    raise RuntimeError("official mincut ran outside the visibility capture")
                candidates = sorted(vertex for vertex in cut.partition[0] if vertex < face_count)
                dmp.i32("asset_mincut_candidate_faces", torch.tensor(candidates, dtype=torch.int32))
                captured["mincut_candidate_count"] = len(candidates)
                return cut

            def __getattr__(self, name):
                return getattr(self._inner, name)

        def capture_fill_holes(fill_vertices, fill_faces, *fill_args, **fill_kwargs):
            # Do not recreate _fill_holes here. Recording the face IDs from
            # its own rasterize calls freezes the exact input consumed by the
            # native visibility/mincut regression.
            num_views = int(fill_kwargs.get("num_views", 500))
            captured["visibility_face_count"] = int(fill_faces.shape[0])
            visibility = torch.zeros(
                fill_faces.shape[0], dtype=torch.int32, device=fill_faces.device
            )
            raster_calls = 0

            def capture_rasterize(*raster_args, **raster_kwargs):
                nonlocal raster_calls
                buffers = official_rasterize(*raster_args, **raster_kwargs)
                face_id = buffers["face_id"][0][buffers["mask"][0] > 0.95] - 1
                face_id = torch.unique(face_id).long()
                if face_id.numel() > 0:
                    visibility[face_id] += 1
                raster_calls += 1
                return buffers

            postprocessing_utils.utils3d.torch.rasterize_triangle_faces = capture_rasterize
            try:
                result = official_fill_holes(fill_vertices, fill_faces, *fill_args, **fill_kwargs)
            finally:
                postprocessing_utils.utils3d.torch.rasterize_triangle_faces = official_rasterize
            if raster_calls != num_views:
                raise RuntimeError(
                    f"official _fill_holes rendered {raster_calls} views; expected {num_views}"
                )
            dmp.t("asset_decimated_vertices_zup", fill_vertices)
            dmp.i32("asset_decimated_faces", fill_faces)
            dmp.t("asset_decimated_visibility", visibility.float() / num_views)
            captured["visibility_views"] = num_views
            return result

        postprocessing_utils.render_multiview = capture_render_multiview
        postprocessing_utils._fill_holes = capture_fill_holes
        postprocessing_utils._meshfix.PyTMesh = CaptureMeshFix
        postprocessing_utils.igraph.Graph = CaptureGraph
        np.random.seed(args.seed)
        np.random.randint = capture_randint
        try:
            final_outputs = pipeline.postprocess_slat_output(
                outputs,
                with_mesh_postprocess=True,
                with_texture_baking=True,
                use_vertex_color=False,
            )
        finally:
            postprocessing_utils.render_multiview = official_render_multiview
            postprocessing_utils._fill_holes = official_fill_holes
            postprocessing_utils.utils3d.torch.rasterize_triangle_faces = official_rasterize
            postprocessing_utils._meshfix.PyTMesh = official_meshfix_type
            postprocessing_utils.igraph.Graph = official_graph_type
            np.random.randint = official_randint

        final_mesh = final_outputs["glb"]
        final_glb = os.path.join(asset_dir, "official_pbr.glb")
        final_mesh.export(final_glb)
        dmp.t("asset_postprocess_vertices", final_mesh.vertices)
        dmp.i32("asset_postprocess_faces", torch.from_numpy(final_mesh.faces))
        uvs = getattr(final_mesh.visual, "uv", None)
        if uvs is None:
            raise RuntimeError("official textured GLB unexpectedly has no UV coordinates")
        dmp.t("asset_uv", uvs)

        material = getattr(final_mesh.visual, "material", None)
        texture = getattr(material, "baseColorTexture", None)
        if texture is None:
            raise RuntimeError("official textured GLB unexpectedly has no PBR base-color texture")
        texture_path = os.path.join(asset_dir, "base_color.png")
        texture.convert("RGBA").save(texture_path)

        if "observations" not in captured:
            raise RuntimeError("official texture bake did not call render_multiview")
        if captured.get("visibility_views") != 1000:
            raise RuntimeError("official mesh cleanup did not record the required 1000 visibility views")
        if not captured.get("pre_repair_saved"):
            raise RuntimeError("official mesh cleanup did not record the pre-MeshFix mincut mesh")
        if "mincut_candidate_count" not in captured:
            raise RuntimeError("official mesh cleanup did not record the mincut partition")
        observations = captured["observations"]
        extrinsics = np.stack([item.detach().cpu().numpy() for item in captured["extrinsics"]])
        intrinsics = np.stack([item.detach().cpu().numpy() for item in captured["intrinsics"]])
        # The NPZ remains convenient for Python, while SAMT makes the exact
        # recorded camera sequence consumable by the native renderer without a
        # NumPy/Python runtime dependency.
        dmp.t("bake_extrinsics", extrinsics)
        dmp.t("bake_intrinsics", intrinsics)
        np.savez_compressed(
            os.path.join(asset_dir, "bake_cameras.npz"),
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )
        if args.dump_bake_observations:
            np.savez_compressed(
                os.path.join(asset_dir, "bake_observations.npz"),
                rgb=np.stack(observations),
            )

        observation_hashes = [
            hashlib.sha256(np.ascontiguousarray(observation).tobytes()).hexdigest()
            for observation in observations
        ]
        asset_manifest = {
            "schema": "sam3d.official-pbr-reference.v1",
            "source": "InferencePipeline.postprocess_slat_output",
            "seed": args.seed,
            "reference_provenance": dmp.manifest["reference_provenance"],
            "mesh_postprocess": {
                "simplify_ratio": 0.95,
                "fill_holes": True,
                "fill_holes_resolution": 1024,
                "fill_holes_num_views": 1000,
                "visibility_frequency": "../asset_decimated_visibility.samt",
                "visibility_frequency_views": captured["visibility_views"],
                "mincut_vertices": "../asset_mincut_vertices_zup.samt",
                "mincut_faces": "../asset_mincut_faces.samt",
                "mincut_candidate_faces": "../asset_mincut_candidate_faces.samt",
            },
            "uv": {"parameterizer": "xatlas", "texture_size": 1024},
            "bake": {
                "mode": "opt",
                "renderer": pipeline.rendering_engine,
                "views": captured["nviews"],
                "resolution": captured["resolution"],
                "total_steps": 2500,
                "optimizer": "Adam(beta1=0.5,beta2=0.9,lr=1e-2)",
                "lr_schedule": "cosine(1e-2,1e-5)",
                "lambda_tv": 0.01,
                "selected_views": selected_views,
                "observation_sha256": observation_hashes,
                "observations_saved": args.dump_bake_observations,
            },
            "outputs": {
                "glb": "official_pbr.glb",
                "base_color": "base_color.png",
                "cameras": "bake_cameras.npz",
                "extrinsics_samt": "../bake_extrinsics.samt",
                "intrinsics_samt": "../bake_intrinsics.samt",
                "decimated_vertices_samt": "../asset_decimated_vertices_zup.samt",
                "decimated_faces_samt": "../asset_decimated_faces.samt",
                "decimated_visibility_samt": "../asset_decimated_visibility.samt",
                "mincut_vertices_samt": "../asset_mincut_vertices_zup.samt",
                "mincut_faces_samt": "../asset_mincut_faces.samt",
                "mincut_candidate_faces_samt": "../asset_mincut_candidate_faces.samt",
            },
        }
        with open(os.path.join(asset_dir, "manifest.json"), "w") as handle:
            json.dump(asset_manifest, handle, indent=2)
        dmp.meta("official_pbr_reference", os.path.relpath(asset_dir, dmp.out_dir))
        dmp.meta("official_pbr_glb", os.path.relpath(final_glb, dmp.out_dir))
        dmp.meta("official_pbr_texture", os.path.relpath(texture_path, dmp.out_dir))

    print(f"[stage2] done in {time.perf_counter()-t0:.1f} s")

    dmp.save()
    print("\n[e2e] ALL STAGES COMPLETE")


if __name__ == "__main__":
    main()
