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
import json
import os
import struct
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    ap.add_argument("--out-dir", default="cpp_ggml/benchmarks/data/e2e")
    ap.add_argument("--config", default="checkpoints/hf/pipeline.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-render", action="store_true", default=True)
    args = ap.parse_args()

    from loguru import logger
    logger.remove()  # quiet the per-step info spam

    from inference import Inference, load_image, load_single_mask  # noqa: E402

    dmp = Dumper(os.path.join(REPO_ROOT, args.out_dir))

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
            model = model.to(dtype=torch.bfloat16)
            dtxt = "bf16"
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
    orig_gen_noise = type(ss_gen)._generate_noise

    def gen_noise_cap(self, x_shape, x_device):
        out = orig_gen_noise(self, x_shape, x_device)
        noise_cap["x0"] = {k: v.detach().float().cpu() for k, v in out.items()}
        noise_cap["keys"] = list(out.keys())
        return out

    type(ss_gen)._generate_noise = gen_noise_cap

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

    torch.manual_seed(args.seed)
    ss_return = pipeline.sample_sparse_structure(ss_input_dict)
    type(ss_gen)._generate_noise = orig_gen_noise

    dmp.meta("ss_noise_keys", noise_cap.get("keys", []))
    for k, v in noise_cap.get("x0", {}).items():
        dmp.t(f"ss_x0_{k}", v)

    if "ss_cond_tokens" in cond_tokens_cap:
        dmp.t("ss_cond_tokens", cond_tokens_cap["ss_cond_tokens"])

    shape_latent = ss_return["shape"]
    dmp.t("ss_shape_latent", shape_latent)
    for k in ("6drotation_normalized", "scale", "translation", "translation_scale"):
        if k in ss_return:
            dmp.t(f"ss_pose_latent_{k}", ss_return[k])
    dmp.meta("ss_return_keys", sorted(ss_return.keys()))

    # ss_decoder I/O
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=pipeline.shape_model_dtype):
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

    outputs = pipeline.decode_slat(slat, ["gaussian", "mesh"])
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

    print(f"[stage2] done in {time.perf_counter()-t0:.1f} s")

    dmp.save()
    print("\n[e2e] ALL STAGES COMPLETE")


if __name__ == "__main__":
    main()
