#!/usr/bin/env python3
"""Dump the official SLat Euler trajectory for end-to-end parity debugging.

The script reuses the immutable tensors produced by ``dump_e2e_stages.py``.
It calls ``FlowMatching._generate_dynamics`` directly so the time sequence,
classifier-free guidance wrapper, and Euler update are exactly the ones used
by the official pipeline while retaining every normalized latent state.
"""
import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
from samt_io import read_samt as load_samt
from samt_io import write_samt_f32
import torch

from gguf_torch_loader import replace_backbone_from_gguf









def load_generator(checkpoint, config, device, gguf=None, weight_report=None):
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    import sam3d_objects  # noqa: F401 - registers Hydra targets.
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    generator = instantiate(OmegaConf.load(config)["module"]["generator"]["backbone"])
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = state_dict.get("state_dict", state_dict)
    prefix = "_base_models.generator."
    state_dict = {name[len(prefix):]: value for name, value in state_dict.items()
                  if name.startswith(prefix)}
    if gguf:
        receipt = replace_backbone_from_gguf(state_dict, gguf)
        if weight_report:
            Path(weight_report).write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    missing, unexpected = generator.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"generator state mismatch: missing={missing}, unexpected={unexpected}")
    return generator.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-dir", default="cpp_ggml/benchmarks/data/e2e")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/hf/slat_generator.ckpt")
    parser.add_argument("--config", default="checkpoints/hf/slat_generator.yaml")
    parser.add_argument("--gguf",
                        help=("dequantize this candidate SLat GGUF into the official Torch "
                              "backbone; enables same-GGUF trajectory localization"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-dtype", choices=("f32", "f16"), default="f16",
                        help=("operator dtype; f16 preserves the historical official autocast "
                              "behavior, while f32 isolates native implementation error"))
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--rescale-t", type=float, default=1.0)
    parser.add_argument("--cfg-strength", type=float, default=1.0)
    parser.add_argument("--cfg-start", type=float, default=0.0)
    parser.add_argument("--cfg-end", type=float, default=500.0)
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.compute_dtype == "f16" and not str(args.device).startswith("cuda"):
        parser.error("--compute-dtype f16 requires a CUDA device")
    os.makedirs(args.out_dir, exist_ok=True)
    generator = load_generator(
        args.checkpoint, args.config, args.device, args.gguf,
        (os.path.join(args.out_dir, "reference_weight_scope.json") if args.gguf else None))
    generator.inference_steps = args.steps
    generator.rescale_t = args.rescale_t
    generator.reverse_fn.strength = args.cfg_strength
    generator.reverse_fn.interval = [args.cfg_start, args.cfg_end]
    generator.reverse_fn.unconditional_handling = "add_flag"

    x_shape, x_data = load_samt(os.path.join(args.e2e_dir, "slat_x0.samt"))
    coord_shape, coord_data = load_samt(os.path.join(args.e2e_dir, "slat_coords.samt"))
    cond_shape, cond_data = load_samt(os.path.join(args.e2e_dir, "slat_cond_tokens.samt"))
    if (len(x_shape) != 3 or x_shape[2] != 1 or len(coord_shape) != 2 or
            len(cond_shape) != 3 or cond_shape[2] != 1):
        raise ValueError("expected batch-1 SLat tensors and rank-2 coordinates")
    x = torch.from_numpy(x_data.reshape(1, x_shape[1], x_shape[0]).copy()).to(args.device)
    coords = coord_data.reshape(coord_shape[1], coord_shape[0]).copy()
    condition = torch.from_numpy(
        cond_data.reshape(1, cond_shape[1], cond_shape[0]).copy()).to(args.device)

    times = generator._prepare_t().to(args.device)
    write_samt_f32(os.path.join(args.out_dir, "slat_torch_x000.samt"),
                   [x_shape[0], x_shape[1]], x[0].float().cpu().numpy())
    autocast_enabled = args.compute_dtype == "f16"
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                         enabled=autocast_enabled):
        for step, (t0, t1) in enumerate(zip(times[:-1], times[1:]), start=1):
            velocity = generator._generate_dynamics(x, t0, condition, coords)
            x = x + (t1 - t0) * velocity
            write_samt_f32(os.path.join(args.out_dir, f"slat_torch_x{step:03d}.samt"),
                           [x_shape[0], x_shape[1]], x[0].float().cpu().numpy())
            print(f"step {step:02d}/{args.steps}: t={t0.item():.6f} dt={(t1 - t0).item():.6f}")


if __name__ == "__main__":
    main()
