#!/usr/bin/env python3
"""Dump the PyTorch SS Euler trajectory used by the raw C++ E2E path.

The inputs and schedule are intentionally explicit: this is a numerical
reference for locating Q8 error propagation, not a replacement inference
pipeline. Every conditional/unconditional velocity and updated latent is
written as SAMT with the same channel-fastest layout consumed by ggml.
"""
import argparse
import os
import struct

import numpy as np
import torch

SAMT_MAGIC = b"SAMT"
MODS = ["6drotation_normalized", "scale", "shape", "translation",
        "translation_scale"]


def load_samt(path):
    with open(path, "rb") as stream:
        if stream.read(4) != SAMT_MAGIC:
            raise ValueError(f"{path}: invalid SAMT header")
        n_dims, = struct.unpack("<i", stream.read(4))
        shape = struct.unpack(f"<{n_dims}q", stream.read(8 * n_dims))
        value_type, = struct.unpack("<i", stream.read(4))
        dtype = "<f4" if value_type == 0 else "<i4"
        return shape, np.frombuffer(stream.read(), dtype=dtype)


def write_samt_f32(path, shape, values):
    data = np.ascontiguousarray(values, dtype="<f4")
    with open(path, "wb") as stream:
        stream.write(SAMT_MAGIC)
        stream.write(struct.pack("<i", len(shape)))
        stream.write(struct.pack(f"<{len(shape)}q", *shape))
        stream.write(struct.pack("<i", 0))
        stream.write(data.tobytes())


def load_generator(checkpoint, config, device, steps, rescale_t, cfg_strength,
                   cfg_start, cfg_end):
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    import sam3d_objects  # noqa: F401 - registers Hydra targets.
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    generator = instantiate(
        OmegaConf.load(config)["module"]["generator"]["backbone"])
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = state_dict.get("state_dict", state_dict)
    prefix = "_base_models.generator."
    state_dict = {name[len(prefix):]: value for name, value in state_dict.items()
                  if name.startswith(prefix)}
    missing, unexpected = generator.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch: missing={missing}, unexpected={unexpected}")
    generator = generator.to(device).eval()
    generator.inference_steps = steps
    generator.rescale_t = rescale_t
    generator.reverse_fn.strength = cfg_strength
    generator.reverse_fn.interval = [cfg_start, cfg_end]
    generator.reverse_fn.unconditional_handling = "add_flag"
    generator.no_shortcut = True
    return generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-dir", default="cpp_ggml/benchmarks/data/e2e")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", default="cpp_ggml/models/pytorch/ss_generator.ckpt")
    parser.add_argument("--config", default="cpp_ggml/models/pytorch/ss_generator.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--rescale-t", type=float, default=3.0)
    parser.add_argument("--cfg-strength", type=float, default=7.0)
    parser.add_argument("--cfg-start", type=float, default=0.0)
    parser.add_argument("--cfg-end", type=float, default=500.0)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    os.makedirs(args.out_dir, exist_ok=True)

    generator = load_generator(args.checkpoint, args.config, args.device,
                               args.steps, args.rescale_t, args.cfg_strength,
                               args.cfg_start, args.cfg_end)
    x = {}
    for mod in MODS:
        shape, data = load_samt(os.path.join(args.e2e_dir, f"ss_x0_{mod}.samt"))
        x[mod] = torch.from_numpy(data.reshape(shape[::-1]).copy()).to(args.device)
        write_samt_f32(os.path.join(args.out_dir, f"ss_torch_x000_{mod}.samt"),
                       [shape[0], shape[1]], x[mod][0].float().cpu().numpy())
    _, cond_data = load_samt(os.path.join(args.e2e_dir, "ss_cond_tokens.samt"))
    cond = torch.from_numpy(cond_data.reshape(1, -1, 1024).copy()).to(args.device)

    # This is FlowMatching._prepare_t() and session.cpp::make_euler_schedule().
    u = torch.linspace(0.0, 1.0, args.steps + 1, device=args.device)
    times = u / (1.0 + (args.rescale_t - 1.0) * (1.0 - u)) if args.rescale_t else u
    wbb = generator.reverse_fn.backbone
    with torch.no_grad():
        for step, (t0, t1) in enumerate(zip(times[:-1], times[1:]), start=1):
            # Shortcut._generate_dynamics scales normalized solver time before
            # invoking the backbone; session.cpp passes the same scalar.
            t = (t0 * 1000.0).reshape(1)
            d = torch.zeros_like(t)
            vc = wbb(x, t, cond, d=d, cfg=False)
            cfg_active = args.cfg_start <= float(t) <= args.cfg_end
            vu = wbb(x, t, cond, d=d, cfg=True) if cfg_active else None
            for index, mod in enumerate(MODS):
                write_samt_f32(
                    os.path.join(args.out_dir, f"ss_torch_vc{step:03d}_{mod}.samt"),
                    [x[mod].shape[-1], x[mod].shape[-2]],
                    vc[mod][0].float().cpu().numpy())
                if vu is not None:
                    write_samt_f32(
                        os.path.join(args.out_dir, f"ss_torch_vu{step:03d}_{mod}.samt"),
                        [x[mod].shape[-1], x[mod].shape[-2]],
                        vu[mod][0].float().cpu().numpy())
                velocity = vc[mod]
                if cfg_active and mod == "shape":
                    velocity = vc[mod] + args.cfg_strength * (vc[mod] - vu[mod])
                x[mod] = x[mod] + (t1 - t0) * velocity
                write_samt_f32(
                    os.path.join(args.out_dir, f"ss_torch_x{step:03d}_{mod}.samt"),
                    [x[mod].shape[-1], x[mod].shape[-2]],
                    x[mod][0].float().cpu().numpy())
            print(f"step {step:02d}/{args.steps}: t={float(t):.6f} "
                  f"dt={float(t1 - t0):.9f} cfg={int(cfg_active)}")


if __name__ == "__main__":
    main()
