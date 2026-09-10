#!/usr/bin/env python3
"""Torch-side per-op distribution for ONE SLat backbone forward, same inputs
and weights as the native slat-step replay. Mirrors slat_step_ref.py loading
without diagnostic hooks; runs the forward under torch.profiler.

Usage: python3 prof_torch_slat.py [--iters 5]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from samt_io import read_samt as load_samt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/hf/slat_generator.ckpt")
    ap.add_argument("--yaml", default="checkpoints/hf/slat_generator.yaml")
    ap.add_argument("--e2e-dir", default="cpp_ggml/benchmarks/data/e2e")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    import sam3d_objects  # registers the hydra targets
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    yconf = OmegaConf.load(args.yaml)
    bb = yconf["module"]["generator"]["backbone"]
    gen = instantiate(bb)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    gen_sd = {k[len("_base_models.generator."):]: v for k, v in sd.items()
              if k.startswith("_base_models.generator.")}
    del sd
    missing, unexpected = gen.load_state_dict(gen_sd, strict=False)
    print(f"[prof] slat generator loaded: missing={len(missing)} unexpected={len(unexpected)}")
    gen = gen.to(args.device).eval()
    gen.inference_steps = 12
    gen.rescale_t = 3
    gen.reverse_fn.strength = 0.0
    gen.no_shortcut = True

    ne, xa = load_samt(os.path.join(args.e2e_dir, "slat_x0.samt"))
    x = torch.from_numpy(xa.reshape(1, ne[1], ne[0]).copy()).to(args.device)
    ne, ca = load_samt(os.path.join(args.e2e_dir, "slat_coords.samt"))
    coords = torch.from_numpy(ca.reshape(ne[1], ne[0]).copy()).to(args.device)
    ne, ta = load_samt(os.path.join(args.e2e_dir, "slat_cond_tokens.samt"))
    cond = torch.from_numpy(ta.reshape(1, ne[1], ne[0]).copy()).to(args.device)
    t = torch.tensor([0.0], device=args.device, dtype=torch.float32)
    wbb = gen.reverse_fn.backbone
    print(f"[prof] inputs: x{x.shape} coords{coords.shape} cond{cond.shape}")

    # warmup
    with torch.no_grad():
        for _ in range(2):
            y = wbb(x, t, cond, coords, cfg=False)
        torch.cuda.synchronize()

    from torch.profiler import profile, ProfilerActivity
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                y = wbb(x, t, cond, coords, cfg=False)
            torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=22))
    evts = prof.key_averages()
    total_cuda = sum(e.self_device_time_total for e in evts)
    print(f"[prof] total self CUDA time over {args.iters} iters: {total_cuda/1e3:.1f} ms "
          f"({total_cuda/1e3/args.iters:.1f} ms/forward)")
    # kernel-name level rows (self device time)
    rows = [(e.key, e.count, e.self_device_time_total) for e in evts
            if e.self_device_time_total > 0 and e.device_type != torch.autograd.DeviceType.CPU]
    rows.sort(key=lambda r: -r[2])
    print(f"[prof] top CUDA rows:")
    for key, cnt, us in rows[:18]:
        print(f"  {key[:90]:90} {cnt:6} {us/1e3:9.1f} ms {100*us/total_cuda:5.1f}%")
    print(f"[prof] output: {y.shape}")


if __name__ == "__main__":
    main()
