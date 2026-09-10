#!/usr/bin/env python3
"""Run the reproducible SS-decoder accuracy and latency regression gate.

The gate intentionally requires every requested backend and precision. A
missing model, executable, incompatible tensor contract, NaN, parity
violation, or latency regression exits non-zero. The default input/reference
pair is emitted together by the official stage dumper and therefore measures
one real SS-decoder forward. It is a stage gate, not an image-to-PLY release
gate; use validate_e2e_benchmark.py for final rendered-output acceptance.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from samt_io import read_samt

from gpu_exclusivity import require_exclusive_gpu




def is_ss_latent_shape(shape: tuple[int, ...]) -> bool:
    """Accept the decoder's four axes with an optional exported batch=1 axis."""
    return shape in ((16, 16, 16, 8), (16, 16, 16, 8, 1))


def is_ss_occupancy_shape(shape: tuple[int, ...]) -> bool:
    """Accept historical and current single-batch occupancy SAMT layouts."""
    return shape in ((64, 64, 64), (64, 64, 64, 1), (64, 64, 64, 1, 1))


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


def canonical_backend(value: object) -> str:
    """Normalize ggml backend labels emitted by different build versions."""
    name = str(value).strip().lower()
    if name.startswith("cuda"):
        return "cuda"
    if name.startswith("vulkan"):
        return "vulkan"
    if name.startswith("metal"):
        return "metal"
    if name.startswith("hip"):
        return "hipblas"
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--input", type=Path,
                    default=root / "benchmarks/data/e2e/ss_latent_dec_input.samt",
                    help="official SS-decoder latent SAMT (default: canonical E2E dump)")
    ap.add_argument("--reference", type=Path,
                    default=root / "benchmarks/data/e2e/ss_occ_f32.samt",
                    help="matching official PyTorch F32 occupancy SAMT")
    ap.add_argument("--models-dir", type=Path, default=root / "models/gguf")
    ap.add_argument("--build-root", type=Path, default=root)
    ap.add_argument("--backends", default="cpu,cuda,vulkan")
    ap.add_argument("--dtypes", default="f32,f16,q4_0,q8_0")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--max-abs", type=float, default=0.25)
    ap.add_argument("--max-rel", type=float, default=0.01)
    ap.add_argument("--baseline", type=Path,
                    help="JSON object or JSONL with graph_ms_mean by backend/dtype")
    ap.add_argument("--max-slowdown", type=float, default=1.10)
    ap.add_argument("--allow-missing-backends", action="store_true")
    ap.add_argument("--require-exclusive-gpu", action="store_true",
                    help="reject CUDA/Vulkan latency measurements with another NVIDIA compute client")
    ap.add_argument("--json", type=Path, help="write machine-readable results")
    args = ap.parse_args()

    if not args.input.exists():
        ap.error(f"missing input: {args.input}")
    if not args.reference.exists():
        ap.error(f"missing reference: {args.reference}")
    input_shape, _ = read_samt(args.input)
    reference_shape, reference = read_samt(args.reference)
    if not is_ss_latent_shape(input_shape):
        ap.error(f"{args.input}: expected SS latent shape (16,16,16,8[,1]), got {input_shape}")
    if not is_ss_occupancy_shape(reference_shape):
        ap.error(f"{args.reference}: expected SS occupancy shape (64,64,64[,1[,1]]), got {reference_shape}")
    backends = [canonical_backend(x) for x in args.backends.split(",") if x.strip()]
    dtypes = [x.strip() for x in args.dtypes.split(",") if x.strip()]
    if not backends or not dtypes:
        ap.error("--backends and --dtypes must not be empty")
    gpu_exclusivity = {
        backend: (require_exclusive_gpu(backend) if args.require_exclusive_gpu else
                  {"required": False, "checked": False, "active_compute_processes": []})
        for backend in backends
    }

    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="sam3d-regression-") as tmp:
        tmpdir = Path(tmp)
        for backend in backends:
            build = args.build_root / f"build-{backend}"
            exe = build / "bin/sam3d-cli"
            if not exe.is_file():
                msg = f"missing {exe}; configure/build with the corresponding backend"
                if args.allow_missing_backends:
                    print("SKIP:", msg, file=sys.stderr)
                    continue
                raise SystemExit(msg)
            for dtype in dtypes:
                model = args.models_dir / f"ss_decoder-{dtype}.gguf"
                if not model.exists():
                    raise SystemExit(f"missing model: {model}")
                out = tmpdir / f"{backend}-{dtype}.samt"
                jsonl = tmpdir / f"{backend}-{dtype}.jsonl"
                env = os.environ.copy()
                # Each build tree contains a matching ggml core and loadable
                # backend. Prefix it so a caller's LD_LIBRARY_PATH cannot
                # accidentally register the wrong backend (for example a
                # Vulkan library while invoking the CUDA executable).
                lib_dir = build / "lib"
                if lib_dir.is_dir():
                    env["LD_LIBRARY_PATH"] = str(lib_dir) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
                cmd = [str(exe), "decode-ss", "--model", str(model),
                       "--input", str(args.input), "--backend", backend,
                       "--threads", str(args.threads), "--warmup", str(args.warmup),
                       "--iters", str(args.iters), "--out", str(out),
                       "--json", str(jsonl)]
                run(cmd, env)
                _, actual = read_samt(out)
                if actual.size != reference.size:
                    raise SystemExit(f"{backend}/{dtype}: shape mismatch ({actual.size} vs {reference.size})")
                if not np.isfinite(actual).all():
                    raise SystemExit(f"{backend}/{dtype}: non-finite output")
                delta = np.abs(actual - reference)
                scale = np.maximum(np.abs(reference), 1e-6)
                max_abs = float(delta.max())
                max_rel = float((delta / scale).max())
                mean_abs = float(delta.mean())
                rmse = float(np.sqrt(np.mean(np.square(actual - reference))))
                p99_abs = float(np.quantile(delta, 0.99))
                if max_abs > args.max_abs and max_rel > args.max_rel:
                    raise SystemExit(
                        f"{backend}/{dtype}: parity failed max_abs={max_abs:.6g} "
                        f"max_rel={max_rel:.6g} (limits {args.max_abs}/{args.max_rel})")
                row = json.loads(jsonl.read_text().splitlines()[-1])
                row.update({
                    "max_abs": max_abs,
                    "max_rel": max_rel,
                    "mean_abs": mean_abs,
                    "rmse": rmse,
                    "p99_abs": p99_abs,
                })
                results.append(row)
                print(f"PASS {backend}/{dtype}: max_abs={max_abs:.6g} "
                      f"mean_abs={mean_abs:.6g} rmse={rmse:.6g} "
                      f"p99_abs={p99_abs:.6g} max_rel={max_rel:.6g} "
                      f"latency={row['graph_ms_mean']:.3f} ms")

    if args.baseline:
        baseline: dict[tuple[str, str], float] = {}
        text = args.baseline.read_text()
        records = [json.loads(x) for x in text.splitlines() if x.strip()] if "\n" in text else [json.loads(text)]
        if isinstance(records[0], dict) and "results" in records[0]:
            records = records[0]["results"]
        for row in records:
            if "backend" in row and "dtype" in row and "graph_ms_mean" in row:
                baseline[(canonical_backend(row["backend"]), row["dtype"])] = float(row["graph_ms_mean"])
        for row in results:
            key = (canonical_backend(row["backend"]), row["dtype"])
            if key not in baseline:
                raise SystemExit(f"baseline missing {key[0]}/{key[1]}")
            limit = baseline[key] * args.max_slowdown
            if float(row["graph_ms_mean"]) > limit:
                raise SystemExit(f"{key[0]}/{key[1]} latency regression: "
                                 f"{row['graph_ms_mean']:.3f} > {limit:.3f} ms")

    payload = {"results": results, "gpu_exclusivity": gpu_exclusivity,
               "limits": {"max_abs": args.max_abs,
               "max_rel": args.max_rel, "max_slowdown": args.max_slowdown}}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"regression gate passed: {len(results)} configurations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
