#!/usr/bin/env python3
"""Run a GGML 3D candidate and compare its PLY with an official oracle.

The default ``frozen-official-condition`` mode is a stage diagnostic: it
replays an official condition dump into the GGML generative stages. With
``--native-image-input``, the candidate is instead the complete native C++
``image + mask -> MoGe -> conditions -> SS -> SLat -> Gaussian PLY`` path.
The official dump is then read only as a render oracle and is not an inference
input. The JSON summary records the selected execution mode explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from gpu_exclusivity import require_exclusive_gpu

REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_ROOT = REPO_ROOT / "cpp_ggml"


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def backend_env(build_dir: Path, backend: str, dtype: str, threads: int,
                gaussian_dtype: str | None = None,
                slat_dtype: str | None = None,
                ss_dtype: str | None = None,
                ss_decoder_dtype: str | None = None,
                fuse_quant_qkv: bool = False,
                flash_attn_kernel: str = "auto",
                profile_jsonl: Path | None = None,
                ss_attention: str = "normal") -> dict[str, str]:
    env = os.environ.copy()
    # The runner owns stage orchestration.  Do not let a shell used for an
    # isolated flow diagnostic make its condition subprocess skip the inputs.
    for name in (
        "SAM3D_E2E_SKIP_COND", "SAM3D_E2E_STAGE", "SAM3D_E2E_REFERENCE_COORDS",
        "SAM3D_E2E_SS_COND_PATH", "SAM3D_E2E_SLAT_COND_PATH",
        "SAM3D_E2E_COORDS_PATH", "SAM3D_E2E_DUMP_SLAT_STEPS",
        "SAM3D_E2E_SLAT_FLOW_ONLY", "SAM3D_E2E_DEBUG_SLAT_FORWARDS",
        "SAM3D_E2E_DEBUG_SLAT_OUTPUT", "SAM3D_E2E_DEBUG_ONCE",
        "SAM3D_E2E_SS_DTYPE", "SAM3D_E2E_SS_DECODER_DTYPE",
        "SAM3D_E2E_SLAT_DTYPE", "SAM3D_E2E_GS_DTYPE", "SAM3D_E2E_MESH_DTYPE",
        "SAM3D_E2E_FUSE_QUANT_QKV", "SAM3D_E2E_PROFILE_JSONL",
        "GGML_CUDA_FATTN_KERNEL", "SAM3D_PYTORCH_PHILOX_DISTRIBUTION_BLOCKS",
    ):
        env.pop(name, None)
    env.pop("SAM3D_SS_STRICT_ATTN", None)
    env["SAM3D_BACKEND"] = backend
    env["SAM3D_E2E_DTYPE"] = dtype
    if gaussian_dtype is not None:
        env["SAM3D_E2E_GS_DTYPE"] = gaussian_dtype
    if slat_dtype is not None:
        env["SAM3D_E2E_SLAT_DTYPE"] = slat_dtype
    if ss_dtype is not None:
        env["SAM3D_E2E_SS_DTYPE"] = ss_dtype
    if ss_decoder_dtype is not None:
        env["SAM3D_E2E_SS_DECODER_DTYPE"] = ss_decoder_dtype
    if fuse_quant_qkv:
        env["SAM3D_E2E_FUSE_QUANT_QKV"] = "1"
    if flash_attn_kernel != "auto":
        env["GGML_CUDA_FATTN_KERNEL"] = flash_attn_kernel
    if ss_attention == "strict":
        env["SAM3D_SS_STRICT_ATTN"] = "1"
    if profile_jsonl is not None:
        env["SAM3D_E2E_PROFILE_JSONL"] = str(profile_jsonl)
    env["SAM3D_NTHREADS"] = str(threads)
    lib_dir = build_dir / "lib"
    env["LD_LIBRARY_PATH"] = str(lib_dir) + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    return env


def discover_pytorch_philox_distribution_blocks(cuda_binary: Path, seed: int) -> int:
    """Read the CUDA PyTorch-distribution launch contract without retaining noise."""
    if not cuda_binary.is_file():
        raise RuntimeError(
            "CPU/Vulkan native-seed sampling requires a built CUDA sampler to record "
            f"the PyTorch Philox distribution contract: {cuda_binary}")
    with tempfile.TemporaryDirectory(prefix="sam3d-philox-contract-") as directory:
        output = Path(directory)
        completed = subprocess.run(
            [str(cuda_binary), "rng-dump", "--implementation", "cuda", "--seed", str(seed),
             "--sizes", "1", "--out-dir", str(output)],
            cwd=REPO_ROOT, text=True, capture_output=True, check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                "CUDA rng-dump failed while recording the PyTorch Philox contract"
                + (f": {detail}" if detail else ""))
        try:
            contract = json.loads((output / "rng_contract.json").read_text(encoding="utf-8"))
            blocks = contract["distribution_blocks"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise RuntimeError("CUDA rng-dump wrote no valid Philox contract") from error
        if not isinstance(blocks, int) or blocks <= 0:
            raise RuntimeError("CUDA rng-dump wrote an invalid Philox distribution-block count")
        return blocks


def input_fingerprint(image: Path, mask: Path, config: Path, mask_index: int,
                      seed: int) -> dict[str, object]:
    def stat(path: Path) -> dict[str, object]:
        info = path.stat()
        return {"path": str(path.resolve()), "size": info.st_size,
                "mtime_ns": info.st_mtime_ns}

    return {"image": stat(image), "mask": stat(mask), "config": stat(config),
            "mask_index": mask_index, "seed": seed}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_provenance(executable: Path, models_dir: Path, dtype: str,
                       ss_dtype: str | None, ss_decoder_dtype: str | None,
                       slat_dtype: str | None,
                       gaussian_dtype: str | None,
                       moge_model: Path | None = None,
                       mesh_dtype: str | None = None) -> dict[str, object]:
    """Fingerprint executable and exact GGUF inputs outside the timed interval."""
    selected = {
        "ss_generator": models_dir / f"ss_generator-{ss_dtype or dtype}.gguf",
        "ss_decoder": models_dir / f"ss_decoder-{ss_decoder_dtype or ss_dtype or dtype}.gguf",
        "slat_generator": models_dir / f"slat_generator-{slat_dtype or dtype}.gguf",
        "slat_decoder_gs": models_dir / f"slat_decoder_gs-{gaussian_dtype or dtype}.gguf",
    }
    if mesh_dtype is not None:
        selected["slat_decoder_mesh"] = models_dir / f"slat_decoder_mesh-{mesh_dtype}.gguf"
    files = {name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
             for name, path in selected.items()}
    if moge_model is not None:
        files["moge"] = {
            "path": str(moge_model.resolve()),
            "sha256": sha256_file(moge_model),
        }
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            capture_output=True, check=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True,
            capture_output=True, check=True).stdout.strip() != ""
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", True
    return {
        "schema": "sam3d.e2e.provenance.v1",
        "source_revision": revision,
        "source_dirty": dirty,
        "sam3d_cli_sha256": sha256_file(executable),
        "models": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--mask-index", type=int, default=0)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), default="cuda")
    parser.add_argument("--dtype", choices=("f32", "f16", "q4_0", "q4_1", "q4_k", "q8_0"), default="f16")
    parser.add_argument("--gaussian-dtype", choices=("f32", "f16", "q4_0", "q4_k", "q8_0"),
                        help="override only the Gaussian decoder dtype; omitted means --dtype")
    parser.add_argument("--slat-dtype", choices=("f32", "f16", "q4_0", "q4_1", "q4_k", "q8_0"),
                        help="override only the SLat condition/flow generator; diagnostic only")
    parser.add_argument("--ss-dtype", choices=("f32", "f16", "q4_0", "q4_1", "q4_k", "q8_0"),
                        help="override SS generator/decoder dtype; diagnostic only")
    parser.add_argument("--ss-decoder-dtype", choices=("f32", "f16", "q4_0", "q4_1", "q4_k", "q8_0"),
                        help="override only the SS occupancy decoder dtype; diagnostic only")
    parser.add_argument("--fuse-quant-qkv", action="store_true",
                        help="diagnostic: fuse quantized self-attention QKV projections")
    parser.add_argument("--flash-attn-kernel", choices=("auto", "mma", "tile", "vec"),
                        default="auto",
                        help="CUDA diagnostic only: select a patched ggml flash-attention kernel")
    parser.add_argument("--ss-attention", choices=("normal", "strict"), default="strict",
                        help=("SS attention policy: strict uses explicit F32 QK^T/softmax/V "
                              "for numeric parity; normal uses the flash-attention throughput path"))
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--models-dir", type=Path, default=CPP_ROOT / "models/gguf")
    parser.add_argument("--moge-model", type=Path,
                        help="native MoGe GGUF; defaults to models-dir/moge_vitl-f16.gguf")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "checkpoints/hf/pipeline.yaml")
    parser.add_argument("--python", type=Path,
                        help="official-environment Python used for MoGe dump")
    parser.add_argument("--out-dir", type=Path, default=CPP_ROOT / "benchmarks/image_to_3d")
    parser.add_argument("--conditions-dir", type=Path,
                        help="use an existing, immutable official stage dump instead of regenerating it")
    parser.add_argument("--native-image-input", action="store_true",
                        help="run the complete native C++ image+mask pipeline; conditions are oracle-only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rng-distribution-blocks", type=int,
                        help=("PyTorch CUDA Philox distribution-block capacity recorded from the "
                              "reference device; required for CPU/Vulkan native-seed sampling"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--reuse-conditions", action="store_true")
    parser.add_argument("--replay-noise", action="store_true",
                        help="replay official noise; requires matching occupancy shape")
    parser.add_argument("--noise-dir", type=Path,
                        help=("native-image-input diagnostic only: require initial SS/SLat noise "
                              "from this immutable official stage directory"))
    parser.add_argument("--strict-reference-coords", action="store_true",
                        help="hold sparse support to the official dump for a focused Q8 parity gate")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--render-frames", type=int, default=60)
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--max-render-mae", type=float, default=None)
    parser.add_argument("--max-e2e-ms", type=float, default=None,
                        help="fail when condition+SS+SLat+Gaussian exceeds this latency")
    parser.add_argument("--require-exclusive-gpu", action="store_true",
                        help="reject CUDA/Vulkan timing runs while another NVIDIA compute client is active")
    parser.add_argument("--profile-jsonl", type=Path,
                        help="write per-stage H2D/D2H/submit/sync measurements as JSONL")
    parser.add_argument("--dtype-contract-out", type=Path,
                        help="native-image-input: write observed ggml graph dtype evidence here")
    args = parser.parse_args()
    if args.rng_distribution_blocks is not None and args.rng_distribution_blocks <= 0:
        parser.error("--rng-distribution-blocks must be positive")
    if args.strict_reference_coords and not args.replay_noise:
        parser.error("--strict-reference-coords requires --replay-noise")
    if args.flash_attn_kernel != "auto" and args.backend != "cuda":
        parser.error("--flash-attn-kernel is supported only with --backend cuda")
    if args.native_image_input:
        incompatible = []
        if args.replay_noise:
            incompatible.append("--replay-noise")
        if args.strict_reference_coords:
            incompatible.append("--strict-reference-coords")
        if args.ss_dtype is not None:
            incompatible.append("--ss-dtype")
        if args.ss_decoder_dtype is not None:
            incompatible.append("--ss-decoder-dtype")
        if args.slat_dtype is not None:
            incompatible.append("--slat-dtype")
        if args.gaussian_dtype is not None:
            incompatible.append("--gaussian-dtype")
        if incompatible:
            parser.error("--native-image-input rejects diagnostic overrides: " + ", ".join(incompatible))
    elif args.noise_dir is not None:
        parser.error("--noise-dir is only supported with --native-image-input; use --replay-noise otherwise")
    if args.noise_dir is not None and not args.noise_dir.is_dir():
        parser.error(f"--noise-dir must be an existing official stage directory: {args.noise_dir}")

    official_python = args.python
    if official_python is None:
        configured = os.environ.get("SAM3D_PYTHON")
        conda_prefix = os.environ.get("CONDA_PREFIX")
        official_python = Path(configured) if configured else (
            Path(conda_prefix) / "bin/python" if conda_prefix else Path(sys.executable)
        )
    official_python = official_python.resolve()
    if not official_python.is_file():
        parser.error(f"official Python does not exist: {official_python}")
    if not args.skip_render:
        # Rendering is part of the E2E contract. Check the selected interpreter
        # before launching minutes of native generation work.
        probe = subprocess.run(
            [str(official_python), "-c", "import gsplat, plyfile"],
            text=True,
            capture_output=True,
        )
        if probe.returncode:
            detail = probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else ""
            parser.error(
                "official renderer dependencies are unavailable; pass --python or set "
                f"SAM3D_PYTHON to the sam3d-objects environment ({detail})"
            )

    build_dir = (args.build_dir or CPP_ROOT / f"build-{args.backend}").resolve()
    executable = build_dir / "bin/sam3d-cli"
    if args.backend != "cuda" and args.rng_distribution_blocks is None:
        try:
            args.rng_distribution_blocks = discover_pytorch_philox_distribution_blocks(
                CPP_ROOT / "build-cuda/bin/sam3d-cli", args.seed)
        except RuntimeError as error:
            parser.error(str(error))
    moge_model = (args.moge_model or args.models_dir / "moge_vitl-f16.gguf").resolve()
    conditions = args.conditions_dir.resolve() if args.conditions_dir else args.out_dir / "conditions"
    ggml_ply = args.out_dir / f"ggml_{args.backend}_{args.dtype}.ply"
    native_pose = args.out_dir / f"ggml_{args.backend}_{args.dtype}_pose.json"
    dtype_contract = (args.dtype_contract_out or
                      args.out_dir / f"ggml_{args.backend}_{args.dtype}_dtype_contract.json")
    dtype_contract_report = args.out_dir / f"ggml_{args.backend}_{args.dtype}_dtype_contract_report.json"
    pytorch_ply = conditions / "output_gs.ply"
    report_dir = args.out_dir / f"render_{args.backend}_{args.dtype}"

    for label, path in (("image", args.image), ("mask directory", args.mask_dir),
                        ("pipeline config", args.config), ("ggml executable", executable),
                        ("GGUF model directory", args.models_dir)):
        if not path.exists():
            parser.error(f"missing {label}: {path}")
    if args.native_image_input and not moge_model.is_file():
        parser.error(f"missing native MoGe GGUF: {moge_model}")
    mask = args.mask_dir / f"{args.mask_index}.png"
    if not mask.is_file():
        parser.error(f"missing mask: {mask}")
    gpu_exclusivity = (
        require_exclusive_gpu(args.backend) if args.require_exclusive_gpu else
        {"required": False, "checked": False, "active_compute_processes": []}
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    profile_jsonl = args.profile_jsonl.resolve() if args.profile_jsonl else None
    if profile_jsonl is not None:
        profile_jsonl.parent.mkdir(parents=True, exist_ok=True)
        profile_jsonl.unlink(missing_ok=True)
    if args.conditions_dir and not (conditions / "manifest.json").is_file():
        parser.error(f"conditions directory has no manifest: {conditions}")
    fingerprint = input_fingerprint(args.image, mask, args.config,
                                     args.mask_index, args.seed)
    metadata_path = conditions / "run_inputs.json"
    reusable = False
    if args.reuse_conditions and (conditions / "manifest.json").is_file():
        try:
            with metadata_path.open(encoding="utf-8") as stream:
                reusable = json.load(stream) == fingerprint
        except (OSError, json.JSONDecodeError):
            reusable = False

    if not args.conditions_dir and not reusable:
        run([
            str(official_python), str(CPP_ROOT / "scripts/dump_e2e_stages.py"),
            "--image", str(args.image.resolve()),
            "--mask-dir", str(args.mask_dir.resolve()),
            "--mask-index", str(args.mask_index),
            "--out-dir", str(conditions.resolve()),
            "--config", str(args.config.resolve()),
            "--seed", str(args.seed),
        ])
        conditions.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(fingerprint, stream, indent=2)

    execution_mode = (
        "native-image-input" if args.native_image_input else "frozen-official-condition"
    )
    condition_ms: float | None = None
    ss_ms: float | None = None
    slat_ms: float | None = None
    if args.native_image_input:
        native_env = backend_env(build_dir, args.backend, args.dtype, args.threads,
                                 fuse_quant_qkv=args.fuse_quant_qkv,
                                 flash_attn_kernel=args.flash_attn_kernel,
                                 profile_jsonl=profile_jsonl,
                                 ss_attention=args.ss_attention)
        if args.rng_distribution_blocks is not None:
            native_env["SAM3D_PYTORCH_PHILOX_DISTRIBUTION_BLOCKS"] = str(
                args.rng_distribution_blocks)
        native_command = [
            str(executable), "image-to-3d", "--model", str(args.models_dir.resolve()),
            "--moge-model", str(moge_model), "--image", str(args.image.resolve()),
            "--mask", str(mask.resolve()), "--backend", args.backend,
            "--dtype", args.dtype, "--out", str(ggml_ply.resolve()),
            "--pose-out", str(native_pose.resolve()),
            "--dtype-contract-out", str(dtype_contract.resolve()),
            "--ss-attention", args.ss_attention,
            "--seed", str(args.seed), "--threads", str(args.threads),
        ]
        if args.noise_dir is not None:
            native_command.extend(("--noise-dir", str(args.noise_dir.resolve())))
        native_started = time.perf_counter()
        run(native_command, env=native_env)
        native_ms = (time.perf_counter() - native_started) * 1000.0
        if not ggml_ply.is_file():
            raise RuntimeError(f"native image-to-3d did not write Gaussian PLY: {ggml_ply}")
        if not native_pose.is_file():
            raise RuntimeError(f"native image-to-3d did not write pose JSON: {native_pose}")
        if not dtype_contract.is_file():
            raise RuntimeError(
                f"native image-to-3d did not write dtype contract: {dtype_contract}")
        run([
            sys.executable, str(CPP_ROOT / "scripts/verify_dtype_contract.py"),
            "--contract", str(dtype_contract.resolve()),
            "--expected-weight-type", args.dtype,
            "--json-out", str(dtype_contract_report.resolve()),
        ])
    else:
        condition_prefix = args.out_dir / "ggml_condition_tokens"
        condition_env = backend_env(build_dir, args.backend, args.dtype, args.threads,
                                    args.gaussian_dtype, args.slat_dtype, args.ss_dtype,
                                    args.ss_decoder_dtype, args.fuse_quant_qkv,
                                    args.flash_attn_kernel, profile_jsonl, args.ss_attention)
        if args.rng_distribution_blocks is not None:
            condition_env["SAM3D_PYTORCH_PHILOX_DISTRIBUTION_BLOCKS"] = str(
                args.rng_distribution_blocks)
        condition_env["SAM3D_E2E_STAGE"] = "cond"
        condition_command = [
            str(executable), "e2e", "--model", str(args.models_dir.resolve()),
            str(conditions.resolve()), "--out", str(condition_prefix.resolve()),
            "--seed", str(args.seed), "--threads", str(args.threads),
        ]
        condition_started = time.perf_counter()
        run(condition_command, env=condition_env)
        condition_ms = (time.perf_counter() - condition_started) * 1000.0

        ss_tokens = Path(str(condition_prefix) + ".ss_cond.samt")
        slat_tokens = Path(str(condition_prefix) + ".slat_cond.samt")
        for label, path in (("SS condition tokens", ss_tokens),
                            ("SLat condition tokens", slat_tokens)):
            if not path.is_file():
                raise RuntimeError(f"ggml condition stage did not write {label}: {path}")

        ss_ms = 0.0
        coords_path = conditions / "coords.samt"
        if not args.strict_reference_coords:
            ss_prefix = args.out_dir / "ggml_ss"
            ss_command = [
                str(executable), "e2e", "--model", str(args.models_dir.resolve()),
                str(conditions.resolve()), "--out", str(ss_prefix.resolve()),
                "--seed", str(args.seed), "--threads", str(args.threads),
            ]
            if args.replay_noise:
                ss_command.extend(("--noise-dir", str(conditions.resolve())))
            ss_env = backend_env(build_dir, args.backend, args.dtype, args.threads,
                                 args.gaussian_dtype, args.slat_dtype, args.ss_dtype,
                                 args.ss_decoder_dtype, args.fuse_quant_qkv,
                                 args.flash_attn_kernel, profile_jsonl, args.ss_attention)
            ss_env["SAM3D_E2E_STAGE"] = "ss"
            ss_env["SAM3D_E2E_SS_COND_PATH"] = str(ss_tokens.resolve())
            ss_started = time.perf_counter()
            run(ss_command, env=ss_env)
            ss_ms = (time.perf_counter() - ss_started) * 1000.0
            coords_path = Path(str(ss_prefix) + ".coords.samt")
            if not coords_path.is_file():
                raise RuntimeError(f"ggml SS stage did not write sparse coordinates: {coords_path}")

        slat_command = [
            str(executable), "e2e", "--model", str(args.models_dir.resolve()),
            str(conditions.resolve()), "--out", str(ggml_ply.resolve()),
            "--seed", str(args.seed), "--threads", str(args.threads),
        ]
        if args.replay_noise:
            slat_command.extend(("--noise-dir", str(conditions.resolve())))
        slat_env = backend_env(build_dir, args.backend, args.dtype, args.threads,
                               args.gaussian_dtype, args.slat_dtype, args.ss_dtype,
                               args.ss_decoder_dtype, args.fuse_quant_qkv,
                               args.flash_attn_kernel, profile_jsonl, args.ss_attention)
        slat_env["SAM3D_E2E_STAGE"] = "slat"
        slat_env["SAM3D_E2E_SLAT_COND_PATH"] = str(slat_tokens.resolve())
        if args.strict_reference_coords:
            slat_env["SAM3D_E2E_REFERENCE_COORDS"] = "1"
        else:
            slat_env["SAM3D_E2E_COORDS_PATH"] = str(coords_path.resolve())
        slat_started = time.perf_counter()
        run(slat_command, env=slat_env)
        slat_ms = (time.perf_counter() - slat_started) * 1000.0
        native_ms = condition_ms + ss_ms + slat_ms

    summary = {
        "image": str(args.image.resolve()),
        "mask": str(mask.resolve()),
        "backend": args.backend,
        "dtype": args.dtype,
        "execution_mode": execution_mode,
        "ss_dtype": args.ss_dtype or args.dtype,
        "ss_decoder_dtype": args.ss_decoder_dtype or args.ss_dtype or args.dtype,
        "fuse_quant_qkv": args.fuse_quant_qkv,
        "flash_attn_kernel": args.flash_attn_kernel,
        "slat_dtype": args.slat_dtype or args.dtype,
        "gaussian_dtype": args.gaussian_dtype or args.dtype,
        "seed": args.seed,
        "sampling_mode": (
            "official-noise-replay-diagnostic" if args.noise_dir is not None else "native-seed"
        ),
        "noise_dir": str(args.noise_dir.resolve()) if args.noise_dir is not None else None,
        "ggml_condition_ms": condition_ms,
        "ggml_ss_ms": ss_ms,
        "ggml_slat_ms": slat_ms,
        "ggml_flow_ms": None if ss_ms is None or slat_ms is None else ss_ms + slat_ms,
        "ggml_e2e_ms": native_ms,
        "timer_contract": (
            {
                "id": "sam3d.cold-request.image-mask-to-gaussian-ply.v2",
                "kind": "cold-request",
                "includes": [
                    "CLI process startup", "model loading", "input image/mask decode", "native preprocessing",
                    "MoGe", "SS", "SLat", "Gaussian decode", "PLY serialization and file close",
                ],
                "excludes": ["Python launcher import", "mesh/PBR postprocessing", "external render scoring"],
            }
            if args.native_image_input else
            {
                "id": "sam3d.stage-process.frozen-condition-to-gaussian-ply.v1",
                "kind": "stage-diagnostic",
                "includes": ["three independent CLI process startups", "stage model loading", "SAMT/PLY I/O"],
                "excludes": ["native image/mask preprocessing", "mesh/PBR postprocessing", "external render scoring"],
            }
        ),
        "ss_attention": args.ss_attention,
        "strict_reference_coords": args.strict_reference_coords,
        "gpu_exclusivity": gpu_exclusivity,
        "ggml_ply": str(ggml_ply.resolve()),
        "native_pose": str(native_pose.resolve()) if args.native_image_input else None,
        "native_dtype_contract": (
            str(dtype_contract.resolve()) if args.native_image_input else None),
        "native_dtype_contract_validation": (
            str(dtype_contract_report.resolve()) if args.native_image_input else None),
        "pytorch_ply": str(pytorch_ply.resolve()),
        "provenance": runtime_provenance(executable, args.models_dir.resolve(), args.dtype,
                                          args.ss_dtype, args.ss_decoder_dtype, args.slat_dtype,
                                          args.gaussian_dtype,
                                          moge_model if args.native_image_input else None),
    }
    if profile_jsonl is not None:
        summary["backend_profile_jsonl"] = str(profile_jsonl)
    if not args.skip_render:
        command = [
            str(official_python), str(CPP_ROOT / "scripts/render_compare.py"),
            "--pytorch-ply", str(pytorch_ply.resolve()),
            "--ggml-ply", str(ggml_ply.resolve()),
            "--condition-dir", str(conditions.resolve()),
            "--out-dir", str(report_dir.resolve()),
            "--frames", str(args.render_frames),
            "--resolution", str(args.render_resolution),
        ]
        run(command)
        summary["render_report"] = str((report_dir / "render_metrics.json").resolve())
        with (report_dir / "render_metrics.json").open(encoding="utf-8") as stream:
            render_metrics = json.load(stream)
        summary["render_mae"] = float(render_metrics["mae"])
        summary["render_quality_gate_enabled"] = args.max_render_mae is not None
        summary["render_quality_pass"] = (
            summary["render_mae"] <= args.max_render_mae
            if args.max_render_mae is not None else None
        )

    with (args.out_dir / "run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))
    if summary.get("render_quality_pass") is False:
        raise RuntimeError(
            f"render quality gate failed: {summary['render_mae']:.9f} "
            f"> {args.max_render_mae:.9f}")
    if args.max_e2e_ms is not None and summary["ggml_e2e_ms"] > args.max_e2e_ms:
        raise RuntimeError(
            f"e2e latency gate failed: {summary['ggml_e2e_ms']:.3f} ms "
            f"> {args.max_e2e_ms:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
