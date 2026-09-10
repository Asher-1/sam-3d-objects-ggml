#!/usr/bin/env python3
"""Run the complete official-versus-native SAM 3D Objects acceptance case.

This script creates every generated artifact below a new caller-selected work
directory. It never writes model files or benchmark baselines. The official
Python process is an oracle generator and scorer; every GGML candidate starts
from ``image + mask`` through the native C++ CLI. The script records every
command, elapsed time, return code, and artifact path in one JSON summary.

It performs these independent checks in dependency order:
  1. no-cuDNN inspection of native CUDA binaries;
  2. fresh official stage/PBR oracle generation for this image, mask, and seed;
  3. controlled native PBR assembly from official raw mesh/PLY inputs;
  4. raw native CUDA F16/Q8_0/Q4_0 image-to-PLY-and-GLB inference and render comparison;
  5. raw Vulkan F16/Q8_0/Q4_0 inference handed to the CUDA PBR binary;
  6. raw native CUDA/Vulkan F16, Q8_0, and Q4_0 latency/quality matrix.

The command is intentionally non-zero if a required step or its release gate
fails. Failed steps remain in the JSON report; no missing measurement is
converted into a passing result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from gpu_exclusivity import require_exclusive_gpu
from run_image_to_3d import runtime_provenance, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_ROOT = REPO_ROOT / "cpp_ggml"
SCRIPTS = CPP_ROOT / "scripts"


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def run_step(name: str, command: list[str], steps: list[dict[str, Any]]) -> int:
    print(f"[{name}] + {command_text(command)}", flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, check=False)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    steps.append({
        "name": name,
        "command": command_text(command),
        "returncode": result.returncode,
        "elapsed_ms": elapsed_ms,
        "status": "PASS" if result.returncode == 0 else "FAIL",
    })
    return result.returncode


def require_exclusive_cuda(name: str, steps: list[dict[str, Any]]) -> bool:
    """Reject a full-E2E measurement before another compute client can taint it."""
    started = time.perf_counter()
    try:
        provenance = require_exclusive_gpu("cuda")
    except RuntimeError as error:
        steps.append({
            "name": name,
            "command": "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory",
            "returncode": 1,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "status": "FAIL",
            "error": str(error),
        })
        return False
    steps.append({
        "name": name,
        "command": "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory",
        "returncode": 0,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "status": "PASS",
        "gpu_exclusivity": provenance,
    })
    return True


def require_file(parser: argparse.ArgumentParser, label: str, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        parser.error(f"missing {label}: {resolved}")
    return resolved


def require_dir(parser: argparse.ArgumentParser, label: str, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        parser.error(f"missing {label}: {resolved}")
    return resolved


def read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def valid_vulkan_cuda_handoff(path: Path) -> bool:
    """Accept only a complete manifest that names the two distinct owners."""
    manifest = read_json_object(path)
    if not manifest or manifest.get("schema") != "sam3d.vulkan-inference-cuda-pbr-handoff.v1":
        return False
    if manifest.get("passed") is not True:
        return False
    if manifest.get("neural_backend") != "vulkan" or manifest.get("pbr_backend") != "cuda":
        return False
    artifacts = manifest.get("artifacts")
    required = {"ply", "mesh_vertices", "mesh_faces", "pose", "dtype_contract", "glb", "texture"}
    if manifest.get("neural_artifacts_unchanged") is not True:
        return False
    if not isinstance(artifacts, dict) or not required.issubset(artifacts):
        return False
    for name in required:
        artifact = artifacts[name]
        if not isinstance(artifact, dict):
            return False
        artifact_path = artifact.get("path")
        checksum = artifact.get("sha256")
        if not isinstance(artifact_path, str) or not Path(artifact_path).is_file():
            return False
        if not isinstance(checksum, str) or len(checksum) != 64:
            return False
        digest = hashlib.sha256()
        try:
            with Path(artifact_path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return False
        if digest.hexdigest() != checksum:
            return False
    return True


def capture_full_provenance(models_dir: Path, moge_model: Path, binaries: dict[str, Path],
                            image: Path, mask: Path) -> dict[str, Any]:
    return {
        "schema": "sam3d.full-e2e-provenance.v1",
        "input_sha256": {"image": sha256_file(image), "mask": sha256_file(mask)},
        "model_directory": str(models_dir),
        "binary_sha256": {name: sha256_file(path) for name, path in binaries.items()},
        "ggml_patch_sha256": sha256_file(CPP_ROOT / "third_party/ggml-patches/0001-sam3d-ggml-combined.patch"),
        "variants": {dtype: runtime_provenance(binaries["native_pbr"], models_dir, dtype,
                     None, None, None, None, moge_model, mesh_dtype=dtype)
                     for dtype in ("f16", "q8_0", "q4_k")},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-python", type=Path, required=True,
                        help="official SAM 3D Objects Python interpreter")
    parser.add_argument("--cuda-binary", type=Path, required=True,
                        help="regular CUDA sam3d-cli used by the latency matrix")
    parser.add_argument("--vulkan-binary", type=Path, required=True,
                        help="Vulkan sam3d-cli used by the latency matrix")
    parser.add_argument("--native-pbr-binary", type=Path, required=True,
                        help="licensed CUDA PBR sam3d-cli for textured GLB checks")
    parser.add_argument("--models-dir", type=Path, default=CPP_ROOT / "models/gguf")
    parser.add_argument("--moge-model", type=Path,
                        help="native MoGe preprocessing GGUF; defaults to models-dir/moge_vitl-f16.gguf")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--mask-index", type=int, default=14)
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "checkpoints/hf/pipeline.yaml")
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="new or empty directory for all generated artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--render-frames", type=int, default=60)
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--ss-attention", choices=("normal", "strict"), default="normal",
                        help=("SS attention path recorded into every raw row: normal is the "
                              "delivery default (F16-KV flash); strict is the parity diagnostic"))
    parser.add_argument("--diagnostic-noise-replay", action="store_true",
                        help=("inject the fresh official stage noise into raw candidates for numerical "
                              "isolation only; the resulting report is not a production native-seed pass"))
    parser.add_argument("--operator-oracle", action="store_true",
                        help=("run SS/SLat block-boundary captures against one GGUF candidate; "
                              "without calibrated thresholds it remains diagnostic"))
    parser.add_argument("--operator-oracle-dtype", choices=("f16", "q8_0", "q4_k"), default="f16")
    parser.add_argument("--operator-oracle-reference-weight-scope",
                        choices=("official-checkpoint", "same-gguf-dequantized"),
                        default="same-gguf-dequantized",
                        help="weight scope for --operator-oracle Torch observations")
    parser.add_argument("--operator-oracle-max-stage-mae", type=float,
                        help="optional per-boundary MAE gate for --operator-oracle")
    parser.add_argument("--operator-oracle-max-stage-abs", type=float,
                        help="optional per-boundary maximum-absolute-error gate for --operator-oracle")
    parser.add_argument("--operator-oracle-trajectory", action="store_true",
                        help=("extend --operator-oracle with all 25 SS/SLat Euler states; "
                              "diagnostic until its thresholds are empirically calibrated"))
    parser.add_argument("--operator-oracle-trajectory-ss-compute-dtype",
                        choices=("f32", "f16"), default="f32")
    parser.add_argument("--operator-oracle-trajectory-slat-compute-dtype",
                        choices=("f32", "f16"), default="f32")
    parser.add_argument("--operator-oracle-trajectory-ss-baseline", type=Path,
                        help="optional non-regression report for the SS trajectory")
    parser.add_argument("--operator-oracle-trajectory-slat-max-terminal-mae", type=float,
                        help="optional final SLat trajectory MAE limit")
    parser.add_argument("--operator-oracle-vulkan", action="store_true",
                        help=("also run the selected operator oracle on Vulkan with the same "
                              "fresh stage directory; it remains diagnostic without thresholds"))
    parser.add_argument("--max-render-mae", type=float, default=0.01,
                        help="raw CUDA PBR render gate for every tested dtype; matrix keeps its release contract")
    parser.add_argument("--max-texture-mae-u8", type=float,
                        help="optional controlled-PBR base-color MAE gate on the 0..255 scale")
    parser.add_argument("--max-texture-rmse-u8", type=float,
                        help="optional controlled-PBR base-color RMSE gate on the 0..255 scale")
    parser.add_argument("--max-texture-abs-u8", type=float,
                        help="optional controlled-PBR base-color maximum-error gate on the 0..255 scale")
    parser.add_argument("--max-normal-angle-deg", type=float,
                        help="optional controlled-PBR maximum vertex-normal angle gate")
    parser.add_argument("--max-glb-rgb-mae-linear", type=float,
                        help="optional raw-GLB fixed-orbit linear-RGB MAE gate")
    parser.add_argument("--min-glb-mask-iou", type=float,
                        help="optional raw-GLB fixed-orbit silhouette IoU gate")
    parser.add_argument("--max-glb-normal-angle-deg", type=float,
                        help="optional raw-GLB fixed-orbit maximum normal-angle gate")
    parser.add_argument("--max-glb-depth-mae-ndc", type=float,
                        help="optional raw-GLB fixed-orbit depth MAE gate")
    parser.add_argument("--measure-official-full-glb", action="store_true",
                        help="also measure the official hot-session image/mask-to-textured-GLB contract")
    parser.add_argument("--skip-neural-diagnostic-matrix", action="store_true",
                        help="omit the extra neural-only benchmark; all six raw full-GLB cases and their Gaussian comparisons still run")
    args = parser.parse_args()

    official_python = require_file(parser, "official Python", args.official_python)
    cuda_binary = require_file(parser, "CUDA executable", args.cuda_binary)
    vulkan_binary = require_file(parser, "Vulkan executable", args.vulkan_binary)
    pbr_binary = require_file(parser, "native PBR executable", args.native_pbr_binary)
    models_dir = require_dir(parser, "GGUF models directory", args.models_dir)
    moge_model = require_file(
        parser, "MoGe preprocessing model",
        args.moge_model or models_dir / "moge_vitl-f16.gguf")
    image = require_file(parser, "image", args.image)
    mask_dir = require_dir(parser, "mask directory", args.mask_dir)
    mask = require_file(parser, "selected mask", mask_dir / f"{args.mask_index}.png")
    config = require_file(parser, "pipeline config", args.config)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.render_frames < 1 or args.render_resolution < 16:
        parser.error("--render-frames must be positive and --render-resolution must be >= 16")
    if any(value is not None and (not math.isfinite(value) or value < 0.0) for value in (
            args.max_render_mae, args.max_texture_mae_u8, args.max_texture_rmse_u8,
            args.max_texture_abs_u8, args.max_normal_angle_deg, args.max_glb_rgb_mae_linear,
            args.min_glb_mask_iou, args.max_glb_normal_angle_deg, args.max_glb_depth_mae_ndc)):
        parser.error("all quality thresholds must be finite and non-negative")
    if args.min_glb_mask_iou is not None and args.min_glb_mask_iou > 1.0:
        parser.error("--min-glb-mask-iou must be <= 1")
    if (args.operator_oracle_max_stage_mae is not None and
            args.operator_oracle_max_stage_mae < 0.0):
        parser.error("--operator-oracle-max-stage-mae must be non-negative")
    if (args.operator_oracle_max_stage_abs is not None and
            args.operator_oracle_max_stage_abs < 0.0):
        parser.error("--operator-oracle-max-stage-abs must be non-negative")
    if (args.operator_oracle_trajectory_slat_max_terminal_mae is not None and
            args.operator_oracle_trajectory_slat_max_terminal_mae < 0.0):
        parser.error("--operator-oracle-trajectory-slat-max-terminal-mae must be non-negative")

    glb_render_thresholds = {
        "max_rgb_mae_linear": args.max_glb_rgb_mae_linear,
        "min_mask_iou": args.min_glb_mask_iou,
        "max_normal_angle_deg": args.max_glb_normal_angle_deg,
        "max_depth_mae_ndc": args.max_glb_depth_mae_ndc,
    }
    # A successful renderer invocation only establishes that a comparison was
    # measured. It is not an acceptance result until all final-asset metrics
    # have explicit, predeclared limits.
    glb_render_gate_configured = all(value is not None for value in glb_render_thresholds.values())
    controlled_pbr_gate_configured = all(value is not None for value in (
        args.max_texture_mae_u8, args.max_texture_rmse_u8,
        args.max_texture_abs_u8, args.max_normal_angle_deg))

    work_dir = args.work_dir.resolve()
    if work_dir.exists() and any(work_dir.iterdir()):
        parser.error(f"refusing to overwrite non-empty work directory: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    provenance = capture_full_provenance(models_dir, moge_model,
        {"cuda": cuda_binary, "vulkan": vulkan_binary, "native_pbr": pbr_binary}, image, mask)
    provenance["capture_phase"] = "before any measured subprocess"
    (work_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    stages = work_dir / "official_stages"
    reference_dir = stages / "official_pbr_reference"
    official_full_timing_dir = work_dir / "official_full_glb_timing"
    controlled_glb = work_dir / "controlled_native_pbr.glb"
    native_pose_oracle = work_dir / "native_pose_oracle.json"
    operator_oracle_dir = work_dir / "operator_oracle"
    operator_oracle_summary = operator_oracle_dir / "operator_oracle_summary.json"
    operator_oracle_vulkan_dir = work_dir / "operator_oracle_vulkan"
    operator_oracle_vulkan_summary = operator_oracle_vulkan_dir / "operator_oracle_summary.json"
    matrix_dir = work_dir / "matrix"
    matrix_report = matrix_dir / "e2e_latency.json"
    cross_backend_rng_report = work_dir / "cross_backend_rng.json"
    steps: list[dict[str, Any]] = []

    initial_exclusive_ok = require_exclusive_cuda("gpu_exclusive_before_oracle", steps)
    unique_cuda_binaries = []
    for candidate in (cuda_binary, pbr_binary):
        if candidate not in unique_cuda_binaries:
            unique_cuda_binaries.append(candidate)
    runtime_ok = initial_exclusive_ok
    if initial_exclusive_ok:
        for index, binary in enumerate(unique_cuda_binaries):
            runtime_ok &= run_step(
                f"native_runtime_no_cudnn_{index}",
                [sys.executable, str(SCRIPTS / "verify_native_runtime.py"), "--binary", str(binary)],
                steps,
            ) == 0
        runtime_ok &= run_step(
            "native_runtime_no_cudnn_vulkan",
            [sys.executable, str(SCRIPTS / "verify_native_runtime.py"), "--binary", str(vulkan_binary)],
            steps,
        ) == 0
    else:
        steps.append({"name": "native_runtime_no_cudnn", "status": "SKIPPED",
                      "reason": "GPU compute exclusivity failed before oracle generation"})

    official_full_timing_ok = False
    if args.measure_official_full_glb and runtime_ok and require_exclusive_cuda(
            "gpu_exclusive_before_official_full_glb_timing", steps):
        official_full_timing_ok = run_step(
            "official_full_image_to_textured_glb_timing",
            [
                str(official_python), str(SCRIPTS / "bench_full_e2e_pipeline.py"),
                "--image", str(image), "--mask-dir", str(mask_dir),
                "--mask-index", str(args.mask_index), "--config", str(config),
                "--out-dir", str(official_full_timing_dir), "--seed", str(args.seed),
                "--warmup", "0", "--iters", "1", "--stream-weights",
                "--require-exclusive-gpu",
            ],
            steps,
        ) == 0
    elif args.measure_official_full_glb:
        steps.append({"name": "official_full_image_to_textured_glb_timing", "status": "SKIPPED",
                      "reason": "native runtime or GPU compute exclusivity check failed"})

    stage_ok = False
    cuda_rng_stream_ok = False
    cuda_randperm_ok = False
    cuda_coordinate_downsample_ok = False
    cross_backend_rng_ok = False
    cross_backend_randperm_ok = False
    cross_backend_coordinate_downsample_ok = False
    pytorch_philox_distribution_blocks: int | None = None
    native_pose_contract_ok = False
    if runtime_ok:
        stage_ok = run_step(
            "official_stage_and_pbr_oracle",
            [
                str(official_python), str(SCRIPTS / "dump_e2e_stages.py"),
                "--image", str(image), "--mask-dir", str(mask_dir),
                "--mask-index", str(args.mask_index), "--out-dir", str(stages),
                "--config", str(config), "--seed", str(args.seed),
                "--dump-final-pbr-reference",
            ],
            steps,
        ) == 0
    else:
        steps.append({"name": "official_stage_and_pbr_oracle", "status": "SKIPPED",
                      "reason": "native runtime or GPU compute exclusivity check failed"})
    if stage_ok and require_exclusive_cuda("gpu_exclusive_before_controlled_pbr", steps):
        stage_ok &= run_step(
            "verify_official_pbr_oracle",
            [
                str(official_python), str(SCRIPTS / "verify_official_pbr_reference.py"),
                "--reference-dir", str(reference_dir), "--stage-dir", str(stages),
            ],
            steps,
        ) == 0

    # Two official entry points are independent runs. Measure their agreement
    # without choosing acceptance limits from the observed candidate errors.
    if stage_ok and official_full_timing_ok:
        run_step(
            "official_timed_vs_stage_glb_diagnostic",
            [str(official_python), str(SCRIPTS / "render_glb_compare.py"),
             "--reference-glb", str(reference_dir / "official_pbr.glb"),
             "--native-glb", str(work_dir / "official_full_glb_timing/official_pbr_00.glb"),
             "--out-dir", str(work_dir / "official_reference_agreement"),
             "--frames", str(args.render_frames), "--resolution", str(args.render_resolution)],
            steps,
        )

    # The stage oracle records the actual SS and SLat draw sizes for this
    # input. Verify the entire draw stream before attributing a later image
    # difference to attention, quantization, or PBR post-processing.
    if stage_ok:
        cuda_rng_stream_ok = True
        cuda_randperm_ok = True
        cuda_coordinate_downsample_ok = True
        for index, binary in enumerate(unique_cuda_binaries):
            cuda_rng_stream_ok &= run_step(
                f"native_cuda_rng_stream_matches_official_{index}",
                [
                    str(official_python), str(SCRIPTS / "verify_pytorch_cuda_rng.py"),
                    "--binary", str(binary), "--seed", str(args.seed),
                    "--reference-dir", str(stages),
                ],
                steps,
            ) == 0
            cuda_randperm_ok &= run_step(
                f"native_cuda_randperm_matches_pytorch_{index}",
                [
                    str(official_python), str(SCRIPTS / "verify_pytorch_cuda_randperm.py"),
                    "--cuda-binary", str(binary), "--portable-binary", str(binary),
                    "--seed", str(args.seed), "--reference-dir", str(stages),
                ],
                steps,
            ) == 0
            cuda_coordinate_downsample_ok &= run_step(
                f"native_cuda_coordinate_downsample_matches_pytorch_{index}",
                [
                    str(official_python), str(SCRIPTS / "verify_pytorch_coordinate_downsample.py"),
                    "--cuda-binary", str(binary), "--portable-binary", str(binary),
                    "--seed", str(args.seed), "--reference-dir", str(stages),
                ],
                steps,
            ) == 0
        stage_ok &= (cuda_rng_stream_ok and cuda_randperm_ok and
                     cuda_coordinate_downsample_ok)
    else:
        steps.append({"name": "native_cuda_rng_stream_matches_official", "status": "SKIPPED",
                      "reason": "official stage oracle did not complete"})
        steps.append({"name": "native_cuda_randperm_matches_pytorch", "status": "SKIPPED",
                      "reason": "official stage oracle did not complete"})
        steps.append({"name": "native_cuda_coordinate_downsample_matches_pytorch", "status": "SKIPPED",
                      "reason": "official stage oracle did not complete"})

    # CPU/Vulkan construct the same Philox state and scatter layout on the
    # host. Freeze the CUDA reference launch capacity and compare both the
    # complete fresh SS/SLat draw stream and coordinate-selection randperm
    # before raw neural output is interpreted.
    if stage_ok:
        cross_backend_rng_ok = run_step(
            "native_vulkan_rng_stream_matches_cuda_philox",
            [
                str(official_python), str(SCRIPTS / "verify_portable_philox_rng.py"),
                "--cuda-binary", str(cuda_binary), "--portable-binary", str(vulkan_binary),
                "--seed", str(args.seed), "--reference-dir", str(stages),
                "--json-out", str(cross_backend_rng_report),
            ],
            steps,
        ) == 0
        cross_backend_rng = read_json_object(cross_backend_rng_report)
        if cross_backend_rng is not None:
            blocks = cross_backend_rng.get("distribution_blocks")
            if isinstance(blocks, int) and blocks > 0:
                pytorch_philox_distribution_blocks = blocks
            else:
                cross_backend_rng_ok = False
        else:
            cross_backend_rng_ok = False
        cross_backend_randperm_ok = run_step(
            "native_vulkan_randperm_matches_pytorch",
            [
                str(official_python), str(SCRIPTS / "verify_pytorch_cuda_randperm.py"),
                "--cuda-binary", str(cuda_binary), "--portable-binary", str(vulkan_binary),
                "--seed", str(args.seed), "--reference-dir", str(stages),
            ],
            steps,
        ) == 0
        cross_backend_coordinate_downsample_ok = run_step(
            "native_vulkan_coordinate_downsample_matches_pytorch",
            [
                str(official_python), str(SCRIPTS / "verify_pytorch_coordinate_downsample.py"),
                "--cuda-binary", str(cuda_binary), "--portable-binary", str(vulkan_binary),
                "--seed", str(args.seed), "--reference-dir", str(stages),
            ],
            steps,
        ) == 0
        cross_backend_rng_ok &= (cross_backend_randperm_ok and
                                 cross_backend_coordinate_downsample_ok)
        stage_ok &= cross_backend_rng_ok
    else:
        steps.append({"name": "native_vulkan_rng_stream_matches_cuda_philox", "status": "SKIPPED",
                      "reason": "official stage oracle or CUDA RNG contract did not complete"})
        steps.append({"name": "native_vulkan_randperm_matches_pytorch", "status": "SKIPPED",
                      "reason": "official stage oracle or CUDA RNG contract did not complete"})
        steps.append({"name": "native_vulkan_coordinate_downsample_matches_pytorch", "status": "SKIPPED",
                      "reason": "official stage oracle or CUDA RNG contract did not complete"})

    # Pose is an official output of the point-map pipeline, independent of
    # Gaussian/mesh decoding. Verify its exact ScaleShiftInvariant conversion
    # against the same fresh stage tensors before accepting later asset scores.
    if stage_ok:
        native_pose_contract_ok = run_step(
            "verify_native_pose_decoder",
            [
                str(official_python), str(SCRIPTS / "verify_native_pose_decoder.py"),
                "--binary", str(pbr_binary),
                "--rotation-6d", str(stages / "ss_pose_latent_6drotation_normalized.samt"),
                "--log-scale", str(stages / "ss_pose_latent_scale.samt"),
                "--translation", str(stages / "ss_pose_latent_translation.samt"),
                "--log-translation-scale", str(stages / "ss_pose_latent_translation_scale.samt"),
                "--scene-scale", str(stages / "ss_input_pointmap_scale.samt"),
                "--scene-shift", str(stages / "ss_input_pointmap_shift.samt"),
                "--out", str(native_pose_oracle),
            ],
            steps,
        ) == 0
        stage_ok &= native_pose_contract_ok
    else:
        steps.append({"name": "verify_native_pose_decoder", "status": "SKIPPED",
                      "reason": "official stage oracle or CUDA RNG contract did not complete"})

    # This oracle observes individual flow boundaries after both reference and
    # native model capture. It is opt-in until repeat runs establish empirical
    # per-boundary limits, and a diagnostic completion never counts as a pass.
    operator_oracle_completed = False
    operator_oracle_passed = False
    operator_trajectory_completed = False
    operator_trajectory_passed = False
    operator_oracle_vulkan_completed = False
    operator_oracle_vulkan_passed = False
    operator_trajectory_vulkan_completed = False
    operator_trajectory_vulkan_passed = False
    if args.operator_oracle and stage_ok and runtime_ok:
        operator_command = [
            str(official_python), str(SCRIPTS / "run_operator_oracle.py"),
            "--official-python", str(official_python), "--binary", str(pbr_binary),
            "--models-dir", str(models_dir), "--e2e-dir", str(stages),
            "--out-dir", str(operator_oracle_dir), "--backend", "cuda",
            "--dtype", args.operator_oracle_dtype, "--threads", str(args.threads),
            "--ss-attention", "strict", "--reference-weight-scope",
            args.operator_oracle_reference_weight_scope,
        ]
        if args.operator_oracle_max_stage_mae is not None:
            operator_command.extend(("--max-stage-mae", str(args.operator_oracle_max_stage_mae)))
        if args.operator_oracle_max_stage_abs is not None:
            operator_command.extend(("--max-stage-abs", str(args.operator_oracle_max_stage_abs)))
        if args.operator_oracle_trajectory:
            operator_command.extend((
                "--trajectory", "--trajectory-ss-compute-dtype",
                args.operator_oracle_trajectory_ss_compute_dtype,
                "--trajectory-slat-compute-dtype",
                args.operator_oracle_trajectory_slat_compute_dtype,
            ))
        if args.operator_oracle_trajectory_ss_baseline is not None:
            operator_command.extend((
                "--trajectory-ss-baseline",
                str(args.operator_oracle_trajectory_ss_baseline)))
        if args.operator_oracle_trajectory_slat_max_terminal_mae is not None:
            operator_command.extend((
                "--trajectory-slat-max-terminal-mae",
                str(args.operator_oracle_trajectory_slat_max_terminal_mae)))
        if require_exclusive_cuda("gpu_exclusive_before_operator_oracle", steps):
            operator_oracle_completed = run_step(
                "official_operator_boundary_oracle", operator_command, steps) == 0
            operator_report = read_json_object(operator_oracle_summary)
            operator_oracle_passed = bool(operator_report and operator_report.get("passed") is True)
            trajectory_report = operator_report.get("trajectory") if operator_report else None
            if isinstance(trajectory_report, dict):
                operator_trajectory_completed = bool(trajectory_report.get("completed"))
                operator_trajectory_passed = trajectory_report.get("passed") is True
        else:
            steps.append({"name": "official_operator_boundary_oracle", "status": "SKIPPED",
                          "reason": "GPU compute exclusivity failed"})
    elif args.operator_oracle:
        steps.append({"name": "official_operator_boundary_oracle", "status": "SKIPPED",
                      "reason": "native runtime or official stage oracle did not complete"})

    if args.operator_oracle_vulkan and args.operator_oracle and stage_ok and runtime_ok:
        vulkan_operator_command = [
            str(official_python), str(SCRIPTS / "run_operator_oracle.py"),
            "--official-python", str(official_python), "--binary", str(vulkan_binary),
            "--models-dir", str(models_dir), "--e2e-dir", str(stages),
            "--out-dir", str(operator_oracle_vulkan_dir), "--backend", "vulkan",
            "--dtype", args.operator_oracle_dtype, "--threads", str(args.threads),
            "--ss-attention", "strict", "--reference-weight-scope",
            args.operator_oracle_reference_weight_scope,
        ]
        if args.operator_oracle_max_stage_mae is not None:
            vulkan_operator_command.extend((
                "--max-stage-mae", str(args.operator_oracle_max_stage_mae)))
        if args.operator_oracle_max_stage_abs is not None:
            vulkan_operator_command.extend((
                "--max-stage-abs", str(args.operator_oracle_max_stage_abs)))
        if args.operator_oracle_trajectory:
            vulkan_operator_command.extend((
                "--trajectory", "--trajectory-ss-compute-dtype",
                args.operator_oracle_trajectory_ss_compute_dtype,
                "--trajectory-slat-compute-dtype",
                args.operator_oracle_trajectory_slat_compute_dtype,
            ))
        if args.operator_oracle_trajectory_ss_baseline is not None:
            vulkan_operator_command.extend((
                "--trajectory-ss-baseline",
                str(args.operator_oracle_trajectory_ss_baseline)))
        if args.operator_oracle_trajectory_slat_max_terminal_mae is not None:
            vulkan_operator_command.extend((
                "--trajectory-slat-max-terminal-mae",
                str(args.operator_oracle_trajectory_slat_max_terminal_mae)))
        if require_exclusive_cuda("gpu_exclusive_before_vulkan_operator_oracle", steps):
            operator_oracle_vulkan_completed = run_step(
                "official_vulkan_operator_boundary_oracle", vulkan_operator_command, steps) == 0
            operator_vulkan_report = read_json_object(operator_oracle_vulkan_summary)
            operator_oracle_vulkan_passed = bool(
                operator_vulkan_report and operator_vulkan_report.get("passed") is True)
            trajectory_report = (
                operator_vulkan_report.get("trajectory") if operator_vulkan_report else None)
            if isinstance(trajectory_report, dict):
                operator_trajectory_vulkan_completed = bool(trajectory_report.get("completed"))
                operator_trajectory_vulkan_passed = trajectory_report.get("passed") is True
        else:
            steps.append({"name": "official_vulkan_operator_boundary_oracle", "status": "SKIPPED",
                          "reason": "GPU compute exclusivity failed"})
    elif args.operator_oracle_vulkan:
        steps.append({"name": "official_vulkan_operator_boundary_oracle", "status": "SKIPPED",
                      "reason": "requires --operator-oracle plus native runtime and fresh stage"})

    controlled_ok = False
    if stage_ok:
        controlled_ok = run_step(
            "controlled_native_pbr_assembly",
            [
                str(pbr_binary), "pbr-assemble", "--vertices", str(stages / "decode_mesh_vertices.samt"),
                "--faces", str(stages / "decode_mesh_faces.samt"), "--ply", str(stages / "output_gs.ply"),
                "--out", str(controlled_glb), "--views", "100", "--resolution", "1024",
                "--texture-size", "1024", "--steps", "2500", "--seed", str(args.seed),
            ],
            steps,
        ) == 0
        if controlled_ok:
            controlled_verify_command = [
                str(sys.executable), str(SCRIPTS / "verify_native_pbr_pipeline.py"),
                "--native-glb", str(controlled_glb), "--stage-dir", str(stages),
                "--reference-dir", str(reference_dir),
                "--require-texture-comparison", "--require-normal-comparison",
                "--json-out", str(work_dir / "controlled_pbr_report.json"),
            ]
            for option, threshold in (
                ("--max-texture-mae-u8", args.max_texture_mae_u8),
                ("--max-texture-rmse-u8", args.max_texture_rmse_u8),
                ("--max-texture-abs-u8", args.max_texture_abs_u8),
                ("--max-normal-angle-deg", args.max_normal_angle_deg),
            ):
                if threshold is not None:
                    controlled_verify_command.extend((option, str(threshold)))
            controlled_ok &= run_step(
                "verify_controlled_native_pbr",
                controlled_verify_command,
                steps,
            ) == 0
    else:
        steps.append({"name": "controlled_native_pbr_assembly", "status": "SKIPPED",
                      "reason": "official oracle generation failed"})

    raw_pbr_results: dict[str, bool] = {}
    raw_pbr_artifacts: dict[str, dict[str, str]] = {}
    raw_glb_structure_results: dict[str, bool] = {}
    raw_glb_render_measurements: dict[str, bool] = {}
    raw_glb_render_results: dict[str, bool] = {}
    raw_dtype_contract_results: dict[str, bool] = {}
    if stage_ok and runtime_ok:
        for dtype in ("f16", "q8_0", "q4_k"):
            case_dir = work_dir / f"raw_native_cuda_{dtype}"
            case_dir.mkdir(parents=True, exist_ok=True)
            raw_ply = case_dir / "output.ply"
            raw_glb = case_dir / "output.glb"
            raw_pose = case_dir / "pose.json"
            raw_dtype_contract = case_dir / "dtype_contract.json"
            raw_render_dir = case_dir / "render_compare"
            raw_glb_render_dir = case_dir / "glb_render_compare"
            raw_pbr_artifacts[dtype] = {
                "ply": str(raw_ply), "glb": str(raw_glb),
                "pose": str(raw_pose),
                "sampling_mode": (
                    "official-noise-replay-diagnostic" if args.diagnostic_noise_replay else "native-seed"),
                "dtype_contract": str(raw_dtype_contract),
                "dtype_contract_validation": str(case_dir / "dtype_contract_report.json"),
                "render_report": str(raw_render_dir / "render_metrics.json"),
                "glb_render_report": str(raw_glb_render_dir / "glb_render_metrics.json"),
            }
            case_ok = require_exclusive_cuda(
                f"gpu_exclusive_before_raw_native_cuda_{dtype}", steps)
            structure_ok = False
            if case_ok:
                raw_command = [
                    str(pbr_binary), "image-to-3d", "--model", str(models_dir),
                    "--moge-model", str(moge_model),
                    "--image", str(image), "--mask", str(mask), "--backend", "cuda", "--dtype", dtype,
                    "--seed", str(args.seed), "--threads", str(args.threads), "--out", str(raw_ply),
                    "--pbr-out", str(raw_glb), "--pose-out", str(raw_pose),
                    "--dtype-contract-out", str(raw_dtype_contract),
                    "--ss-attention", args.ss_attention,
                ]
                if args.diagnostic_noise_replay:
                    raw_command.extend(("--noise-dir", str(stages)))
                case_ok = run_step(
                    f"raw_native_cuda_{dtype}_image_to_pbr",
                    raw_command,
                    steps,
                ) == 0
            dtype_contract_ok = False
            if case_ok:
                dtype_contract_ok = run_step(
                    f"verify_raw_native_cuda_{dtype}_dtype_contract",
                    [
                        str(sys.executable), str(SCRIPTS / "verify_dtype_contract.py"),
                        "--contract", str(raw_dtype_contract),
                        "--expected-weight-type", dtype,
                        "--require-stage", "mesh_decoder",
                        "--json-out", str(case_dir / "dtype_contract_report.json"),
                    ],
                    steps,
                ) == 0
                case_ok &= dtype_contract_ok
            raw_dtype_contract_results[dtype] = dtype_contract_ok
            if case_ok:
                structure_ok = run_step(
                    f"verify_raw_native_cuda_{dtype}_pbr_structure",
                    [
                        str(sys.executable), str(SCRIPTS / "verify_native_pbr_pipeline.py"),
                        "--native-glb", str(raw_glb), "--structural-only",
                        "--json-out", str(case_dir / "pbr_structure_report.json"),
                    ],
                    steps,
                ) == 0
                case_ok &= structure_ok
            raw_glb_structure_results[dtype] = structure_ok
            if case_ok:
                glb_compare_command = [
                    str(official_python), str(SCRIPTS / "render_glb_compare.py"),
                    "--reference-glb", str(reference_dir / "official_pbr.glb"),
                    "--native-glb", str(raw_glb), "--out-dir", str(raw_glb_render_dir),
                    "--frames", str(args.render_frames), "--resolution", str(args.render_resolution),
                ]
                for option, threshold in (
                    ("--max-rgb-mae-linear", args.max_glb_rgb_mae_linear),
                    ("--min-mask-iou", args.min_glb_mask_iou),
                    ("--max-normal-angle-deg", args.max_glb_normal_angle_deg),
                    ("--max-depth-mae-ndc", args.max_glb_depth_mae_ndc),
                ):
                    if threshold is not None:
                        glb_compare_command.extend((option, str(threshold)))
                glb_render_ok = run_step(
                    f"render_compare_raw_native_cuda_{dtype}_glb", glb_compare_command, steps) == 0
                raw_glb_render_measurements[dtype] = bool(read_json_object(
                    raw_glb_render_dir / "glb_render_metrics.json"))
                raw_glb_render_results[dtype] = glb_render_ok and glb_render_gate_configured
                case_ok &= raw_glb_render_results[dtype]
            else:
                raw_glb_render_measurements[dtype] = False
                raw_glb_render_results[dtype] = False
            if raw_ply.is_file():
                case_ok &= run_step(
                    f"render_compare_raw_native_cuda_{dtype}",
                    [
                        str(official_python), str(SCRIPTS / "render_compare.py"),
                        "--pytorch-ply", str(stages / "output_gs.ply"), "--ggml-ply", str(raw_ply),
                        "--condition-dir", str(stages), "--out-dir", str(raw_render_dir),
                        "--frames", str(args.render_frames), "--resolution", str(args.render_resolution),
                        "--max-mae", str(args.max_render_mae),
                    ],
                    steps,
                ) == 0
            else:
                steps.append({
                    "name": f"verify_raw_native_cuda_{dtype}_ply_artifact",
                    "command": f"test -f {raw_ply}",
                    "returncode": 1,
                    "elapsed_ms": 0.0,
                    "status": "FAIL",
                    "error": "native image-to-3d did not produce its required Gaussian PLY artifact",
                })
                case_ok = False
            raw_pbr_results[dtype] = case_ok
    else:
        for dtype in ("f16", "q8_0", "q4_k"):
            raw_pbr_results[dtype] = False
            raw_glb_structure_results[dtype] = False
            raw_glb_render_measurements[dtype] = False
            raw_glb_render_results[dtype] = False
            raw_dtype_contract_results[dtype] = False
            steps.append({"name": f"raw_native_cuda_{dtype}_image_to_pbr", "status": "SKIPPED",
                          "reason": "native runtime or official oracle check failed"})

    vulkan_cuda_pbr_results: dict[str, bool] = {}
    vulkan_cuda_handoff_results: dict[str, bool] = {}
    vulkan_cuda_pbr_artifacts: dict[str, dict[str, str]] = {}
    vulkan_cuda_pbr_structure_results: dict[str, bool] = {}
    vulkan_cuda_pbr_render_measurements: dict[str, bool] = {}
    vulkan_cuda_pbr_render_results: dict[str, bool] = {}
    vulkan_cuda_pbr_dtype_contract_results: dict[str, bool] = {}
    if stage_ok and runtime_ok and pytorch_philox_distribution_blocks is not None:
        for dtype in ("f16", "q8_0", "q4_k"):
            case_dir = work_dir / f"raw_vulkan_cuda_pbr_{dtype}"
            handoff_manifest = case_dir / "handoff_manifest.json"
            handoff_ply = case_dir / "vulkan_output.ply"
            handoff_glb = case_dir / "cuda_pbr.glb"
            handoff_dtype_contract = case_dir / "vulkan_dtype_contract.json"
            handoff_render_dir = case_dir / "render_compare"
            handoff_glb_render_dir = case_dir / "glb_render_compare"
            vulkan_cuda_pbr_artifacts[dtype] = {
                "manifest": str(handoff_manifest), "ply": str(handoff_ply),
                "glb": str(handoff_glb), "dtype_contract": str(handoff_dtype_contract),
                "sampling_mode": (
                    "official-noise-replay-diagnostic" if args.diagnostic_noise_replay else "native-seed"),
                "render_report": str(handoff_render_dir / "render_metrics.json"),
                "glb_render_report": str(handoff_glb_render_dir / "glb_render_metrics.json"),
            }
            case_ok = require_exclusive_cuda(
                f"gpu_exclusive_before_raw_vulkan_cuda_pbr_{dtype}", steps)
            handoff_ok = False
            if case_ok:
                handoff_command = [
                    str(sys.executable), str(SCRIPTS / "run_vulkan_cuda_pbr.py"),
                    "--vulkan-binary", str(vulkan_binary), "--cuda-pbr-binary", str(pbr_binary),
                    "--models-dir", str(models_dir), "--moge-model", str(moge_model),
                    "--image", str(image), "--mask", str(mask), "--out-dir", str(case_dir),
                    "--dtype", dtype, "--seed", str(args.seed), "--threads", str(args.threads),
                    "--ss-attention", "strict", "--rng-distribution-blocks",
                    str(pytorch_philox_distribution_blocks),
                ]
                if args.diagnostic_noise_replay:
                    handoff_command.extend(("--noise-dir", str(stages)))
                handoff_ok = run_step(
                    f"raw_vulkan_cuda_pbr_{dtype}_handoff", handoff_command, steps) == 0
                handoff_ok &= valid_vulkan_cuda_handoff(handoff_manifest)
                if not handoff_ok:
                    steps.append({
                        "name": f"verify_raw_vulkan_cuda_pbr_{dtype}_handoff_manifest",
                        "command": f"validate {handoff_manifest}",
                        "returncode": 1,
                        "elapsed_ms": 0.0,
                        "status": "FAIL",
                        "error": "handoff manifest is incomplete or does not name Vulkan inference plus CUDA PBR",
                    })
            vulkan_cuda_handoff_results[dtype] = handoff_ok
            dtype_contract_ok = False
            if handoff_ok:
                dtype_contract_ok = run_step(
                    f"verify_raw_vulkan_cuda_pbr_{dtype}_dtype_contract",
                    [
                        str(sys.executable), str(SCRIPTS / "verify_dtype_contract.py"),
                        "--contract", str(handoff_dtype_contract), "--expected-weight-type", dtype,
                        "--require-stage", "mesh_decoder",
                        "--json-out", str(case_dir / "dtype_contract_report.json"),
                    ],
                    steps,
                ) == 0
                case_ok &= dtype_contract_ok
            vulkan_cuda_pbr_dtype_contract_results[dtype] = dtype_contract_ok
            structure_ok = False
            if handoff_ok and dtype_contract_ok:
                structure_ok = run_step(
                    f"verify_raw_vulkan_cuda_pbr_{dtype}_structure",
                    [
                        str(sys.executable), str(SCRIPTS / "verify_native_pbr_pipeline.py"),
                        "--native-glb", str(handoff_glb), "--structural-only",
                        "--json-out", str(case_dir / "pbr_structure_report.json"),
                    ],
                    steps,
                ) == 0
                case_ok &= structure_ok
            vulkan_cuda_pbr_structure_results[dtype] = structure_ok
            glb_render_ok = False
            if handoff_ok and dtype_contract_ok and structure_ok:
                glb_compare_command = [
                    str(official_python), str(SCRIPTS / "render_glb_compare.py"),
                    "--reference-glb", str(reference_dir / "official_pbr.glb"),
                    "--native-glb", str(handoff_glb), "--out-dir", str(handoff_glb_render_dir),
                    "--frames", str(args.render_frames), "--resolution", str(args.render_resolution),
                ]
                for option, threshold in (
                    ("--max-rgb-mae-linear", args.max_glb_rgb_mae_linear),
                    ("--min-mask-iou", args.min_glb_mask_iou),
                    ("--max-normal-angle-deg", args.max_glb_normal_angle_deg),
                    ("--max-depth-mae-ndc", args.max_glb_depth_mae_ndc),
                ):
                    if threshold is not None:
                        glb_compare_command.extend((option, str(threshold)))
                glb_render_ok = run_step(
                    f"render_compare_raw_vulkan_cuda_pbr_{dtype}_glb", glb_compare_command, steps) == 0
                vulkan_cuda_pbr_render_measurements[dtype] = bool(read_json_object(
                    handoff_glb_render_dir / "glb_render_metrics.json"))
                vulkan_cuda_pbr_render_results[dtype] = glb_render_ok and glb_render_gate_configured
                case_ok &= vulkan_cuda_pbr_render_results[dtype]
            else:
                vulkan_cuda_pbr_render_measurements[dtype] = False
                vulkan_cuda_pbr_render_results[dtype] = False
            if handoff_ok and handoff_ply.is_file():
                case_ok &= run_step(
                    f"render_compare_raw_vulkan_cuda_pbr_{dtype}",
                    [
                        str(official_python), str(SCRIPTS / "render_compare.py"),
                        "--pytorch-ply", str(stages / "output_gs.ply"), "--ggml-ply", str(handoff_ply),
                        "--condition-dir", str(stages), "--out-dir", str(handoff_render_dir),
                        "--frames", str(args.render_frames), "--resolution", str(args.render_resolution),
                        "--max-mae", str(args.max_render_mae),
                    ],
                    steps,
                ) == 0
            else:
                case_ok = False
            vulkan_cuda_pbr_results[dtype] = case_ok
    else:
        reason = "native runtime, official oracle, or CUDA Philox distribution-block contract did not complete"
        for dtype in ("f16", "q8_0", "q4_k"):
            vulkan_cuda_pbr_results[dtype] = False
            vulkan_cuda_pbr_structure_results[dtype] = False
            vulkan_cuda_pbr_render_measurements[dtype] = False
            vulkan_cuda_pbr_render_results[dtype] = False
            vulkan_cuda_pbr_dtype_contract_results[dtype] = False
            steps.append({"name": f"raw_vulkan_cuda_pbr_{dtype}_handoff", "status": "SKIPPED",
                          "reason": reason})

    matrix_ok = False
    if args.skip_neural_diagnostic_matrix:
        steps.append({"name": "raw_native_cuda_vulkan_matrix", "status": "NOT_REQUESTED",
                      "reason": "extra neural-only benchmark omitted; six raw full-GLB cases remain mandatory"})
    elif stage_ok and runtime_ok and require_exclusive_cuda("gpu_exclusive_before_matrix", steps):
        matrix_ok = run_step(
            "raw_native_cuda_vulkan_matrix",
            [
                str(official_python), str(SCRIPTS / "run_e2e_matrix.py"),
                "--official-python", str(official_python), "--image", str(image),
                "--mask-dir", str(mask_dir), "--mask-index", str(args.mask_index),
                "--models-dir", str(models_dir), "--conditions-dir", str(stages),
                "--moge-model", str(moge_model),
                "--cuda-build-dir", str(cuda_binary.parent.parent),
                "--vulkan-build-dir", str(vulkan_binary.parent.parent),
                "--work-dir", str(matrix_dir), "--output-report", str(matrix_report),
                "--seed", str(args.seed), "--threads", str(args.threads),
                "--rng-distribution-blocks", str(pytorch_philox_distribution_blocks),
                "--render-frames", str(args.render_frames),
                "--render-resolution", str(args.render_resolution),
            ] + (["--diagnostic-noise-replay"] if args.diagnostic_noise_replay else []),
            steps,
        ) == 0
    else:
        steps.append({"name": "raw_native_cuda_vulkan_matrix", "status": "SKIPPED",
                      "reason": "native runtime, official oracle, or GPU compute exclusivity check failed"})

    required_raw_dtypes = ("f16", "q8_0", "q4_k")
    coverage = {
        "controlled_pbr_texture_and_normals": controlled_ok and controlled_pbr_gate_configured,
        "cuda_rng_stream_matches_official": cuda_rng_stream_ok,
        "cuda_randperm_matches_pytorch": cuda_randperm_ok,
        "cuda_coordinate_downsample_matches_pytorch": cuda_coordinate_downsample_ok,
        "raw_cuda_glb_structure": all(raw_glb_structure_results.get(dtype, False)
                                       for dtype in required_raw_dtypes),
        "final_material_contract": all(
            raw_glb_structure_results.get(dtype, False) and
            vulkan_cuda_pbr_structure_results.get(dtype, False) for dtype in required_raw_dtypes),
        "raw_cuda_glb_multiview_render": all(raw_glb_render_results.get(dtype, False)
                                              for dtype in required_raw_dtypes),
        "official_full_glb_timing": official_full_timing_ok,
        "native_full_glb_hot_timing": False,
        "cross_backend_rng_contract": cross_backend_rng_ok,
        "cross_backend_randperm_contract": cross_backend_randperm_ok,
        "cross_backend_coordinate_downsample_contract": cross_backend_coordinate_downsample_ok,
        "official_operator_boundary_oracle": operator_oracle_passed,
        "per_operator_dtype_contract": all(
            raw_dtype_contract_results.get(dtype, False) and
            vulkan_cuda_pbr_dtype_contract_results.get(dtype, False) for dtype in required_raw_dtypes),
        "native_pose_contract": native_pose_contract_ok,
        "vulkan_inference_cuda_pbr_handoff": all(
            vulkan_cuda_handoff_results.get(dtype, False) for dtype in required_raw_dtypes),
    }
    checks_passed = all(step.get("status") == "PASS" for step in steps
                        if not (step.get("name") == "raw_native_cuda_vulkan_matrix" and
                                step.get("status") == "NOT_REQUESTED"))
    report = {
        "schema": "sam3d.full-e2e-parity.v2",
        "input": {"image": str(image), "mask": str(mask), "seed": args.seed},
        "candidate_sampling_mode": (
            "official-noise-replay-diagnostic" if args.diagnostic_noise_replay else "native-seed"),
        "preprocessing_model": str(moge_model),
        "provenance": provenance,
        "pytorch_philox_distribution_blocks": pytorch_philox_distribution_blocks,
        "pytorch_cuda_randperm_matches_official": cuda_randperm_ok,
        "cross_backend_randperm_matches_pytorch": cross_backend_randperm_ok,
        "pytorch_coordinate_downsample_matches_official": cuda_coordinate_downsample_ok,
        "cross_backend_coordinate_downsample_matches_pytorch": cross_backend_coordinate_downsample_ok,
        "quantization_scope": "dtype selects all generative GGUF stages; the recorded MoGe preprocessing GGUF is identical for every dtype row",
        "binaries": {
            "cuda": str(cuda_binary), "vulkan": str(vulkan_binary), "native_pbr": str(pbr_binary),
        },
        "artifacts": {
            "official_stages": str(stages), "official_pbr_reference": str(reference_dir),
            "cross_backend_rng_report": str(cross_backend_rng_report),
            "official_full_glb_timing": str(official_full_timing_dir),
            "controlled_native_glb": str(controlled_glb), "native_pose_oracle": str(native_pose_oracle),
            "operator_oracle": str(operator_oracle_summary),
            "operator_oracle_vulkan": str(operator_oracle_vulkan_summary),
            "raw_cuda_pbr": raw_pbr_artifacts,
            "raw_vulkan_cuda_pbr": vulkan_cuda_pbr_artifacts,
            "matrix_report": str(matrix_report),
        },
        "controlled_native_pbr_measured": controlled_ok,
        "controlled_native_pbr_passed": controlled_ok and controlled_pbr_gate_configured,
        "controlled_native_pbr_gate_configured": controlled_pbr_gate_configured,
        "raw_native_cuda_pbr_passed": raw_pbr_results,
        "raw_native_cuda_glb_structure_passed": raw_glb_structure_results,
        "raw_native_cuda_glb_render_measured": raw_glb_render_measurements,
        "raw_native_cuda_glb_render_gate_passed": raw_glb_render_results,
        "glb_render_acceptance_thresholds": glb_render_thresholds,
        "glb_render_acceptance_gate_configured": glb_render_gate_configured,
        "raw_native_cuda_dtype_contract_passed": raw_dtype_contract_results,
        "raw_vulkan_cuda_pbr_passed": vulkan_cuda_pbr_results,
        "raw_vulkan_cuda_handoff_passed": vulkan_cuda_handoff_results,
        "raw_vulkan_cuda_pbr_structure_passed": vulkan_cuda_pbr_structure_results,
        "raw_vulkan_cuda_pbr_render_measured": vulkan_cuda_pbr_render_measurements,
        "raw_vulkan_cuda_pbr_render_gate_passed": vulkan_cuda_pbr_render_results,
        "raw_vulkan_cuda_pbr_dtype_contract_passed": vulkan_cuda_pbr_dtype_contract_results,
        "official_operator_boundary_oracle_completed": operator_oracle_completed,
        "official_operator_boundary_oracle_passed": operator_oracle_passed,
        "official_operator_boundary_oracle_reference_weight_scope": (
            args.operator_oracle_reference_weight_scope if args.operator_oracle else None),
        "official_operator_trajectory_requested": (
            args.operator_oracle and args.operator_oracle_trajectory),
        "official_operator_trajectory_completed": operator_trajectory_completed,
        "official_operator_trajectory_passed": operator_trajectory_passed,
        "official_operator_vulkan_requested": (
            args.operator_oracle and args.operator_oracle_vulkan),
        "official_operator_vulkan_completed": operator_oracle_vulkan_completed,
        "official_operator_vulkan_passed": operator_oracle_vulkan_passed,
        "official_operator_trajectory_vulkan_completed": operator_trajectory_vulkan_completed,
        "official_operator_trajectory_vulkan_passed": operator_trajectory_vulkan_passed,
        "raw_neural_diagnostic_matrix_requested": not args.skip_neural_diagnostic_matrix,
        "raw_cuda_vulkan_matrix_passed": None if args.skip_neural_diagnostic_matrix else matrix_ok,
        "coverage": coverage,
        "steps": steps,
        "checks_passed": checks_passed,
        "passed": checks_passed and all(coverage.values()),
    }
    report_path = work_dir / "full_e2e_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
