#!/usr/bin/env python3
"""Idempotent multi-variant scene benchmark driver.

Runs the full-scene reconstruction (official demo_multi_object flow) for the
PyTorch reference and the native variants, then renders every variant on the
same orbit contract (300 frames, radius 1, fov 60, 512 px). Every object and
every assemble step is skipped when its artifact already exists, so the
driver can be interrupted and re-run at any time:

    SAM3D_PYTHON=... python cpp_ggml/scripts/run_scene_benchmark.py \
        --work-dir output/scene-benchmark

Variant layout under --work-dir:
    <variant>/objects/obj_<idx>/{output.ply,pose.json}   (native; tensor cache for pytorch)
    <variant>/scene_posed.ply, frames/frame_*.png, scene_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent

VARIANTS = {
    # name: (backend, dtype)
    "pytorch": (None, None),
    "cuda-f16": ("cuda", "f16"),
    "cuda-q8_0": ("cuda", "q8_0"),
    "cuda-q4_k": ("cuda", "q4_k"),
    "vulkan-q8_0": ("vulkan", "q8_0"),
}


def enumerate_masks(mask_dir: Path, indices: list[int] | None) -> list[tuple[int, Path]]:
    if indices is None:
        index = 0
        indices = []
        while (mask_dir / f"{index}.png").exists():
            indices.append(index)
            index += 1
    masks = [(index, mask_dir / f"{index}.png") for index in indices]
    if not masks:
        raise SystemExit(f"error: no masks found in {mask_dir}")
    for index, path in masks:
        if not path.is_file():
            raise SystemExit(f"error: missing mask: {path}")
    return masks


def run(command: list[str], log: Path, check: bool = True) -> int:
    with log.open("a") as stream:
        stream.write(f"\n$ {' '.join(str(part) for part in command)}\n")
        stream.flush()
        completed = subprocess.run(
            [str(part) for part in command], stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if check and completed.returncode != 0:
        raise SystemExit(f"error: command failed ({completed.returncode}); see {log}")
    return completed.returncode


def philox_blocks(cli: Path, seed: int, work_dir: Path, log: Path) -> int:
    contract_dir = work_dir / "rng_contract"
    if not (contract_dir / "rng_contract.json").is_file():
        run([cli, "rng-dump", "--implementation", "cuda", "--seed", str(seed),
             "--sizes", "1", "--out-dir", contract_dir], log)
    contract = json.loads((contract_dir / "rng_contract.json").read_text(encoding="utf-8"))
    blocks = contract.get("distribution_blocks")
    if not isinstance(blocks, int) or blocks <= 0:
        raise SystemExit("error: invalid CUDA Philox contract")
    return blocks


def run_pytorch(args: argparse.Namespace, masks, variant_dir: Path, log: Path) -> None:
    command = [
        sys.executable, str(SCRIPTS / "run_scene_pipeline.py"),
        "--runner", "python", "--image", args.image, "--mask-dir", args.mask_dir,
        "--config", args.config, "--out-dir", variant_dir, "--seed", args.seed,
        "--num-frames", args.num_frames, "--resolution", args.resolution,
        "--radius", args.radius, "--fov", args.fov, "--resume",
    ]
    if args.mask_indices:
        command += ["--mask-indices", args.mask_indices]
    run(command, log)


def object_dir(objects_dir: Path, index: int) -> Path:
    """Locate an object's outputs; the batch tree drops the zero padding."""
    unpadded = objects_dir / f"obj_{index}"
    if (unpadded / "output.ply").is_file():
        return unpadded
    return objects_dir / f"obj_{index:02d}"


def run_native(args: argparse.Namespace, masks, variant_dir: Path,
               backend: str, dtype: str, log: Path) -> None:
    cli = args.cli_vulkan if backend == "vulkan" else args.cli_cuda
    blocks = None
    if backend == "vulkan":
        blocks = philox_blocks(args.cli_cuda, int(args.seed), variant_dir, log)
    objects_dir = variant_dir / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)
    list_path = variant_dir / "scene_objects.txt"
    entries = []
    failed = []
    if not args.per_process:
        # Session batch (default): one CLI process serves every mask with the
        # shared model resources and the RGB-only MoGe point map reuse. The
        # batch manifest records input/output digests per object, so its
        # resume only skips objects whose stored digests still match the
        # files on disk - stale artifacts can no longer pass as fresh.
        pending = []
        for index, mask_path in masks:
            case_dir = objects_dir / f"obj_{index:02d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            if (case_dir / "FAILED").is_file():
                # Previous attempt diverged; excluded from the scene until the
                # marker is removed.
                failed.append(index)
                continue
            pending.append((index, mask_path))
        if pending:
            mask_list = variant_dir / "batch_mask_list.txt"
            mask_list.write_text(
                "".join(f"{index}\t{mask_path}\n" for index, mask_path in pending),
                encoding="utf-8",
            )
            command = [
                cli, "image-to-3d",
                "--model", args.models_dir, "--moge-model", args.moge_model,
                "--image", args.image,
                "--mask-list", str(mask_list), "--out-dir", str(variant_dir),
                "--backend", backend, "--dtype", dtype,
                "--seed", args.seed, "--threads", args.threads,
                # Scene preset: F16-KV flash attention. A/B on mask 14 and
                # obj 15 after the MoGe strict-KV repair showed the free-run
                # pose parity is attention-mode independent (3.839 vs 3.835
                # deg; 14.30 vs 15.72 deg) while the strict F32-KV path costs
                # 2x SS-flow time. normal is the delivery default since
                # 2026-09-18 (27-scene acceptance + single-object A/B).
                "--ss-attention", "normal",
            ]
            if blocks is not None:
                command += ["--philox-blocks", str(blocks)]
            started = time.time()
            returncode = run(command, log, check=False)
            elapsed = time.time() - started
            manifest_path = variant_dir / "batch_manifest.json"
            if returncode != 0 or not manifest_path.is_file():
                raise SystemExit(
                    f"error: image-to-3d batch exit {returncode} after {elapsed:.1f}s "
                    f"for {variant_dir.name}; see {log}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for obj in manifest["objects"]:
                case_dir = object_dir(objects_dir, obj["id"])
                if obj["status"] == "ok":
                    with log.open("a") as stream:
                        stream.write(f"obj_{obj['id']:02d} done in {obj['elapsed_s']:.1f}s "
                                     f"(moge_reused={obj['moge_pointmap_reused']})\n")
                else:
                    # Divergence mid-batch keeps the same contract as the
                    # per-process runner: record, exclude, keep going.
                    (case_dir / "FAILED").write_text(
                        obj.get("error", "batch object failed") + "\n")
                    failed.append(obj["id"])
                    with log.open("a") as stream:
                        stream.write(f"obj_{obj['id']:02d} FAILED\n")
        else:
            with log.open("a") as stream:
                stream.write(f"variant {variant_dir.name}: nothing to run (all excluded)\n")
    else:
        # Explicit A/B baseline: one process per object, resume only by file
        # presence. Kept to measure the session-batch saving.
        for index, mask_path in masks:
            case_dir = objects_dir / f"obj_{index:02d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            ply_path = case_dir / "output.ply"
            pose_path = case_dir / "pose.json"
            if (case_dir / "FAILED").is_file():
                # Previous attempt diverged; excluded from the scene until the
                # marker is removed.
                failed.append(index)
                continue
            if not (ply_path.is_file() and ply_path.stat().st_size > 0 and
                    pose_path.is_file() and pose_path.stat().st_size > 0):
                # Free-running reconstruction can diverge on individual objects
                # (f16 is the least stable dtype). Record the failure, exclude the
                # object from the scene, and keep the variant going; delete the
                # FAILED marker to retry on the next resume.
                command = [
                    cli, "image-to-3d",
                    "--model", args.models_dir, "--moge-model", args.moge_model,
                    "--image", args.image, "--mask", mask_path,
                    "--backend", backend, "--dtype", dtype,
                    "--seed", args.seed, "--threads", args.threads,
                    # Scene preset: F16-KV flash attention. A/B on mask 14 and
                    # obj 15 after the MoGe strict-KV repair showed the free-run
                    # pose parity is attention-mode independent (3.839 vs 3.835
                    # deg; 14.30 vs 15.72 deg) while the strict F32-KV path costs
                    # 2x SS-flow time. normal is the delivery default since
                    # 2026-09-18 (27-scene acceptance + single-object A/B).
                    "--ss-attention", "normal",
                    "--out", ply_path, "--pose-out", pose_path,
                ]
                if blocks is not None:
                    command += ["--philox-blocks", str(blocks)]
                started = time.time()
                returncode = run(command, log, check=False)
                elapsed = time.time() - started
                if returncode != 0 or not (ply_path.is_file() and pose_path.is_file()):
                    (case_dir / "FAILED").write_text(
                        f"image-to-3d exit {returncode} after {elapsed:.1f}s\n")
                    failed.append(index)
                    with log.open("a") as stream:
                        stream.write(f"obj_{index:02d} FAILED (exit {returncode})\n")
                    continue
                with log.open("a") as stream:
                    stream.write(f"obj_{index:02d} done in {elapsed:.1f}s\n")
            entries.append(f"{ply_path}\t{pose_path}")
    if not args.per_process:
        # The batch tree already lives under objects/obj_<ID>; the scene list
        # mirrors the per-process layout so scene-assemble stays unchanged.
        for index, mask_path in masks:
            case_dir = object_dir(objects_dir, index)
            if (case_dir / "FAILED").is_file():
                continue
            if (case_dir / "output.ply").is_file() and (case_dir / "pose.json").is_file():
                entries.append(f"{case_dir / 'output.ply'}\t{case_dir / 'pose.json'}")
    list_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    if failed:
        with log.open("a") as stream:
            stream.write(f"variant {variant_dir.name}: {len(failed)} diverged "
                         f"objects excluded from the scene: {failed}\n")
    if (variant_dir / "scene_manifest.json").is_file():
        return
    assemble = [
        args.cli_cuda, "scene-assemble", "--objects-list", list_path,
        "--out-dir", variant_dir, "--num-frames", args.num_frames,
        "--resolution", args.resolution, "--radius", args.radius,
        "--fov", args.fov,
    ]
    # Render every variant on the reference's recorded scene frame: the
    # free-running objects shift the per-variant bounds, so re-normalizing
    # each scene independently would compare different camera framings.
    reference_normalization = args.work_dir / args.reference / "normalization.json"
    if reference_normalization.is_file():
        assemble += ["--normalization-json", reference_normalization]
    run(assemble, log)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "output/scene-benchmark")
    parser.add_argument("--image", type=Path,
                        default=REPO_ROOT / "notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
    parser.add_argument("--mask-dir", type=Path,
                        default=REPO_ROOT / "notebook/images/shutterstock_stylish_kidsroom_1640806567")
    parser.add_argument("--mask-indices", type=str, default="")
    parser.add_argument("--variants", type=str,
                        default="pytorch,cuda-f16,cuda-q8_0,cuda-q4_k,vulkan-q8_0")
    parser.add_argument("--reference", type=str, default="pytorch",
                        help="variant whose normalization contract anchors the orbit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--num-frames", type=str, default="300")
    parser.add_argument("--resolution", type=str, default="512")
    parser.add_argument("--radius", type=str, default="1.0")
    parser.add_argument("--fov", type=str, default="60.0")
    # native
    parser.add_argument("--per-process", action="store_true",
                        help="A/B baseline: one CLI process per object instead "
                             "of the default single-process session batch")
    parser.add_argument("--cli-cuda", type=Path,
                        default=REPO_ROOT / "cpp_ggml/build-cuda-pbr/bin/sam3d-cli")
    parser.add_argument("--cli-vulkan", type=Path,
                        default=REPO_ROOT / "cpp_ggml/build-vulkan/bin/sam3d-cli")
    parser.add_argument("--models-dir", type=Path,
                        default=REPO_ROOT / "cpp_ggml/models/gguf")
    parser.add_argument("--moge-model", type=Path)
    # pytorch
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "checkpoints/hf/pipeline.yaml")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"missing image: {args.image}")
    if not args.mask_dir.is_dir():
        parser.error(f"missing mask directory: {args.mask_dir}")
    if not args.cli_cuda.is_file():
        parser.error(f"missing native binary: {args.cli_cuda}")
    if args.moge_model is None:
        args.moge_model = args.models_dir / "moge_vitl-f16.gguf"
    if not args.moge_model.is_file():
        parser.error(f"missing MoGe GGUF: {args.moge_model}")
    indices = ([int(part) for part in args.mask_indices.split(",") if part]
               if args.mask_indices else None)
    masks = enumerate_masks(args.mask_dir, indices)
    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    for name in variants:
        if name not in VARIANTS:
            parser.error(f"unknown variant: {name}")

    logs = args.work_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    overview = {"schema": "sam3d.scene-benchmark-driver.v1", "seed": args.seed,
                "image": str(args.image.resolve()), "mask_dir": str(args.mask_dir.resolve()),
                "mask_indices": [index for index, _ in masks], "variants": {}}
    for name in variants:
        backend, dtype = VARIANTS[name]
        variant_dir = args.work_dir / name
        started = time.time()
        log = logs / f"{name}.log"
        with log.open("a") as stream:
            stream.write(f"\n===== variant {name} start {time.strftime('%F %T')} =====\n")
        try:
            if backend is None:
                run_pytorch(args, masks, variant_dir, log)
            else:
                run_native(args, masks, variant_dir, backend, dtype, log)
        except SystemExit as error:
            # One variant failing (e.g. the known Vulkan get_rows defect) must
            # not block the remaining variants; the publisher simply skips it.
            overview["variants"][name] = {"error": str(error), "elapsed_s": time.time() - started}
            (args.work_dir / "driver_state.json").write_text(
                json.dumps(overview, indent=2) + "\n", encoding="utf-8")
            print(f"[driver] {name} FAILED: {error}", flush=True)
            continue
        elapsed = time.time() - started
        overview["variants"][name] = {"elapsed_s": elapsed}
        (args.work_dir / "driver_state.json").write_text(
            json.dumps(overview, indent=2) + "\n", encoding="utf-8")
        print(f"[driver] {name} complete in {elapsed / 60:.1f} min", flush=True)
    print(f"[driver] all variants complete: {args.work_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
