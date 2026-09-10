#!/usr/bin/env python3
"""Publish the full-scene reconstruction benchmark under benchmarks/.

Reads the driver layout (run_scene_benchmark.py) and emits
benchmarks/e2e_comparison/scene_current/:
  - one directory per variant: scene.gif, two representative frames, and an
    8-frame contact sheet (the posed PLY and full frame set stay in the
    work directory; benchmarks only carry compressed evidence)
  - per-native-variant side-by-side sheets and per-frame RGB-MAE / foreground
    IoU metrics against the PyTorch reference, plus a metrics plot
  - README.md and provenance.json

The scene objects are free-running reconstructions: the documented
implementation-inherent drift (GEMM accumulation order) shifts shape and pose
within the same object family, so the metrics are appearance/layout evidence
with that band - not a per-pixel oracle gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE = "pytorch"
SHEET_FRAMES = (0, 75, 150, 225)
GRID_FRAMES = (0, 45, 90, 135, 150, 195, 240, 285)
FOREGROUND_THRESHOLD = 16  # u8 luminance


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def foreground_mask(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.int32).sum(axis=2) > FOREGROUND_THRESHOLD * 3


def frame_metrics(reference: np.ndarray, variant: np.ndarray) -> dict:
    diff = np.abs(reference.astype(np.int16) - variant.astype(np.int16))
    fg_ref = foreground_mask(reference)
    fg_var = foreground_mask(variant)
    union = fg_ref | fg_var
    return {
        "rgb_mae_u8": float(diff.mean()),
        "rgb_max_abs_u8": int(diff.max()),
        "foreground_iou": (float((fg_ref & fg_var).sum() / union.sum())
                           if union.any() else 1.0),
    }


def contact_sheet(frames: dict[int, np.ndarray], order: list[int]) -> np.ndarray:
    tiles = [frames[index] for index in order if index in frames]
    rows = [tiles[i:i + 4] for i in range(0, len(tiles), 4)]
    height, width = tiles[0].shape[:2]
    gap = 4
    canvas = np.zeros((len(rows) * (height + gap) - gap,
                       4 * (width + gap) - gap, 3), dtype=np.uint8)
    for row_index, row in enumerate(rows):
        for column, tile in enumerate(row):
            y = row_index * (height + gap)
            x = column * (width + gap)
            canvas[y:y + height, x:x + width] = tile
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path,
                        default=REPO_ROOT / "output/scene-benchmark")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "cpp_ggml/benchmarks/e2e_comparison/scene_current")
    parser.add_argument("--reference", type=str, default=REFERENCE)
    parser.add_argument("--variants", type=str, default="")
    args = parser.parse_args()

    reference_dir = args.work_dir / args.reference
    if not (reference_dir / "scene_manifest.json").is_file():
        raise SystemExit(f"error: reference variant is missing: {reference_dir}")
    available = sorted(
        entry.name for entry in args.work_dir.iterdir()
        if entry.is_dir() and (entry / "scene_manifest.json").is_file()
        and entry.name != "logs")
    variants = ([name.strip() for name in args.variants.split(",") if name.strip()]
                or available)
    if args.reference not in variants:
        variants.insert(0, args.reference)

    args.out.mkdir(parents=True, exist_ok=True)
    reference_manifest = json.loads(
        (reference_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    reference_frames_dir = reference_dir / "frames"
    frame_count = len(list(reference_frames_dir.glob("frame_*.png")))
    if frame_count == 0:
        raise SystemExit("error: reference variant has no rendered frames")

    started = time.time()
    variant_rows = []
    metrics_by_variant: dict[str, dict] = {}

    def manifest_summary(manifest: dict) -> tuple[int | None, int]:
        """Normalize the pytorch and native manifest shapes.

        The native manifest lists objects as [{ply, pose, gaussians}]; the
        pytorch manifest stores timing payloads and a separate
        gaussian_counts list.
        """
        counts = manifest.get("gaussian_counts") or []
        total = manifest.get("gaussians_total")
        if total is None and counts:
            total = sum(int(item.get("gaussians", 0)) for item in counts)
        if total is None:
            entries = manifest.get("objects", [])
            if isinstance(entries, dict):
                entries = entries.get("objects", [])
            total = sum(int(item.get("gaussians", 0))
                        for item in entries if isinstance(item, dict)) or None
        entries = manifest.get("objects", [])
        if isinstance(entries, dict):
            entries = entries.get("objects", [])
        if counts:
            object_count = len(counts)
        else:
            object_count = (len(entries) if entries
                            else len(manifest.get("mask_indices", [])))
        return total, object_count

    for name in variants:
        variant_dir = args.work_dir / name
        manifest = json.loads(
            (variant_dir / "scene_manifest.json").read_text(encoding="utf-8"))
        frames_dir = variant_dir / "frames"
        out_dir = args.out / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Compressed per-variant evidence. The native scene-assemble writes
        # PNG frames (GIF encoding stays out of the native runtime); encode
        # the orbit GIF here from those frames, matching the reference GIF.
        frames_dir = variant_dir / "frames"
        frame_paths = sorted(frames_dir.glob("frame_*.png")) if frames_dir.is_dir() else []
        if not (out_dir / "scene.gif").is_file():
            if frame_paths:
                import imageio
                with imageio.get_writer(out_dir / "scene.gif", mode="I",
                                        duration=1000 / 30, loop=0) as writer:
                    for frame_path in frame_paths:
                        writer.append_data(np.asarray(Image.open(frame_path).convert("RGB")))
            elif (variant_dir / "scene.gif").is_file():
                shutil.copy2(variant_dir / "scene.gif", out_dir / "scene.gif")
        frames = {index: load_rgb(frames_dir / f"frame_{index:04d}.png")
                  for index in GRID_FRAMES
                  if (frames_dir / f"frame_{index:04d}.png").is_file()}
        order = sorted(frames)
        if frames:
            Image.fromarray(contact_sheet(frames, order)).save(
                out_dir / "orbit_contact_sheet.png")
            for index in SHEET_FRAMES:
                if index in frames:
                    Image.fromarray(frames[index]).save(
                        out_dir / f"frame_{index:04d}.png")

        gaussians_total, object_count = manifest_summary(manifest)
        row = {
            "id": name,
            "variant": name,
            "runner": ("PyTorch staged mixed (official)" if name == args.reference
                       else name.replace("-", " ").replace("q8_0", "Q8_0")
                       .replace("q4_0", "Q4_0").replace("f16", "F16")),
            "gaussians_total": gaussians_total,
            "objects": object_count,
            "scene_elapsed_s": manifest.get("scene_elapsed_s"),
            "assets": {
                "scene_gif": "scene.gif" if (out_dir / "scene.gif").is_file() else None,
                "contact_sheet": "orbit_contact_sheet.png",
                "manifest": "scene_manifest.json",
            },
        }
        if name != args.reference:
            shutil.copy2(variant_dir / "scene_manifest.json",
                         out_dir / "scene_manifest.json")
        else:
            shutil.copy2(variant_dir / "scene_manifest.json",
                         out_dir / "scene_manifest.json")
        variant_rows.append(row)

        # Per-frame metrics against the reference orbit.
        if name != args.reference:
            per_frame = []
            for index in range(frame_count):
                reference_path = reference_frames_dir / f"frame_{index:04d}.png"
                variant_path = frames_dir / f"frame_{index:04d}.png"
                if not (reference_path.is_file() and variant_path.is_file()):
                    continue
                per_frame.append(
                    {"frame": index,
                     **frame_metrics(load_rgb(reference_path), load_rgb(variant_path))})
            if per_frame:
                maes = [item["rgb_mae_u8"] for item in per_frame]
                ious = [item["foreground_iou"] for item in per_frame]
                aggregate = {
                    "frames_compared": len(per_frame),
                    "rgb_mae_u8_mean": float(np.mean(maes)),
                    "rgb_mae_u8_p50": float(np.median(maes)),
                    "rgb_mae_u8_max": float(np.max(maes)),
                    "foreground_iou_mean": float(np.mean(ious)),
                    "foreground_iou_min": float(np.min(ious)),
                }
                metrics_by_variant[name] = {"aggregate": aggregate,
                                            "per_frame": per_frame}
                row["rgb_mae_u8_mean"] = aggregate["rgb_mae_u8_mean"]
                row["foreground_iou_mean"] = aggregate["foreground_iou_mean"]

                # Side-by-side sheets at the representative frames.
                for index in SHEET_FRAMES:
                    reference_path = reference_frames_dir / f"frame_{index:04d}.png"
                    variant_path = frames_dir / f"frame_{index:04d}.png"
                    if not (reference_path.is_file() and variant_path.is_file()):
                        continue
                    left = load_rgb(reference_path)
                    right = load_rgb(variant_path)
                    gap = np.full((left.shape[0], 8, 3), 40, dtype=np.uint8)
                    Image.fromarray(np.concatenate([left, gap, right], axis=1)).save(
                        out_dir / f"side_by_side_frame_{index:04d}.png")

    if metrics_by_variant:
        plot_path = args.out / "scene_render_metrics.png"
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            figure, (axis_mae, axis_iou) = plt.subplots(2, 1, figsize=(9, 7),
                                                        sharex=True)
            # The reference is the PyTorch render itself, so the vertical
            # distance from each curve to the zero/identity baseline is
            # exactly the ggml-vs-PyTorch gap for that frame.
            for name, payload in metrics_by_variant.items():
                frames = [item["frame"] for item in payload["per_frame"]]
                maes = [item["rgb_mae_u8"] for item in payload["per_frame"]]
                ious = [item["foreground_iou"] for item in payload["per_frame"]]
                axis_mae.plot(frames, maes, label=name, linewidth=1.2)
                axis_iou.plot(frames, ious, label=name, linewidth=1.2)
                axis_mae.annotate(f"mean {np.mean(maes):.1f}",
                                  (frames[-1], maes[-1]),
                                  xytext=(4, 0), textcoords="offset points",
                                  va="center", fontsize=8)
            frames = next(iter(metrics_by_variant.values()))["per_frame"]
            frame_axis = [item["frame"] for item in frames]
            axis_mae.axhline(0, color="#62676d", linestyle="--", linewidth=1,
                             label="PyTorch reference (0 = identical)")
            axis_iou.axhline(1.0, color="#62676d", linestyle="--", linewidth=1,
                             label="PyTorch reference (1.0 = identical)")
            axis_mae.set_ylabel("RGB MAE (u8)\nvs PyTorch reference")
            axis_mae.set_title("Full-scene orbit: distance to the PyTorch render "
                               "(lower MAE / higher IoU = closer)")
            axis_mae.legend(fontsize=8)
            axis_mae.grid(alpha=0.3)
            axis_iou.set_ylabel("Foreground IoU\nvs PyTorch reference")
            axis_iou.set_xlabel("Orbit frame")
            axis_iou.set_ylim(0, 1)
            axis_iou.legend(fontsize=8)
            axis_iou.grid(alpha=0.3)
            figure.tight_layout()
            figure.savefig(plot_path, dpi=130)
            plt.close(figure)
        except Exception as error:  # noqa: BLE001 - the plot is optional evidence
            print(f"[publish] metrics plot skipped: {error}")
            plot_path = None

    # Provenance.
    provenance = {
        "schema": "sam3d.scene-benchmark-provenance.v1",
        "generated_at": time.strftime("%F %T"),
        "reference": args.reference,
        "variants": variants,
        "frame_count": frame_count,
        "render_contract": {"radius": 1.0, "fov_degrees": 60.0,
                            "resolution": 512, "frames": frame_count,
                            "yaw_start_deg": -90.0, "pitch_deg": 0.0},
        "seed": reference_manifest.get("seed"),
        "image_sha256": sha256(Path(reference_manifest["image"])),
        "mask_dir": reference_manifest.get("mask_dir"),
        "metrics_note": (
            "Scene objects are free-running reconstructions; the documented "
            "implementation-inherent drift shifts shape and pose within the "
            "same object family. Metrics are appearance/layout evidence, not "
            "a per-pixel oracle gate."),
        "driver": "cpp_ggml/scripts/run_scene_benchmark.py",
        "publisher": "cpp_ggml/scripts/publish_scene_benchmark.py",
    }
    (args.out / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (args.out / "scene_render_metrics.json").write_text(
        json.dumps({"schema": "sam3d.scene-render-metrics.v1",
                    "reference": args.reference,
                    "variants": metrics_by_variant}, indent=2) + "\n",
        encoding="utf-8")

    # Per-object pose parity: the native pose receipt vs the PyTorch
    # reference cache. This is the layout-level evidence the user sees in the
    # scene frames, reported independently of the rendering metrics.
    try:
        import torch

        def quat_angle_degrees(a, b):
            dot = abs(float(torch.dot(a.flatten().float(), b.flatten().float())))
            return math.degrees(2 * math.acos(min(1.0, dot)))

        for row in variant_rows:
            if row["runner"] == "pytorch":
                continue
            angles, translations, scales = [], [], []
            for pose_path in sorted(
                    (args.work_dir / row["variant"] / "objects").glob("obj_*/pose.json")):
                object_dir = pose_path.parent.name
                cache_path = args.work_dir / args.reference / "objects" / object_dir / "object.pt"
                if not cache_path.is_file():
                    continue
                receipt = json.loads(pose_path.read_text(encoding="utf-8"))
                cache = torch.load(cache_path, map_location="cpu", weights_only=False)
                angles.append(quat_angle_degrees(
                    torch.tensor(receipt["rotation"]), cache["rotation"]))
                translations.append(float(torch.norm(
                    torch.tensor(receipt["translation"]) - cache["translation"])))
                reference_scale = float(cache["scale"].mean())
                scales.append(abs(float(torch.tensor(receipt["scale"]).mean()))
                              / reference_scale * 100 - 100)
            if angles:
                angles.sort()
                translations.sort()
                scales.sort()
                row["pose_objects"] = len(angles)
                row["pose_median_rot_deg"] = round(angles[len(angles) // 2], 2)
                row["pose_median_trans"] = round(translations[len(translations) // 2], 4)
                row["pose_median_scale_pct"] = round(scales[len(scales) // 2], 2)
    except Exception as error:  # noqa: BLE001 - pose parity is optional evidence
        print(f"[publish] pose parity skipped: {error}")

    # Scene wall time: the native scene-assemble manifest does not carry a
    # timer; fall back to the driver log's per-variant completion lines. The
    # log is append-only across benchmark revisions, so the LAST completion
    # line per variant is the current evidence.
    driver_log = args.work_dir.parent / "scene-benchmark-driver.log"
    if driver_log.is_file():
        completions: dict[str, float] = {}
        for line in driver_log.read_text(encoding="utf-8").splitlines():
            match = re.search(r"\[driver\] (\S+) complete in ([0-9.]+) min", line)
            if match:
                completions[match.group(1)] = float(match.group(2)) * 60
        for row in variant_rows:
            if not row.get("scene_elapsed_s") and row["variant"] in completions:
                row["scene_elapsed_s"] = completions[row["variant"]]

    # README.
    lines = [
        "# Full-Scene Reconstruction Comparison (Current)",
        "",
        "Multi-object scene reconstruction of the kidsroom image: **every mask",
        f"is an independent reconstruction**, the objects are placed with the",
        "official `make_scene` pose semantics, and every variant is rendered",
        f"on the same {frame_count}-frame orbit (radius 1, fov 60, 512 px; seed",
        f"{provenance['seed']}). The PyTorch variant is the official streamed",
        "reference; the native variants run the pure-C++ pipeline",
        "(`run_ggml.sh --mask-dir`, native `scene-assemble`).",
        "",
        "## Orbit overview (8 sampled views per variant)",
        "",
        "| " + " | ".join(
            ("PyTorch (official reference)" if row["variant"] == args.reference
             else row["runner"]) for row in variant_rows) + " |",
        "| " + " | ".join("---" for _ in variant_rows) + " |",
        "| " + " | ".join(
            f"![{row['variant']} orbit]({row['variant']}/orbit_contact_sheet.png)"
            for row in variant_rows) + " |",
        "",
        "## Headline numbers",
        "",
        "| Variant | Objects | Gaussians | Scene time | RGB MAE (u8) | Foreground IoU | Median pose drift (rot / scale) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in variant_rows:
        elapsed = row.get("scene_elapsed_s") or 0
        if row["runner"] == "pytorch":
            pose_col = "-"
        elif row.get("pose_median_rot_deg") is not None:
            pose_col = (f"{row['pose_median_rot_deg']} deg / "
                        f"{row.get('pose_median_scale_pct', '-')}%")
        else:
            pose_col = "-"
        mae = row.get("rgb_mae_u8_mean")
        iou = row.get("foreground_iou_mean")
        lines.append(
            "| {} | {} | {} | {:.1f} min | {} | {} | {} |".format(
                row["runner"], row["objects"], row["gaussians_total"],
                elapsed / 60,
                f"{mae:.1f}" if mae is not None else "-",
                f"{iou:.3f}" if iou is not None else "-",
                pose_col))
    lines += [
        "",
        "Pose drift compares each object's native pose receipt against the",
        "PyTorch reference cache (rotation angle, translation distance and",
        "relative scale error, medians over the reconstructed objects).",
        "",
        "## What changed in this revision (downsample scale rescale)",
        "",
        "Cross-checking every object's native pose receipt against the PyTorch",
        "reference cache exposed two objects (masks 2 and 23) whose scale was",
        "exactly half the reference while their rotation and translation were",
        "near-exact. Root cause: both pipelines downsample the sparse support",
        "when the pruned coordinates exceed 42000 (grid halved, factor 2), and",
        "the official pipeline then rescales the decoded instance scale by that",
        "factor (inference_pipeline.py scale *= downsample_factor) - the native",
        "e2e chain decoded the pose before the downsample branch and never",
        "applied the rescale, so every object whose run triggered downsampling",
        "rendered at half size. The fix pre-computes the effective factor and",
        "passes it to the pose decode (pose_decoder_test.cpp covers the rescale),",
        "and the affected receipts (f16/q8_0 masks 2+23, q4_k mask 23 - the ones",
        "where both sides triggered) were rescaled by exactly x2, bit-identical",
        "to a full fixed-binary rerun. Controlled decomposition with the fixed",
        "receipts, all rendered through the parity-fixed native renderer:",
        "reference assets through the native renderer hold IoU 0.955 / MAE 2.82",
        "(residual renderer difference), native assets with the reference poses",
        "hold IoU 0.947 / MAE 3.86 (per-object content difference ~0.008 IoU),",
        "and the published scene keeps IoU 0.834 (free-run pose drift ~0.113",
        "IoU). An official-pipeline seed-sensitivity calibration (masks 4 and 15",
        "re-run with seeds 43/44 through the official staged pipeline) shows the",
        "official generator itself moves far more between seeds than native",
        "moves from the official seed-42 output: dRotation 2.9-17.0 deg,",
        "dScale -7..-23%, dGaussian-count +40..+155% (cf. native max 12.3 deg,",
        "-2.4%, -6.6%), and the SAM 3D paper itself documents best-of-N sampling",
        "(~50 seeds for hard inputs). The free-run pose drift is therefore inside",
        "the generator's inherent multi-solution band, not an unresolved native",
        "defect; the renderer residual (MAE 2.82) is deterministic and traced to",
        "concrete implementation deltas (alpha clamp 0.99 vs 0.999, __expf vs",
        "exp, tile-sort key quantization) documented in the fork's UPSTREAM.md.",
        "",
        "## What changed in this revision (gsplat render parity)",
        "",
        "The controlled decomposition above attributed ~0.07 IoU to the native",
        "renderer. Two independent root causes were found and fixed:",
        "",
        "1. **Orbit camera yaw drift.** The official orbit is",
        "torch.linspace(0, 2*pi, frames) - inclusive of both endpoints - so the",
        "per-frame step is 2*pi/(frames-1). make_orbit_cameras divided by",
        "frames instead, drifting every frame by 0.0034 deg * frame and",
        "reaching ~1 deg at frame 299. This alone accounted for nearly all of",
        "the renderer-attributed gap: with the fixed camera, the reference",
        "assets rendered natively improve from MAE 2.82 to 0.23 and IoU 0.955",
        "to 0.996 (300-frame mean; frame 0 was already 0.25 before the fix",
        "because its yaw matched).",
        "2. **Rasterizer semantics.** The official scene render goes through",
        "the gsplat backend (render_frames backend=\"gsplat\"), where",
        "pipe.kernel_size is never consumed and the rasterizer applies its",
        "default eps2d=0.3 low-pass with antialiased=False (no compensation).",
        "The native wrapper fed the inria-style kernel_size=0.1 into the",
        "mip-splatting fork, whose unconditional determinant compensation",
        "scaled every splat's alpha down (a systematic darkening, 75% of",
        "foreground pixels). The fork's forward pass now keeps the cov2d",
        "low-pass (kernel_size=0.3) with the compensation fixed to 1 and the",
        "gsplat opacity-aware elliptical tile bbox; alpha clamp 0.999 and",
        "__expf match gsplat exactly (see the fork's UPSTREAM.md \"Local",
        "deltas\"). An ablation confirmed the continuous items (clamp/exp) are",
        "invisible at u8 precision and a +-1 ULP depth probe confirmed sorting",
        "is robust to view-transform rounding.",
        "",
        "Final decomposition, all rendered through the fully fixed native",
        "pipeline: reference assets natively hold IoU 0.996 / MAE 0.23",
        "(renderer+camera residual 0.004 IoU), native assets with the reference",
        "poses hold IoU 0.982 / MAE 2.18 (per-object content 0.014 IoU), and",
        "the published scene keeps IoU 0.833 (free-run pose drift 0.149 IoU).",
        "An official-pipeline seed-sensitivity calibration (masks 4 and 15",
        "re-run with seeds 43/44) shows the official generator itself moves far",
        "more between seeds than native moves from the official seed-42 output:",
        "dRotation 2.9-17.0 deg, dScale -7..-23%, dGaussian-count +40..+155%",
        "(cf. native max 12.3 deg, -2.4%, -6.6%), and the SAM 3D paper itself",
        "documents best-of-N sampling (~50 seeds for hard inputs) - the",
        "free-run pose drift is inside the generator's inherent multi-solution",
        "band, not an unresolved native defect.",
        "",
        "## Hybrid-conditions experiment (route C, negative result)",
        "",
        "The pose-drift share was probed with the strongest available",
        "intervention: the OFFICIAL per-mask MoGe pointmap exported as SAMT,",
        "injected through `preprocess-conditions`, and the full native",
        "cond/SS/SLat/GS chain run on top. Single-object validation on the",
        "worst object (mask 25) succeeded - rotation 173.68 deg dropped to",
        "2.13 deg with surface-support distance 0.0037 (better than the",
        "official seed-42 output). But the full 27-object scene regressed:",
        "MAE 8.84 -> 12.69, IoU 0.833 -> 0.782. Re-sampling the condition",
        "re-rolls every object's SS trajectory, so the fixed flip created new",
        "ones (mask 19: ~1 deg -> 172.8 deg; mask 22 -> 171.7 deg; mask 02 ->",
        "114.4 deg; 16 objects above 10 deg). A 3D surface-support metric",
        "(splat centers to the pointmap field) improved for all 27 objects,",
        "which only proves the flipped solutions are equally well supported -",
        "rotational symmetry makes per-object metrics blind to flips that are",
        "obvious in the render. Conclusion: the free-run layout difference is",
        "chaotic re-rolling under sub-threshold condition differences, present",
        "in every configuration tried (native/official/hybrid pointmaps;",
        "strict/normal attention; f16/q8_0/q4_k weights; seeds). The published",
        "native scene remains the best obtainable without copying the official",
        "pose output itself.",
        "",
        "## What changed in this revision (root-cause repair)",
        "",
        "The first publication attributed the scene layout gap to",
        "\"implementation-inherent drift\". A condition-chain bisect traced",
        "the real dominant source to **native MoGe running its flash",
        "attention with F16 K/V while the official depth model runs F32**: on",
        "the DINOv2 outlier-carrying residual stream that rounding compounds",
        "to a raw pointmap MAE of 0.14, a +13.8% scene-scale error and",
        "pose-head drift (median 10.9 deg, worst 174 deg). Repairing MoGe to",
        "the strict F32 K/V contract cut the pointmap MAE to 5.4e-4 (258x),",
        "and the free-running pose parity improved accordingly (mask 14:",
        "11.9 deg to 3.8 deg; scale error 17.1% to 1.6%; obj 15: 77.8 deg to",
        "14.3 deg). The A/B also showed the SS-flow attention precision no",
        "longer measurably affects free-run pose parity, so the scene preset",
        "uses the faster F16-KV flash path (1.7x SS-flow speedup).",
        "",
        "This revision adds two measured changes: the condition-embedder",
        "weights (DINO/PointPatch/fuser inside ss_generator) are pinned to",
        "F16 storage regardless of the deployment dtype - as q8_0 they had",
        "amplified the mask-6 token error 2.6x - and the DINO flash path now",
        "uses the same F32-KV strict contract. Median pose drift improved to",
        "1.14 deg (Q8_0) / 1.36 deg (F16). The Q4 row switched from the",
        "deleted q4_0 weights to q4_k, the measured winner of the Q4 family",
        "(best decoder accuracy, fastest flow); its 4-bit flow quantization",
        "remains visible as a larger median drift (5.39 deg), reported as-is.",
        "",
        "## Side-by-side against the reference",
        "",
    ]
    for row in variant_rows:
        if row["variant"] == args.reference:
            continue
        lines += [
            f"### {row['runner']}",
            "",
            f"![{row['runner']} side-by-side frame 0]({row['variant']}/side_by_side_frame_0000.png)",
            "",
            f"![{row['runner']} side-by-side frame 150]({row['variant']}/side_by_side_frame_0150.png)",
            "",
        ]
    lines += [
        "Left: official PyTorch reference. Right: the native variant. The",
        "framing is shared through the reference's recorded normalization",
        "contract (`normalization.json`), so differences you see are the",
        "pipeline, not the camera.",
        "",
        "## Per-frame metrics",
        "",
    ]
    if plot_path is not None:
        lines += ["![Per-frame RGB MAE and foreground IoU](scene_render_metrics.png)", ""]
    lines += [
        "Full orbit videos: " + " \u00b7 ".join(
            f"[{row['runner']}](pytorch/scene.gif)" if row["runner"] == "pytorch"
            else f"[{row['runner']}]({row['variant']}/scene.gif)"
            for row in variant_rows),
        "",
        "## Reading the metrics",
        "",
        "Every variant renders the **same cameras** on the **same scene",
        "frame**, and the scene assembly itself was verified field-by-field",
        "against the official Python chain to float32 rounding. The remaining",
        "layout difference is the **free-running pose head**: each object is an",
        "independent reconstruction whose pose output is sensitive to",
        "rounding-level input differences, amplified by the 25-step Euler",
        "trajectory at CFG strength 7. Individual object assets are much",
        "closer to the reference than these layout numbers suggest - the",
        "controlled single-object evidence lives in `full_glb_current/`.",
        "",
        "All 27 masks now complete in every CUDA variant; the earlier",
        "seven-object non-finite divergence (masks 6, 7, 10, 11, 16, 18, 21)",
        "disappeared after the MoGe strict-F32-KV repair. The historical",
        "arbitration matrix recorded before that repair stays in the parity",
        "contract for reference: with the OFFICIAL pointmap even native-graph",
        "tokens converge onto the official pose, while the native MoGe",
        "pointmap's F32 tile-order floor (2.6e-4) propagates through the",
        "condition nonlinearity and shows up today as the free-running pose",
        "drift quantified in the decomposition above;",
        "",
        "The Vulkan variant remains absent from this scene publication: its",
        "SLat flow hits the Vulkan `mul_mat` defect documented in the parity",
        "contract (the earlier `get_rows` attribution was a misdiagnosis),",
        "which corrupts most per-object PLYs with non-finite values.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "# ~2.5 h GPU (3 native variants x 27 objects), idempotent and resumable",
        "python cpp_ggml/scripts/run_scene_benchmark.py --work-dir output/scene-benchmark",
        "python cpp_ggml/scripts/publish_scene_benchmark.py --work-dir output/scene-benchmark",
        "```",
        "",
        "Provenance: generated_at={}, image_sha256={}...".format(
            provenance["generated_at"], provenance["image_sha256"][:16]),
        "",
    ]
    (args.out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[publish] wrote {args.out} ({len(variants)} variants) "
          f"in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
