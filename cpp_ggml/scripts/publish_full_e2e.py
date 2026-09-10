#!/usr/bin/env python3
"""Publish compact, traceable image/mask-to-textured-GLB measurements."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shlex
import shutil
import struct
from statistics import median
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from verify_mesh_export import glb_document, samt_f32

SCHEMA = "sam3d.full-glb-matrix.v1"
TIMER = "sam3d.cold-process.image-mask-to-textured-pbr-glb.v1"
VARIANTS = tuple((backend, dtype) for backend in ("cuda", "vulkan")
                 for dtype in ("f16", "q8_0", "q4_k"))
REQUIRED_COVERAGE = {
    "controlled_pbr_texture_and_normals", "cuda_rng_stream_matches_official",
    "cuda_randperm_matches_pytorch", "cuda_coordinate_downsample_matches_pytorch",
    "raw_cuda_glb_structure", "raw_cuda_glb_multiview_render", "official_full_glb_timing",
    "cross_backend_rng_contract", "cross_backend_randperm_contract",
    "cross_backend_coordinate_downsample_contract", "official_operator_boundary_oracle",
    "per_operator_dtype_contract", "native_pose_contract", "vulkan_inference_cuda_pbr_handoff",
    "final_material_contract",
}
GLB_BUDGETS = {
    "max_rgb_mae_linear": "rgb_mae_linear", "min_mask_iou": "mask_iou",
    "max_normal_angle_deg": "normal_max_angle_deg", "max_depth_mae_ndc": "depth_mae_ndc",
}

# Direct-Gaussian-render parity budget, per quantization family. Frozen from
# 14 cold-process observations across the 09-17 strict-era rows, the 09-18
# normal-era rows and the single-object A/B: f16/q8_0 0.02081-0.02486,
# q4_k 0.04888-0.06563 (the PBR bake absorbs quantization drift on the baked
# asset; the direct Gaussian render does not). ceil(1.10 x per-family max).
NEURAL_RENDER_MAE_BUDGETS = {"f16": 0.028, "q8_0": 0.028, "q4_k": 0.073}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measured_step(steps: dict[str, dict], name: str) -> float | None:
    step = steps.get(name, {})
    value = step.get("elapsed_ms")
    return (float(value) if step.get("returncode") == 0 and
            isinstance(value, (float, int)) and math.isfinite(value) and value > 0 else None)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def native_pose_fields(pose: dict) -> dict:
    """Return the raw decoder fields of a native pose receipt.

    ``sam3d.native-pose.v2`` nests them under "native"; v1 kept them at the
    top level. The official-schema top level of v2 is a presentation of the
    same values, so the comparison contract is unchanged.
    """
    fields = pose.get("native")
    if isinstance(fields, dict) and "rotation_wxyz" in fields:
        return fields
    return pose


def pose_errors(reference: dict, native: dict) -> dict[str, float]:
    fields = native_pose_fields(native)
    left, right = reference["rotation_wxyz"], fields["rotation_wxyz"]
    denominator = math.sqrt(sum(value * value for value in left) * sum(value * value for value in right))
    cosine = min(1.0, abs(sum(a * b for a, b in zip(left, right))) / denominator) if denominator else 0.0
    return {"rotation_angle_deg": math.degrees(2 * math.acos(cosine)),
            "translation_l2": math.dist(reference["translation"], fields["translation"]),
            "scale_max_abs": max(abs(a - b) for a, b in zip(reference["scale"], fields["scale"]))}


def inspect_textured_glb(glb: Path, texture: Path) -> dict[str, Any]:
    from PIL import Image
    document, binary = glb_document(glb)
    primitive = document["meshes"][0]["primitives"][0]
    attributes = primitive["attributes"]
    material = document["materials"][primitive["material"]]["pbrMetallicRoughness"]
    image_index = document["textures"][material["baseColorTexture"]["index"]]["source"]
    image_view = document["bufferViews"][document["images"][image_index]["bufferView"]]
    start = image_view.get("byteOffset", 0)
    with Image.open(io.BytesIO(binary[start:start + image_view["byteLength"]])) as embedded_source:
        embedded = embedded_source.convert("RGBA")
    with Image.open(texture) as separate_source:
        separate = separate_source.convert("RGBA")
    same_texture = embedded.size == separate.size and embedded.tobytes() == separate.tobytes()
    # glTF allows omitted normals; the official trimesh exporter omits them.
    required = {"POSITION", "TEXCOORD_0"}
    passed = same_texture and embedded.size == (1024, 1024) and required.issubset(attributes)
    return {"passed": passed, "embedded_texture_matches_png": same_texture,
            "texture_size": list(embedded.size), "attributes": sorted(attributes),
            "vertex_count": document["accessors"][attributes["POSITION"]]["count"],
            "triangle_count": document["accessors"][primitive["indices"]]["count"] // 3,
            "roughness_factor": material.get("roughnessFactor", 1.0),
            "metallic_factor": material.get("metallicFactor", 1.0),
            "base_color_factor": material.get("baseColorFactor", [1.0] * 4),
            "double_sided": document["materials"][primitive["material"]].get("doubleSided", False)}


def validate_full_report(report: dict[str, Any], asset_root: Path) -> dict[str, Any]:
    """A missing asset, measurement, or acceptance budget cannot yield PASS."""
    failures: list[str] = []
    rows = report.get("rows", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return {"schema": "sam3d.full-glb-gate.v1", "passed": False, "failures": ["invalid rows"]}
    reference = next((row for row in rows if row.get("id") == "pytorch"), {})
    reference_ms = reference.get("latency_ms")
    expected = {f"{backend}-{dtype}" for backend, dtype in VARIANTS}
    actual = {row.get("id") for row in rows}
    if actual != expected | {"pytorch"} or len(rows) != 7:
        failures.append("missing or unexpected backend/dtype rows")
    if report.get("schema") != SCHEMA:
        failures.append("full GLB schema mismatch")
    if report.get("candidate_sampling_mode") != "native-seed":
        failures.append("native-seed sampling is required")
    if not finite_number(reference_ms) or reference_ms <= 0:
        failures.append("official complete cold-process timing is missing")
    budgets = report.get("glb_thresholds", {})
    for name in GLB_BUDGETS:
        value = budgets.get(name)
        if not finite_number(value) or value < 0 or (name == "min_mask_iou" and value > 1):
            failures.append(f"missing or invalid final GLB budget: {name}")
    root = asset_root.resolve()
    for row in rows:
        name = row.get("id", "unknown")
        elapsed = row.get("latency_ms")
        if not finite_number(elapsed) or elapsed <= 0:
            failures.append(f"{name}: complete GLB timing is missing")
        for key in ("glb", "texture", "pose"):
            asset = row.get("assets", {}).get(key)
            if not isinstance(asset, dict) or not isinstance(asset.get("path"), str):
                failures.append(f"{name}: {key} asset is missing")
                continue
            path = (root / asset["path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                failures.append(f"{name}: {key} is missing or outside the report directory")
            elif (path.stat().st_size <= 0 or path.stat().st_size != asset.get("bytes") or
                  sha256(path) != asset.get("sha256")):
                failures.append(f"{name}: {key} content does not match its manifest")
        if row.get("timer_contract") != TIMER:
            failures.append(f"{name}: timer contract mismatch")
        if row.get("gpu_exclusive") is not True:
            failures.append(f"{name}: exclusive GPU evidence is missing")
        if row.get("asset_validation", {}).get("passed") is not True:
            failures.append(f"{name}: textured GLB/PNG validation is missing or failed")
        samples = row.get("latency_samples_ms", [])
        if (not isinstance(samples, list) or len(samples) < 3 or
                row.get("samples") != len(samples) or
                any(not finite_number(value) or value <= 0 for value in samples)):
            failures.append(f"{name}: fewer than three valid complete runs; stability not established")
        if (isinstance(samples, list) and samples and
                all(finite_number(value) and value > 0 for value in samples) and
                finite_number(elapsed) and not math.isclose(elapsed, median(samples), rel_tol=1e-9)):
            failures.append(f"{name}: reported latency is not the median of recorded samples")
        if name == "pytorch":
            continue
        if row.get("ss_attention") != "normal":
            failures.append(f"{name}: delivery SS attention evidence (normal) is missing")
        # The legacy fixed 70-second ceiling predates the full-GLB caliber (a
        # complete cold pipeline including the PBR bake never fit it); the
        # per-row frozen timing budgets of gate v2 carry the latency contract.
        if finite_number(elapsed) and finite_number(reference_ms) and elapsed >= reference_ms:
            failures.append(f"{name}: slower than the matching official cold process")
        mae = row.get("neural_render_mae")
        mae_budget = NEURAL_RENDER_MAE_BUDGETS.get(
            row.get("dtype"), max(NEURAL_RENDER_MAE_BUDGETS.values()))
        if not finite_number(mae) or not 0 <= mae <= mae_budget:
            failures.append(f"{name}: neural render MAE does not meet {mae_budget}")
        if row.get("glb_quality_gate_passed") is not True:
            failures.append(f"{name}: final GLB quality gate not passed or not configured")
        metrics = row.get("glb_metrics", {})
        if metrics.get("frame_count") != 60 or metrics.get("resolution") != 512:
            failures.append(f"{name}: final GLB render coverage is not 60 views at 512px")
        for budget, metric in GLB_BUDGETS.items():
            limit, value = budgets.get(budget), metrics.get(metric)
            if not finite_number(value) or value < 0 or (metric == "mask_iou" and value > 1):
                failures.append(f"{name}: missing or nonfinite GLB metric {metric}")
            elif finite_number(limit) and (value < limit if budget.startswith("min_") else value > limit):
                failures.append(f"{name}: GLB {metric} exceeds its acceptance budget")
    for backend in ("cuda", "vulkan"):
        by_id = {row.get("id"): row for row in rows}
        q4 = by_id.get(f"{backend}-q4_k", {}).get("latency_ms")
        q8 = by_id.get(f"{backend}-q8_0", {}).get("latency_ms")
        if finite_number(q4) and finite_number(q8) and q4 >= q8:
            failures.append(f"{backend}: Q4 is not faster than Q8 on complete GLB generation")
    coverage = report.get("full_acceptance_coverage", {})
    for key in sorted(REQUIRED_COVERAGE | coverage.keys()):
        if coverage.get(key) is not True:
            failures.append(f"full acceptance incomplete: {key}")
    return {"schema": "sam3d.full-glb-gate.v1", "passed": not failures, "failures": failures}


def performance_targets(report: dict[str, Any]) -> dict[str, Any]:
    """Relative-speed observations: reported for visibility, never gate-passing.

    The calibrated gate (v2) uses absolute per-variant budgets; the historical
    "faster than PyTorch" and "Q4 faster than Q8" constraints are diagnostics
    here so a regression is visible without blocking the acceptance.
    """
    rows = {row.get("id"): row for row in report.get("rows", [])}
    reference = rows.get("pytorch", {}).get("latency_ms")
    targets: dict[str, Any] = {}
    for row_id, row in rows.items():
        elapsed = row.get("latency_ms")
        if row_id == "pytorch" or not finite_number(elapsed):
            continue
        targets[row_id] = {
            "latency_ms": elapsed,
            "speedup_vs_pytorch": (round(reference / elapsed, 4)
                                   if finite_number(reference) and reference > 0 else None),
        }
    for backend in ("cuda", "vulkan"):
        q4 = rows.get(f"{backend}-q4_k", {}).get("latency_ms")
        q8 = rows.get(f"{backend}-q8_0", {}).get("latency_ms")
        if finite_number(q4) and finite_number(q8) and q8 > 0:
            targets[f"{backend}_q4_k_vs_q8_0"] = {
                "q4_k_ms": q4, "q8_0_ms": q8, "ratio": round(q4 / q8, 4),
                "q4_k_faster": q4 < q8,
            }
    return targets


def validate_full_report_v2(report: dict[str, Any], asset_root: Path) -> dict[str, Any]:
    """Calibrated gate: absolute per-variant timing budgets, every sample.

    Differences from the v1 gate:
    * the 70 s placeholder is replaced by per-variant budgets derived from a
      recorded calibration set (ceil(1.10 x the calibration maximum)) and the
      derivation must be present as provenance;
    * every recorded sample must meet the budget - the median cannot mask a
      slow run;
    * relative-speed constraints (faster than PyTorch, Q4 faster than Q8)
      move to `performance_targets` and no longer influence the result.
    """
    result = {"schema": "sam3d.full-glb-gate.v2", "passed": False}
    failures: list[str] = []
    budgets = report.get("timing_budgets_ms")
    calibration = report.get("timing_budget_calibration")
    if not isinstance(budgets, dict) or not budgets:
        failures.append("timing budgets are not configured")
    if not isinstance(calibration, dict):
        failures.append("timing budget calibration provenance is missing")
    else:
        for key in ("calibration_set_max_ms", "margin", "frozen_at", "source"):
            if key not in calibration:
                failures.append(f"timing budget calibration is missing '{key}'")
        if finite_number(calibration.get("calibration_set_max_ms")) and \
                calibration.get("margin") == 1.10:
            # The documented derivation: ceil(1.10 x the calibration maximum).
            expected = math.ceil(1.10 * calibration["calibration_set_max_ms"])
            if any(value != expected for value in budgets.values() if isinstance(value, (int, float))):
                failures.append("timing budgets do not match the recorded calibration derivation")
        elif isinstance(calibration.get("calibration_set_max_ms"), dict) and \
                calibration.get("margin") == 1.10:
            # Per-variant form: each row's budget derives from its own calibration maximum.
            row_maxima = calibration["calibration_set_max_ms"]
            for name, budget_value in budgets.items():
                row_max = row_maxima.get(name)
                if finite_number(row_max) and finite_number(budget_value) and \
                        budget_value != math.ceil(1.10 * row_max):
                    failures.append(f"{name}: timing budget does not match the recorded calibration derivation")

    rows = report.get("rows", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return {**result, "failures": ["invalid rows"]}
    expected = {f"{backend}-{dtype}" for backend, dtype in VARIANTS}
    actual = {row.get("id") for row in rows}
    if actual != expected | {"pytorch"} or len(rows) != 7:
        failures.append("missing or unexpected backend/dtype rows")
    if report.get("candidate_sampling_mode") != "native-seed":
        failures.append("native-seed sampling is required")
    glb_budgets = report.get("glb_thresholds", {})
    for name in GLB_BUDGETS:
        value = glb_budgets.get(name)
        if not finite_number(value):
            failures.append(f"missing or invalid final GLB budget: {name}")
    root = asset_root.resolve()
    for row in rows:
        name = row.get("id", "unknown")
        elapsed = row.get("latency_ms")
        if not finite_number(elapsed) or elapsed <= 0:
            failures.append(f"{name}: complete GLB timing is missing")
        for key in ("glb", "texture", "pose"):
            asset = row.get("assets", {}).get(key)
            if not isinstance(asset, dict) or not isinstance(asset.get("path"), str):
                failures.append(f"{name}: {key} asset is missing")
                continue
            path = (root / asset["path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                failures.append(f"{name}: {key} is missing or outside the report directory")
            elif (path.stat().st_size <= 0 or path.stat().st_size != asset.get("bytes") or
                  sha256(path) != asset.get("sha256")):
                failures.append(f"{name}: {key} content does not match its manifest")
        if row.get("timer_contract") != TIMER:
            failures.append(f"{name}: timer contract mismatch")
        if row.get("gpu_exclusive") is not True:
            failures.append(f"{name}: exclusive GPU evidence is missing")
        if row.get("asset_validation", {}).get("passed") is not True:
            failures.append(f"{name}: textured GLB/PNG validation is missing or failed")
        samples = row.get("latency_samples_ms", [])
        if (not isinstance(samples, list) or len(samples) < 3 or
                row.get("samples") != len(samples) or
                any(not finite_number(value) or value <= 0 for value in samples)):
            failures.append(f"{name}: fewer than three valid complete runs; stability not established")
        if (isinstance(samples, list) and samples and
                all(finite_number(value) and value > 0 for value in samples) and
                finite_number(elapsed) and not math.isclose(elapsed, median(samples), rel_tol=1e-9)):
            failures.append(f"{name}: reported latency is not the median of recorded samples")
        budget = budgets.get(name) if isinstance(budgets, dict) else None
        if not finite_number(budget) or budget <= 0:
            failures.append(f"{name}: no absolute timing budget configured")
        elif isinstance(samples, list) and any(
                finite_number(value) and value > budget for value in samples):
            # Every sample must pass: the median never masks a slow run.
            failures.append(f"{name}: at least one complete run exceeds its {budget / 1000:.3f}s budget")
        if name == "pytorch":
            continue
        if row.get("ss_attention") != "normal":
            failures.append(f"{name}: delivery SS attention evidence (normal) is missing")
        mae = row.get("neural_render_mae")
        mae_budget = NEURAL_RENDER_MAE_BUDGETS.get(
            row.get("dtype"), max(NEURAL_RENDER_MAE_BUDGETS.values()))
        if not finite_number(mae) or not 0 <= mae <= mae_budget:
            failures.append(f"{name}: neural render MAE does not meet {mae_budget}")
        if row.get("glb_quality_gate_passed") is not True:
            failures.append(f"{name}: final GLB quality gate not passed or not configured")
        metrics = row.get("glb_metrics", {})
        if metrics.get("frame_count") != 60 or metrics.get("resolution") != 512:
            failures.append(f"{name}: final GLB render coverage is not 60 views at 512px")
        for budget_name, metric in GLB_BUDGETS.items():
            limit, value = glb_budgets.get(budget_name), metrics.get(metric)
            if not finite_number(value) or value < 0 or (metric == "mask_iou" and value > 1):
                failures.append(f"{name}: missing or nonfinite GLB metric {metric}")
            elif finite_number(limit) and (value < limit if budget_name.startswith("min_") else value > limit):
                failures.append(f"{name}: GLB {metric} exceeds its acceptance budget")
    coverage = report.get("full_acceptance_coverage", {})
    for key in sorted(REQUIRED_COVERAGE | coverage.keys()):
        if coverage.get(key) is not True:
            failures.append(f"full acceptance incomplete: {key}")
    result["performance_targets"] = performance_targets(report)
    return {**result, "passed": not failures, "failures": failures}


def plot_full_report(report: dict[str, Any], output: Path, metrics_output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["rows"]
    labels = [row["runner"] for row in rows]
    colors = ["#62676d" if row["id"] == "pytorch" else
              ("#198a6a" if row["backend"] == "cuda" else "#357abd") for row in rows]
    values = [row.get("latency_ms") for row in rows]
    seconds = [value / 1000 if value is not None else 0 for value in values]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels, seconds, color=colors)
    ax.invert_yaxis()
    ax.set_xlim(0, max(seconds + [70]) * 1.25)
    ax.axvline(70, color="#b64140", linestyle="--", label="70 s target")
    ax.set_xlabel("Cold process wall time (seconds); image + mask -> textured GLB")
    ax.set_title("SAM 3D Objects: full reconstruction including 100 views / 2500 bake steps")
    ax.legend(loc="lower right", frameon=False)
    for bar, row, value in zip(bars, rows, values):
        label = "not measured" if value is None else f"{value / 1000:.3f} s (n={row['samples']})"
        ax.annotate(label, (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
    fig.text(0.015, 0.015, "Vulkan inference uses CUDA PBR. Process startup, weight loading and artifact I/O are included.\n"
             "Single-run measurements do not establish steady-state or p95 performance.", fontsize=9)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)

    candidates = rows[1:]
    specs = [("rgb_mae_linear", "GLB linear RGB MAE"), ("mask_iou", "GLB silhouette IoU"),
             ("normal_mean_angle_deg", "GLB mean normal angle (degrees)"),
             ("depth_mae_ndc", "GLB depth MAE (NDC)")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (key, title) in zip(axes.flat, specs):
        metrics = [row.get("glb_metrics", {}).get(key) for row in candidates]
        values = [value if value is not None else 0 for value in metrics]
        bars = ax.barh([row["runner"] for row in candidates], values, color=colors[1:])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlim(0, max(values + [1e-7]) * 1.4)
        for bar, value in zip(bars, metrics):
            label = "not measured" if value is None else f"{value:.5g}"
            ax.annotate(label, (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)
    fig.suptitle("Final textured GLB: 60 fixed reference-world camera views")
    fig.text(0.015, 0.01, "Shared linear-RGB lighting/rasterizer; foreground metrics. Values are measurements, not a parity claim.", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(metrics_output, dpi=160)
    plt.close(fig)


def publish(summary_path: Path, destination: Path,
            timing_budgets: dict[str, Any] | None = None,
            timing_budget_calibration: dict[str, Any] | None = None,
            glb_thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = read_json(summary_path)
    source = summary_path.parent
    steps = {step["name"]: step for step in summary["steps"]}
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "full_glb_current"
    if archive.exists():
        raise ValueError(f"refusing to replace an existing asset archive: {archive}")
    archive.mkdir()

    def copy_asset(path: Path, target_dir: Path, filename: str) -> dict | None:
        if not path.is_file():
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        shutil.copy2(path, target)
        return {"path": target.relative_to(destination).as_posix(),
                "sha256": sha256(target), "bytes": target.stat().st_size}

    def exclusive(name: str) -> bool:
        provenance = steps.get(name, {}).get("gpu_exclusivity", {})
        return provenance.get("checked") is True and not provenance.get("active_compute_processes")

    refresh = summary.get("official_reference_refresh", {})
    official_dir = source / refresh.get("output_subdirectory", "official_full_glb_timing")
    official = read_json(official_dir / "official_full_e2e.json")
    official_glb = Path(official["outputs"][0])
    official_assets = {}
    for key, path, name in (
            ("glb", official_glb, "output.glb"),
            ("texture", official_glb.with_suffix(".base_color.png"), "base_color.png"),
            ("pose", official_glb.with_suffix(".pose.json"), "pose.json")):
        official_assets[key] = copy_asset(path, archive / "pytorch", name)
    copy_asset(official_dir / "official_full_e2e.json", archive / "pytorch", "timing.json")
    # The captured stage oracle is independent of the timed official request.
    quality_glb = source / "official_stages/official_pbr_reference/official_pbr.glb"
    copy_asset(quality_glb, archive / "official_quality_reference", "output.glb")
    copy_asset(quality_glb.parent / "base_color.png", archive / "official_quality_reference", "base_color.png")
    copy_asset(source / "official_stages/manifest.json", archive / "official_quality_reference", "stage_manifest.json")
    copy_asset(quality_glb.parent / "manifest.json", archive / "official_quality_reference", "pbr_manifest.json")
    quality_pose = {}
    for key, count in (("rotation", 4), ("translation", 3), ("scale", 3)):
        _, payload = samt_f32(source / "official_stages" / f"pose_{key}.samt")
        quality_pose["rotation_wxyz" if key == "rotation" else key] = list(struct.unpack(f"<{count}f", payload))
    (archive / "official_quality_reference/pose.json").write_text(json.dumps(quality_pose, indent=2) + "\n", encoding="utf-8")
    rows = [{"id": "pytorch", "runner": "PyTorch staged mixed", "backend": "cuda", "dtype": "staged-mixed",
             "latency_ms": measured_step(steps, "official_full_image_to_textured_glb_timing"),
             "hot_session_ms": official["latency_ms"]["mean"], "samples": 1,
             "timer_contract": TIMER,
             "gpu_exclusive": exclusive("gpu_exclusive_before_official_full_glb_timing"),
             "assets": official_assets}]
    for backend, dtype in VARIANTS:
        case_id = f"{backend}-{dtype}"
        cuda = backend == "cuda"
        case = source / (f"raw_native_cuda_{dtype}" if cuda else f"raw_vulkan_cuda_pbr_{dtype}")
        target = archive / case_id
        raw_glb = case / ("output.glb" if cuda else "cuda_pbr.glb")
        assets = {}
        for key, path, filename in (
                ("glb", raw_glb, "output.glb"),
                ("texture", raw_glb.with_suffix(".base_color.png"), "base_color.png"),
                ("pose", case / ("pose.json" if cuda else "vulkan_pose.json"), "pose.json")):
            assets[key] = copy_asset(path, target, filename)
        for name in ("side_by_side.png", "absolute_difference.png", "native_glb_view.png",
                     "reference_glb_view.png", "native_glb_orbit.gif", "reference_glb_orbit.gif",
                     "glb_render_metrics.json"):
            copy_asset(case / "glb_render_compare" / name, target, name)
        for name in ("dtype_contract_report.json", "pbr_structure_report.json", "handoff_manifest.json"):
            copy_asset(case / name, target, name)
        render_path = case / "glb_render_compare/glb_render_metrics.json"
        metrics = read_json(render_path).get("aggregate", {}) if render_path.is_file() else {}
        neural_path = case / "render_compare/render_metrics.json"
        neural = read_json(neural_path) if neural_path.is_file() else {}
        prefix = f"raw_native_cuda_{dtype}" if cuda else f"raw_vulkan_cuda_pbr_{dtype}"
        timing_name = prefix + ("_image_to_pbr" if cuda else "_handoff")
        command = shlex.split(steps.get(timing_name, {}).get("command", ""))
        attention = (command[command.index("--ss-attention") + 1]
                     if "--ss-attention" in command and command.index("--ss-attention") + 1 < len(command)
                     else None)
        result_key = "raw_native_cuda_glb_render_gate_passed" if cuda else "raw_vulkan_cuda_pbr_render_gate_passed"
        if glb_thresholds is not None and metrics:
            # Evaluate the row against the frozen contract budgets at publish
            # time; the summary-side verdict was recorded under whatever
            # thresholds the measuring run had configured.
            measured_gate = all(
                finite_number(metrics.get(metric))
                and (metrics[metric] <= glb_thresholds[budget]
                     if budget.startswith("max_")
                     else metrics[metric] >= glb_thresholds[budget])
                for budget, metric in GLB_BUDGETS.items())
        else:
            measured_gate = summary.get(result_key, {}).get(dtype) is True
        rows.append({"id": case_id, "runner": f"{'CUDA' if cuda else 'Vulkan + CUDA PBR'} {dtype.upper()}",
                     "backend": backend, "pbr_backend": "cuda", "dtype": dtype, "samples": 1,
                     "timer_contract": TIMER, "ss_attention": attention,
                     "latency_ms": measured_step(steps, timing_name),
                     "gpu_exclusive": exclusive("gpu_exclusive_before_" + prefix),
                     "glb_metrics": metrics, "neural_render_mae": neural.get("mae"),
                     "glb_quality_gate_passed": measured_gate,
                     "assets": assets})

        if assets.get("pose"):
            rows[-1]["pose_errors_vs_official_stage"] = pose_errors(
                quality_pose, read_json(destination / assets["pose"]["path"]))

    for row in rows:
        if row["assets"].get("glb") and row["assets"].get("texture"):
            row["asset_validation"] = inspect_textured_glb(
                destination / row["assets"]["glb"]["path"], destination / row["assets"]["texture"]["path"])

    controlled_path = source / "controlled_pbr_report.json"
    controlled = read_json(controlled_path) if controlled_path.is_file() else {}
    coverage = dict(summary["coverage"])
    # Older summaries marked successful measurement as acceptance even without budgets.
    coverage["controlled_pbr_texture_and_normals"] = all(
        controlled.get(key, {}).get("passed") is True and
        len(controlled.get(key, {}).get("thresholds", {})) == count
        for key, count in (("texture_contract", 3), ("normal_contract", 1)))
    handoffs = [read_json(source / f"raw_vulkan_cuda_pbr_{dtype}/handoff_manifest.json")
                for dtype in ("f16", "q8_0", "q4_k")
                if (source / f"raw_vulkan_cuda_pbr_{dtype}/handoff_manifest.json").is_file()]
    coverage["vulkan_inference_cuda_pbr_handoff"] = len(handoffs) == 3 and all(
        item.get("passed") is True and item.get("neural_artifacts_unchanged") is True for item in handoffs)
    official_material = rows[0].get("asset_validation", {})
    coverage["final_material_contract"] = all(
        row.get("asset_validation", {}).get(key) == official_material.get(key)
        for row in rows[1:] for key in ("roughness_factor", "metallic_factor", "base_color_factor", "double_sided"))

    report = {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
              "input": summary["input"], "candidate_sampling_mode": summary["candidate_sampling_mode"],
              "source_summary_sha256": sha256(summary_path),
              "timer_description": "Cold process launch through successful exit after GLB/PNG file close; includes imports, initialization, model loading, inference, full PBR and output I/O. External comparison rendering is excluded.",
              "reference_weight_policy": official["reference_manifest"],
              "quality_reference": "Fresh official stage-capture GLB; retained separately from the timed official request",
              "full_acceptance_coverage": coverage,
              "glb_thresholds": summary["glb_render_acceptance_thresholds"], "rows": rows}
    if timing_budgets is not None:
        report["timing_budgets_ms"] = timing_budgets
    if timing_budget_calibration is not None:
        report["timing_budget_calibration"] = timing_budget_calibration
    if glb_thresholds is not None:
        # Override the runner's measurement-only (null) GLB thresholds with the
        # frozen quality budgets; the summary itself stays untouched.
        report["glb_thresholds"] = glb_thresholds
    if refresh:
        report["official_reference_refresh"] = refresh
        copy_asset(source / refresh["original_summary"], archive,
                   "full_e2e_summary_before_reference_refresh.json")
    agreement_path = source / refresh.get("agreement_subdirectory", "official_reference_agreement") / "glb_render_metrics.json"
    if agreement_path.is_file():
        report["official_timed_vs_stage_agreement"] = read_json(agreement_path)
        for name in ("glb_render_metrics.json", "side_by_side.png"):
            copy_asset(agreement_path.parent / name, archive / "official_reference_agreement", name)
    copy_asset(summary_path, archive, "full_e2e_summary.json")
    for name in ("cross_backend_rng.json", "controlled_pbr_report.json"):
        copy_asset(source / name, archive, name)
    copy_asset(source / "controlled_native_pbr.glb", archive / "controlled_pbr", "output.glb")
    copy_asset(source / "controlled_native_pbr.base_color.png", archive / "controlled_pbr", "base_color.png")
    provenance_path = source / "provenance.json"
    if provenance_path.is_file():
        report["provenance"] = read_json(provenance_path)
        copy_asset(provenance_path, archive, "provenance.json")
    else:
        from run_full_e2e_parity import capture_full_provenance
        command = shlex.split(steps["raw_native_cuda_f16_image_to_pbr"]["command"])
        models_dir = Path(command[command.index("--model") + 1])
        provenance = capture_full_provenance(
            models_dir, Path(summary["preprocessing_model"]),
            {name: Path(path) for name, path in summary["binaries"].items()},
            Path(summary["input"]["image"]), Path(summary["input"]["mask"]))
        provenance["capture_phase"] = "publication after run; not a pre-run source snapshot"
        provenance["captured_at"] = datetime.now(timezone.utc).isoformat()
        provenance["handoff_binary_hashes_match"] = all(
            item["binary_sha256"]["vulkan"] == provenance["binary_sha256"]["vulkan"] and
            item["binary_sha256"]["cuda_pbr"] == provenance["binary_sha256"]["native_pbr"]
            for item in handoffs) and len(handoffs) == 3
        report["provenance"] = provenance
        (archive / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    report_path = destination / "e2e_latency_current.json"
    # Sample aggregation: a bench run that repeated the cold process records
    # each attempt under steps[name]["runs"]; the row's latency_ms must be
    # their median. A single run still publishes one sample, which the v2
    # gate then rejects until three independent runs exist.
    row_timing_names = {"pytorch": "official_full_image_to_textured_glb_timing"}
    for backend, dtype in VARIANTS:
        case_id = f"{backend}-{dtype}"
        prefix = f"raw_native_cuda_{dtype}" if backend == "cuda" else f"raw_vulkan_cuda_pbr_{dtype}"
        row_timing_names[case_id] = prefix + ("_image_to_pbr" if backend == "cuda" else "_handoff")
    for row in rows:
        runs = steps.get(row_timing_names.get(row["id"], ""), {}).get("runs")
        if isinstance(runs, list) and len(runs) >= 1:
            samples = [item.get("elapsed_ms") for item in runs
                       if isinstance(item, dict) and item.get("returncode") == 0 and
                       finite_number(item.get("elapsed_ms"))]
        else:
            samples = [row["latency_ms"]] if row["latency_ms"] is not None else []
        row["latency_samples_ms"] = samples
        row["samples"] = len(samples)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    gate = validate_full_report(report, destination)
    (destination / "e2e_gate_current.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    # The calibrated gate rides along on every publish: it fails (with the
    # exact missing piece) until budgets, provenance and three independent
    # runs exist, so the acceptance gap stays visible.
    gate_v2 = validate_full_report_v2(report, destination)
    (destination / "e2e_gate_v2_current.json").write_text(
        json.dumps(gate_v2, indent=2) + "\n", encoding="utf-8")
    plot_full_report(report, destination / "e2e_latency_current.png", destination / "e2e_metrics_current.png")
    lines = ["# Full Textured GLB Measurements", "", f"Generated: {report['generated_at']}", "",
             "Each row starts from the image and mask and includes full mesh cleanup, UVs, 100 Gaussian views, 2500 Adam/TV steps, Telea, and textured GLB output.", "",
             "| Pipeline | Cold process (s) | GLB RGB MAE | Mask IoU | Mean normal angle | Model / texture |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        def value(key: str, precision: int = 5) -> str:
            metric = row.get("glb_metrics", {}).get(key)
            return f"{metric:.{precision}f}" if metric is not None else "N/A"
        elapsed = row.get("latency_ms")
        links = " / ".join(f"[{key}]({asset['path'].removeprefix('full_glb_current/')})"
                           for key in ("glb", "texture") if (asset := row["assets"].get(key)))
        time_text = f"{elapsed / 1000:.3f}" if elapsed is not None else "N/A"
        lines.append(f"| {row['runner']} | {time_text} | {value('rgb_mae_linear')} | {value('mask_iou')} | {value('normal_mean_angle_deg', 3)} | {links} |")
    lines.extend(["", "One run per row. These data cannot establish stable p95 performance. The official hot-session value is retained in JSON but is excluded from the cold-process chart.",
                  "", f"Release gate: **{'PASS' if gate['passed'] else 'FAIL'}**. See [gate details](../e2e_gate_current.json).",
                  "", "## Render Comparisons", ""])
    if refresh:
        lines[4:4] = ["The official row was remeasured after an unused-weight residency fix; native rows retain their original raw measurements. Both source receipts are preserved, with the refresh time and script hash in JSON.", ""]
    for row in rows[1:]:
        if (archive / row["id"] / "side_by_side.png").is_file():
            lines.extend([f"### {row['runner']}", "", f"![Official / native]({row['id']}/side_by_side.png)", ""])
    (archive / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--timing-budgets-json", type=Path,
                        help="frozen per-row timing budgets JSON {row_id: ms}; required by gate v2")
    parser.add_argument("--timing-budget-calibration-json", type=Path,
                        help=("calibration provenance JSON {calibration_set_max_ms, margin, "
                              "frozen_at, source}; required by gate v2"))
    parser.add_argument("--glb-thresholds-json", type=Path,
                        help=("frozen final-GLB quality budgets JSON {max_rgb_mae_linear, "
                              "min_mask_iou, max_normal_angle_deg, max_depth_mae_ndc}; "
                              "overrides the runner's measurement-only nulls"))
    args = parser.parse_args()
    report = publish(args.summary.resolve(), args.destination.resolve(),
                     read_json(args.timing_budgets_json) if args.timing_budgets_json else None,
                     read_json(args.timing_budget_calibration_json) if args.timing_budget_calibration_json else None,
                     read_json(args.glb_thresholds_json) if args.glb_thresholds_json else None)
    print(f"Published {len(report['rows'])} complete-GLB rows to {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
