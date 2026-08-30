#!/usr/bin/env python3
"""Plot measured full image-to-3D latency and rendered-quality comparisons.

The input is a small JSON document with one row per runner.  A missing
PyTorch latency is represented as null and is rendered as an explicit OOM /
not-measured annotation instead of an invented number.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metrics-output", type=Path, default=None,
        help="render MAE/PSNR output (default: sibling e2e_metrics_*.png)",
    )
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    rows = report.get("rows", [])
    if not rows:
        parser.error("input contains no latency rows")

    labels = [str(row["runner"]) for row in rows]
    values = [row.get("latency_ms") for row in rows]
    measured = [float(value) if value is not None else 0.0 for value in values]
    def bar_color(row: dict[str, object], value: object) -> str:
        if value is None:
            return "#B0B0B0"
        status = str(row.get("status", "")).lower()
        if "failed" in status:
            return "#C44E52"
        return "#4C78A8"

    colors = [bar_color(row, value) for row, value in zip(rows, values)]

    figure_height = max(5.2, 0.52 * len(labels) + 0.9)
    fig, ax = plt.subplots(figsize=(11.0, figure_height))
    positions = list(range(len(labels)))
    bars = ax.barh(positions, measured, color=colors, edgecolor="white", linewidth=0.8)
    xmax = max(measured + [1.0])
    ax.set_xlim(0.0, xmax * 1.25)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("end-to-end latency (ms)")
    ax.set_title("SAM 3D Objects - end-to-end latency (image + mask -> Gaussian PLY)")
    ax.grid(axis="y", alpha=0.3)
    gate_ms = report.get("latency_gate_ms", 70000)
    if gate_ms is not None:
        ax.axvline(float(gate_ms), color="#C44E52", linestyle="--", linewidth=1.2,
                   label=f"gate {float(gate_ms) / 1000:.0f}s")
        ax.legend(frameon=False, loc="lower right")
    for bar, row, value in zip(bars, rows, values):
        if value is None:
            text = str(row.get("status", "not measured"))
            x = xmax * 0.02
        else:
            text = f"{float(value):,.0f} ms"
            x = float(value)
        ax.annotate(text, (x, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=9)
    fig.text(0.01, 0.01, str(report.get("note", "")), fontsize=8,
             color="#555555", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)

    metrics_output = args.metrics_output
    if metrics_output is None:
        stem = args.output.stem.replace("latency", "metrics")
        metrics_output = args.output.with_name(stem + args.output.suffix)
    quality_rows = []
    for row in rows:
        metrics = {
            "mae": row.get("render_mae"),
            "psnr_db": row.get("render_psnr_db"),
        }
        report_path = row.get("render_report")
        if report_path:
            path = Path(str(report_path))
            if not path.is_absolute():
                path = args.input.parent / path
            if path.is_file():
                metrics.update(json.loads(path.read_text(encoding="utf-8")))
        if metrics["mae"] is not None or metrics["psnr_db"] is not None:
            quality_rows.append((row, metrics))
    if any(str(row.get("runner", "")).lower().startswith("pytorch") for row in rows):
        quality_rows.insert(0, ({"runner": "PyTorch official reference", "latency_ms": 0},
                                {"mae": 0.0, "psnr_db": None}))
    if quality_rows:
        labels = [str(row["runner"]) for row, _ in quality_rows]
        maes = [float(metrics["mae"]) if metrics.get("mae") is not None else math.nan
                for _, metrics in quality_rows]
        psnrs = [float(metrics["psnr_db"]) if metrics.get("psnr_db") is not None else math.nan
                 for _, metrics in quality_rows]
        colors = [bar_color(row, row.get("latency_ms")) for row, _ in quality_rows]
        figure_height = max(5.2, 0.52 * len(labels) + 0.9)
        fig, axes = plt.subplots(1, 2, figsize=(13.0, figure_height))
        ypos = list(range(len(labels)))
        mae_values = [0.0 if math.isnan(value) else value for value in maes]
        mae_bars = axes[0].barh(ypos, mae_values, color=colors, edgecolor="white")
        axes[0].axvline(0.01, color="#C44E52", linestyle="--", linewidth=1.0,
                        label="quality gate 0.01")
        axes[0].set_xlabel("render RGB MAE (lower is better)")
        axes[0].set_title("End-to-end rendered quality")
        axes[0].legend(frameon=False)
        for bar, value in zip(mae_bars, maes):
            if not math.isnan(value):
                axes[0].annotate(f"{value:.5f}", (value, bar.get_y() + bar.get_height() / 2),
                                 xytext=(5, 0), textcoords="offset points", va="center")
        psnr_values = [0.0 if math.isnan(value) else value for value in psnrs]
        psnr_bars = axes[1].barh(ypos, psnr_values, color=colors, edgecolor="white")
        axes[1].set_xlabel("render PSNR (dB, higher is better)")
        axes[1].set_title("End-to-end rendered quality")
        for bar, value in zip(psnr_bars, psnrs):
            if not math.isnan(value):
                axes[1].annotate(f"{value:.2f}", (value, bar.get_y() + bar.get_height() / 2),
                                 xytext=(5, 0), textcoords="offset points", va="center")
        for quality_ax in axes:
            quality_ax.set_yticks(ypos, labels)
            quality_ax.invert_yaxis()
            quality_ax.grid(axis="x", alpha=0.3)
        fig.suptitle("SAM 3D Objects - end-to-end render comparison", fontsize=13)
        fig.text(0.01, 0.01,
                 "Metrics are from the same input/seed and official camera path; "
                 "PyTorch MAE is the self-reference (0 by definition); PSNR is N/A.",
                 fontsize=8, color="#555555")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        metrics_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(metrics_output, dpi=160)
        plt.close(fig)
        print(f"wrote {metrics_output}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
