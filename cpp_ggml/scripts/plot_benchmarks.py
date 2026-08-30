#!/usr/bin/env python3
"""Render SS-decoder diagnostics from checked-in measurements.

Inputs (under cpp_ggml/benchmarks/diagnostics/ss_decoder/):
  latency.jsonl            rows written by sam3d-cli decode-ss --json (append-only;
                           the latest row per (backend, threads, dtype) wins)
  ss_decoder_parity.md     node-by-node parity table vs the PyTorch reference
  data/ss_occ_{dtype}.samt occupancy outputs dumped by bench_decode_ss.sh

Outputs (outside the repository by default):
  latency_by_backend.png   grouped SS-decoder diagnostic latency bars
  latency_matrix.png       SS-decoder backend/precision heatmap
  stage_breakdown.png      SS-decoder setup/compute diagnostic breakdown
  ss_decoder_parity_nodes.png  node-by-node max|d| vs PyTorch (from the md table)
  ss_decoder_quality.png   3D occupancy renders per precision + logit agreement

These outputs are module-level diagnostics and are deliberately kept outside
the release E2E evidence. The script validates the full key matrix before
writing anything and fails loudly on missing data.
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
SS_BENCH_DIR = BENCH_DIR / "diagnostics" / "ss_decoder"
OUTPUT_DIR = Path(os.environ.get("SAM3D_SS_DIAGNOSTIC_DIR", "/tmp/sam3d-ss-diagnostics"))
JSONL = SS_BENCH_DIR / "latency.jsonl"
PARITY_MD = SS_BENCH_DIR / "ss_decoder_parity.md"
DATA_DIR = BENCH_DIR / "data"

DTYPES = ["f32", "f16", "q4_0", "q8_0"]
CONFIGS = [("cpu", 8, "CPU 8T"), ("cpu", 32, "CPU 32T"),
           ("cuda", None, "CUDA"), ("vulkan", None, "Vulkan")]
CONFIG_COLORS = {"CPU 8T": "#4C72B0", "CPU 32T": "#64B5CD",
                 "CUDA": "#DD8452", "Vulkan": "#55A868"}


def read_samt_f32(path: Path) -> tuple[tuple[int, ...], np.ndarray]:
    with open(path, "rb") as f:
        if f.read(4) != b"SAMT":
            raise ValueError(f"{path}: bad magic")
        (ndims,) = struct.unpack("<i", f.read(4))
        ne = struct.unpack(f"<{ndims}q", f.read(8 * ndims))  # ne is int64 on disk
        (ttype,) = struct.unpack("<i", f.read(4))
        if ttype != 0:
            raise ValueError(f"{path}: expected F32 payload, type={ttype}")
        data = np.frombuffer(f.read(), dtype="<f4")
    return ne, data


def norm_backend(name: str) -> str:
    name = name.lower()
    if name.startswith("cuda"):
        return "cuda"
    if name.startswith("vulkan"):
        return "vulkan"
    return name


def load_rows() -> dict[tuple[str, int, str], dict]:
    if not JSONL.exists():
        sys.exit(f"missing {JSONL}; run scripts/bench_decode_ss.sh first")
    rows: dict[tuple[str, int, str], dict] = {}
    for line in JSONL.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        key = (norm_backend(r["backend"]), int(r["n_threads"]), r["dtype"])
        rows[key] = r  # append-only: latest row per key wins
    return rows


def resolve_row(rows, backend: str, threads: int | None, dtype: str) -> dict:
    for (b, t, d), r in rows.items():
        if b == backend and d == dtype and (threads is None or t == threads):
            return r
    raise KeyError(f"no latency row for backend={backend} threads={threads} dtype={dtype}")


def validate(rows) -> None:
    missing = []
    for backend, threads, _label in CONFIGS:
        for dt in DTYPES:
            try:
                resolve_row(rows, backend, threads, dt)
            except KeyError:
                missing.append(f"{backend}/{threads or '-'}/{dt}")
    required = ["load_ms", "build_ms", "alloc_ms", "upload_ms",
                "graph_ms_mean", "graph_ms_min", "graph_ms_p50", "graph_ms_max"]
    for key, r in rows.items():
        for f in required:
            if f not in r:
                missing.append(f"{key} lacks {f}")
    if missing:
        sys.exit("incomplete benchmark matrix, refusing to render:\n  " + "\n  ".join(missing))


def parse_parity_table() -> list[dict]:
    """Parse the node-by-node table of ss_decoder_parity.md."""
    nodes, in_section = [], False
    for line in PARITY_MD.read_text().splitlines():
        if line.startswith("## Node-by-node"):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if not in_section or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4 or cells[0] in ("node",) or set(cells[0]) <= {"-", ":"}:
            continue
        try:
            nodes.append({"node": int(cells[0]), "module": cells[1],
                          "max_abs": float(cells[2]), "mean_abs": float(cells[3])})
        except ValueError:
            continue
    if not nodes:
        sys.exit(f"failed to parse the node table from {PARITY_MD}")
    return nodes


def fmt_ms(v: float) -> str:
    return f"{v:.2f}" if v < 100 else f"{v:,.0f}"


# ── charts ────────────────────────────────────────────────────────────────────
def chart_latency_by_backend(rows) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.arange(len(DTYPES))
    width = 0.26
    for i, (_b, _t, label) in enumerate(CONFIGS):
        means = [resolve_row(rows, *CONFIGS[i][:2], dt)["graph_ms_mean"] for dt in DTYPES]
        bars = ax.bar(x + (i - 1) * width, means, width, label=label,
                      color=CONFIG_COLORS[label], edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, means):
            ax.annotate(fmt_ms(v), (bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 3)
    ax.set_xticks(x, DTYPES)
    ax.set_xlabel("weight precision")
    ax.set_ylabel("graph compute time (ms, log)")
    ax.set_title("SAM 3D SS decoder — graph latency by backend and precision\n"
                 "64³ occupancy decode, 8×16³ synthetic latent, fixed input")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "latency_by_backend.png", dpi=150)
    plt.close(fig)


def chart_latency_matrix(rows) -> None:
    mat = np.zeros((len(CONFIGS), len(DTYPES)))
    for i, (b, t, _l) in enumerate(CONFIGS):
        for j, dt in enumerate(DTYPES):
            mat[i, j] = resolve_row(rows, b, t, dt)["graph_ms_mean"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.imshow(mat, cmap="viridis", norm=LogNorm(vmin=mat.min(), vmax=mat.max()))
    ax.set_xticks(range(len(DTYPES)), DTYPES)
    ax.set_yticks(range(len(CONFIGS)), [l for _b, _t, l in CONFIGS])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, fmt_ms(mat[i, j]), ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")
    ax.set_title("SS decoder latency matrix (ms, mean of timed runs)")
    fig.colorbar(im, ax=ax, label="graph ms (log)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "latency_matrix.png", dpi=150)
    plt.close(fig)


def chart_stage_breakdown(rows) -> None:
    configs_flat = [(b, t, label, dt) for (b, t, label) in CONFIGS for dt in DTYPES]
    labels = [f"{label}\n{dt}" for _b, _t, label, dt in configs_flat]
    stages = [("load_ms", "#55A868"), ("build_ms", "#C44E52"),
              ("alloc_ms", "#8172B3"), ("upload_ms", "#CCB974")]
    setup = [sum(resolve_row(rows, b, t, dt)[s] for s, _c in stages)
             for b, t, _l, dt in configs_flat]
    graph = [resolve_row(rows, b, t, dt)["graph_ms_mean"]
             for b, t, _l, dt in configs_flat]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    bottom = np.zeros(len(labels))
    for s, color in stages:
        vals = np.array([resolve_row(rows, b, t, dt)[s]
                         for b, t, _l, dt in configs_flat])
        axes[0].bar(x, vals, bottom=bottom, color=color, label=s.removesuffix("_ms"))
        bottom += vals
    axes[0].set_title("one-time setup per config (model load + graph build +\nalloc + input upload)")
    axes[0].legend(frameon=False)
    bar_colors = [CONFIG_COLORS[label] for (_b, _t, label, _dt) in configs_flat]
    axes[1].bar(x, graph, color=bar_colors)
    for xi, v in zip(x, graph):
        axes[1].annotate(fmt_ms(v), (xi, v), xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8)
    axes[1].set_title("timed graph compute (mean ms)")
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("ms")
    axes[0].set_ylim(top=max(setup) * 1.25)
    fig.suptitle("SS decoder — setup cost vs steady-state compute")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stage_breakdown.png", dpi=150)
    plt.close(fig)


def chart_parity_nodes() -> None:
    nodes = parse_parity_table()
    short = {"middle_block": "midblk", "input_layer conv": "input conv",
             "out_layer (final logits)": "out_layer\n(final logits)",
             " upsample ": "\nup "}
    def module_label(name: str) -> str:
        out = name
        for long, brief in short.items():
            out = out.replace(long, brief)
        return out
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    x = np.arange(len(nodes))
    vals = [n["max_abs"] for n in nodes]
    bars = ax.bar(x, vals, color="#4C72B0", edgecolor="white")
    for bar, n in zip(bars, nodes):
        ax.annotate(f"{n['max_abs']:.4f}", (bar.get_x() + bar.get_width() / 2, n["max_abs"]),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-4, top=max(vals) * 2.5)
    ax.set_xticks(x, [f"{n['node']}\n{module_label(n['module'])}" for n in nodes], fontsize=8)
    ax.set_ylabel("max |Δ| vs PyTorch fp32 (log)")
    ax.set_title("SS decoder node-by-node parity vs official PyTorch (CUDA fp32)\n"
                 "f32 GGUF graph, fp32 accumulation-order noise floor ≈ 1e-3")
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ss_decoder_parity_nodes.png", dpi=150)
    plt.close(fig)


def render_voxel(ax, occ: np.ndarray, title: str) -> dict:
    pos = np.argwhere(occ > 0)  # (N, 3) as (d0, d1, d2) = (w, h, d) axes
    stats = {"n": int(occ.size), "n_pos": int((occ > 0).sum()),
             "lo": float(occ.min()), "hi": float(occ.max())}
    if pos.size:
        step = max(1, len(pos) // 20000)
        pos = pos[::step]
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=pos[:, 2], cmap="viridis",
                   s=4, depthshade=False, linewidths=0)
    ax.set_xlim(0, occ.shape[0] - 1)
    ax.set_ylim(0, occ.shape[1] - 1)
    ax.set_zlim(0, occ.shape[2] - 1)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-60)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(f"{title}\n{stats['n_pos']}/{stats['n']} voxels occupied "
                 f"(logits {stats['lo']:.1f}..{stats['hi']:.1f})", fontsize=9)
    return stats


def chart_quality() -> dict:
    fig = plt.figure(figsize=(16.0, 8.2))
    gs = fig.add_gridspec(2, len(DTYPES), height_ratios=[1.35, 1.0])
    occs, stats = {}, {}
    titles = {"f32": "ggml CPU 8T f32 (parity reference)",
              "f16": "ggml CUDA f16 (deployment format)",
              "q4_0": "ggml CUDA q4_0 (4-bit format)",
              "q8_0": "ggml CPU 8T q8_0 (compact format)"}
    for j, dt in enumerate(DTYPES):
        path = DATA_DIR / f"ss_occ_{dt}.samt"
        if not path.exists():
            sys.exit(f"missing {path}; run scripts/bench_decode_ss.sh (without --skip-outputs)")
        ne, data = read_samt_f32(path)
        occ = data.reshape([d for d in ne if d > 1] or [1])
        occs[dt] = occ
        ax = fig.add_subplot(gs[0, j], projection="3d")
        stats[dt] = render_voxel(ax, occ, titles[dt])
    ref = occs["f32"].ravel()
    agree = {}
    for j, dt in enumerate([dt for dt in DTYPES if dt != "f32"], start=1):
        got = occs[dt].ravel()
        ax = fig.add_subplot(gs[1, j])
        step = max(1, ref.size // 8000)
        ax.scatter(ref[::step], got[::step], s=4, alpha=0.35, linewidths=0,
                   color=CONFIG_COLORS["CUDA"] if dt == "f16" else CONFIG_COLORS["CPU 32T"])
        lims = [min(ref.min(), got.min()), max(ref.max(), got.max())]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.6)
        ax.set_xlabel("f32 logits"); ax.set_ylabel(f"{dt} logits")
        sign_agree = 100.0 * float(np.mean(np.sign(ref) == np.sign(got)))
        agree[dt] = sign_agree
        ax.set_title(f"f32 vs {dt}: sign agreement {sign_agree:.4f}%", fontsize=10)
        ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[1, 0])
    for dt, color in (("f16", CONFIG_COLORS["CUDA"]),
                      ("q4_0", "#DD8452"),
                      ("q8_0", CONFIG_COLORS["CPU 32T"])):
        d = occs[dt].ravel() - ref
        ax.hist(d, bins=80, alpha=0.55, label=f"{dt} − f32 (max|Δ|={np.abs(d).max():.4f})",
                color=color)
    ax.set_xlabel("logit delta"); ax.set_ylabel("count")
    ax.set_title("logit deltas vs f32", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle("SS decoder — decoded occupancy field per precision (synthetic latent, "
                 "fixed seed; occupancy = logits > 0)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ss_decoder_quality.png", dpi=150)
    plt.close(fig)
    return stats, agree


def write_speedup_table(rows) -> None:
    """Emit speedup_table.md: per-config latency breakdown + speedup vs the
    CPU 8T f32 baseline (no PyTorch row exists here: the official checkpoints
    are gated, see README)."""
    baseline = resolve_row(rows, "cpu", 8, "f32")["graph_ms_mean"]
    lines = [
        "# SS decoder latency and speedup",
        "",
        "Graph compute time for one 8×16³ → 1×64³ decode (synthetic fixed-seed",
        "latent). Speedup is relative to the CPU 8T f32 baseline; the official",
        "PyTorch checkpoints are gated on HuggingFace, so no PyTorch reference",
        "row exists in this table (numerical parity evidence lives in",
        "[ss_decoder_parity.md](ss_decoder_parity.md)).",
        "",
        "| config | dtype | load ms | build ms | alloc ms | upload ms | graph mean | min | p50 | max | speedup vs CPU 8T f32 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b, t, label in CONFIGS:
        for dt in DTYPES:
            r = resolve_row(rows, b, t, dt)
            speed = baseline / r["graph_ms_mean"]
            lines.append(
                f"| {label} | {dt} | {r['load_ms']:.0f} | {r['build_ms']:.1f} | "
                f"{r['alloc_ms']:.2f} | {r['upload_ms']:.2f} | "
                f"{r['graph_ms_mean']:.2f} | {r['graph_ms_min']:.2f} | "
                f"{r['graph_ms_p50']:.2f} | {r['graph_ms_max']:.2f} | x{speed:.2f} |")
    lines.append("")
    (OUTPUT_DIR / "speedup_table.md").write_text("\n".join(lines))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    validate(rows)
    chart_latency_by_backend(rows)
    chart_latency_matrix(rows)
    chart_stage_breakdown(rows)
    chart_parity_nodes()
    stats, agree = chart_quality()
    write_speedup_table(rows)
    print(f"rendered SS diagnostics outside the repository: {OUTPUT_DIR}")
    for dt in DTYPES:
        s = stats[dt]
        print(f"  {dt}: occupied {s['n_pos']}/{s['n']} logits [{s['lo']:.2f}, {s['hi']:.2f}]")
    for dt, a in agree.items():
        print(f"  sign agreement f32 vs {dt}: {a:.4f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
