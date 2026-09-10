#!/usr/bin/env python3
"""Re-run one refused official timing sample and append it to the summary.

The bench_full_e2e_pipeline.py process refuses to run (fast rc=1) when its own
exclusivity probe sees an external compute client; such a refusal leaves the
official row one valid sample short of the >=3 gate requirement.  This tool
waits for a free GPU, replays the recorded official command into a fresh
directory, and appends the receipt to ``steps[name]["runs"]``.
"""
from __future__ import annotations

import json
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_exclusivity import require_exclusive_gpu  # noqa: E402

REPO = Path("/home/ludahai/develop/code/github/dl/sam-3d-objects-ggml")
W = REPO / "output/full-glb-respin-gsplat-20260917d"
SUMMARY = W / "full_e2e_summary_repeated.json"
STEP = "official_full_image_to_textured_glb_timing"
NEW_DIR = W / "repeat_2b/official_full_glb_timing"


def main() -> int:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    step = next(s for s in summary["steps"] if s["name"] == STEP)
    # Point the recorded --out-dir at the fresh directory.
    tokens = shlex.split(step["command"])
    out_idx = tokens.index("--out-dir")
    tokens[out_idx + 1] = str(NEW_DIR)
    command = tokens

    deadline = time.monotonic() + 60 * 60
    while True:
        try:
            require_exclusive_gpu("cuda")
            break
        except RuntimeError as error:
            if time.monotonic() >= deadline:
                raise SystemExit(f"GPU stayed busy for 60min: {error}")
            print(f"GPU busy, waiting ({str(error)[:100]})", flush=True)
            time.sleep(30)

    NEW_DIR.mkdir(parents=True, exist_ok=False)
    print("+", " ".join(command), flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=REPO, text=True, check=False)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    receipt = {
        "attempt": 4, "run_id": "repeat_2b", "step": STEP,
        "command": " ".join(command), "dir": str(NEW_DIR),
        "returncode": result.returncode, "elapsed_ms": elapsed_ms,
        "dtype_contract_ok": None,
        "note": "refusal repair for repeat_2 (external client seen at launch)",
    }
    step.setdefault("runs", []).append(receipt)
    ok = [r["elapsed_ms"] for r in step["runs"]
          if r.get("returncode") == 0 and isinstance(r.get("elapsed_ms"), (int, float))]
    if len(ok) >= 2:
        step["elapsed_ms_single_run"] = step["elapsed_ms"]
        step["elapsed_ms"] = statistics.median(ok)
    step["repeat_summary"] = {
        "samples": len(step["runs"]), "successful": len(ok),
        "median_ms": step["elapsed_ms"],
        "min_ms": min(ok) if ok else None, "max_ms": max(ok) if ok else None,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"official: samples={len(step['runs'])} successful={len(ok)} "
          f"median={step['elapsed_ms']/1000:.1f}s rc={result.returncode}")
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
