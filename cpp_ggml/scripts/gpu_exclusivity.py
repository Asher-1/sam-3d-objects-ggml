"""NVIDIA compute-client provenance for reproducible timing runs."""
from __future__ import annotations

import subprocess


def active_nvidia_compute_processes() -> list[str]:
    """Return active NVIDIA compute clients, or fail with an actionable error."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("--require-exclusive-gpu requires nvidia-smi on PATH") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip()
        raise RuntimeError(
            "--require-exclusive-gpu could not query NVIDIA compute clients"
            + (f": {detail}" if detail else "")
        ) from error
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def require_exclusive_gpu(backend: str) -> dict[str, object]:
    """Reject CUDA/Vulkan timing runs contaminated by another compute client."""
    if backend not in ("cuda", "vulkan"):
        return {"required": False, "checked": False, "active_compute_processes": []}
    active = active_nvidia_compute_processes()
    if active:
        raise RuntimeError(
            "GPU is not exclusive; do not use this run for latency evidence. "
            "Active compute clients: " + "; ".join(active)
        )
    return {"required": True, "checked": True, "active_compute_processes": []}
