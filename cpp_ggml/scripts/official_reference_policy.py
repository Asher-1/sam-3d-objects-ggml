"""Shared precision policy for memory-bounded official reference runs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch


def precision_label(checkpoint: str | Path) -> str:
    """Return the checkpoint-specific execution precision in the 12 GiB reference."""
    name = Path(checkpoint).name
    if name.startswith(("ss_decoder", "slat_decoder")):
        # These architectures contain explicit *Norm32 boundaries and apply
        # their own supported F16 conversion internally when configured.
        return "native"
    if name.startswith("slat_generator"):
        return "f16"
    return "bf16"


def install_staged_mixed_precision_loader(
        original_loader: Callable[..., torch.nn.Module]) -> Callable[..., torch.nn.Module]:
    """Keep models on CPU and preserve the official per-family dtype layout.

    The benchmark moves exactly the active stage to CUDA. Returning CPU models
    here prevents all checkpoints from becoming resident during construction.
    """
    def staged_loader(self: Any, config: Any, ckpt_path: str, state_dict_fn: Any = None,
                      state_dict_key: str = "state_dict", device: str = "cuda") -> torch.nn.Module:
        del device
        model = original_loader(
            self, config, ckpt_path, state_dict_fn=state_dict_fn,
            state_dict_key=state_dict_key, device="cpu")
        label = precision_label(ckpt_path)
        if label == "f16":
            model = model.to(dtype=torch.float16)
        elif label == "bf16":
            model = model.to(dtype=torch.bfloat16)
        return model

    return staged_loader


def reference_weight_policy(fp32_weights: bool) -> str:
    return "official-module-f32" if fp32_weights else "staged-mixed-bf16-f16-native"
