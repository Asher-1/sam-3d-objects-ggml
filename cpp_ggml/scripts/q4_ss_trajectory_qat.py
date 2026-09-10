#!/usr/bin/env python3
"""Train sparse SS Q4_0 overrides against a PyTorch Euler trajectory.

Every aligned frozen matrix is encoded from the original F32 checkpoint before
the student enters F16 compute, matching GGUF conversion exactly. Requested
sensitive matrices own F32 master weights with a straight-through gradient.
The output checkpoint is intentionally sparse and is consumed by
``convert_sam3d_to_gguf.py --weight-overrides``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from gguf_schema import quantize_q4_0
from ss_trajectory_ref import MODS, load_generator, load_samt


def q4_0_dequant_ste(weight: torch.Tensor) -> torch.Tensor:
    """Emulate GGML Q4_0's F16 scale and codes while passing an STE gradient."""
    if weight.ndim != 2 or weight.shape[-1] % 32:
        raise ValueError(f"Q4_0 master weight must be rank-2 and 32-aligned, got {tuple(weight.shape)}")
    blocks = weight.float().reshape(-1, 32)
    max_index = blocks.abs().argmax(dim=1, keepdim=True)
    max_value = blocks.gather(1, max_index)
    scale_f32 = max_value / -8.0
    reciprocal = torch.where(scale_f32 != 0, scale_f32.reciprocal(), torch.zeros_like(scale_f32))
    codes = torch.trunc(blocks * reciprocal + 8.5).clamp_(0, 15)
    dequantized = scale_f32.to(torch.float16).float() * (codes - 8.0)
    dequantized = dequantized.reshape_as(weight)
    return weight + (dequantized - weight).detach()


def reference_q4_0(weight: torch.Tensor) -> torch.Tensor:
    """Decode the repository's byte-for-byte GGUF Q4_0 reference encoding."""
    source = np.ascontiguousarray(weight.detach().float().cpu().numpy())
    packed = quantize_q4_0(source).reshape(-1, 18)
    scales = packed[:, :2].copy().reshape(-1).view(np.float16).astype(np.float32)
    codes = packed[:, 2:]
    unpacked = np.concatenate((codes & 0x0F, codes >> 4), axis=1).astype(np.float32)
    return torch.from_numpy((scales[:, None] * (unpacked - 8.0)).reshape(source.shape))


def self_test() -> None:
    torch.manual_seed(17)
    weight = torch.randn(8, 64, dtype=torch.float32, requires_grad=True)
    actual = q4_0_dequant_ste(weight)
    expected = reference_q4_0(weight)
    error = (actual.detach().cpu() - expected).abs().max().item()
    if error != 0.0:
        raise RuntimeError(f"Q4_0 fake quantizer differs from GGUF encoder: max|delta|={error}")
    actual.square().mean().backward()
    if weight.grad is None or not torch.isfinite(weight.grad).all():
        raise RuntimeError("Q4_0 STE did not produce finite master-weight gradients")
    print("q4_0_ste_self_test=PASS")


def prepare_fp16_student(generator: torch.nn.Module) -> torch.nn.Module:
    """Restore the official mixed-precision boundary after compacting weights.

    The input mappings are outside ``convert_to_fp16()``, so they must be
    compacted together with the transformer torso. Timestep embeddings
    deliberately construct FP32 Fourier features, however, and must be
    restored to FP32 before their output is cast at the backbone boundary.
    """
    from sam3d_objects.model.backbone.tdfy_dit.modules.utils import convert_module_to_f16

    generator.apply(convert_module_to_f16)
    backbone = generator.reverse_fn.backbone
    backbone.t_embedder.float()
    if hasattr(backbone, "d_embedder"):
        backbone.d_embedder.float()
    if backbone.share_mod:
        backbone.adaLN_modulation.float()
    for latent in backbone.latent_mapping.values():
        # ``to_output`` intentionally applies FP32 layer_norm before this projection.
        latent.out_layer.float()
    backbone.dtype = torch.float16
    backbone.use_fp16 = True
    return generator.eval()


def compute_dtype(generator: torch.nn.Module) -> torch.dtype:
    return generator.reverse_fn.backbone.dtype


def enable_activation_checkpointing(generator: torch.nn.Module) -> int:
    """Trade frozen-block recomputation for bounded QAT activation memory."""
    blocks = generator.reverse_fn.backbone.blocks
    for block in blocks:
        block.use_checkpoint = True
    return len(blocks)


def state_from_samt(e2e_dir: Path, step: int, device: torch.device) -> dict[str, torch.Tensor]:
    suffix = "x0" if step == 0 else f"state{step}"
    state: dict[str, torch.Tensor] = {}
    for modality in MODS:
        path = e2e_dir / f"ss_{suffix}_{modality}.samt"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}; run ss_trajectory_ref.py first to create the teacher trajectory"
            )
        shape, values = load_samt(path)
        state[modality] = torch.from_numpy(values.reshape(shape[::-1]).copy()).to(device=device)
    return state


def state_from_reference_trajectory(reference_dir: Path, step: int,
                                    device: torch.device) -> dict[str, torch.Tensor]:
    """Load the independent full-precision trajectory emitted by ss_trajectory_ref.

    That utility records public tensors with their PyTorch batch dimension in
    the SAMT header.  QAT consumes one sample, so remove that leading singleton
    while retaining the token-major (tokens, channels) layout used by the
    student and the C++ flow graph.
    """
    state: dict[str, torch.Tensor] = {}
    for modality in MODS:
        path = reference_dir / f"ss_torch_x{step:03d}_{modality}.samt"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}; run ss_trajectory_ref.py with --steps >= {step}"
            )
        shape, values = load_samt(path)
        if len(shape) == 3 and shape[0] == 1:
            rows, columns = shape[1], shape[2]
        elif len(shape) == 2 and shape[0] == 1:
            rows, columns = shape
        elif len(shape) == 2:
            # Native C++/SAMT dumps expose {channels, tokens}.
            columns, rows = shape
        else:
            raise ValueError(f"unsupported reference state shape in {path}: {shape}")
        if values.size != rows * columns:
            raise ValueError(f"truncated reference state in {path}: {values.size} values")
        state[modality] = torch.from_numpy(values.reshape(rows, columns).copy()).to(device=device)
    return state


def q4_sensitivity_report(generator: torch.nn.Module, e2e_dir: Path,
                          device: torch.device, output: Path) -> None:
    """Rank aligned linear leaves by their actual first-step Q4 output error.

    A weight-only norm is not sufficient for this model: the same Q4 error is
    amplified differently by the conditional and CFG-unconditional branches.
    This report executes both branches on the canonical initial state and
    accumulates the error of each individual F.linear replacement.
    """
    state = state_from_samt(e2e_dir, 0, device)
    model_dtype = compute_dtype(generator)
    state = {name: value.to(dtype=model_dtype) for name, value in state.items()}
    _, condition_data = load_samt(e2e_dir / "ss_cond_tokens.samt")
    condition = torch.from_numpy(condition_data.reshape(1, -1, 1024).copy()).to(
        device=device, dtype=model_dtype)
    records: dict[str, dict[str, float | int]] = {}
    hooks = []

    for name, module in generator.named_modules():
        if not isinstance(module, torch.nn.Linear) or module.weight.ndim != 2:
            continue
        if module.weight.shape[1] % 32:
            continue

        def hook(current: torch.nn.Module, inputs: tuple[torch.Tensor, ...], result: torch.Tensor,
                 *, _name: str = name) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(result, torch.Tensor):
                return
            activation = inputs[0]
            quantized = q4_0_dequant_ste(current.weight).detach().to(dtype=activation.dtype)
            delta = F.linear(activation, quantized - current.weight.to(dtype=activation.dtype))
            row = records.setdefault(_name, {"calls": 0, "elements": 0,
                                             "error_sum_sq": 0.0, "signal_sum_sq": 0.0})
            row["calls"] = int(row["calls"]) + 1
            row["elements"] = int(row["elements"]) + delta.numel()
            row["error_sum_sq"] = float(row["error_sum_sq"]) + float(delta.float().square().sum())
            row["signal_sum_sq"] = float(row["signal_sum_sq"]) + float(result.float().square().sum())

        hooks.append(module.register_forward_hook(hook))

    try:
        backbone = generator.reverse_fn.backbone
        time = torch.zeros(1, device=device, dtype=torch.float32)
        with torch.no_grad():
            backbone(state, time, condition, d=torch.zeros_like(time), cfg=False)
            backbone(state, time, condition, d=torch.zeros_like(time), cfg=True)
    finally:
        for hook in hooks:
            hook.remove()

    rows = []
    for name, row in records.items():
        elements = int(row["elements"])
        error_rmse = (float(row["error_sum_sq"]) / elements) ** 0.5
        signal_rmse = (float(row["signal_sum_sq"]) / elements) ** 0.5
        rows.append({"module": name, "calls": int(row["calls"]), "elements": elements,
                     "q4_output_rmse": error_rmse, "signal_rmse": signal_rmse,
                     "relative_rmse": error_rmse / signal_rmse if signal_rmse else 0.0})
    rows.sort(key=lambda row: row["q4_output_rmse"], reverse=True)
    payload = {"schema": "sam3d.ss_q4_first_step_sensitivity.v1",
               "reference": "canonical_ss_x0_and_ss_cond_tokens",
               "branches": ["conditional", "cfg_unconditional"], "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    for row in rows[:20]:
        print(f"{row['module']}: q4_rmse={row['q4_output_rmse']:.7g} "
              f"relative={row['relative_rmse']:.7g}", flush=True)


def save_teacher_trajectory(generator: torch.nn.Module, e2e_dir: Path, out_dir: Path,
                            device: torch.device, schedule_steps: int, save_steps: int, rescale_t: float,
                            cfg_strength: float, cfg_start: float, cfg_end: float) -> None:
    """Write mixed-precision PyTorch Euler states for the requested window."""
    from ss_trajectory_ref import write_samt_f32

    out_dir.mkdir(parents=True, exist_ok=True)
    x = state_from_samt(e2e_dir, 0, device)
    _, cond_data = load_samt(e2e_dir / "ss_cond_tokens.samt")
    model_dtype = compute_dtype(generator)
    cond = torch.from_numpy(cond_data.reshape(1, -1, 1024).copy()).to(device=device, dtype=model_dtype)
    x = {key: value.to(dtype=model_dtype) for key, value in x.items()}
    u = torch.linspace(0.0, 1.0, schedule_steps + 1, device=device)
    times = u / (1.0 + (rescale_t - 1.0) * (1.0 - u)) if rescale_t else u
    backbone = generator.reverse_fn.backbone
    with torch.no_grad():
        for index, value in x.items():
            write_samt_f32(out_dir / f"ss_x0_{index}.samt", [value.shape[-1], value.shape[-2]],
                            value[0].float().cpu().numpy())
        for step, (t0, t1) in enumerate(zip(times[:save_steps], times[1:save_steps + 1]), start=1):
            t = (t0 * 1000.0).reshape(1)
            cond_velocity = backbone(x, t, cond, d=torch.zeros_like(t), cfg=False)
            uncond_velocity = backbone(x, t, cond, d=torch.zeros_like(t), cfg=True)
            for modality in MODS:
                # Match inference-time ClassifierFreeGuidance, whose scalar
                # strength is mapped over every output modality.
                velocity = (cond_velocity[modality] + cfg_strength *
                            (cond_velocity[modality] - uncond_velocity[modality])
                            if cfg_start <= float(t) <= cfg_end
                            else cond_velocity[modality])
                x[modality] = x[modality] + (t1 - t0) * velocity
                write_samt_f32(out_dir / f"ss_state{step}_{modality}.samt",
                                [x[modality].shape[-1], x[modality].shape[-2]],
                                x[modality][0].float().cpu().numpy())
            # The output projection restores the public latent to F32. Keep
            # that reference on disk, then return to the student's FP16 input
            # projection contract before the next Euler evaluation.
            x = {name: value.to(dtype=model_dtype) for name, value in x.items()}


def prepare_deployment_q4_student(
        generator: torch.nn.Module, targets: list[str],
        ) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    """Match the converter's F32-to-Q4 encoding before reducing compute precision.

    GGUF conversion quantizes the original F32 checkpoint.  Casting the model
    to F16 first can change a 32-element Q4 block's extremum and therefore its
    scale and codes.  That creates a different student from the exported GGUF.
    Keep only selected masters on the host, replace every other aligned matrix
    with its exact F32-source Q4 decode, then establish the normal F16 compute
    boundary.
    """
    modules = dict(generator.named_modules())
    unknown = [target for target in targets if target not in modules]
    if unknown:
        raise ValueError(f"unknown QAT targets: {', '.join(unknown)}")
    master_sources: dict[str, torch.Tensor] = {}
    for name, module in modules.items():
        if not isinstance(module, torch.nn.Linear) or module.weight.ndim != 2:
            continue
        if module.weight.shape[1] % 32:
            if name in targets:
                raise ValueError(f"QAT target is not Q4_0 aligned: {name} {tuple(module.weight.shape)}")
            continue
        if name in targets:
            # Host storage avoids duplicating the full-precision generator on
            # the constrained training GPU before the model is compacted.
            master_sources[name] = module.weight.detach().float().cpu().clone()
        else:
            with torch.no_grad():
                module.weight.copy_(q4_0_dequant_ste(module.weight).detach())
    missing = [target for target in targets if target not in master_sources]
    if missing:
        raise TypeError(f"QAT targets must be Q4_0-aligned torch.nn.Linear modules: {', '.join(missing)}")
    return prepare_fp16_student(generator), master_sources


def install_qat_weights(generator: torch.nn.Module, targets: list[str],
                        master_sources: dict[str, torch.Tensor]) -> dict[str, torch.nn.Parameter]:
    """Make the training graph faithful to the all-Q4_0 deployment graph."""
    for parameter in generator.parameters():
        parameter.requires_grad_(False)

    masters: dict[str, torch.nn.Parameter] = {}
    modules = dict(generator.named_modules())
    unknown = [target for target in targets if target not in modules]
    if unknown:
        raise ValueError(f"unknown QAT targets: {', '.join(unknown)}")

    for name, module in modules.items():
        if not isinstance(module, torch.nn.Linear) or module.weight.ndim != 2:
            continue
        if module.weight.shape[1] % 32:
            if name in targets:
                raise ValueError(f"QAT target is not Q4_0 aligned: {name} {tuple(module.weight.shape)}")
            continue
        if name in targets:
            source = master_sources.get(name)
            if source is None:
                raise ValueError(f"missing F32 QAT master source for {name}")
            master = torch.nn.Parameter(source.to(device=module.weight.device))
            masters[name] = master
            weight_source = master
            trainable = True
        else:
            # The original FP16 parameter and its exact Q4_0-dequantized form
            # have the same storage size. Keep only the latter: regenerating a
            # FP32 dequantized tensor at every layer exhausted 12 GiB before
            # backward could reach the selected masters.
            with torch.no_grad():
                weight_source = q4_0_dequant_ste(module.weight).detach().to(module.weight.dtype)
            trainable = False
        bias = module.bias

        def forward(input_tensor: torch.Tensor, *, _weight=weight_source, _bias=bias,
                    _trainable=trainable) -> torch.Tensor:
            quantized = (q4_0_dequant_ste(_weight) if _trainable else _weight).to(
                dtype=input_tensor.dtype)
            return F.linear(input_tensor, quantized, _bias)

        module.forward = forward  # type: ignore[method-assign]
        # The closure owns either the master or the Q4_0-dequantized cache.
        # Releasing the pre-QAT Parameter prevents holding both copies.
        module.register_parameter("weight", None)

    missing = [target for target in targets if target not in masters]
    if missing:
        raise TypeError(f"QAT targets must be Q4_0-aligned torch.nn.Linear modules: {', '.join(missing)}")
    return masters


def install_low_rank_residual_weights(
        generator: torch.nn.Module, targets: list[str], master_sources: dict[str, torch.Tensor],
        rank: int) -> dict[str, tuple[torch.nn.Parameter, torch.nn.Parameter]]:
    """Train zero-initialized ``B @ A`` corrections over exact-Q4 base weights.

    ``B`` starts at zero, making the first forward exactly the deployed Q4
    baseline. This is intentional: static SVD residuals already failed the
    conditional/unconditional deployment gate, so the trajectory loss must
    decide whether any correction is useful.
    """
    if rank < 1:
        raise ValueError("low-rank residual rank must be positive")
    for parameter in generator.parameters():
        parameter.requires_grad_(False)

    factors: dict[str, tuple[torch.nn.Parameter, torch.nn.Parameter]] = {}
    modules = dict(generator.named_modules())
    unknown = [target for target in targets if target not in modules]
    if unknown:
        raise ValueError(f"unknown QAT targets: {', '.join(unknown)}")
    for name, module in modules.items():
        if not isinstance(module, torch.nn.Linear) or module.weight.ndim != 2:
            continue
        if module.weight.shape[1] % 32:
            if name in targets:
                raise ValueError(f"QAT target is not Q4_0 aligned: {name} {tuple(module.weight.shape)}")
            continue
        if name in targets:
            source = master_sources.get(name)
            if source is None:
                raise ValueError(f"missing F32 QAT source for {name}")
            out_features, in_features = source.shape
            if rank > min(out_features, in_features):
                raise ValueError(f"low-rank residual {name} rank {rank} exceeds {tuple(source.shape)}")
            base = q4_0_dequant_ste(source.to(device=module.weight.device)).detach().to(module.weight.dtype)
            factor_a = torch.nn.Parameter(
                torch.randn(rank, in_features, device=module.weight.device, dtype=torch.float32) /
                float(in_features) ** 0.5)
            factor_b = torch.nn.Parameter(
                torch.zeros(out_features, rank, device=module.weight.device, dtype=torch.float32))
            factors[name] = (factor_a, factor_b)

            def forward(input_tensor: torch.Tensor, *, _base=base, _bias=module.bias,
                        _factor_a=factor_a, _factor_b=factor_b) -> torch.Tensor:
                projected = F.linear(input_tensor, _base.to(dtype=input_tensor.dtype), _bias)
                correction = F.linear(F.linear(input_tensor, _factor_a.to(dtype=input_tensor.dtype)),
                                      _factor_b.to(dtype=input_tensor.dtype))
                return projected + correction
        else:
            with torch.no_grad():
                cached = q4_0_dequant_ste(module.weight).detach().to(module.weight.dtype)

            def forward(input_tensor: torch.Tensor, *, _weight=cached, _bias=module.bias) -> torch.Tensor:
                return F.linear(input_tensor, _weight.to(dtype=input_tensor.dtype), _bias)

        module.forward = forward  # type: ignore[method-assign]
        module.register_parameter("weight", None)

    missing = [target for target in targets if target not in factors]
    if missing:
        raise TypeError(f"QAT targets must be Q4_0-aligned torch.nn.Linear modules: {', '.join(missing)}")
    return factors


def trajectory_loss(generator: torch.nn.Module, states: dict[int, dict[str, torch.Tensor]],
                    condition: torch.Tensor, start: int, rollout_steps: int,
                    times: torch.Tensor, cfg_strength: float, cfg_start: float,
                    cfg_end: float, loss_mode: str) -> torch.Tensor:
    model_dtype = compute_dtype(generator)
    x = {name: value.to(device=condition.device, dtype=model_dtype) for name, value in states[start].items()}
    backbone = generator.reverse_fn.backbone
    losses: list[torch.Tensor] = []
    for offset in range(rollout_steps):
        t0, t1 = times[start + offset], times[start + offset + 1]
        t = (t0 * 1000.0).reshape(1)
        cond_velocity = backbone(x, t, condition, d=torch.zeros_like(t), cfg=False)
        uncond_velocity = backbone(x, t, condition, d=torch.zeros_like(t), cfg=True)
        for modality in MODS:
            velocity = (cond_velocity[modality] + cfg_strength *
                        (cond_velocity[modality] - uncond_velocity[modality])
                        if cfg_start <= float(t) <= cfg_end
                        else cond_velocity[modality])
            x[modality] = x[modality] + (t1 - t0) * velocity
            # SAMT stores one sample without the batch dimension; backbone output retains it.
            reference = states[start + offset + 1][modality].unsqueeze(0).to(
                device=condition.device, dtype=model_dtype)
            if loss_mode == "all_steps" or offset + 1 == rollout_steps:
                losses.append(F.mse_loss(x[modality].float(), reference.float()))
        # Match the next-step contract used while producing the teacher states.
        # Public latents are FP32 after the output projection, whereas the
        # non-Q4-aligned input projections retain FP16 weights.
        x = {name: value.to(dtype=model_dtype) for name, value in x.items()}
    return torch.stack(losses).mean()


def parse_targets(values: list[str]) -> list[str]:
    return [name for item in values for name in item.split(",") if name]


def restore_masters(masters: dict[str, torch.nn.Parameter], checkpoint: Path) -> int:
    """Restore a prior sparse QAT checkpoint with an exact target contract."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"QAT checkpoint has no state_dict: {checkpoint}")
    state_dict = payload["state_dict"]
    expected = {"_base_models.generator." + name + ".weight": master
                for name, master in masters.items()}
    unexpected = sorted(set(state_dict) - set(expected))
    missing = sorted(set(expected) - set(state_dict))
    if unexpected or missing:
        raise ValueError(
            f"QAT checkpoint target mismatch: unexpected={unexpected}, missing={missing}")
    with torch.no_grad():
        for key, master in expected.items():
            value = state_dict[key]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(master.shape):
                raise ValueError(
                    f"QAT checkpoint shape mismatch for {key}: "
                    f"{getattr(value, 'shape', None)} vs {tuple(master.shape)}")
            master.copy_(value.to(device=master.device, dtype=master.dtype))
    return len(expected)


def seed_masters(masters: dict[str, torch.nn.Parameter], checkpoint: Path) -> int:
    """Seed an expanded target set without weakening strict restore semantics.

    ``--initial-overrides`` remains an exact continuation contract.  This path
    is deliberately separate: it admits only a shape-checked subset so an
    established trunk correction can seed newly added output heads.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"QAT checkpoint has no state_dict: {checkpoint}")
    expected = {"_base_models.generator." + name + ".weight": master
                for name, master in masters.items()}
    unexpected = sorted(set(payload["state_dict"]) - set(expected))
    if unexpected:
        raise ValueError(f"QAT seed has targets outside the requested contract: {unexpected}")
    if not payload["state_dict"]:
        raise ValueError("QAT seed checkpoint is empty")
    with torch.no_grad():
        for key, value in payload["state_dict"].items():
            master = expected[key]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(master.shape):
                raise ValueError(
                    f"QAT seed shape mismatch for {key}: "
                    f"{getattr(value, 'shape', None)} vs {tuple(master.shape)}")
            master.copy_(value.to(device=master.device, dtype=master.dtype))
    return len(payload["state_dict"])


def scalar_error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Return stable F32 errors for one Euler-window terminal latent."""
    delta = actual.float() - reference.float()
    return {
        "mae": float(delta.abs().mean()),
        "mse": float(delta.square().mean()),
        "max_abs": float(delta.abs().max()),
    }


def sample_terminal_state(generator: torch.nn.Module, initial: dict[str, torch.Tensor],
                          condition: torch.Tensor, start: int, rollout_steps: int,
                          times: torch.Tensor, cfg_strength: float, cfg_start: float,
                          cfg_end: float) -> dict[str, torch.Tensor]:
    """Run a no-grad Euler window and retain only its terminal public latents."""
    model_dtype = compute_dtype(generator)
    x = {name: value.to(device=condition.device, dtype=model_dtype) for name, value in initial.items()}
    backbone = generator.reverse_fn.backbone
    with torch.no_grad():
        for offset in range(rollout_steps):
            t0, t1 = times[start + offset], times[start + offset + 1]
            t = (t0 * 1000.0).reshape(1)
            cond_velocity = backbone(x, t, condition, d=torch.zeros_like(t), cfg=False)
            uncond_velocity = backbone(x, t, condition, d=torch.zeros_like(t), cfg=True)
            for modality in MODS:
                velocity = (cond_velocity[modality] + cfg_strength *
                            (cond_velocity[modality] - uncond_velocity[modality])
                            if cfg_start <= float(t) <= cfg_end
                            else cond_velocity[modality])
                x[modality] = x[modality] + (t1 - t0) * velocity
            x = {name: value.to(dtype=model_dtype) for name, value in x.items()}
    return {name: value.detach().float().cpu() for name, value in x.items()}


def build_ss_decoder(checkpoint_dir: Path, device: torch.device) -> torch.nn.Module:
    """Load the official F32 occupancy decoder for a generator-only proxy gate."""
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    import yaml
    import sam3d_objects  # noqa: F401 - registers Hydra targets.
    from hydra.utils import instantiate

    with (checkpoint_dir / "ss_decoder.yaml").open() as stream:
        decoder = instantiate(yaml.safe_load(stream))
    state_dict = torch.load(checkpoint_dir / "ss_decoder.ckpt", map_location="cpu", weights_only=True)
    decoder.load_state_dict(state_dict.get("state_dict", state_dict), strict=True)
    return decoder.to(device).eval()


def sparse_masks(shape_latent: torch.Tensor, decoder: torch.nn.Module,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw and normal-pipeline surface support masks for a terminal shape latent."""
    latent = shape_latent.to(device=device).permute(0, 2, 1).contiguous().view(1, 8, 16, 16, 16)
    with torch.no_grad():
        raw = (decoder(latent)[0, 0] > 0).detach().cpu()
    return raw, surface_support(raw, device)


def support_error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | int]:
    """Measure binary voxel-support agreement without an order-dependent coordinate table."""
    actual = actual.bool()
    reference = reference.bool()
    intersection = int((actual & reference).sum())
    union = int((actual | reference).sum())
    predicted = int(actual.sum())
    expected = int(reference.sum())
    precision = intersection / predicted if predicted else float(expected == 0)
    recall = intersection / expected if expected else float(predicted == 0)
    return {
        "predicted_voxels": predicted,
        "reference_voxels": expected,
        "intersection_voxels": intersection,
        "union_voxels": union,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "iou": intersection / union if union else 1.0,
    }


def surface_support(raw: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Apply the pipeline's surface-prune policy to one binary occupancy field."""
    from sam3d_objects.pipeline.inference_utils import prune_sparse_structure

    raw = raw.to(device=device, dtype=torch.bool)
    coords = torch.argwhere(raw).to(dtype=torch.int32)
    batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=device)
    surface = prune_sparse_structure(torch.cat((batch, coords), dim=1))
    surface_mask = torch.zeros_like(raw, dtype=torch.bool)
    if surface.numel():
        surface_mask[surface[:, 1], surface[:, 2], surface[:, 3]] = True
    return surface_mask.cpu()


def canonical_terminal_state(e2e_dir: Path) -> dict[str, torch.Tensor]:
    """Load the official BF16-pipeline final public latents from the E2E dump."""
    files = {
        "shape": "ss_shape_latent.samt",
        "6drotation_normalized": "ss_pose_latent_6drotation_normalized.samt",
        "scale": "ss_pose_latent_scale.samt",
        "translation": "ss_pose_latent_translation.samt",
        "translation_scale": "ss_pose_latent_translation_scale.samt",
    }
    result: dict[str, torch.Tensor] = {}
    for modality, filename in files.items():
        shape, values = load_samt(e2e_dir / filename)
        value = torch.from_numpy(values.reshape(shape[::-1]).copy())
        if value.ndim and value.shape[0] == 1:
            value = value[0]
        result[modality] = value
    return result


def canonical_occupancy_support(e2e_dir: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the independent official occupancy dump and apply normal surface pruning."""
    shape, values = load_samt(e2e_dir / "ss_occ_f32.samt")
    occupancy = torch.from_numpy(values.reshape(shape[::-1]).copy())
    while occupancy.ndim > 3 and occupancy.shape[0] == 1:
        occupancy = occupancy[0]
    if tuple(occupancy.shape) != (64, 64, 64):
        raise ValueError(f"canonical occupancy has unexpected shape {tuple(occupancy.shape)}")
    raw = occupancy > 0
    return raw, surface_support(raw, device)


def validate_long_windows(generator: torch.nn.Module, states: dict[int, dict[str, torch.Tensor]],
                          condition: torch.Tensor, starts: list[int], rollout_steps: int,
                          times: torch.Tensor, decoder_checkpoint: Path, e2e_dir: Path,
                          device: torch.device) -> list[dict[str, object]]:
    """Evaluate terminal latent and decoded support consistency on bounded-memory windows."""
    terminal_states: list[tuple[int, int, dict[str, torch.Tensor]]] = []
    for start in starts:
        final_state = sample_terminal_state(generator, states[start], condition, start, rollout_steps,
                                            times, 7.0, 0.0, 500.0)
        terminal_states.append((start, start + rollout_steps, final_state))
        print(f"validated SS window {start}->{start + rollout_steps}", flush=True)

    # The generator's Q4-dequantized cache is several GiB. Decode the saved
    # terminal latents with the official F32 decoder on CPU, so this validation
    # does not compete with the long-window sampler for limited GPU memory.
    decoder_device = torch.device("cpu")
    decoder = build_ss_decoder(decoder_checkpoint, decoder_device)
    windows: list[dict[str, object]] = []
    for start, end, actual in terminal_states:
        # The canonical dump uses the official BF16 stage boundary. The trainer
        # uses FP16 to fit QAT on 12 GiB, so only this full 0->25 terminal can
        # be an independent end-to-end anchor. Other windows remain useful
        # same-precision diagnostics, but must not be presented as that anchor.
        canonical_terminal = start == 0 and end == 25
        if canonical_terminal:
            reference = canonical_terminal_state(e2e_dir)
            ref_raw, ref_surface = canonical_occupancy_support(e2e_dir, decoder_device)
            reference_kind = "canonical_pytorch_bf16_e2e"
        else:
            reference = states[end]
            ref_raw, ref_surface = sparse_masks(reference["shape"].unsqueeze(0), decoder, decoder_device)
            reference_kind = "fp16_teacher_replay"
        raw, surface = sparse_masks(actual["shape"], decoder, decoder_device)
        windows.append({
            "start": start,
            "end": end,
            "reference_kind": reference_kind,
            "final_latent": {name: scalar_error(actual[name], reference[name].unsqueeze(0)) for name in MODS},
            "support": {
                "raw_occupancy": support_error(raw, ref_raw),
                "surface_pruned": support_error(surface, ref_surface),
            },
        })
    return windows


def selection_against_baseline(windows: list[dict[str, object]], baseline_path: Path,
                               targets: list[str]) -> dict[str, object]:
    """Require every long-window terminal metric to be at least baseline quality."""
    baseline = json.loads(baseline_path.read_text())
    if baseline.get("schema") != "sam3d.ss_q4_long_window_validation.v1":
        raise ValueError(f"unsupported baseline schema in {baseline_path}")
    if baseline.get("targets") != targets:
        raise ValueError("baseline target contract differs from candidate")
    baseline_windows = {(row["start"], row["end"]): row for row in baseline.get("windows", [])}
    failures: list[str] = []
    for window in windows:
        key = (window["start"], window["end"])
        reference = baseline_windows.get(key)
        if reference is None:
            failures.append(f"missing baseline window {key[0]}->{key[1]}")
            continue
        if reference.get("reference_kind") != window.get("reference_kind"):
            failures.append(f"{key[0]}->{key[1]} reference kind differs from baseline")
            continue
        for modality in MODS:
            candidate_mse = window["final_latent"][modality]["mse"]
            baseline_mse = reference["final_latent"][modality]["mse"]
            if candidate_mse > baseline_mse:
                failures.append(f"{key[0]}->{key[1]} {modality} mse regressed")
        for stage in ("raw_occupancy", "surface_pruned"):
            candidate_iou = window["support"][stage]["iou"]
            baseline_iou = reference["support"][stage]["iou"]
            if candidate_iou < baseline_iou:
                failures.append(f"{key[0]}->{key[1]} {stage} iou regressed")
    return {"baseline": str(baseline_path), "pareto_non_regressing": not failures, "failures": failures}


def absolute_final_gate(windows: list[dict[str, object]], max_shape_mse: float | None,
                        min_raw_iou: float | None, min_surface_iou: float | None) -> dict[str, object]:
    """Apply optional absolute gates only to the independently anchored terminal."""
    limits = {"max_shape_mse": max_shape_mse, "min_raw_iou": min_raw_iou,
              "min_surface_iou": min_surface_iou}
    failures: list[str] = []
    if any(value is not None for value in limits.values()):
        anchored = [row for row in windows if row["reference_kind"] == "canonical_pytorch_bf16_e2e"]
        if not anchored:
            failures.append("absolute gates require a 0->25 canonical terminal window")
        for window in anchored:
            if max_shape_mse is not None and window["final_latent"]["shape"]["mse"] > max_shape_mse:
                failures.append(f"{window['start']}->{window['end']} shape mse exceeds limit")
            if min_raw_iou is not None and window["support"]["raw_occupancy"]["iou"] < min_raw_iou:
                failures.append(f"{window['start']}->{window['end']} raw support iou below limit")
            if min_surface_iou is not None and window["support"]["surface_pruned"]["iou"] < min_surface_iou:
                failures.append(f"{window['start']}->{window['end']} surface support iou below limit")
    return {"limits": limits, "passed": not failures, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("cpp_ggml/models/pytorch/ss_generator.ckpt"))
    parser.add_argument("--config", type=Path, default=Path("cpp_ggml/models/pytorch/ss_generator.yaml"))
    parser.add_argument("--e2e-dir", type=Path, default=Path("cpp_ggml/benchmarks/data/e2e"))
    parser.add_argument("--teacher-dir", type=Path, required=False)
    parser.add_argument("--reference-trajectory-dir", type=Path,
                        help="independent full-precision ss_trajectory_ref.py dump used for QAT supervision")
    parser.add_argument("--sensitivity-report", type=Path,
                        help="write canonical first-step activation-weighted Q4 linear sensitivity JSON and exit")
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--validate-only", action="store_true",
                        help="run bounded-memory long-window terminal-latent/support validation; do not train")
    parser.add_argument("--validation-json", type=Path,
                        help="machine-readable result path required by --validate-only")
    parser.add_argument("--validation-starts", default="0,8,16",
                        help="comma-separated teacher-state starts for --validate-only")
    parser.add_argument("--validation-rollout-steps", type=int, default=8,
                        help="Euler steps per --validate-only window")
    parser.add_argument("--decoder-checkpoint-dir", type=Path,
                        default=Path("cpp_ggml/models/pytorch"),
                        help="directory containing official ss_decoder.yaml and ss_decoder.ckpt")
    parser.add_argument("--baseline-json", type=Path,
                        help="prior validate-only JSON; candidates must not regress against it")
    parser.add_argument("--max-final-shape-mse", type=float,
                        help="optional absolute MSE limit for canonical 0->25 shape latent")
    parser.add_argument("--min-final-raw-iou", type=float,
                        help="optional absolute raw-occupancy IoU floor for canonical 0->25")
    parser.add_argument("--min-final-surface-iou", type=float,
                        help="optional absolute surface-pruned IoU floor for canonical 0->25")
    parser.add_argument("--fail-on-selection-regression", action="store_true",
                        help="return nonzero after writing validation JSON when --baseline-json is regressed")
    parser.add_argument("--initial-overrides", type=Path, metavar="CHECKPOINT",
                        help="strictly restore sparse masters emitted by an earlier QAT run")
    parser.add_argument("--seed-overrides", type=Path, metavar="CHECKPOINT",
                        help="seed an expanded target set from a shape-checked sparse QAT subset")
    parser.add_argument("--target", action="append", default=[], metavar="MODULE")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--loss-scale", type=float, default=4096.0,
                        help="scale the FP16 trajectory loss before backward, then unscale masters")
    parser.add_argument("--starts", default="0,8,16,23")
    parser.add_argument("--batch-all-starts", action="store_true",
                        help="average gradients from every --starts window before each optimizer step")
    parser.add_argument("--loss-mode", choices=("all_steps", "final_state"), default="all_steps",
                        help="supervise every Euler state or only the final state of each rollout")
    parser.add_argument("--low-rank-rank", type=int,
                        help="train zero-initialized Q4 residual factors B@A instead of dense QAT masters")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.reference_trajectory_dir is not None and args.teacher_dir is not None:
        parser.error("--reference-trajectory-dir and --teacher-dir are mutually exclusive")
    if args.initial_overrides is not None and args.seed_overrides is not None:
        parser.error("--initial-overrides and --seed-overrides are mutually exclusive")
    if args.low_rank_rank is not None and args.low_rank_rank < 1:
        parser.error("--low-rank-rank must be positive")
    if args.low_rank_rank is not None and (args.initial_overrides is not None or args.seed_overrides is not None):
        parser.error("low-rank residual training does not accept dense master override checkpoints")
    diagnostic_only = args.sensitivity_report is not None
    if not diagnostic_only and (not args.target or (not args.validate_only and args.output is None)):
        parser.error("--target and --output are required for training; --validate-only requires --target")
    if args.validate_only and args.validation_json is None:
        parser.error("--validation-json is required with --validate-only")
    if args.validate_only and args.teacher_dir is None and args.reference_trajectory_dir is None:
        parser.error("--validate-only requires --teacher-dir or --reference-trajectory-dir")
    if args.iterations < 1 or args.rollout_steps < 1 or args.loss_scale <= 0:
        parser.error("--iterations, --rollout-steps, and --loss-scale must be positive")
    if args.validation_rollout_steps < 1:
        parser.error("--validation-rollout-steps must be positive")
    if (args.max_final_shape_mse is not None and args.max_final_shape_mse < 0) or \
       any(value is not None and not 0 <= value <= 1
           for value in (args.min_final_raw_iou, args.min_final_surface_iou)):
        parser.error("absolute MSE must be nonnegative and absolute IoU limits must lie in [0, 1]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    targets = parse_targets(args.target)
    starts = [int(value) for value in args.starts.split(",") if value]
    if any(start < 0 or start + args.rollout_steps > 25 for start in starts):
        parser.error("every --starts value must permit the requested rollout within 25 Euler steps")

    validation_starts = [int(value) for value in args.validation_starts.split(",") if value]
    if args.validate_only and any(start < 0 or start + args.validation_rollout_steps > 25
                                  for start in validation_starts):
        parser.error("every --validation-starts value must permit the requested window within 25 Euler steps")
    generator = load_generator(args.checkpoint, args.config, str(device), 25, 3.0,
                               7.0, 0.0, 500.0)
    if diagnostic_only:
        generator = prepare_fp16_student(generator)
        q4_sensitivity_report(generator, args.e2e_dir, device, args.sensitivity_report)
        return 0
    generator, master_sources = prepare_deployment_q4_student(generator, targets)
    required_teacher_steps = max(
        [start + args.rollout_steps for start in starts]
        + ([start + args.validation_rollout_steps for start in validation_starts] if args.validate_only else [])
    )
    if args.reference_trajectory_dir is not None:
        states = {step: state_from_reference_trajectory(args.reference_trajectory_dir, step,
                                                        device=torch.device("cpu"))
                  for step in range(required_teacher_steps + 1)}
        teacher_precision = "independent_full_precision_ss_trajectory_ref"
    else:
        teacher_dir = args.teacher_dir or args.output.with_suffix(".teacher")
        if not all((teacher_dir / f"ss_state{required_teacher_steps}_{modality}.samt").is_file()
                   for modality in MODS):
            save_teacher_trajectory(generator, args.e2e_dir, teacher_dir, device, 25,
                                    required_teacher_steps, 3.0, 7.0, 0.0, 500.0)
        states = {step: state_from_samt(teacher_dir, step, device=torch.device("cpu"))
                  for step in range(required_teacher_steps + 1)}
        teacher_precision = "mixed_fp32_embedding_fp16_transformer"
    _, cond_data = load_samt(args.e2e_dir / "ss_cond_tokens.samt")
    condition = torch.from_numpy(cond_data.reshape(1, -1, 1024).copy()).to(
        device=device, dtype=compute_dtype(generator))
    low_rank_factors: dict[str, tuple[torch.nn.Parameter, torch.nn.Parameter]] = {}
    if args.low_rank_rank is None:
        masters = install_qat_weights(generator, targets, master_sources)
        trainables = list(masters.values())
    else:
        masters = {}
        low_rank_factors = install_low_rank_residual_weights(
            generator, targets, master_sources, args.low_rank_rank)
        trainables = [factor for pair in low_rank_factors.values() for factor in pair]
    restored_masters = 0
    if args.initial_overrides is not None:
        restored_masters = restore_masters(masters, args.initial_overrides)
        print(f"restored {restored_masters} QAT master tensors from {args.initial_overrides}", flush=True)
    seeded_masters = 0
    if args.seed_overrides is not None:
        seeded_masters = seed_masters(masters, args.seed_overrides)
        print(f"seeded {seeded_masters} QAT master tensors from {args.seed_overrides}", flush=True)
    if args.validate_only:
        u = torch.linspace(0.0, 1.0, 26, device=device)
        times = u / (1.0 + 2.0 * (1.0 - u))
        windows = validate_long_windows(generator, states, condition, validation_starts,
                                        args.validation_rollout_steps, times,
                                        args.decoder_checkpoint_dir, args.e2e_dir, device)
        payload: dict[str, object] = {
            "schema": "sam3d.ss_q4_long_window_validation.v1",
            "targets": targets,
            "initial_overrides": str(args.initial_overrides) if args.initial_overrides else None,
            "student_compute_precision": str(compute_dtype(generator)).removeprefix("torch."),
            "window_steps": args.validation_rollout_steps,
            "windows": windows,
        }
        selection: dict[str, object] = {}
        if args.baseline_json is not None:
            selection["relative"] = selection_against_baseline(windows, args.baseline_json, targets)
        absolute = absolute_final_gate(windows, args.max_final_shape_mse,
                                      args.min_final_raw_iou, args.min_final_surface_iou)
        if any(value is not None for value in absolute["limits"].values()):
            selection["absolute"] = absolute
        if selection:
            relative_ok = selection.get("relative", {}).get("pareto_non_regressing", True)
            absolute_ok = selection.get("absolute", {}).get("passed", True)
            selection["accepted"] = relative_ok and absolute_ok
            payload["selection"] = selection
        args.validation_json.parent.mkdir(parents=True, exist_ok=True)
        args.validation_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote long-window validation to {args.validation_json}", flush=True)
        if args.fail_on_selection_regression and not payload.get("selection", {}).get("accepted", True):
            return 2
        return 0
    checkpointed_blocks = enable_activation_checkpointing(generator)
    optimizer = torch.optim.AdamW(trainables, lr=args.learning_rate, weight_decay=0.0)
    u = torch.linspace(0.0, 1.0, 26, device=device)
    times = u / (1.0 + 2.0 * (1.0 - u))
    history = []
    for iteration in range(args.iterations):
        iteration_starts = starts if args.batch_all_starts else [starts[iteration % len(starts)]]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for start in iteration_starts:
            loss = trajectory_loss(generator, states, condition, start, args.rollout_steps,
                                   times, 7.0, 0.0, 500.0, args.loss_mode)
            losses.append(loss.detach())
            # All transformer activations are FP16, while the trajectory loss
            # can be about 1e-5. Scale before FP16 backward so the selected
            # master gradients do not underflow to zero. Backward each window
            # immediately to keep checkpointed activation memory bounded.
            (loss * (args.loss_scale / len(iteration_starts))).backward()
        gradient_norm_sq = 0.0
        for parameter in trainables:
            if parameter.grad is not None:
                parameter.grad.div_(args.loss_scale)
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("QAT produced a non-finite master gradient")
                gradient_norm_sq += float(parameter.grad.float().square().sum().detach().cpu())
        if gradient_norm_sq == 0.0:
            raise RuntimeError(
                "all QAT master gradients are zero after loss scaling; "
                "select an active target or increase --loss-scale")
        torch.nn.utils.clip_grad_norm_(trainables, 1.0)
        optimizer.step()
        value = float(torch.stack(losses).mean().cpu())
        history.append({"iteration": iteration + 1, "starts": iteration_starts, "loss": value,
                        "gradient_norm": gradient_norm_sq ** 0.5})
        print(json.dumps(history[-1]), flush=True)

    if args.low_rank_rank is None:
        state_dict = {"_base_models.generator." + name + ".weight": master.detach().float().cpu()
                      for name, master in masters.items()}
        schema = "sam3d.ss_q4_0_trajectory_qat.v1"
    else:
        state_dict = {
            "_base_models.generator." + name + ".weight.lora_a": factor_a.detach().float().cpu()
            for name, (factor_a, _) in low_rank_factors.items()
        }
        state_dict.update({
            "_base_models.generator." + name + ".weight.lora_b": factor_b.detach().float().cpu()
            for name, (_, factor_b) in low_rank_factors.items()
        })
        schema = "sam3d.ss_q4_0_low_rank_trajectory_qat.v1"
    payload = {"state_dict": state_dict, "sam3d_qat_metadata": {
        "schema": schema, "targets": targets,
        "low_rank_rank": args.low_rank_rank,
        "student_compute_precision": str(compute_dtype(generator)).removeprefix("torch."),
        "teacher_precision": teacher_precision,
        "reference_trajectory_dir": (str(args.reference_trajectory_dir)
                                       if args.reference_trajectory_dir else None),
        "activation_checkpoint": True,
        "checkpointed_blocks": checkpointed_blocks,
        "initial_overrides": str(args.initial_overrides) if args.initial_overrides else None,
        "restored_masters": restored_masters,
        "seed_overrides": str(args.seed_overrides) if args.seed_overrides else None,
        "seeded_masters": seeded_masters,
        "iterations": args.iterations, "rollout_steps": args.rollout_steps,
        "starts": starts, "learning_rate": args.learning_rate,
        "loss_scale": args.loss_scale, "batch_all_starts": args.batch_all_starts,
        "loss_mode": args.loss_mode,
        "history": history,
    }}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"saved {len(state_dict)} QAT master tensors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
