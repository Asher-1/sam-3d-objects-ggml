# Backend Numerics: Attention Precision And Cross-Backend Sampling

This guide consolidates the two global numerical policies that apply to every
native run: which attention formulation each graph uses, and how the sampling
RNG stays reproducible across CUDA, Vulkan and CPU. The stage-by-stage
acceptance status derived from these policies lives in the
[parity contract](NATIVE_E2E_PARITY_CONTRACT.md).

## SS attention modes

The condition encoders always use their scoped F32 attention path. The SS
diffusion graph has an explicit, recorded policy:

| Mode | Command | Graph | Intended use |
| --- | --- | --- | --- |
| `normal` | `--ss-attention normal` | ggml flash attention with F16 K/V | delivery default (throughput) |
| `strict` | `--ss-attention strict` | SS-only F32 `QK^T -> softmax -> V` | numerical-parity diagnosis |

`strict` materializes attention scores and therefore has a larger transient
allocation and a lower throughput than `normal`; it changes neither DINO,
MoGe, SLat nor native post-processing. Since 2026-09-18 `normal` is the CLI
and runner default: the 27-scene acceptance (RGB MAE 9.6 / foreground IoU
0.816 vs strict 9.9/0.804, 27/27 converged) and the single-object A/B
(neural render MAE 0.02081 vs strict 0.02115; pose 3.5717 vs 3.5607 deg)
show no quality cost at ~2.9x SS speed, and the parity contract records
SS-flow attention-mode independence. The release matrix records the choice
in every JSON row. Attention formulation is selected per graph through
explicit options (`AttentionOptions` in the graph builder;
`--gs-portable-attention` for the GS decoder A/B) — there is no
process-wide attention environment switch.

For the direct CLI, append `--ss-attention strict` to `image-to-3d` for
parity diagnosis. The conditioned `e2e` command keeps the strict default
(diagnostic tool); pass `--ss-fast-attention` for throughput runs.
Parity-diagnostic entry points (operator oracle, controlled RNG row,
trajectory validators) pin strict explicitly.

### MoGe attention contract

The same policy class applies to the MoGe depth model, with the opposite
resolution: the official depth model runs F32 K/V, so the native MoGe graph
passes `AttentionOptions{strict_kv=true}` (F32 K/V plus `GGML_PREC_F32`).
Running MoGe flash attention with F16 K/V produced a raw pointmap MAE of
0.14 versus 5.4e-4 for the strict contract and amplified into scene pose
drift; the repair is recorded in the parity contract (2026-09-13) and the
single-object pipeline keeps the strict gate. The scene branch uses the
faster SS F16-KV preset because the post-repair A/B showed no measurable
free-run pose difference for the SS flow itself.

## Cross-backend sampling contract

CUDA uses the verified PyTorch Philox distribution path directly. CPU and
Vulkan use the same Philox4x32-10 counter/scatter sequence and require the
CUDA reference device's `distribution_blocks` value, because that launch
capacity affects the PyTorch draw ordering. `quickstart.sh` and the matrix
runner record it automatically with the CUDA `rng-dump` command; direct
CPU/Vulkan CLI use must provide it explicitly:

```bash
cpp_ggml/build-vulkan/bin/sam3d-cli image-to-3d ... --backend vulkan \
  --philox-blocks 168

python3 cpp_ggml/scripts/verify_portable_philox_rng.py \
  --cuda-binary cpp_ggml/build-cuda/bin/sam3d-cli \
  --portable-binary cpp_ggml/build-vulkan/bin/sam3d-cli --seed 42

python3 cpp_ggml/scripts/verify_pytorch_cuda_randperm.py \
  --cuda-binary cpp_ggml/build-cuda/bin/sam3d-cli \
  --portable-binary cpp_ggml/build-vulkan/bin/sam3d-cli --seed 42

python3 cpp_ggml/scripts/verify_pytorch_coordinate_downsample.py \
  --cuda-binary cpp_ggml/build-cuda/bin/sam3d-cli \
  --portable-binary cpp_ggml/build-vulkan/bin/sam3d-cli --seed 42
```

`168` is an example from the verified RTX 3060 run, not a portable constant.
The verifier compares the entire SS/SLat draw sequence. CUDA device math and
host libm are not bitwise-identical: the current CUDA-to-Vulkan result covers
243,029 F32 values with maximum absolute error `2.205e-6` and mean absolute
error `1.674e-7`, within the verifier's `3e-6` and `4e-7` limits.

The second verifier covers the other random operation in the native path. With
`--reference-dir`, it reads the five SS normal draw sizes from a fresh official
stage manifest, stops before SLat `x0`, then compares the exact
`torch.randperm` output used to select the first 42,000 sparse coordinates. It
runs both a 42,000-element case and a 100,000-element case (which exercises the
64-bit-key branch), and requires byte-for-byte agreement with CUDA PyTorch for
CUDA, Vulkan, and CPU.

The third verifier exercises the complete production
`downsample_sparse_coords_pytorch` path with negative and duplicated I32 sparse
coordinates. It loads the repository's canonical
`downsample_sparse_structure` CUDA function body as the oracle, then requires
byte-for-byte equality for the F32 rescale, ties-to-even rounding, truncation,
clamp, exact four-component coordinate identity, lexicographic
`torch.unique(..., dim=0)` result, and subsequent `randperm` selection. Thus
the release gate no longer has a host-library RNG, lossy coordinate key, or
container-order dependency in this input path.
