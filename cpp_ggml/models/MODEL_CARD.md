# SAM 3D Objects — GGUF Model Card

Pre-converted GGUF weights for the **cpp_ggml** C++ inference engine, covering
every model of the SAM 3D Objects generation pipeline in F32, F16, Q8_0, and
the Q4_0/Q4_1/Q4_K variants.

The native runtime consumes the deterministic condition directory produced by
the official preprocessing path, then generates Gaussian PLY. It does not yet
run raw-image MoGe preprocessing or export a textured PBR GLB in C++.

- **HF repo**: <https://huggingface.co/Asher-1/SAM_3D_OBJECTS_GGUF>
- **Upstream source**: [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects) (official PyTorch checkpoints)
- **Converter**: `cpp_ggml/scripts/convert_sam3d_to_gguf.py` (deterministic; see
  the "Convert models" section of `cpp_ggml/README.md` to regenerate)
- **License**: conversions inherit the terms of the upstream SAM 3D Objects
  release. Use of the weights is subject to Meta's SAM 3D Objects license —
  obtain the official checkpoints and review the license before downloading.

## Download

All files live at the repository root:

```bash
BASE=https://huggingface.co/Asher-1/SAM_3D_OBJECTS_GGUF/resolve/main
# single file, e.g. the recommended f16 set:
curl -L -o ss_generator-f16.gguf  $BASE/ss_generator-f16.gguf
curl -L -o ss_decoder-f16.gguf    $BASE/ss_decoder-f16.gguf
```

or with `huggingface-hub` (resumable, parallel):

```bash
pip install -U "huggingface-hub[cli]"
hf download Asher-1/SAM_3D_OBJECTS_GGUF \
    --include "ss_generator-f16.gguf" "ss_decoder-f16.gguf" \
    --local-dir cpp_ggml/models/gguf
```

A helper script is available: `cpp_ggml/scripts/download_gguf.sh` (supports
`--model`, `--dtype` and `--out` filters).

The public mirror contains Q4 variants. The checked-in Q4 milestone may carry
a different local conversion or QAT revision under the same filename; its
provenance is retained in
[`benchmarks/q4_best/runs/cuda/run_summary.json`](../benchmarks/q4_best/runs/cuda/run_summary.json).
The download helper refuses to replace such a file by default. Regenerate a
local Q4 model from official checkpoints with:

```bash
SAM3D_PYTHON=/absolute/path/to/sam3d-objects/bin/python
"$SAM3D_PYTHON" \
  cpp_ggml/scripts/convert_sam3d_to_gguf.py \
  --checkpoint-dir cpp_ggml/models/pytorch --model all --dtype q4_0 \
  --output cpp_ggml/models/gguf
```

## Files

Sizes in GiB / MiB. Every file was verified loadable via
`cpp_ggml/build-cpu/bin/sam3d-cli info --model <file>`.

| Model (pipeline stage) | GGUF arch | f16 | f32 | q4_0 | q8_0 | Input → Output |
|---|---|---|---|---|---|---|
| ss_generator (condition embedder + MOT DiT) | `sam3d.ss` | 2.98 GiB | 5.96 GiB | 0.85 GiB | 1.59 GiB | condition tensors + timestep → sparse-structure latent (4096×8) |
| ss_decoder (3D conv U-Net) | `sam3d.ssdec` | 282 MiB | 423 MiB | 181 MiB | 216 MiB | latent 8×16³ → occupancy 1×64³ |
| slat_generator (condition embedder + sparse DiT) | `sam3d.slat` | 2.29 GiB | 4.57 GiB | 664 MiB | 1.22 GiB | condition tensors + sparse coordinates + timestep → structured latent |
| slat_decoder_gs (Gaussian decoder) | `sam3d.gsdec` | 163 MiB | 326 MiB | 46.1 MiB | 86.8 MiB | structured latent → 3D Gaussians |
| slat_decoder_gs_4 (Gaussian decoder, stride-4 variant) | `sam3d.gsdec` | 162 MiB | 324 MiB | 45.9 MiB | 86.4 MiB | structured latent → 3D Gaussians |
| slat_decoder_mesh (Mesh decoder) | `sam3d.meshdec` | 174 MiB | 347 MiB | 49.1 MiB | 92.4 MiB | structured latent → mesh |

Per-stage metadata lives under the `cemb.*`, `dit.*`, `dec.*`, `gsdec.*` KV
namespaces (see "GGUF tensor naming" in `cpp_ggml/README.md`).

Per-dtype totals: **f16 ≈ 5.94 GiB**, **f32 ≈ 11.9 GiB**, **q4_0 ≈ 1.94 GiB**, **q8_0 ≈ 3.25 GiB**.

## Quantizations

| dtype | Weights | Activations | Notes |
|---|---|---|---|
| f32 | FP32 | FP32 | reference for parity work; largest |
| f16 | FP16 | FP32 | recommended; gather tables / 1-D params stay FP32 |
| q8_0 | 8-bit block quant | FP32 | exercised on CPU+CUDA by the ss_decoder graph (quantized weights are dequantized once at graph build; see `benchmarks/README.md`) |
| q4_0 | 4-bit block quant (32 values/block) | FP32 | native CUDA/Vulkan matrix GEMM for generator projections; generated with ggml's Q4_0 reference rule |

1-D parameters (norms, biases), gather index tables and pixel-shuffle tables
are always stored in F32/I32 regardless of the selected dtype, so binary ops
never mix precisions on the CPU backend.

## Verification status

| Stage | Parity vs official PyTorch |
|---|---|
| ss_decoder | ✅ node-by-node (f32 max\|d\| = 0.035 @ ±115, f16 0.022, 100% sign agreement); q8_0 CPU+CUDA+Vulkan runs verified (logit sign agreement 99.92% vs f32 on the benchmark input) |
| dino condition embedder | ✅ full 24-block graph vs torch (f32 max\|d\| = 0.0046, f16 0.010; CPU + CUDA identical) |
| ss_generator / slat_generator / gs decoder | ✅ graph replay from conditioned stage dumps; run `sam3d-cli e2e` and `scripts/run_regression.py` |

## Not included

The VAE **encoders** (`ss_encoder`, `slat_encoder`) are intentionally absent:
they are used only for training / latent-space research, not by the
generation pipeline (see the official `pipeline.yaml` — it references no
encoder).

## Intended use & limitations

- Intended: running the conditioned SAM 3D Objects generation graph through
  the GGML C++ engine on CPU / CUDA / Vulkan.
- Raw image/mask input and official-camera comparison are available through the
  hybrid wrapper `scripts/run_image_to_3d.py`, which invokes official Python
  preprocessing and rendering around the native GGML generation stages.
- Textured PBR GLB follows the official Python postprocessing path; C++ writes
  Gaussian PLY only.
- Out of scope: training, fine-tuning, distillation.
- Latency evidence for the ss_decoder stage: `cpp_ggml/benchmarks/`
  (CPU 8T/32T, CUDA and Vulkan, f32/f16/q4_0/q8_0; regenerate on each machine).
