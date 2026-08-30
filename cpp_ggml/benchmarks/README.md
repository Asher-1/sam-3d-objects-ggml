# End-to-end benchmark evidence

This directory retains only full image-and-mask -> GGML inference -> Gaussian
PLY -> official 60-view render evidence. Operator-only profiles, intermediate
latents, and superseded backend experiments are intentionally removed because
they cannot establish an end-to-end release result.

## Current snapshot

The current machine-readable report is
[`e2e_comparison/e2e_latency_current.json`](e2e_comparison/e2e_latency_current.json).
Its latency and quality plots are
[`e2e_comparison/e2e_latency_current.png`](e2e_comparison/e2e_latency_current.png)
and
[`e2e_comparison/e2e_metrics_current.png`](e2e_comparison/e2e_metrics_current.png).
The latest complete render comparison is
[`e2e_comparison/render_cuda_q8_current/side_by_side.png`](e2e_comparison/render_cuda_q8_current/side_by_side.png),
with per-frame metrics in
[`e2e_comparison/render_cuda_q8_current/render_metrics.json`](e2e_comparison/render_cuda_q8_current/render_metrics.json).

| Runner | Format | E2E latency | Render RGB MAE | Provenance | Gate status |
| --- | --- | ---: | ---: | --- | --- |
| Official PyTorch | F16 streamed | 75.994 s | 0 (self-reference) | exclusive GPU | reference |
| GGML CUDA | Q8_0 | 72.726 s | 0.012434 | exclusive GPU | fails MAE <= 0.01 and 70 s |
| GGML Vulkan | Q8_0 | not measured | not measured | shared GPU observed | pending rerun |
| GGML CUDA | Q4 retained best | 111.273 s (historical) | 0.040692 | historical, pre-current hash | fails; rerun required |

Only the first two rows have fresh exclusive-GPU evidence in this checkout.
The release verifier deliberately requires all six CUDA/Vulkan F16, Q8_0 and
Q4 rows, so the current report remains a failing release signal until the
missing rows are regenerated on an idle GPU and meet the quality, latency and
relative-speed contract.

## Reproduce

All model files belong in [`../models/gguf/`](../models/gguf/); no GGUF file is
stored under `benchmarks/`. From the repository root:

```bash
SAM3D_PYTHON=/absolute/path/to/sam3d-objects/bin/python \
  bash cpp_ggml/scripts/quickstart.sh cuda matrix
```

The matrix runs the complete image/mask-to-PLY path and the same official
renderer for every format/backend. It records GPU exclusivity, model hashes,
stage timings, render MAE and PSNR. Rebuild plots from a report with:

```bash
python3 cpp_ggml/scripts/plot_e2e_latency.py \
  --input cpp_ggml/benchmarks/e2e_comparison/e2e_latency_current.json \
  --output cpp_ggml/benchmarks/e2e_comparison/e2e_latency_current.png \
  --metrics-output cpp_ggml/benchmarks/e2e_comparison/e2e_metrics_current.png
python3 cpp_ggml/scripts/validate_e2e_benchmark.py \
  --input cpp_ggml/benchmarks/e2e_comparison/e2e_latency_current.json \
  --output cpp_ggml/benchmarks/e2e_comparison/e2e_gate_current.json
```

The verifier exits nonzero for missing, shared-GPU, over-70-second, or
render-MAE-over-0.01 rows. That failure is expected for the incomplete current
snapshot and is kept visible rather than replaced with partial module timings.

## Reproducibility

GGML source changes are delivered by the single patch
[`third_party/ggml-patches/0001-sam3d-ggml-combined.patch`](../third_party/ggml-patches/0001-sam3d-ggml-combined.patch)
and applied automatically during CMake configuration. Verify its replay before
benchmarking:

```bash
bash cpp_ggml/scripts/apply_ggml_patches.sh --check
git -C cpp_ggml/third_party/ggml diff --check
```
