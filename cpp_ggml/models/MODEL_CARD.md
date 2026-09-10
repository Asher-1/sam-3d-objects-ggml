# SAM 3D Objects GGUF Models

All GGUF weights belong in [gguf/](gguf/), never in benchmarks. Native raw
reconstruction consumes the MoGe model plus five generative stages and exports
a Gaussian PLY, a textured GLB and its base-color PNG. Vulkan inference hands
its raw mesh and Gaussian outputs to a separate CUDA PBR process.

## Download and conversion

The matching model repository is
[Asher-1/SAM_3D_OBJECTS_GGUF](https://huggingface.co/Asher-1/SAM_3D_OBJECTS_GGUF).
Weights inherit the upstream SAM 3D Objects license. Use the helper from the
checkout root; it refuses to overwrite an existing local model by default:

```bash
bash cpp_ggml/scripts/download_gguf.sh --help
bash cpp_ggml/scripts/download_gguf.sh --dtype q8_0
SAM3D_PYTHON=/path/to/sam3d-objects/bin/python \
  bash cpp_ggml/scripts/prepare_moge_gguf.sh
```

The published MoGe ViT-L file is the F16 GGUF (`moge_vitl-f16.gguf`), also in
the public model repository and part of the downloader's default f16 set. The
helper still reconverts from the official `Ruicheng/moge-vitl` checkpoint
(`huggingface_hub`) when another dtype or a fresh validation against upstream
is needed. It preserves existing GGUF files unless explicitly given `--force`.

For conversion from local official checkpoints (conversion writes the chosen
output filenames; do not overwrite a benchmark baseline unintentionally):

```bash
SAM3D_PYTHON=/path/to/sam3d-objects/bin/python
"$SAM3D_PYTHON" cpp_ggml/scripts/convert_sam3d_to_gguf.py \
  --checkpoint-dir checkpoints/hf --model all --dtype q8_0 \
  --keep-f16-regex '^cemb\.' \
  --output cpp_ggml/models/gguf
```

Repeat with `--dtype f16` or `--dtype q8_0` for the other full-matrix families.
`--model all` converts the six generative stages, not MoGe; download the
published F16 file with the downloader or prepare it with the separate helper
above. The checkpoint directory must contain the
converter's expected stage files;
see [the runtime guide](../README.md). Do not substitute LingBot-Map GGUF
weights: that is a different architecture.

## Current local inventory

Sizes below are actual local file sizes after the 2026-09-14 asset ruling, in
MiB (2^20 bytes). They are disk sizes, not inference VRAM requirements or a
promise that a model downloaded under the same filename has identical
contents.

| Stage | F16 MiB | F32 MiB | Q4_K MiB | Q8_0 MiB | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| MoGe ViT-L | 600.0 | removed | not distributed | not distributed | image/mask point-map conditioning |
| SS generator | 3052.4 | removed | 864.8 | 2191.6 | condition encoder, structure diffusion and pose |
| SS decoder | 140.6 | removed | 39.7 | 74.8 | sparse support |
| SLat generator | 2341.9 | removed | 663.9 | 1247.5 | structured latent diffusion |
| Gaussian decoder | 163.7 | removed | 47.2 | 87.7 | Gaussian attributes |
| Gaussian stride-4 decoder | 162.4 | removed | 45.9 | 86.4 | optional alternative, not the current raw matrix |
| Mesh decoder | 173.5 | removed | 49.1 | 92.4 | FlexiCubes mesh features |

The 2026-09-14 ruling deleted every Q4_0/Q4_1 GGUF (12 files, 3.6 GB) and
`moge_vitl-f32.gguf` after measured A/B arbitration (see "Weight-Asset
Rulings" in [the parity contract](../docs/NATIVE_E2E_PARITY_CONTRACT.md)):
q4_k is the accuracy/speed winner inside the Q4 family and matches the
historical q4_best selection; the F16-weight MoGe matches the official F32
checkpoint to MAE 2.6e-4 - better than the F32-weight GGUF - at half the
size. `ss_generator-q8_0.gguf` was re-exported with
`--keep-f16-regex '^cemb\.'` so the DINO/PointPatch/fuser weights stay F16
regardless of the deployment dtype (hence 2191.6 MiB, not 1625.7).
`convert_sam3d_to_gguf.py` keeps `q4_0` in its choices only for the QAT
retraining pipeline; `q4_1` was removed from it.

## Precision policy

F16, Q8_0 and Q4_K select the five generative files of that suffix; the raw
matrix always uses the same explicitly recorded MoGe F16 file. Norm/bias and
other tensors unsupported by a quantization layout retain the converter's
floating-point types. Execution uses mixed intermediate dtypes, not uniformly
F32 activations. The actual graph is recorded by `--dtype-contract-out`.
The historical sensitive-layer SS Q4_K candidate provenance remains in
[q4_best](../benchmarks/q4_best/README.md).
No GGUF is moved, replaced or retrained when publishing benchmarks.

## End-to-end verification

Current per-model latency, final GLB renders, standalone textures and explicit
gate results are linked from [benchmarks](../benchmarks/README.md). Only these
complete raw-input results represent release evidence. A conditioned-stage
replay, a load test or an improved Q4 trajectory does not establish complete
reconstruction parity.

The root `run_ggml.sh` launcher uses strict SS attention. Full output matches
the official material scope: baked base color with a PBR material, not
independently inferred metallic/roughness/normal maps. Native CUDA or
Vulkan-to-CUDA reconstruction does not require Torch or cuDNN at runtime.
VTK >= 9.3, OpenCV and the explicit native rasterizer/MeshFix license options
are required for complete post-processing.

VAE encoders are training components and are not required by generation.
