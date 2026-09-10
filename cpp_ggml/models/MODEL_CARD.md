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

The public generative weight listing does not include MoGe. Its helper uses
`huggingface_hub` to obtain the official checkpoint and converts it offline.
It preserves existing GGUF files unless explicitly given `--force`.

For conversion from local official checkpoints (conversion writes the chosen
output filenames; do not overwrite a benchmark baseline unintentionally):

```bash
SAM3D_PYTHON=/path/to/sam3d-objects/bin/python
"$SAM3D_PYTHON" cpp_ggml/scripts/convert_sam3d_to_gguf.py \
  --checkpoint-dir checkpoints/hf --model all --dtype q4_0 \
  --output cpp_ggml/models/gguf
```

Repeat with `--dtype f16` or `--dtype q8_0` for the other full-matrix families.
`--model all` converts the six generative stages, not MoGe; prepare MoGe with
the separate helper above. The checkpoint directory must contain the
converter's expected stage files;
see [the runtime guide](../README.md). Do not substitute LingBot-Map GGUF
weights: that is a different architecture.

## Current local inventory

Sizes below are actual local file sizes on 2026-09-10, in MiB (2^20 bytes).
They are disk sizes, not inference VRAM requirements or a promise that a model
downloaded under the same filename has identical contents.

| Stage | F16 MiB | F32 MiB | Q4_0 MiB | Q8_0 MiB | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| MoGe ViT-L | 600.0 | 1198.5 | not distributed | not distributed | image/mask point-map conditioning |
| SS generator | 3052.4 | 6098.5 | 864.8 | 1625.7 | condition encoder, structure diffusion and pose |
| SS decoder | 140.6 | 281.0 | 39.7 | 74.8 | sparse support |
| SLat generator | 2341.9 | 4678.8 | 663.9 | 1247.5 | structured latent diffusion |
| Gaussian decoder | 163.7 | 325.7 | 47.2 | 87.7 | Gaussian attributes |
| Gaussian stride-4 decoder | 162.4 | 324.5 | 45.9 | 86.4 | optional alternative, not the current raw matrix |
| Mesh decoder | 173.5 | 346.9 | 49.1 | 92.4 | FlexiCubes mesh features |

The active five-stage deployment plus MoGe F16 totals **6.320 GiB (F16)**,
**3.641 GiB (Q8_0)** and **2.212 GiB (Q4_0)**. These totals exclude the unused
stride-4 decoder. They do not estimate activations, attention workspaces or
the texture baker's allocations.

## Precision policy

F16, Q8_0 and Q4_0 select the five generative files of that suffix; the raw
matrix always uses the same explicitly recorded MoGe F16 file. Norm/bias and
other tensors unsupported by a quantization layout retain the converter's
floating-point types. Execution uses mixed intermediate dtypes, not uniformly
F32 activations. The actual graph is recorded by `--dtype-contract-out`.

Q4_1 and Q4_K files also exist locally, but they are not rows in the default
full-GLB matrix. In particular, the retained sensitive-layer SS Q4_K candidate
is **not** the current `ss_generator-q4_0.gguf`; its selection policy and
historical provenance are in [q4_best](../benchmarks/q4_best/README.md).
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
