# SAM 3D Objects GGML Runtime

This directory contains the C++/GGML stages for SAM 3D Objects. The supported
native path is image/mask conditions -> GGML condition/SS/SLat/Gaussian stages
-> Gaussian PLY. The checked-in raw image workflow still calls the official
Python MoGe preprocessing and official gsplat renderer; it is therefore a
hybrid end-to-end workflow, not a claim that every official Python operation
has been rewritten in C++.

## What is supported

| Area | Current implementation | Boundary |
| --- | --- | --- |
| GGML inference | condition, SS, SLat and Gaussian decoder | CPU, CUDA and Vulkan builds |
| Quantization | F32, F16, Q8_0 and Q4 model files | Q4 raw E2E quality is not yet a release pass |
| Output | Gaussian PLY; native non-baked mesh GLB export | `mesh-export` validates official decoder topology and `COLOR_0` |
| Official mesh/PBR path | Python `to_glb` postprocessing | no native mesh decoder, UV unwrap, texture baker, simplifier or hole repair |

The upstream pipeline defaults to Gaussian and mesh decoding, mesh
simplification/hole repair, UV parametrization and texture baking. Its GLB
uses a glTF PBR material with a base-color texture and roughness 1.0. The C++
runtime can export an already-decoded mesh to a non-baked GLB with the official
vertex colors. Textured PBR GLB parity is not implemented in C++ yet: its mesh
decoder, simplification/hole repair, UV parametrization and texture baking are
still Python-only. Use the official Python pipeline when a textured mesh is
required.

The native CMake targets do not link cuDNN. CUDA builds use GGML CUDA,
CUDA-runtime and cuBLAS libraries only. The raw-image wrapper can still load
the official PyTorch preprocessing/render environment. The repository contains
a separately tested native MoGe neural graph, but it is not yet wired through
the complete official condition-preprocessing and mesh/PBR chain; therefore
the checked-in raw-image wrapper remains hybrid.

## Clone and bootstrap

```bash
git clone --recursive https://github.com/Asher-1/sam-3d-objects-ggml.git
cd sam-3d-objects-ggml
bash cpp_ggml/scripts/quickstart.sh bootstrap
```

`bootstrap` initializes submodules, replays the single GGML patch through
CMake's patch helper, and prepares the `sam3d-objects` reference environment.
For an existing environment, set its absolute interpreter explicitly:

```bash
export SAM3D_PYTHON=/absolute/path/to/sam3d-objects/bin/python
```

The setup helper discovers `conda` from `PATH`; alternatively set
`SAM3D_CONDA_EXE`. It does not rely on a developer-specific absolute conda
installation.

## Models

All GGUF files belong in [`models/gguf/`](models/gguf/). Download the published
files with:

```bash
bash cpp_ggml/scripts/download_gguf.sh
```

The conversion/model card is in [`models/MODEL_CARD.md`](models/MODEL_CARD.md).
Do not place model files in `benchmarks/`; benchmark directories contain only
reports and rendered evidence.

## Build and test

The one-command interface caps build parallelism at six jobs for reproducible
resource use:

```bash
bash cpp_ggml/scripts/quickstart.sh cpu test
bash cpp_ggml/scripts/quickstart.sh cuda test
bash cpp_ggml/scripts/quickstart.sh vulkan test
```

The CUDA test suite also runs `sam3d_native_cuda_no_cudnn`, which inspects the
linked native executable and fails if cuDNN appears in its dependency tree.

Equivalent manual commands are:

```bash
cmake -S cpp_ggml -B cpp_ggml/build-cuda -DSAM3D_GGML_CUDA=ON
cmake --build cpp_ggml/build-cuda --parallel 6
ctest --test-dir cpp_ggml/build-cuda --output-on-failure
```

Every configure automatically runs
[`third_party/ggml-patches/0001-sam3d-ggml-combined.patch`](third_party/ggml-patches/0001-sam3d-ggml-combined.patch).
The helper is idempotent and verifies the final tree. To check a clean
submodule replay manually:

```bash
bash cpp_ggml/scripts/apply_ggml_patches.sh --check
git -C cpp_ggml/third_party/ggml diff --check
```

## End-to-end inference

For one canonical image/mask, the wrapper writes a Gaussian PLY and optional
official gsplat comparison assets:

```bash
SAM3D_PYTHON=/absolute/path/to/sam3d-objects/bin/python \
  bash cpp_ggml/scripts/quickstart.sh cuda e2e
```

Set `SAM3D_DTYPE=f16`, `SAM3D_DTYPE=q8_0` or `SAM3D_DTYPE=q4_0` to select a
model family. Set `SAM3D_WORK_DIR` to keep outputs at a known location. CUDA
and Vulkan latency runs require an idle GPU; the wrapper records this
provenance and refuses to label shared-GPU timings as release evidence.

The complete six-row release matrix and plots are generated with:

```bash
SAM3D_PYTHON=/absolute/path/to/sam3d-objects/bin/python \
  bash cpp_ggml/scripts/quickstart.sh cuda matrix
```

The matrix compares total image-to-PLY latency and the same 60-view official
render path. It is intentionally a failing release signal until every required
CUDA/Vulkan F16, Q8 and Q4 row has fresh exclusive-GPU timing, render MAE <=
0.01, latency <= 70 s, and the required relative-speed checks.

Current checked-in evidence and regeneration commands are documented in
[`benchmarks/README.md`](benchmarks/README.md). Run the standalone verifier with:

```bash
python3 cpp_ggml/scripts/validate_e2e_benchmark.py \
  --input cpp_ggml/benchmarks/e2e_comparison/e2e_latency_current.json \
  --output cpp_ggml/benchmarks/e2e_comparison/e2e_gate_current.json
```

The verifier reports failures instead of hiding missing or non-exclusive rows.

## Native mesh asset export

`mesh-export` is the C++ boundary after mesh decoding. It consumes the official
decoder's F32 SAMT tensors (`vertices`, `faces`, and optional six-channel
vertex attributes), applies the same Z-up-to-Y-up transform as Python
`to_glb`, and writes a non-baked GLB with `POSITION` and `COLOR_0`:

```bash
cpp_ggml/build-cpu/bin/sam3d-cli mesh-export \
  --vertices cpp_ggml/benchmarks/data/e2e/decode_mesh_vertices.samt \
  --faces cpp_ggml/benchmarks/data/e2e/decode_mesh_faces.samt \
  --attrs cpp_ggml/benchmarks/data/e2e/decode_mesh_vertex_attrs.samt \
  --out /tmp/sam3d_mesh.glb
```

This command is regression-tested against the real canonical decoder output.
It does not run the mesh decoder itself and it intentionally does not label
the result as PBR-texture parity: the native FlexiCubes, simplification/hole
repair, xatlas UV unwrap, Gaussian multiview renderer, and optimized texture
baking stages remain required for a fully native official-equivalent GLB.
