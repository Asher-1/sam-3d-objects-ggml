# SAM 3D Objects GGML Runtime

This directory contains the C++/GGML stages for SAM 3D Objects. It has native
FlexiCubes extraction, VTK decimation, xatlas seams, CUDA Gaussian rendering,
and Telea texture repair, but it does **not** yet have a complete native
image-to-textured-PBR-GLB pipeline. The checked-in raw image workflow remains
hybrid where it invokes official Python preprocessing or final texture baking.
It must not be presented as fully native end-to-end reconstruction.

## What is supported

| Area | Current implementation | Boundary |
| --- | --- | --- |
| GGML inference | condition, SS, SLat and Gaussian decoder | CPU, CUDA and Vulkan builds |
| Quantization | F32, F16, Q8_0 and Q4 model files | Q4 raw E2E quality is not yet a release pass |
| Raw mesh neural decode | 101-channel sparse cube features at 256^3 | Native graph exists, but the full official mesh-decoder fixture still diverges after its first transformer block |
| FlexiCubes | Native cube extraction | Synthetic Python/C++ mesh fixtures match; full raw mesh parity depends on the unresolved neural decoder divergence |
| Mesh cleanup | Exact VTK quadric decimation; official visibility/mincut replay; opt-in exact MeshFix repair | The mincut implementation consumes frozen official face frequencies while the native CUDA face-visibility rasterizer is incomplete. MeshFix is exact only with the explicit GPL/commercial build option |
| UV | Native xatlas parameterization | Canonical postprocessed mesh vertices, faces, and UVs match the Python xatlas result exactly |
| Gaussian observations | Native CUDA Inria/Mip renderer | Recorded-camera rendering is native; two official RGBA8 camera fixtures match exactly. The 100-view command accepts the frozen camera tensors |
| Texture repair | Native OpenCV Telea | Cross-version C++/official-cv2 comparison is exact on the fixed regression input |
| Output | Gaussian PLY; non-baked GLB export; PBR-capable GLB writer | Native UV rasterization, 2500-step Adam/TV baking, and final orchestration are still required before a final textured PBR GLB can be claimed |

The upstream pipeline decodes Gaussian and mesh assets, decimates and removes
invisible mesh regions, creates xatlas UVs, renders 100 Gaussian observations,
optimizes a 1024px texture for 2500 Adam/TV steps, repairs uncovered texels,
and writes a glTF PBR material. The native implementation now covers several
of those individual operations, but the missing operations are connected ones:
CUDA triangle visibility, differentiable UV rasterization/sampling, the
optimizer, and the final orchestrator. Use the
official Python path whenever a finished textured PBR asset is required.

The exact native-completion definition, official-source anchors, frozen PBR
reference bundle, and stage-by-stage acceptance procedure are in
[`docs/NATIVE_E2E_PARITY_CONTRACT.md`](docs/NATIVE_E2E_PARITY_CONTRACT.md).
That document is the source of truth for finishing the native textured-GLB
path; the current non-baked `mesh-export` output is not a substitute.

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

## Native mesh decode and asset export

`mesh-decode` consumes a real SLat latent and sparse 64^3 support, then runs
the native mesh transformer plus two official-order `SparseSubdivide` blocks.
It writes raw 101-channel cube features and, optionally, their 256^3 sparse
coordinates:

```bash
cpp_ggml/build-cuda/bin/sam3d-cli mesh-decode \
  --model cpp_ggml/models/gguf/slat_decoder_mesh-f16.gguf \
  --input cpp_ggml/benchmarks/data/e2e/slat_feats_final.samt \
  --coords cpp_ggml/benchmarks/data/e2e/slat_coords.samt \
  --coords-out /tmp/mesh_coords.samt --out /tmp/mesh_raw.samt --backend cuda
```

Create a source-of-truth reference by adding `--dump-mesh-decoder-reference`
to the official dumper. The final raw feature fixture is large, so it is
deliberately opt-in:

```bash
"$SAM3D_PYTHON" cpp_ggml/scripts/dump_e2e_stages.py \
  --out-dir /tmp/sam3d_reference --dump-mesh-decoder-reference
python3 cpp_ggml/scripts/verify_mesh_decoder_reference.py \
  --reference-dir /tmp/sam3d_reference \
  --native-features /tmp/mesh_raw.samt --native-coords /tmp/mesh_coords.samt
```

The verifier first requires exact sparse coordinates, then reports measured
MAE, RMSE, maximum absolute error and its token/channel location. It assigns
no numeric tolerance itself: a release gate must record one from repeated
official runs as described in the parity contract. `--stage input_layer`,
`--stage block11`, `--stage upsample0`, and `--stage upsample1` expose the
same graph boundaries for fault isolation.

For a transformer block whose QKV coordinates already agree, isolate the
attention kernel from the linear projections with the same-QKV semantic gate:

```bash
"$SAM3D_PYTHON" cpp_ggml/scripts/verify_mesh_attention_semantics.py \
  --reference-dir /tmp/sam3d_reference \
  --native-qkv /tmp/block0_qkv.samt \
  --native-qkv-coords /tmp/block0_qkv-coords.samt \
  --native-attention /tmp/block0_attention.samt \
  --native-attention-coords /tmp/block0_attention-coords.samt \
  --block 0 --device cuda
```

It requires byte-identical sparse coordinates, replays the released PyTorch
windowed-SDPA function on the native QKV tensor, and reports projection,
same-QKV attention, and fixture errors separately. This is a diagnostic gate,
not a substitute for the complete mesh or final-asset gate.

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
It intentionally does not label the result as PBR-texture parity: raw neural
mesh parity, native visibility rasterization, boundary repair, UV texture
optimization, and final assembly are still required.

## Native PBR Stage Regression

Create a complete official reference bundle before testing native PBR stages.
First dump the neural stages. The dumper records the source Gaussian, raw mesh,
and inputs needed by the standalone official PBR reference generator:

```bash
"$SAM3D_PYTHON" cpp_ggml/scripts/dump_e2e_stages.py \
  --image INPUT.png --mask-dir MASK_DIR --mask-index 0 --seed 42 \
  --out-dir /tmp/sam3d_reference

"$SAM3D_PYTHON" cpp_ggml/scripts/generate_official_pbr_reference.py \
  --stage-dir /tmp/sam3d_reference \
  --reference-dir /tmp/sam3d_reference/official_pbr_reference \
  --seed 42 --save-observations --save-bake-raster

python3 cpp_ggml/scripts/verify_official_pbr_reference.py \
  --reference-dir /tmp/sam3d_reference/official_pbr_reference \
  --stage-dir /tmp/sam3d_reference
```

`--save-bake-raster` writes the exact 100 per-view `uv`, `uv_dr`, and coverage
maps that the official 2500-step differentiable baker consumes. It is a large,
external debugging fixture and is deliberately excluded from `benchmarks/` and
`models/`. The verifier checks every recorded array's shape and SHA-256 before
a native baker may use it for a focused regression.

The following runs the native visibility/mincut decision against the frozen
official post-VTK mesh and compares the pre-MeshFix arrays exactly. It is a
stage regression only: a successful command does not include MeshFix boundary
repair or texture baking.

```bash
python3 cpp_ggml/scripts/verify_mesh_visibility_reference.py \
  --binary cpp_ggml/build-cuda/bin/sam3d-cli \
  --stage-dir /tmp/sam3d_reference --out-dir /tmp/native_visibility_check
```

The final official cleanup uses `pymeshfix.PyTMesh`, whose TMesh source is
GPL-3.0-or-commercial. It is deliberately absent from the default runtime.
After accepting a compatible license, enable the exact implementation and
compare it with the captured official output:

```bash
cmake -S cpp_ggml -B cpp_ggml/build-cuda-meshfix \
  -DSAM3D_GGML_CUDA=ON -DSAM3D_GGML_NATIVE_PBR=ON \
  -DSAM3D_GGML_MESHFIX_GPL=ON
cmake --build cpp_ggml/build-cuda-meshfix --parallel 6
python3 cpp_ggml/scripts/verify_meshfix_reference.py \
  --binary cpp_ggml/build-cuda-meshfix/bin/sam3d-cli \
  --stage-dir /tmp/sam3d_reference --out-dir /tmp/native_meshfix_check
```

This gate requires exact vertices and faces at the pre-xatlas boundary. It
does not cover native visibility rasterization, texture baking, or final GLB
assembly.

When only this boundary is being changed, avoid conflating it with the long
texture bake by generating the same official cleanup reference alone:

```bash
"$SAM3D_PYTHON" cpp_ggml/scripts/generate_official_pbr_reference.py \
  --stage-dir /tmp/sam3d_reference --reference-dir /tmp/mesh_cleanup_reference \
  --mesh-cleanup-only
```

Use the recorded rather than regenerated Gaussian cameras when exercising 100
native observations:

```bash
cpp_ggml/build-cuda/bin/sam3d-cli gaussian-render \
  --ply /tmp/sam3d_reference/output_gs.ply --out-dir /tmp/native_observations \
  --views 100 --resolution 1024 \
  --extrinsics /tmp/sam3d_reference/bake_extrinsics.samt \
  --intrinsics /tmp/sam3d_reference/bake_intrinsics.samt
```
