# SAM 3D Objects GGML Runtime

This directory contains the C++/GGML stages for SAM 3D Objects. The native
CUDA PBR build includes FlexiCubes, official mesh cleanup, xatlas seams,
100-view Gaussian observations, 2500-step Adam/TV texture baking, Telea
repair, and textured PBR GLB export. A Vulkan build owns neural inference only:
it writes Gaussian PLY plus the raw FlexiCubes mesh, which a separate CUDA PBR
binary consumes without Python, Torch or cuDNN at runtime. The resulting GLB is
therefore a `Vulkan inference -> CUDA PBR` artifact, never a Vulkan renderer
claim.

## What is supported

| Area | Current implementation | Boundary |
| --- | --- | --- |
| GGML inference | condition, SS, SLat and Gaussian decoder | CPU, CUDA and Vulkan builds |
| Quantization | F32, F16, Q8_0 and Q4 model files | Complete raw numerical acceptance remains incomplete for F16/Q8/Q4 |
| Raw mesh neural decode | 101-channel sparse cube features at 256^3 | Native graph is wired into `image-to-3d`; intermediate parity reports remain available because F16/Q8/Q4 quantization is not bitwise identical to the official BF16 graph |
| FlexiCubes | Native cube extraction | Uses the official sparse feature layout and native lookup tables |
| Mesh cleanup | Exact VTK quadric decimation; native CUDA 1,000-view visibility/mincut; exact MeshFix repair | The canonical visibility camera table and face frequencies match the official fixture bit-for-bit. MeshFix requires the explicit GPL/commercial build option |
| UV | Native xatlas parameterization | Canonical postprocessed mesh vertices, faces, and UVs match the Python xatlas result exactly |
| Gaussian observations | Native CUDA Inria/Mip renderer | Recorded-camera rendering is native; two official RGBA8 camera fixtures match exactly. The 100-view command accepts the frozen camera tensors |
| Texture repair | Native OpenCV Telea | Cross-version C++/official-cv2 comparison is exact on the fixed regression input |
| Output | Gaussian PLY and raw mesh on CPU/CUDA/Vulkan; textured PBR GLB on licensed CUDA | Vulkan exports `--mesh-vertices-out` and `--mesh-faces-out`; `run_vulkan_cuda_pbr.py` hands them and the PLY to a separate CUDA PBR binary. `--pbr-out` remains deliberately rejected in a Vulkan process |

The upstream pipeline decodes Gaussian and mesh assets, decimates and removes
invisible mesh regions, creates xatlas UVs, renders 100 Gaussian observations,
optimizes a 1024px texture for 2500 Adam/TV steps, repairs uncovered texels,
and writes a glTF PBR material. The native implementation executes the same
sequence. The official neural reference uses BF16/SDPA, while GGUF F16/Q8/Q4
execution is a separate numerical policy; parity reports therefore record
support, latent and rendered-image errors instead of claiming bitwise equality
where it has not been measured.

The exact native-completion definition, official-source anchors, frozen PBR
reference bundle, and stage-by-stage acceptance procedure are in
[`docs/NATIVE_E2E_PARITY_CONTRACT.md`](docs/NATIVE_E2E_PARITY_CONTRACT.md).
That document is the source of truth for the native textured-GLB path and its
license/build boundaries. The non-baked `mesh-export` command remains useful
for inspecting decoder tensors, but it is not the production PBR entry point.

The 2026-09-10 root-cause audit plan has been fully executed and merged into
the [native E2E parity contract](docs/NATIVE_E2E_PARITY_CONTRACT.md): every
pipeline boundary is now either verified in-band against the official oracle
or characterized as implementation-inherent, and the non-numerical
acceptance boundaries (UV/material export, quantization entry point,
performance authorization) live in its "Non-Numerical Acceptance Boundaries"
section.

The native CMake targets do not link cuDNN. CUDA builds use GGML CUDA,
CUDA-runtime, cuBLAS and the explicitly enabled nvdiffrast-compatible raster
core only. The official Python environment is used by reference-generation
and comparison scripts, never by the native `image-to-3d` runtime.

## Documentation map

This guide covers deployment, models, build and the one-command inference
paths. Deeper reference material lives in [`docs/`](docs/):

| Document | Scope |
| --- | --- |
| [`NATIVE_E2E_PARITY_CONTRACT.md`](docs/NATIVE_E2E_PARITY_CONTRACT.md) | source of truth: acceptance definition, official anchors, numerical boundary status, gate history |
| [`BACKEND_NUMERICS.md`](docs/BACKEND_NUMERICS.md) | attention precision policy (SS strict/normal, MoGe F32-K/V) and the cross-backend Philox sampling contract |
| [`NATIVE_PBR_STAGE_REGRESSION.md`](docs/NATIVE_PBR_STAGE_REGRESSION.md) | per-stage regressions: mesh decode/export, visibility, MeshFix, camera contract, Gaussian observations |
| [`../benchmarks/README.md`](benchmarks/README.md) | published evidence: single-object GLB matrix and full-scene reconstruction, with reproduce/publish commands |

## Clone and bootstrap

### System prerequisites

The tested deployment is Linux x86-64 with an NVIDIA CUDA GPU (12 GiB in the
recorded run). Full textured reconstruction needs CUDA even when inference
uses Vulkan. Install a C++17 compiler, Git, CMake, libjpeg headers, CUDA Toolkit
and its compatible driver. Vulkan inference additionally needs Vulkan headers,
loader and `glslc`. Native post-processing needs **VTK >= 9.3** (the quadric
decimation implementation is part of the numerical contract) and OpenCV >= 4.2
with `core` and `photo`. A system VTK 9.1 package is not sufficient.

On Debian/Ubuntu, common non-CUDA prerequisites are:

```bash
sudo apt-get install build-essential git cmake libjpeg-dev libopencv-dev \
  libvulkan-dev glslc
```

Install VTK 9.3 or newer separately when the distribution is older; expose its
installation with `CMAKE_PREFIX_PATH=/path/to/vtk-prefix`. The native build
uses VTK 9.3 and OpenCV 4.5.4; the separately installed Python oracle uses
VTK 9.6.2 and OpenCV 5.0.0. CUDA and VTK are not silently installed with
root privileges by the launchers. CMake reports a missing dependency before
inference starts. The Python reference requires Conda and the official model
access approval documented in [`../doc/setup.md`](../doc/setup.md).

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

The installer pins NumPy 1.26 with OpenCV 4.9 and plyfile 1.0.3 to avoid
NumPy-2-only package requirements in the Kaolin environment. This differs
from the existing measured environment (OpenCV 5 / plyfile 1.1.5), which has
declared dependency conflicts despite completing the recorded inference.
A clean-machine installation and a fresh oracle comparison for the corrected
dependency set remain separate validation tasks; this guide does not claim
that a successful run proves the entire environment passes `pip check`.

## Models

All GGUF files belong in [`models/gguf/`](models/gguf/). Download the published
files with:

```bash
bash cpp_ggml/scripts/download_gguf.sh --dtype q8_0
SAM3D_PYTHON=/absolute/path/to/sam3d-objects/bin/python \
  bash cpp_ggml/scripts/prepare_moge_gguf.sh
```

The conversion/model card is in [`models/MODEL_CARD.md`](models/MODEL_CARD.md).
Do not place model files in `benchmarks/`; benchmark directories contain only
reports and rendered evidence. The published MoGe ViT-L file is the F16 GGUF
and is part of the default f16 download set; `prepare_moge_gguf.sh` remains
the offline reconversion path from the official `Ruicheng/moge-vitl`
checkpoint when another dtype is needed. Native execution still uses no
Python. Prepare F16/Q4 files separately when selecting those precisions.
Public files can be an older conversion revision than the local benchmark;
the downloader preserves different local files and checks download sizes and
headers, but this is not a numerical-equivalence guarantee. To reproduce the
current converter format from official checkpoints, follow the model card.

### One-command textured reconstruction

From the repository root, after installing the prerequisites and downloading
the matching checkpoints/GGUF weights:

```bash
# Official Python; --setup installs its environment on the first run.
bash run_python.sh --setup --out-dir output/python

# Native CUDA: configure, apply the one ggml patch, build, and reconstruct.
bash run_ggml.sh --backend cuda --dtype q8_0 \
  --accept-pbr-licenses --out-dir output/cuda-q8

# Vulkan inference followed by a separate native CUDA PBR process.
bash run_ggml.sh --backend vulkan --dtype q8_0 \
  --accept-pbr-licenses --out-dir output/vulkan-q8

# Multi-object scene mode (the official demo_multi_object flow), fully
# native: every '<idx>.png' under --mask-dir is reconstructed once by the
# native binary, then the pure-C++ scene-assemble command applies the
# official make_scene pose semantics, normalizes, and renders the orbit with
# the same CUDA Gaussian rasterizer as the 100-view bake (PNG frames;
# scene.gif when ffmpeg is available). No PBR bake, no Python.
bash run_ggml.sh --skip-build --backend cuda --dtype q8_0 \
  --mask-dir notebook/images/shutterstock_stylish_kidsroom_1640806567 \
  --mask-indices 14,17 --out-dir output/scene-ggml
```

The license flag explicitly enables the existing nvdiffrast non-commercial
and MeshFix GPL/commercial dependencies. Review their terms before using it.
After building once, use `--skip-build`. To choose another object, pass
`--image /path/image.png --mask /path/mask.png`; the mask must select one object
and match the image dimensions. Use a new output directory for each invocation:
the scripts refuse to overwrite an earlier reconstruction. `--help` lists
paths, seed, thread and precision options. Defaults are the kidsroom image,
mask 14, seed 42 and strict SS attention.

Python produces `official_pbr_00.glb`, `official_pbr_00.base_color.png`,
`official_pbr_00.pose.json` and the `official_full_e2e.json` timing report.
Native reconstruction produces `output.glb`, `output.base_color.png`,
`output.ply`, `pose.json`, `dtype_contract.json`, `native_full_e2e.json` and
`run.log`. `pose.json` carries the official pose receipt (`rotation`,
`translation`, `scale`; the per-axis scale collapses to its uniform mean like
the official decoder) with the raw native decoder fields preserved under a
`native` block. The native timing report mirrors the official report's
`latency_ms` structure under the documented cold-process timer contract, so
the two launchers' artifacts stay directly comparable.

Scene mode (`--mask-dir`) instead produces `scene_posed.ply` (the objects
transformed into the scene frame), `frames/frame_*.png` (the 300-frame orbit
rendered by the same CUDA rasterizer as the 100-view bake; `scene.gif` when
`ffmpeg` is installed) and `scene_manifest.json`, with one
`objects/obj_<idx>/{output.ply,pose.json}` per mask. The scene assembly lives
in the native `sam3d-cli scene-assemble` command: it reproduces the official
`make_scene` pose semantics (including the pytorch3d row-vector transform
convention and quaternion standardization), the official `normalized_gaussian`
rescale, and the official `render_video` orbit, verified field-by-field
against the official Python chain to float32 rounding. The Python script
`cpp_ggml/scripts/run_scene_pipeline.py` (`--runner python|ggml`) remains
available as the official-notebook reference flow for comparisons.

The Vulkan launcher
also saves raw mesh tensors and verifies that
CUDA post-processing does not change its inference outputs. A GLB embeds its
base-color PNG and can be opened on its own. The separate PNG is the same
atlas, not another neural output. Both paths perform 100-view observations and
2500 texture optimization steps; they do not use a reduced-quality bake.

The official Python implementation assigns a glTF PBR material with a baked
base-color atlas and fixed roughness. It does **not** reconstruct independent
metallic, roughness or tangent-space normal textures. Native output follows
that scope; calling the asset PBR must not imply those additional maps exist.
The native runtime needs neither Python, Torch nor cuDNN. The independent
official Python reference uses its own Torch dependencies and streamed mixed
precision to fit the test GPU; it is not an all-F32 reference.
The streamed Python path releases the SLat generator and conditioner before
FlexiCubes decoding. The root-launcher regression measured 2.28 GiB of freed
Torch allocations at that boundary after diagnosing an OOM there; this does
not promise that every input/support size fits on a 12 GiB device.

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

For one image and mask, the native command writes a Gaussian PLY and, when
the MeshFix and nvdiffrast options are enabled, the final textured PBR GLB:

```bash
SAM3D_PYTHON=/absolute/path/to/sam3d-objects/bin/python \
  bash cpp_ggml/scripts/quickstart.sh cuda e2e
```

That `e2e` wrapper is a frozen-condition diagnostic: its candidate consumes
official intermediate tensors, so it cannot prove full raw image-input parity.
For a production run with no Python/Torch process, use the native one-command
wrapper instead:

```bash
SAM3D_DTYPE=f16 \
  bash cpp_ggml/scripts/quickstart.sh cuda native-e2e
```

Set both `SAM3D_NATIVE_PBR=1` and `SAM3D_ACCEPT_PBR_LICENSES=1` to build the
licensed CUDA path and also write `output.glb`. This explicit acceptance is
required because it enables the nvdiffrast non-commercial source and the
MeshFix GPL-3.0-or-commercial boundary-repair source. The direct command below
is equivalent and exposes all paths.

The direct native entry point is:

```bash
cpp_ggml/build-cuda-pbr/bin/sam3d-cli image-to-3d \
  --model cpp_ggml/models/gguf \
  --moge-model cpp_ggml/models/gguf/moge_vitl-f16.gguf \
  --image notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png \
  --mask notebook/images/shutterstock_stylish_kidsroom_1640806567/14.png \
  --dtype f16 --backend cuda --seed 42 --threads 6 --ss-attention strict \
  --out /tmp/sam3d_native.ply \
  --pbr-out /tmp/sam3d_native.glb \
  --pose-out /tmp/sam3d_native_pose.json \
  --dtype-contract-out /tmp/sam3d_native_dtype_contract.json
```

### Attention precision

Attention formulation is selected per graph through explicit options, never
by a process-wide environment switch. The SS diffusion graph defaults to the
strict F32 `QK^T -> softmax -> V` path so a raw quality run cannot silently
use the faster F16-KV flash path; the condition encoders always use their
scoped F32 path; MoGe is pinned to the F32-K/V contract that matches the
official depth model. The mode table, the MoGe repair rationale and the
per-command flags are documented in
[`docs/BACKEND_NUMERICS.md`](docs/BACKEND_NUMERICS.md).

`--dtype-contract-out` writes a machine-readable inventory from the graphs that
were actually constructed: stage, ggml op, source/output dtype, aggregated node
count and largest output tensor. It is an execution-type audit, not a claim of
PyTorch numerical parity. Validate an artifact with:

```bash
python3 cpp_ggml/scripts/verify_dtype_contract.py \
  --contract /tmp/sam3d_native_dtype_contract.json \
  --expected-weight-type f16 --require-stage mesh_decoder
```

`--pbr-out` is intentionally a hard error in builds without
`SAM3D_GGML_NVDIFFRAST_NONCOMMERCIAL=ON` and `SAM3D_GGML_MESHFIX_GPL=ON`; it
never silently falls back to a non-textured asset.

Set `SAM3D_DTYPE=f16`, `SAM3D_DTYPE=q8_0` or `SAM3D_DTYPE=q4_0` to select a
model family. Set `SAM3D_WORK_DIR` to keep outputs at a known location. CUDA
and Vulkan latency runs require an idle GPU; the wrapper records this
provenance and refuses to label shared-GPU timings as release evidence.

### Cross-backend sampling contract

CUDA, Vulkan and CPU share the verified PyTorch Philox draw sequence;
portable backends additionally need the CUDA reference device's
`distribution_blocks` value (`--philox-blocks`). Three verifiers pin the
contract — the full SS/SLat draw sequence, the `torch.randperm` coordinate
selection, and the production downsample path — and are documented with
commands and measured tolerances in
[`docs/BACKEND_NUMERICS.md`](docs/BACKEND_NUMERICS.md).

The neural-only diagnostic matrix is generated with:

```bash
SAM3D_PYTHON=/absolute/path/to/sam3d-objects/bin/python \
  bash cpp_ggml/scripts/quickstart.sh cuda matrix
```

This command ends at Gaussian PLY; it does not replace full GLB acceptance
below and its charts are not published as product E2E metrics. `--dtype`
selects generative GGUF stages; all rows keep the same separate MoGe F16
preprocessing model. Stage directories are oracle-only, never candidate
support, pose or latent replacements in a raw run.

For the complete release acceptance case, including a fresh official PBR oracle,
controlled native mesh/UV/texturing comparison, raw CUDA F16/Q8_0/Q4_0 PBR
outputs, Vulkan F16/Q8_0/Q4_0 inference handed to CUDA PBR, and the six raw
CUDA/Vulkan neural rows, run:

```bash
SAM3D_PYTHON=/absolute/path/to/sam3d-objects/bin/python \
SAM3D_ACCEPT_PBR_LICENSES=1 \
SAM3D_MEASURE_OFFICIAL_FULL_GLB=1 \
SAM3D_WORK_DIR=/tmp/sam3d-full-e2e \
  bash cpp_ggml/scripts/quickstart.sh cuda full-e2e
```

The command writes `full_e2e_summary.json`, raw PLY/GLB artifacts, reference-
world GLB render comparisons, matrix JSON, and plots below `SAM3D_WORK_DIR`.
The GLB reports include linear RGB, silhouette, depth and normal metrics; final
thresholds remain explicit options until official repeat variation is measured.
`SAM3D_MEASURE_OFFICIAL_FULL_GLB=1` also records the official hot-session
timing. The published main chart compares complete cold subprocesses on both
sides, including imports, initialization, model loading and GLB/PNG output.
The hot official value is not compared to cold native values. Native
hot-session timing and repeated-run stability remain unimplemented/unproven.

Use a new work directory for each run. A failing quality gate still produces
reports and assets. Publish those measurements, including their failures, with:

```bash
"$SAM3D_PYTHON" cpp_ggml/scripts/publish_full_e2e.py \
  --summary /tmp/sam3d-full-e2e/full_e2e_summary.json \
  --destination /tmp/sam3d-full-e2e-published
```

The publisher refuses to overwrite an existing asset archive. See
[benchmarks/README.md](benchmarks/README.md) for the canonical checked-in
archive and explicit texture/final-GLB acceptance budget environment variables.
Unconfigured budgets never count as passed acceptance.

The official full-GLB report also writes `reference_manifest`: pipeline-config
SHA-256, interpreter, PyTorch/CUDA version, device, TF32/SDP settings,
weight-residency policy, and per-module dtypes. A latency or quality result
without a matching manifest is diagnostic evidence only, not a comparable
official baseline. Before raw candidates are generated, `full-e2e` compares
the exact stage-oracle SS/SLat Philox draw stream from the CUDA sampler against
the Vulkan portable sampler, freezes that device contract, and forwards it to
all raw candidates. This covers sampling-state alignment only; it is not
evidence that neural operators or final assets are numerically aligned.

For a controlled **diagnostic** run that holds the official initial SS/SLat
noise fixed while comparing CUDA and Vulkan numerical behavior, set
`SAM3D_DIAGNOSTIC_NOISE_REPLAY=1` on the same `full-e2e` command. The generated
summary records `official-noise-replay-diagnostic`, and the native-seed release
gate intentionally rejects it. This mode isolates later graph/quantization
error; it is not a substitute for the portable production RNG regression.

To localize a numerical difference before changing an operator, also set
`SAM3D_OPERATOR_ORACLE=1`. The full runner then creates fresh SS/SLat
block-0 observations and compares every public native debug boundary against
the selected GGUF model. By default it loads **every** generator
`reverse_fn.backbone` tensor from that GGUF through `gguf`'s dequantizer into
the official Torch architecture (`same-gguf-dequantized`), so the report can
separate conversion/quantization error from native graph error. Its reference
directories contain `reference_weight_scope.json`, including the exact GGUF,
loaded tensor count, stored tensor types, and layout inversions. The current
Q4 models have no low-rank residual factors; a future residual-bearing GGUF is
rejected by this oracle until Torch models the native two-GEMM `Wq@x + B@(A@x)`
topology exactly. Set `SAM3D_OPERATOR_REFERENCE_WEIGHTS=checkpoint` only for
the broader official-checkpoint-versus-GGUF comparison. Until repeated runs
establish limits, leave `SAM3D_OPERATOR_MAX_STAGE_MAE` and
`SAM3D_OPERATOR_MAX_STAGE_ABS` unset; the report then records `passed: null`
and full-E2E coverage stays false.

Set `SAM3D_OPERATOR_TRAJECTORY=1` together with
`SAM3D_OPERATOR_ORACLE=1` to extend that same provenance-checked reference
to all 25 Euler states and all SS conditional/unconditional velocities. SS and
SLat are each executed by the native `e2e` stage on the selected backend and
compared against a freshly generated Torch trajectory. The report stores
`reference_weight_scope`, candidate GGUF paths, and per-step errors. This is
available for either standalone CUDA or Vulkan `run_operator_oracle.py`
invocation. Set `SAM3D_OPERATOR_ORACLE_VULKAN=1` to make `full-e2e` run
the same selected oracle on Vulkan after CUDA; its raw neural matrix always
exercises both backends. Leave
`SAM3D_OPERATOR_TRAJECTORY_SS_BASELINE` and
`SAM3D_OPERATOR_TRAJECTORY_SLAT_MAX_TERMINAL_MAE` unset until repeat runs
produce a same-workload, same-scope calibration. Without one of those limits,
the trajectory result is intentionally `passed: null` and cannot satisfy a
release coverage field.
Native PBR baking is intentionally CUDA-only because this source tree contains
CUDA implementations for Gaussian observation rendering, differentiable UV
rasterization/sampling, and the 2500-step Adam/TV loop, but no Vulkan versions
of those computations. `full-e2e` therefore exports Vulkan PLY and raw
FlexiCubes vertices/faces, invokes CUDA `pbr-assemble`, and records the two
owners plus SHA-256 values in `handoff_manifest.json`. The Vulkan rows cover
image/mask through CUDA-generated textured GLB, plus the separate neural
render and runtime-dependency checks.
The command returns
non-zero for any failed prerequisite, metric gate, or missing artifact and does
not modify `benchmarks/`. The latest complete assets are retained at
[`benchmarks/e2e_comparison/full_glb_current/`](benchmarks/e2e_comparison/full_glb_current/).
The multi-object scene evidence (every kidsroom mask, orbit renders,
per-frame metrics and per-object pose parity against the official PyTorch
reference) is published at
[`benchmarks/e2e_comparison/scene_current/`](benchmarks/e2e_comparison/scene_current/)
and is regenerated idempotently with
`run_scene_benchmark.py` + `publish_scene_benchmark.py`. After the full
divergence repair (2026-09-14c: the native chain now feeds the condition
embedders a fully finite point map exactly like the official receipt, see
"REVISION 2" in the parity contract) all three native variants reconstruct
**27/27 objects** in 47-51 min per variant with a median free-run pose
drift of 1.0-1.2 deg and foreground IoU 0.80-0.82 (was 20/27 objects,
IoU 0.48); the scene branch uses the faster F16-KV attention preset while
the single-object pipeline keeps the strict F32-KV gate.
The full-E2E gate remains non-passing until calibrated budgets and
same-contract hot-session timing are configured; the numerical boundary
status is tracked in the
[parity contract](docs/NATIVE_E2E_PARITY_CONTRACT.md). Successful export is
not numerical acceptance.

Current checked-in evidence and regeneration commands are documented in
[`benchmarks/README.md`](benchmarks/README.md). Run the standalone verifier with:

```bash
python3 cpp_ggml/scripts/validate_e2e_benchmark.py \
  --input cpp_ggml/benchmarks/e2e_comparison/e2e_latency_current.json \
  --output cpp_ggml/benchmarks/e2e_comparison/e2e_gate_current.json
```

The verifier reports failures instead of hiding missing or non-exclusive rows.

To run only the explicit mixed-backend artifact boundary, without the official
oracle/scoring workflow, use separate configured build directories:

```bash
python3 cpp_ggml/scripts/run_vulkan_cuda_pbr.py \
  --vulkan-binary cpp_ggml/build-vulkan/bin/sam3d-cli \
  --cuda-pbr-binary cpp_ggml/build-cuda-pbr/bin/sam3d-cli \
  --models-dir cpp_ggml/models/gguf \
  --moge-model cpp_ggml/models/gguf/moge_vitl-f16.gguf \
  --image INPUT.png --mask MASK.png --dtype q8_0 \
  --rng-distribution-blocks CUDA_RECORDED_BLOCKS --out-dir /tmp/sam3d-vulkan-cuda-pbr
```

The command creates `vulkan_output.ply`, raw `vulkan_flexicubes_*.samt`,
`cuda_pbr.glb`, and `handoff_manifest.json`. It is a functional handoff run;
use `cuda full-e2e` for official render and accuracy gates.

## Native mesh decode and PBR stage regression

Beyond the end-to-end commands above, every native post-processing stage has
a focused official-reference regression: `mesh-decode` and the C++
`mesh-export` boundary against the decoder fixtures, the visibility/mincut
and MeshFix cleanup boundaries, the frozen 1,000-view camera contract, and
the 100-view Gaussian observation comparison against the recorded official
cameras. The per-stage commands, reference-bundle generation, the
`--save-bake-raster` fixture policy and the MeshFix license gate are
documented in
[`docs/NATIVE_PBR_STAGE_REGRESSION.md`](docs/NATIVE_PBR_STAGE_REGRESSION.md).
