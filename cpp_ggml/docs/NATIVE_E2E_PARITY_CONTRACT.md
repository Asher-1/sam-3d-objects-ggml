# Native End-To-End Parity Contract

This contract defines completion for the C++/ggml implementation. A Gaussian
PLY, a decodable GLB, or a stage-local tensor match is not a completed result.
The required result is the same final textured PBR GLB and the same rendered
appearance as the official SAM 3D Objects Python pipeline for the same image,
mask, checkpoint revision, seed, and rendering conditions.

The [current root-cause audit and implementation plan](E2E_ROOT_CAUSE_AND_IMPLEMENTATION_PLAN.md)
records the source-confirmed gaps, existing measurements, and proposed work
needed to satisfy this contract. It is not an implementation or validation
completion report.

## Source Of Truth

The primary behavioural specification is the checked-in official implementation:

| Contract | Official source |
| --- | --- |
| image/mask merge and whole pipeline | `sam3d_objects/pipeline/inference_pipeline.py:InferencePipeline.run` |
| crop, background removal, square padding, and resize | `sam3d_objects/pipeline/preprocess_utils.py` |
| point-map conditioning | `sam3d_objects/pipeline/inference_pipeline_pointmap.py` |
| mesh extraction and decoded attributes | `sam3d_objects/model/backbone/tdfy_dit/representations/mesh/cube2mesh.py` |
| mesh cleanup, UV parameterization, views, texture bake, and GLB | `sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py` |

`../trellis-ggml` is an auxiliary engineering reference only. Its vendored
xatlas integration, deterministic GLB writer, atlas validation, and
source-controlled dependency pattern are useful. Its dense per-vertex PBR
field and fallback projection atlas are not semantically equivalent to SAM's
Gaussian-observation texture optimization and must not be used as a substitute.

## Required Native Data Flow

```text
image + alpha/mask
  -> native decode / official crop-pad-resize
  -> native MoGe point map + camera normalization
  -> native DINO + PointPatch conditions
  -> native SS flow, occupancy, support, pose
  -> native SLat flow and sparse decoder
  -> native Gaussian and 101-channel FlexiCubes mesh decoder
  -> official-equivalent mesh cleanup and coordinate conversion
  -> xatlas UV charts
  -> 100 Gaussian observations at 1024x1024 + cameras
  -> 2500-step texture optimization + TV + Telea inpaint
  -> embedded baseColorTexture PBR GLB
```

Release inference must execute every arrow in C++ and must have no Python,
Torch, PyTorch3D, or cuDNN runtime dependency. The licensed native
nvdiffrast-compatible CUDA core is vendored and built as C++/CUDA source; the
Python `nvdiffrast` package is never imported at runtime. Python is allowed
only to produce frozen reference artifacts and to run comparisons.

## Reference Bundle

Generate a full official asset reference with:

```bash
export SAM3D_PYTHON=/path/to/sam3d-python/bin/python
"$SAM3D_PYTHON" cpp_ggml/scripts/dump_e2e_stages.py \
  --image INPUT.png --mask-dir MASK_DIR --mask-index 0 --seed 42 \
  --out-dir /tmp/sam3d_e2e_case

"$SAM3D_PYTHON" cpp_ggml/scripts/generate_official_pbr_reference.py \
  --stage-dir /tmp/sam3d_e2e_case \
  --reference-dir /tmp/sam3d_e2e_case/official_pbr_reference \
  --seed 42 --save-observations --save-bake-raster
```

Validate the resulting reference bundle before using it for native parity:

```bash
python cpp_ggml/scripts/verify_official_pbr_reference.py \
  --reference-dir /tmp/sam3d_e2e_case/official_pbr_reference \
  --stage-dir /tmp/sam3d_e2e_case
```

The normal stage tensors remain in `e2e_case`. The optional, large final-asset
bundle is placed in `e2e_case/official_pbr_reference`:

| Artifact | Purpose |
| --- | --- |
| `official_pbr.glb` | final official binary asset |
| `base_color.png` | final 1024px PBR base-color atlas |
| `bake_cameras.npz` | all baked-view extrinsic and intrinsic matrices |
| `bake_observations.npz` | optional 100 official RGB observations; required for pixel-level texture regression |
| `bake_raster/view_###.npz` | optional external UV, UV-derivative, and coverage inputs to each official differentiable texture sample |
| `manifest.json` | fixed seed, mesh cleanup parameters, bake hyperparameters, 2500 selected view indices, and hashes for each recorded input |
| `asset_parameterized_vertices_zup.samt`, `asset_parameterized_faces.samt` | legacy postprocess/xatlas comparison fixture before final z-up to y-up export rotation |
| `asset_postprocess_*.samt`, `asset_uv.samt` | stage mesh/topology and pre-export UVs; final GLB UV validation must additionally account for trimesh's `V = 1 - V` export transform and compare the actual official GLB |
| `asset_decimated_*.samt`, `asset_mincut_*.samt` | official VTK output, 1000-view face frequencies, and exact pre-MeshFix mincut boundary |

The generator seeds NumPy immediately before calling the official baking path
and records every sampled view. This converts the upstream random view choice
into a replayable acceptance input without changing the official algorithm.
It also records both z-up parameterized geometry and the y-up geometry written
by `to_glb()`, so a coordinate conversion cannot be mistaken for a mesh error.
`--save-bake-raster` is deliberately opt-in because the 100 full-resolution
maps are large; they belong in an external reference directory, never in
`benchmarks/` or the GGUF model directory. When requested, the verifier checks
all 100 shapes and SHA-256 values before they are used to separate native UV
rasterization from texture-optimizer error.

## Acceptance Method

No numeric threshold is invented in this document. For each canonical case,
first run the official reference twice on the same software and hardware and
record its own variation with
`compare_official_pbr_references.py`. Then set each native tolerance from that
observed variation, with the exact command, checkpoint hashes, driver, GPU,
and output hashes stored beside the result. The fixed seed makes the sampled
views and source observations exact; CUDA texture optimization may still have
small atomic-reduction variation in the final PNG.

The native regression must check all of these in order:

1. Input/preprocessing tensors and MoGe point map.
2. SS and SLat steps, including sparse coordinates and support set.
3. Raw FlexiCubes vertices, faces, and 101-channel decoded attributes.
4. Postprocessed vertices/faces and xatlas UV topology.
   First require exact native equality at the captured pre-MeshFix mincut
   boundary; only then assess the separate boundary-repair implementation.
5. The 100 rendered Gaussian observations, using recorded cameras and their
   SHA-256 values.
6. Texture optimization at recorded selected views, final atlas pixels, and
   Telea-filled atlas pixels.
7. GLB validation: topology, UVs, embedded `baseColorTexture`, material factors,
   and multiview linear-RGB renders against the official GLB.

Discrete sparse support is evaluated with exact coordinate set differences as
well as precision, recall, and IoU. Continuous tensors, texture pixels, and
linear renders report maximum absolute error, MAE, RMSE, PSNR, and the full
worst-pixel location. A final pass requires every stage and final renders to
meet the empirically recorded contract; a count match or visual inspection alone
does not pass.

## One-Command Acceptance

The repository provides one command that constructs a fresh oracle and records
the currently implemented checks without overwriting checked-in benchmark snapshots:

```bash
SAM3D_PYTHON=/absolute/path/to/sam3d-objects/bin/python \
SAM3D_ACCEPT_PBR_LICENSES=1 \
SAM3D_MEASURE_OFFICIAL_FULL_GLB=1 \
SAM3D_WORK_DIR=/tmp/sam3d-full-e2e \
  bash cpp_ggml/scripts/quickstart.sh cuda full-e2e
```

`full-e2e` executes these boundaries in order and writes the exact commands,
return codes, elapsed times, paths, and pass/fail state to
`$SAM3D_WORK_DIR/full_e2e_summary.json`:

1. Inspect the CUDA and Vulkan native binaries with `ldd` and reject a cuDNN
   dependency. Before the oracle and before every timed native CUDA case,
   reject any other NVIDIA compute client.
2. Generate a fresh official stage dump and official 100-view/2500-step PBR
   oracle from the supplied image, mask, and seed.
3. Feed that oracle's raw mesh and Gaussian PLY to native `pbr-assemble`; this
   isolates mesh cleanup, xatlas UV, observation rendering, texture baking,
   Telea, and GLB export from neural error.
4. Execute native CUDA F16, Q8_0, and Q4_0 `image-to-3d --pbr-out` from the
   same image and mask, structurally validate each GLB, score each Gaussian
   PLY with the official renderer, and render each final GLB on the reference
   world-space orbit. Each case must also export its actual ggml dtype/op graph
   contract and prove that its selected dtype is an observed `MUL_MAT` source,
   including the mesh decoder. The final-asset report includes linear RGB,
   silhouette, depth and normal metrics plus visual artifacts.
5. Execute native Vulkan F16, Q8_0, and Q4_0 `image-to-3d` with raw
   FlexiCubes vertices/faces export. Feed exactly those artifacts and the
   Vulkan PLY to the separately configured CUDA `pbr-assemble` binary, then
   validate the manifest owner fields, SHA-256 receipts, PBR structure, PLY
   render, and final-GLB multi-view render. This is the required
   `vulkan_inference_cuda_pbr_handoff` boundary, not a Vulkan PBR claim.
6. Execute native raw-image CUDA/Vulkan F16, Q8_0, and Q4_0 rows and apply the
   matrix latency, quality, GPU-exclusivity, and relative-speed gate.

Set `SAM3D_OPERATOR_ORACLE=1` to add a sixth, deliberately separate
localization boundary. It creates SS/SLat block-0 Torch captures from the
fresh stage directory, invokes native `ss-step`/`slat-step` for every public
debug boundary, and writes shape plus MAE/RMSE/maximum-absolute-error reports
below `$SAM3D_WORK_DIR/operator_oracle/`. The default
`same-gguf-dequantized` scope replaces every generator-backbone tensor in the
Torch reference with the selected GGUF's exact `gguf`-library dequantization.
The runner requires its per-model `reference_weight_scope.json` receipts,
including loaded tensor count and GGUF path. This removes checkpoint-to-GGUF
weight error from the compared block-0 dependencies. A GGUF with low-rank
residual factors is rejected until the reference reproduces the native two
GEMM residual topology, rather than silently folding it into one matrix.
Set `SAM3D_OPERATOR_REFERENCE_WEIGHTS=checkpoint` to retain the separate
official-checkpoint-versus-GGUF localization mode. Supply only
repeat-calibrated values through `SAM3D_OPERATOR_MAX_STAGE_MAE` and
`SAM3D_OPERATOR_MAX_STAGE_ABS`; without at least one limit the report has
`passed: null`, and coverage remains false by design.

With `SAM3D_OPERATOR_TRAJECTORY=1`, the same call also emits fresh all-step
SS and SLat Torch trajectories from the selected reference-weight scope and
runs `validate_ss_gguf_trajectory.py` plus
`validate_slat_gguf_trajectory.py`. The SLat validator consumes
`slat_coords.samt`, which is the immutable sparse-coordinate stage tensor;
it must not substitute a non-existent generic `coords.samt`. Each trajectory
report records the reference scope, candidate GGUF scope, and, for same-GGUF,
the required `reference_weight_scope.json` receipt. Optional trajectory
gates are `SAM3D_OPERATOR_TRAJECTORY_SS_BASELINE` and
`SAM3D_OPERATOR_TRAJECTORY_SLAT_MAX_TERMINAL_MAE`; do not set them from one
candidate run. The raw full-E2E entry invokes this optional localization on
CUDA; set `SAM3D_OPERATOR_ORACLE_VULKAN=1` to repeat it on Vulkan from the
same fresh stage directory. A direct
`run_operator_oracle.py --backend vulkan --trajectory` invocation remains
available for focused diagnosis. Neither form replaces raw render acceptance.

`SAM3D_DIAGNOSTIC_NOISE_REPLAY=1` adds the fresh official initial SS/SLat noise
to raw candidate commands. It is useful only to separate RNG error from later
operator or quantization error: the matrix records the sampling mode as
`official-noise-replay-diagnostic`, while its release contract requires
`native-seed` and therefore rejects the report. It must never be cited as a
production cross-backend RNG pass.

The PBR build requires explicit license acceptance because it enables the
vendored nvdiffrast-compatible CUDA source under NVIDIA's non-commercial terms
and the MeshFix GPL-3.0-or-commercial source. The command returns non-zero if
an artifact is missing or any gate fails; a generated report with failures is
evidence of an unmet contract, never a release pass.

Native PBR baking is CUDA-only. The Vulkan contract therefore validates the
native image/mask-to-PLY graph with the identical official-render, latency, and
runtime-dependency gates, then hands PLY plus raw FlexiCubes mesh to a distinct
CUDA PBR process. The resulting GLB is explicitly labeled
`Vulkan inference -> CUDA PBR`; no Vulkan raster/texturing backend is claimed.

Set `SAM3D_MEASURE_OFFICIAL_FULL_GLB=1` to measure the official hot-session
`image + mask -> textured GLB` path and its outer cold-process wall time.
The publisher compares the latter with complete native cold-process times.
The hot value cannot be compared to a cold native value; native hot-session
measurement and configured numerical quality gates remain incomplete.

The official timing artifact carries a versioned reference-execution manifest:
the pipeline-config hash, Python/PyTorch/CUDA versions, selected device,
TF32/SDP settings, residency policy, and actual module dtypes. The full runner
validates every CUDA binary's continuous SS/SLat Philox draws against the fresh
stage oracle. It also validates CUDA, CPU, and Vulkan against the same Philox
counter/scatter contract, compares their 42,000- and 100,000-element coordinate
`randperm` outputs byte-for-byte with CUDA PyTorch, and separately compares the
complete sparse-coordinate downsample path (F32 rescale, ties-to-even rounding,
truncation/clamp, exact four-I32-component uniqueness, lexical unique ordering,
and random selection) to the official CUDA implementation. Normal values use a
strict numerical bound because CUDA and host math libraries differ; coordinate
permutations and selected I32 rows are exact. This resolves the RNG-input
coverage gap, but it does not prove neural or final-asset parity.

This is the current command's coverage, not numerical completion. Raw CUDA and
Vulkan-inference/CUDA-PBR GLBs are rendered without pose/scale
normalization, while controlled texture and angle-weighted normal comparisons
can be made mandatory. Final GLB thresholds remain opt-in until they are
calibrated against repeated official runs. The command's report therefore keeps
`passed: false` while same-contract native hot timing or configured numerical
gates are missing. See the [root-cause plan](E2E_ROOT_CAUSE_AND_IMPLEMENTATION_PLAN.md).

## Implementation Boundaries

Native image/mask preprocessing, MoGe, the sparse 101-channel mesh decoder,
FlexiCubes, cleanup, xatlas, Gaussian observations, Adam/TV and Telea are wired
into the production CUDA `image-to-3d --pbr-out` path. Vulkan produces neural
outputs for the separate CUDA assembler. These are implemented stages, not
future porting tasks. Current raw CUDA/Vulkan assets exercise that complete
sequence; controlled fixtures separately test matching intermediate inputs.

Numerical alignment, multi-image/repeated-run acceptance and a CPU full-model
raw regression remain distinct from implementation coverage. Remaining work
and evidence are maintained in the linked audit, not a second stale task list.

Any ggml source change required by these stages stays in the single
`third_party/ggml-patches/0001-sam3d-ggml-combined.patch` and is applied by
CMake through `scripts/apply_ggml_patches.sh`. Native SAM code, asset code,
and third-party non-ggml dependencies must not modify the ggml submodule.

## Current Status

The 2026-09-10 raw run produces complete textured assets for CUDA
F16/Q8_0/Q4_0 and Vulkan F16/Q8_0/Q4_0 followed by CUDA PBR. The current
[full GLB report](../benchmarks/e2e_comparison/full_glb_current/README.md)
contains all seven pipelines' assets, standalone texture PNGs, pose receipts,
matching cold-process timings and 60-view final-GLB comparisons. Historical
neural-only latency tables are no longer product E2E evidence.

Root launchers `run_python.sh` and `run_ggml.sh` drive these existing paths.
The native build requires explicit nvdiffrast and MeshFix license options.
Vulkan owns inference only; `pbr-assemble` owns CUDA cleanup/raster/baking.
All handoff inputs are hashed before and after assembly. Native linked
dependencies are checked for missing libraries, Python, Torch and cuDNN.

The release gate still fails. F16 has measurable raw render error as well as
Q8/Q4; controlled mesh/UV equality does not imply neural or texture equality.
Final numerical budgets need calibration, native hot-session timing remains
unimplemented, and one cold-process sample per row cannot establish stable
latency. The optional operator oracle was not run in this snapshot.

Actual GLB parsing revealed missing V inversion at export and official
effective metallicFactor=1 versus native=0. The PBR assembly boundary now
flips V only after baking and sets the matching material factors. The
controlled verifier compares directly to the exported official GLB, not only
the pre-export stage UV tensor. The final snapshot reruns raw reconstruction
after this fix. Full physical PBR shading is still outside the shared
Lambertian renderer; material-field equality is a separate coverage item.
Official NORMAL omission is valid glTF; the acceptance renderer derives
angle-weighted normals when absent.

See the [audit](E2E_ROOT_CAUSE_AND_IMPLEMENTATION_PLAN.md) for the remaining
implementation work and the [benchmark guide](../benchmarks/README.md) for
reproduction, publishing, and gate commands. Successful export or CTest must
never be described as full numerical acceptance.
