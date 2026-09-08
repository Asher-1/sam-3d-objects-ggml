# Native End-To-End Parity Contract

This contract defines completion for the C++/ggml implementation. A Gaussian
PLY, a decodable GLB, or a stage-local tensor match is not a completed result.
The required result is the same final textured PBR GLB and the same rendered
appearance as the official SAM 3D Objects Python pipeline for the same image,
mask, checkpoint revision, seed, and rendering conditions.

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
Torch, nvdiffrast, PyTorch3D, or cuDNN runtime dependency. Python is allowed
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
| `asset_parameterized_vertices_zup.samt`, `asset_parameterized_faces.samt` | postprocess/xatlas comparison before final z-up to y-up export rotation |
| `asset_postprocess_*.samt`, `asset_uv.samt` | final GLB-coordinate mesh, topology, and UV comparisons |
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

## Implementation Order

1. Complete native raw image/mask preprocessing and MoGe integration.
2. Validate the native sparse 101-channel mesh decoder against official stage
   fixtures, then port FlexiCubes extraction.
3. Add the official cleanup semantics and pinned xatlas dependency as an
   auditable third-party source.
4. Implement the GPU face-visibility and UV rasterizers, recorded camera path,
   and Adam/TV texture optimizer in C++. The optional GPL/commercial MeshFix
   adapter is exact against the captured official post-mincut mesh; Telea
   inpainting is already implemented and independently cross-checked.
5. Connect the atlas to `sam3d::NativeMesh` and validate the final GLB.
6. Run CPU, CUDA, and Vulkan raw-input E2E regressions. CUDA and Vulkan may
   accelerate neural inference; the final asset contract remains identical.

Any ggml source change required by these stages stays in the single
`third_party/ggml-patches/0001-sam3d-ggml-combined.patch` and is applied by
CMake through `scripts/apply_ggml_patches.sh`. Native SAM code, asset code,
and third-party non-ggml dependencies must not modify the ggml submodule.

## Current Status

The repository has native FlexiCubes extraction, xatlas parameterization, the
unmodified Inria/Mip CUDA Gaussian renderer, and Telea repair. The canonical
raw mesh input has also been verified byte-for-byte through official VTK
quadric decimation and xatlas seam expansion, and two official camera renders
match exactly after RGBA8 conversion. The C++ mincut stage now reproduces the
official component threshold, weighted dual graph, cut validation, and mesh
compaction when supplied the face frequencies captured from the official
1000-view rasterization loop. These are stage-level facts only.

The sparse 101-channel mesh decoder still fails its full official fixture
comparison. Native CUDA face visibility, differentiable UV
rasterization/sampling, optimized 2500-step Adam/TV texture baking, and the
final textured native GLB orchestrator are not implemented. The exact
MeshFix-compatible boundary step is available only under
`SAM3D_GGML_MESHFIX_GPL=ON`, because its upstream TMesh dependency is
GPL-3.0-or-commercial; the default distributable runtime remains free of it.
Therefore no current benchmark is allowed to claim complete native PBR
end-to-end parity.

`sam3d-cli moge-smoke --input image.png` now accepts a decoded native RGB image
for the MoGe neural graph. It is intentionally limited to the direct RGB-to-MoGe
boundary: official mask handling, crop/pad/resize, point-map normalization, and
all downstream stages remain incomplete native work and are not bypassed by this
command.
