# Native Mesh Decode, Asset Export And PBR Stage Regression

This guide consolidates the per-stage native verification workflow between
the neural pipeline and the final textured GLB: mesh decoding, the C++ mesh
export boundary, and the official-reference regression for every native PBR
stage (cleanup, visibility, MeshFix, cameras and Gaussian observations).
The complete release acceptance built on top of these gates is the
`full-e2e` case in [the main guide](../README.md); the numerical boundary
status is tracked in the
[parity contract](NATIVE_E2E_PARITY_CONTRACT.md).

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
It is a diagnostic export; production textured output is emitted by
`image-to-3d --pbr-out` after the native mesh and baking stages.

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
focused stage regression; the complete production path also invokes MeshFix,
xatlas, Gaussian rendering and baking.

```bash
python3 cpp_ggml/scripts/verify_mesh_visibility_reference.py \
  --binary cpp_ggml/build-cuda/bin/sam3d-cli \
  --stage-dir /tmp/sam3d_reference --out-dir /tmp/native_visibility_check
```

The camera contract behind that gate can be checked independently. The
official 1,000-view `_fill_holes` matrices are generated once from the pinned
PyTorch environment into a read-only C++ table, so this command has no Python
or PyTorch runtime dependency in the native renderer:

```bash
"$SAM3D_PYTHON" cpp_ggml/scripts/verify_mesh_camera_contract.py \
  --binary cpp_ggml/build-cuda/bin/sam3d-cli --contract visibility --views 1000
```

The final official cleanup uses `pymeshfix.PyTMesh`, whose TMesh source is
GPL-3.0-or-commercial. It is deliberately absent from the default runtime.
After accepting a compatible license, enable the exact implementation and
compare it with the captured official output:

```bash
cmake -S cpp_ggml -B cpp_ggml/build-cuda-meshfix \
  -DSAM3D_GGML_CUDA=ON -DSAM3D_GGML_NATIVE_PBR=ON \
  -DSAM3D_GGML_NVDIFFRAST_NONCOMMERCIAL=ON \
  -DSAM3D_GGML_MESHFIX_GPL=ON
cmake --build cpp_ggml/build-cuda-meshfix --parallel 6
python3 cpp_ggml/scripts/verify_meshfix_reference.py \
  --binary cpp_ggml/build-cuda-meshfix/bin/sam3d-cli \
  --stage-dir /tmp/sam3d_reference --out-dir /tmp/native_meshfix_check
```

This gate requires exact vertices and faces at the pre-xatlas boundary. It
does not cover texture baking or final GLB
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
