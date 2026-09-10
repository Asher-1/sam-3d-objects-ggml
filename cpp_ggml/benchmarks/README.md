# Complete Textured GLB Benchmarks

The current snapshot is the **2026-09-10 raw image/mask-to-textured-GLB** run.
The release gate is **FAIL**, not a claim of official numerical parity or
stable 70-second reconstruction. See the [remaining-issues audit](../docs/E2E_ROOT_CAUSE_AND_IMPLEMENTATION_PLAN.md).

## Current Evidence

- [Seven-pipeline measurement table, downloadable GLBs/PNGs and six comparisons](e2e_comparison/full_glb_current/README.md)
- [Complete timing and quality JSON](e2e_comparison/e2e_latency_current.json)
- [Machine-readable failure reasons](e2e_comparison/e2e_gate_current.json)
- [Original commands, durations and return codes](e2e_comparison/full_glb_current/full_e2e_summary.json)
- [Model, executable and patch provenance](e2e_comparison/full_glb_current/provenance.json)

![Complete image-to-textured-GLB cold latency](e2e_comparison/e2e_latency_current.png)

![Final GLB multiview quality](e2e_comparison/e2e_metrics_current.png)

All rows use kidsroom mask 14, seed 42, RTX 3060 12 GiB, strict SS attention
and native sampling. Native candidates do not replay official noise, support,
pose or latent. CUDA runs the whole native path; Vulkan owns inference and
hands immutable Gaussian/raw-mesh artifacts to a separate CUDA PBR process.
All generative stages, including the mesh decoder, use the indicated GGUF
family; MoGe stays F16. Q4 means Q4_0 here, not the separate retained SS Q4_K
sensitive-layer candidate described in [q4_best](q4_best/README.md).

The workload includes mesh cleanup, xatlas UV, 100 Gaussian views at 1024px,
2500 Adam/TV updates, Telea, GLB output and a standalone 1024px base-color PNG.
The timer starts at subprocess launch and ends after successful exit. Imports,
initialization, weight loading and file I/O are included; external comparison
rendering is excluded. Each row has **one** sample. Exclusive-GPU receipts
check other NVIDIA compute clients at each launch, not continuous isolation
from the desktop or every possible graphics client.

The official reference uses the documented streamed mixed BF16/F16/native
policy to fit this GPU. Its hot-session measurement remains in JSON but is
not mixed into the cold-process chart. The independently captured official
quality GLB is retained separately from the timed official GLB. This is not a
claim of unchanged full-F32 official inference or bitwise parity.

The two official outputs are also compared directly:
[timed-versus-stage diagnostic](e2e_comparison/full_glb_current/official_reference_agreement/glb_render_metrics.json).
Their measured RGB MAE is `0.0057263`, IoU `0.9991254`; neither identical
reference outputs nor a stable tolerance can be assumed from one pair.

The official row was remeasured after releasing unused SLat weights before
FlexiCubes to fix a root-launcher OOM. Its current complete cold time is
234.303 seconds; both the original and refreshed measurement receipts are
preserved. Native binaries did not change, so their six raw measurements
remain unchanged. Q8/Q4 are faster than this reference; both F16 rows are
slower. No row has been promoted to stable or numerically accepted.

Every final GLB is compared over 60 fixed reference-world camera views at
512px. Foreground linear RGB, silhouette, depth and normal errors are reported.
These are final-asset comparisons, not PLY pictures. The common Lambertian
renderer does not model complete glTF metallic/roughness response; material
fields are inspected independently. The audit identified and corrected a
missing export-time V flip and mismatched metallicFactor. The current archive
is regenerated from raw input after that fix, not edited GLB JSON. The files
can also be inspected in a normal GLB viewer.

## Reproduce And Publish

Deploy dependencies and models using [the main guide](../README.md). To
generate just one complete model from the repository root:

```bash
bash run_python.sh --python /absolute/path/to/sam3d-objects/bin/python
bash run_ggml.sh --backend cuda --dtype q8_0 --accept-pbr-licenses
bash run_ggml.sh --backend vulkan --dtype q8_0 --accept-pbr-licenses
```

For a fresh long raw regression, reserve an idle GPU and use a **new** work
directory. The command builds required targets and runs the official oracle,
controlled postprocessing, all six complete raw GLB candidates and the
additional neural diagnostic matrix:

```bash
export SAM3D_PYTHON=/absolute/path/to/sam3d-objects/bin/python
export SAM3D_WORK_DIR=/tmp/sam3d-full-e2e-new
SAM3D_ACCEPT_PBR_LICENSES=1 SAM3D_MEASURE_OFFICIAL_FULL_GLB=1 \
  bash cpp_ggml/scripts/quickstart.sh cuda full-e2e
```

Set `SAM3D_SKIP_NEURAL_DIAGNOSTIC_MATRIX=1` to omit only the extra neural-only
benchmark. All six complete raw GLB reconstructions and their Gaussian/final
GLB comparisons still run. The post-export-fix snapshot uses this option;
the earlier run already executed the separate neural diagnostic matrix.

The nonzero return code records a failed gate; it does not discard measured
assets. Do not wrap it in `|| true` in acceptance CI. Once the command has
finished, publish the compact results to a new directory even when parity
failed:

```bash
"$SAM3D_PYTHON" cpp_ggml/scripts/publish_full_e2e.py \
  --summary "$SAM3D_WORK_DIR/full_e2e_summary.json" \
  --destination /tmp/sam3d-full-e2e-published
```

This writes the seven GLB/PNG/pose triples, comparison images/GIFs, source
receipts, current JSON, gate and both charts. It refuses an existing archive.
The final snapshot captures input, binary, model and patch hashes before any
measured subprocess. Older summaries without that field receive explicitly
labelled publication-time provenance, never a fabricated pre-run snapshot.

Final numerical budgets must be chosen and documented **before acceptance**,
not adjusted to the candidate's errors. `quickstart.sh` forwards these
environment variables to the full runner:

| Scope | Variables |
| --- | --- |
| Final rendered GLB | `SAM3D_MAX_GLB_RGB_MAE_LINEAR`, `SAM3D_MIN_GLB_MASK_IOU`, `SAM3D_MAX_GLB_NORMAL_ANGLE_DEG`, `SAM3D_MAX_GLB_DEPTH_MAE_NDC` |
| Controlled texture | `SAM3D_MAX_TEXTURE_MAE_U8`, `SAM3D_MAX_TEXTURE_RMSE_U8`, `SAM3D_MAX_TEXTURE_ABS_U8` |
| Controlled normals | `SAM3D_MAX_NORMAL_ANGLE_DEG` |

Unconfigured budgets mean measurement only. Existing neural MAE <= 0.01 is
not relaxed. Optional calibrated operator/trajectory checks are documented in
[the runtime guide](../README.md). Native hot-session timing, sufficient
repeats and numerical parity remain required but missing; setting these budget
variables alone cannot make the full release gate pass.

Regenerate charts and validate archived content without rerunning inference:

```bash
"$SAM3D_PYTHON" cpp_ggml/scripts/plot_e2e_latency.py \
  --input cpp_ggml/benchmarks/e2e_comparison/e2e_latency_current.json \
  --output cpp_ggml/benchmarks/e2e_comparison/e2e_latency_current.png \
  --metrics-output cpp_ggml/benchmarks/e2e_comparison/e2e_metrics_current.png
"$SAM3D_PYTHON" cpp_ggml/scripts/validate_e2e_benchmark.py \
  --input cpp_ggml/benchmarks/e2e_comparison/e2e_latency_current.json \
  --output cpp_ggml/benchmarks/e2e_comparison/e2e_gate_current.json
```

The validator rejects missing or changed assets, nonfinite values, wrong
timer/sampling scope, missing budgets, unpassed quality and incomplete
coverage. It also reports the 70-second, relative-PyTorch and Q4/Q8 requirements
and missing repeated-run evidence. Synthetic validator tests are not model
parity tests.

## Retention And Reproducibility

All GGUF files live in `../models/gguf/`. The current archive keeps complete
viewable assets and small receipts, not giant latent/raster/observation dumps.
Historical normal-attention GLBs, old raw runs and neural-only presentation
images are superseded by this snapshot. The retained `data/e2e` directory is
a test fixture, not a current performance result. The Q4 milestone retains
its small provenance/strategy receipts, not obsolete output images or models.

Original receipts may mention removed scratch paths; archived model links
and SHA-256 values are authoritative. Regenerate intermediate tensors through
the raw command when needed. No unrelated user temporary directories are
included in benchmark cleanup.

GGML changes are delivered by the single
[combined patch](../third_party/ggml-patches/0001-sam3d-ggml-combined.patch),
automatically applied by CMake. Before measuring:

```bash
bash cpp_ggml/scripts/apply_ggml_patches.sh --check
python3 cpp_ggml/scripts/test_ggml_patch_delivery.py
```

The clean-clone test verifies exact final-tree equivalence, idempotency,
read-only checking and rejection of unrecorded ggml source edits.
