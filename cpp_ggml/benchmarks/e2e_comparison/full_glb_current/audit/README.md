# Release Audit Evidence

These receipts accompany the complete raw GLB matrix in the parent directory.
They do not replace its latency samples, quality gates or asset manifest.

## Confirmed Fixes

- `uv_export_before_fix.json` and `uv_export_after_fix.json`: final exported
  GLB UVs, not just the pre-export xatlas tensor. The missing V flip was a real
  export bug. `controlled_export_fix_render.json` records RGB MAE 0.00119204,
  IoU 1 and zero depth error for identical official intermediate inputs.
- The final native material now matches official effective glTF base color,
  metallic, roughness and double-sided settings. The shared comparison
  renderer is Lambertian, not a full physically based material renderer.
- `launcher_python_oom.log` records a real 512 MiB allocation failure during
  FlexiCubes. Moving the completed SLat generator and conditioner off the GPU
  before decoding freed 2,452,979,200 allocated bytes on the successful retry.
  No dtype, diffusion-step or baking-quality setting changed.

## Executed Checks

- `root_launcher_checks.json`: real root Python, CUDA Q8 and Vulkan Q8 plus
  CUDA PBR requests; GLB/PNG pixel identity, material and pose-asset checks.
  These runs did not collect formal pre-run exclusive-GPU receipts and are
  deliberately excluded from the primary timing matrix.
- `test_full_e2e_report.log`: negative tests for evidence validation. Missing
  budgets, missing/changed assets, nonfinite metrics, inconsistent sample
  counts and non-strict SS attention must not become release PASS.
- `test_ggml_patch_delivery.log`: one combined patch, clean-clone replay,
  identical final tree, idempotency, read-only checking and rejection of
  unrecorded edits or extra patch files.
- `ctest-*.log`: native unit/integration checks for the recorded build. Their
  success is not proof that the full neural precision gate passes.
  Final results: CUDA-PBR 28/28, Vulkan 17/17, CPU 17/17; report validation
  24 tests and clean patch delivery 1 test also passed.
- `final_asset_check.json`: all seven primary GLB/PNG/pose triples were
  rechecked after publication; file sizes/hashes and embedded PNG pixels
  match. This is artifact integrity, not numerical parity.
- `*_dependencies.json`: native linked-library trees and executable hashes.
  `cuda_runtime_maps.txt` and `vulkan_runtime_maps.txt` are snapshots from
  actual inference processes. No cuDNN, Torch or Python library was found in
  these snapshots. They are not continuous traces of all future `dlopen`s.

## Remaining Limits

The full release gate still fails. All primary native cold-process runs
exceed 70 seconds, F16/Q8/Q4 raw neural errors exceed the existing MAE budget,
and native hot-session timing, sufficient repeats, final quality budgets,
calibrated operator checks and full physical-PBR render parity remain open.
Raw pose differences are reported separately from the decoder-only oracle.

`python_packages.json` records the measured environment. `python_pip_check.log`
is a failure receipt, not a successful deployment check: the existing OpenCV
5 / plyfile 1.1.5 requirements conflict with NumPy 1.26.4, in addition to
unused training dependencies. The installer has been corrected and geometry
versions pinned, but a fresh install and full oracle rerun on that corrected
dependency set have not been performed.

## Retention

`obsolete_benchmark_inventory.txt` lists superseded benchmark outputs removed
after the new complete assets were verified. Models and still-used stage
fixtures were retained. Original command receipts can reference deleted
scratch paths; usable primary GLBs/PNGs and their hashes are in the parent
matrix. The small pre-fix summaries are failure provenance, not current
performance results. No unrelated user temporary directories were removed.

Cleanup removed 2,652,694,460 bytes of superseded benchmark data and the
4,466,293,834-byte task-owned directory
`/tmp/sam3d-release-20260910.DriMmt` after copying the retained evidence.
These are pre-deletion logical sizes, not a filesystem free-space measurement.
Tracked old content remains recoverable through Git; removed untracked raw
intermediates must be regenerated using the documented regression command.
