# Retained Q4 Candidate Provenance

The retained SS candidate uses Q4_K for quantized matrices and F16 for
sensitive latent mapping, pose attention/MLP, AdaLN/time embedding and
cross-attention matrices. Other stages use Q4_0. Its weights remain exclusively
under [models/gguf](../../models/gguf/); no model is stored here.

This candidate is **not** the uniform-generative Q4_0 configuration in the
current raw full-GLB matrix. The frozen-condition images previously kept here
did not capture the input model hashes at inference time, so they could not
prove the current candidate's quality. Those obsolete PLY images and large
intermediates have been removed; the small historical
[run receipt](runs/cuda/run_summary.json) and
[QAT selection record](q4_qat_report.json) remain for provenance only.

Use the [current complete-GLB results](../README.md) for actual CUDA/Vulkan
latency and final textured-model comparisons. Do not attribute its Q4_0 rows
to this Q4_K candidate or treat a trajectory screen as a release pass.

The retained optimization tools are
[q4_ss_trajectory_qat.py](../../scripts/q4_ss_trajectory_qat.py) and
[validate_ss_gguf_trajectory.py](../../scripts/validate_ss_gguf_trajectory.py).
A replacement candidate must use a separate temporary GGUF directory, record
its input hashes before execution, pass the canonical 0-to-25 support screen,
and then pass the same raw full-GLB CUDA/Vulkan regression. Publishing reports
does not alter the selected weights or relax accuracy thresholds.
