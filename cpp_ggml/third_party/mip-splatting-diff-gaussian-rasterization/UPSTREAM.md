# Mip-Splatting Differential Gaussian Rasterizer

This directory contains the **unmodified CUDA rasterizer core** used by the
official SAM 3D Objects Python renderer when its `inria` backend is installed.
It is imported from the following immutable Mip-Splatting source snapshot:

- Repository: `https://github.com/autonomousvision/mip-splatting`
- Commit: `dda02ab5ecf45d6edb8c540d9bb65c7e451345a9`
- Source directory:
  `submodules/diff-gaussian-rasterization`
- Imported files: `cuda_rasterizer/`, `LICENSE.md`, and the matching GLM 0.9.9.9
  headers under `third_party/glm/glm/`.

The non-build GLM documentation, test, CMake, and utility directories are not
copied. They are not included by the rasterizer and omitting them keeps this
repository source-only dependency to approximately 3.3 MiB.

The rasterizer and this integration are restricted to research/evaluation use
by the included Inria Gaussian-Splatting license. `LICENSE.md` must accompany
redistribution. This directory is not ggml source and is not modified by the
ggml patch mechanism.

The `sam3d_mip_gaussian_rasterizer` CMake target builds these files unchanged
and exposes their pure C++/CUDA `CudaRasterizer::Rasterizer` API. It does not
use PyTorch, Python, nvdiffrast, or cuDNN.
