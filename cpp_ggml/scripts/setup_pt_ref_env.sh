#!/usr/bin/env bash
#
# setup_pt_ref_env.sh — PyTorch reference environment for the SAM 3D Objects
# end-to-end benchmark (image -> 3D asset) used by cpp_ggml/benchmarks.
#
# Creates/uses the `sam3d-objects` conda env (python 3.11, CUDA 12.1 toolchain)
# and installs only the inference-path dependencies: the full requirements.txt
# of the upstream project is the training set and is deliberately skipped
# (`pip install --no-deps` for the project itself).
#
# Usage: bash cpp_ggml/scripts/setup_pt_ref_env.sh
# NOTE: no `set -u` — the conda gcc activation scripts reference unset vars.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_EXE="${SAM3D_CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "${CONDA_EXE}" ]]; then
    echo "error: conda was not found; install Miniconda/Anaconda or set SAM3D_CONDA_EXE" >&2
    exit 1
fi
CONDA_BASE="$("${CONDA_EXE}" info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if [[ ! -x "${CONDA_BASE}/envs/sam3d-objects/bin/python" ]]; then
    conda create -y -n sam3d-objects python=3.11
fi
conda activate sam3d-objects

# CUDA 12.1 toolchain in the env (nvcc + dev headers incl. cublas/cusparse,
# which torch's CUDAContext headers require) — needed to compile pytorch3d /
# gsplat. Idempotent: skips packages that are already present.
if [[ ! -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
    conda install -y -c conda-forge gcc_linux-64=12 gxx_linux-64=12
    conda install -y -c "nvidia/label/cuda-12.1.1" \
        cuda-nvcc cuda-cudart-dev cuda-crt cuda-cccl cuda-profiler-api \
        libcublas-dev libcusparse-dev libcufft-dev libcurand-dev \
        libcusolver-dev libnvjitlink-dev
fi

PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
KAOLIN_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

step() { echo -e "\n=== [$(date +%H:%M:%S)] $1 ==="; }

step "torch 2.5.1 + cu121"
python -c "import torch; assert torch.__version__ == '2.5.1+cu121' and torch.version.cuda == '12.1'" 2>/dev/null || \
    pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url "${PYTORCH_INDEX}"

step "xformers 0.0.28.post3 (torch 2.5.1 build)"
python -c "import xformers" 2>/dev/null || \
    pip install xformers==0.0.28.post3 --index-url "${PYTORCH_INDEX}"

step "spconv-cu121 + kaolin 0.17.0"
python -c "import spconv" 2>/dev/null || pip install spconv-cu121==2.3.8
python -c "import kaolin" 2>/dev/null || pip install kaolin==0.17.0 --find-links "${KAOLIN_LINKS}"

step "light python deps"
# Keep the NumPy 1.x ABI required by Kaolin in the same resolver transaction.
# New OpenCV/plyfile releases require NumPy 2 and cannot be mixed with it.
pip install numpy==1.26.4 hydra-core==1.3.2 omegaconf opencv-python==4.9.0.80 easydict gradio imageio \
    loguru pillow plotly plyfile==1.0.3 pymeshfix==0.18.1 pyvista==0.48.4 vtk==9.6.2 safetensors scipy seaborn \
    open3d optree astor timm lightning utils3d einops ninja igraph fvcore \
    xatlas==0.0.11 roma trimesh==5.0.0 point-cloud-utils

step "MoGe (microsoft/MoGe @a8c3734)"
python -c "import moge" 2>/dev/null || \
    pip install "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"

# Compile for the attached device unless the caller requests an explicit set.
export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(python -c 'import torch; major, minor = torch.cuda.get_device_capability(); print(f"{major}.{minor}")')}"
export MAX_JOBS="${MAX_JOBS:-6}"
export FORCE_CUDA=1

step "pytorch3d @75ebeeae (compiled, ~10-20 min)"
python -c "import pytorch3d" 2>/dev/null || {
    pytorch3d_source="$(mktemp -d /tmp/sam3d-pytorch3d.XXXXXX)"
    git clone https://github.com/facebookresearch/pytorch3d.git "${pytorch3d_source}"
    git -C "${pytorch3d_source}" checkout 75ebeeaea0908c5527e7b1e305fbc7681382db47
    pip install --no-build-isolation "${pytorch3d_source}"
}

step "gsplat @2323de59 (compiled, ~5-10 min)"
python -c "import gsplat" 2>/dev/null || \
    pip install --no-build-isolation \
        "git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"

# The official textured-GLB path imports these two CUDA extensions directly:
# `nvdiffrast` for differentiable UV rasterization/texture sampling and the
# Mip Gaussian rasterizer for its 100 baked observations. Pinning them makes a
# frozen Python reference reproducible; the native release binary never links
# against Python, Torch, or either extension.
step "nvdiffrast @253ac4fc (official texture-bake reference)"
python -c "import nvdiffrast.torch" 2>/dev/null || \
    pip install --no-build-isolation \
        "git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae"

step "Mip Gaussian rasterizer @dda02ab5 (official observation reference)"
python -c "import diff_gaussian_rasterization" 2>/dev/null || \
    pip install --no-build-isolation \
        "git+https://github.com/autonomousvision/mip-splatting.git@dda02ab5ecf45d6edb8c540d9bb65c7e451345a9#subdirectory=submodules/diff-gaussian-rasterization"

step "sam3d_objects (metadata only, --no-deps)"
pip install --no-deps -e "${REPO_ROOT}"

step "hydra 1.3.2 patch (facebookresearch/hydra#2863 workaround)"
python "${REPO_ROOT}/patching/hydra"

step "import smoke test"
LIDRA_SKIP_INIT=true python - <<'EOF'
import torch, xformers, spconv, kaolin, pytorch3d, gsplat, hydra, utils3d
import nvdiffrast.torch, diff_gaussian_rasterization
import sam3d_objects
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("E2E_ENV_OK")
EOF
echo "SETUP COMPLETE"
