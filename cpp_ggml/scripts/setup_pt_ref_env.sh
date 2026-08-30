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
CONDA_BASE="$(${CONDA_EXE} info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate sam3d-objects

# CUDA 12.1 toolchain in the env (nvcc + dev headers incl. cublas/cusparse,
# which torch's CUDAContext headers require) — needed to compile pytorch3d /
# gsplat. Idempotent: skips packages that are already present.
if ! command -v nvcc >/dev/null 2>&1; then
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
python -c "import torch; assert torch.__version__.startswith('2.5.1')" 2>/dev/null || \
    pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url "${PYTORCH_INDEX}"

step "xformers 0.0.28.post3 (torch 2.5.1 build)"
python -c "import xformers" 2>/dev/null || \
    pip install xformers==0.0.28.post3 --index-url "${PYTORCH_INDEX}"

step "spconv-cu121 + kaolin 0.17.0"
python -c "import spconv" 2>/dev/null || pip install spconv-cu121==2.3.8
python -c "import kaolin" 2>/dev/null || pip install kaolin==0.17.0 --find-links "${KAOLIN_LINKS}"

step "light python deps"
pip install hydra-core==1.3.2 omegaconf opencv-python easydict gradio imageio \
    loguru pillow plotly plyfile pymeshfix pyvista safetensors scipy seaborn \
    open3d optree astor timm lightning utils3d einops ninja igraph fvcore \
    xatlas roma trimesh point-cloud-utils
# kaolin 0.17.0 ships Cython extensions built against numpy 1.x — keep numpy<2
pip install "numpy==1.26.4"

step "MoGe (microsoft/MoGe @a8c3734)"
python -c "import moge" 2>/dev/null || \
    pip install "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"

# CUDA extensions are compiled with the env's nvcc 12.1 for sm_86 (RTX 3060)
export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS="${MAX_JOBS:-24}"
export FORCE_CUDA=1

step "pytorch3d @75ebeeae (compiled, ~10-20 min)"
python -c "import pytorch3d" 2>/dev/null || {
    rm -rf /tmp/pytorch3d-src
    git clone https://github.com/facebookresearch/pytorch3d.git /tmp/pytorch3d-src
    git -C /tmp/pytorch3d-src checkout 75ebeeaea0908c5527e7b1e305fbc7681382db47
    pip install --no-build-isolation /tmp/pytorch3d-src
}

step "gsplat @2323de59 (compiled, ~5-10 min)"
python -c "import gsplat" 2>/dev/null || \
    pip install --no-build-isolation \
        "git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"

step "sam3d_objects (metadata only, --no-deps)"
pip install --no-deps -e "${REPO_ROOT}"

step "hydra 1.3.2 patch (facebookresearch/hydra#2863 workaround)"
python "${REPO_ROOT}/patching/hydra"

step "import smoke test"
python - <<'EOF'
import torch, xformers, spconv, kaolin, pytorch3d, gsplat, hydra, utils3d
import sam3d_objects
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("E2E_ENV_OK")
EOF
echo "SETUP COMPLETE"
