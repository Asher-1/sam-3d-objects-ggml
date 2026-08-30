#!/bin/bash
# Create the sam3d inference python env with GPU torch.
# This environment runs the official PyTorch pipeline (reference) and the
# GGUF conversion scripts. It is NOT required for C++ inference itself.
set -e

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.1}
export PATH=$CUDA_HOME/bin:$PATH
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d .venv-sam3d ]; then
    python3 -m virtualenv .venv-sam3d
fi
source .venv-sam3d/bin/activate
pip install --upgrade pip wheel setuptools

# torch stack cu121
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# core inference deps (trimmed from requirements.txt: inference-relevant only)
pip install hydra-core==1.3.2 omegaconf loguru==0.7.2 safetensors timm==0.9.16 \
    opencv-python==4.9.0.80 "numpy<2" pillow seaborn matplotlib gradio==5.49.0 \
    open3d==0.18.0 trimesh xatlas pymeshfix pyrender \
    optree==0.14.1 einops ftfy utils3d huggingface-hub tqdm
pip install spconv-cu121==2.3.8
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html

# MoGe (depth model)
pip install "MoGe @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"

# pytorch3d from source (needs nvcc 12.1)
pip install "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47" --no-build-isolation

# gsplat from source
pip install "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7" --no-build-isolation

# install sam3d_objects package itself (deps already installed manually)
pip install -e . --no-deps

echo "=== ENV READY ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
