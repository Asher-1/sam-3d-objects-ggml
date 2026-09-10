# Setup

## Prerequisites

* A linux 64-bits architecture (i.e. `linux-64` platform in `mamba info`).
* A NVIDIA GPU with at least 32 Gb of VRAM.

These are the upstream resident-pipeline prerequisites. The repository's
[streamed Python and native launchers](../cpp_ggml/README.md) have separately
recorded complete GLB runs on a 12 GiB RTX 3060; they use the documented
mixed-precision/residency policy, not an unchanged resident run.

## 1. Setup Python Environment

The following will install the default environment. If you use `conda` instead of `mamba`, replace its name in the first two lines. Note that you may have to build the environment on a compute node with GPU (e.g., you may get a `RuntimeError: Not compiled with GPU support` error when running certain parts of the code that use Pytorch3D).

```bash
# create sam3d-objects environment
mamba env create -f environments/default.yml
mamba activate sam3d-objects

# for pytorch/cuda dependencies
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"

# install sam3d-objects and core dependencies
pip install -e '.[dev]'
pip install -e '.[p3d]' # pytorch3d dependency on pytorch is broken, this 2-step approach solves it

# for inference
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'

# patch things that aren't yet in official pip packages
./patching/hydra # https://github.com/facebookresearch/hydra/pull/2863
```

## 2. Getting Checkpoints

### From HuggingFace

⚠️ Before using SAM 3D Objects, please request access to the checkpoints on the SAM 3D Objects
Hugging Face [repo](https://huggingface.co/facebook/sam-3d-objects). Once accepted, you
need to be authenticated to download the checkpoints. You can do this by running
the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
(e.g. `hf auth login` after generating an access token).

⚠️ SAM 3D Objects is available via HuggingFace globally, **except** in comprehensively sanctioned jurisdictions.
Sanctioned jurisdiction will result in requests being **rejected**.

```bash
pip install 'huggingface-hub[cli]<1.0'

TAG=hf
hf download \
  --repo-type model \
  --local-dir checkpoints/${TAG}-download \
  --max-workers 1 \
  facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
```

### GGUF weights (C++ inference, optional)

The `cpp_ggml` C++ engine does not need the PyTorch checkpoints above; it loads
pre-converted GGUF weights hosted at
[Asher-1/SAM_3D_OBJECTS_GGUF](https://huggingface.co/Asher-1/SAM_3D_OBJECTS_GGUF)
(F32/F16/Q8_0/Q4 variants; full GLB generation uses five generative stages
plus MoGe, about 6.32 GiB for F16 or 3.64 GiB for Q8_0 with MoGe F16):

```bash
bash cpp_ggml/scripts/download_gguf.sh --dtype q8_0
SAM3D_PYTHON=/path/to/sam3d-objects/bin/python \
  bash cpp_ggml/scripts/prepare_moge_gguf.sh
```

See [`cpp_ggml/models/MODEL_CARD.md`](../cpp_ggml/models/MODEL_CARD.md) for the
file manifest and quantization notes.

For image/mask-to-textured-GLB launchers, system prerequisites and licensed
native PBR dependencies, follow [the C++ deployment guide](../cpp_ggml/README.md).
From the repository root, `bash run_python.sh --help` and
`bash run_ggml.sh --help` list the complete reconstruction options.
