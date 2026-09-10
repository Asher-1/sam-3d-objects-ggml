#!/usr/bin/env bash
# Shared root launcher. Native reconstruction itself runs only C++ binaries.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CPP="${ROOT}/cpp_ggml"
runner="$1"
shift

usage() {
    cat <<'EOF'
Usage: bash run_python.sh [options]
       bash run_ggml.sh [options]

  --image PATH          Input RGB/RGBA image (default: repository kidsroom image)
  --mask PATH           Binary object mask (default: kidsroom mask 14)
  --out-dir PATH        New or empty output directory (default: output/<runner>-<timestamp>)
  --seed N              Random seed, 0..2147483647 (default: 42)
  --threads N           CPU/build threads, 1..6 (default: 6)
  --backend cuda|vulkan GGML inference backend (default: cuda); PBR always uses CUDA
  --dtype f16|q8_0|q4_0 GGUF precision (default: q8_0)
  --models-dir PATH     GGUF directory (default: cpp_ggml/models/gguf)
  --moge-model PATH     MoGe GGUF (default: <models-dir>/moge_vitl-f16.gguf)
  --python PATH         Official Python interpreter; also accepts SAM3D_PYTHON
  --config PATH         Official pipeline YAML (default: checkpoints/hf/pipeline.yaml)
  --skip-build          Use existing GGML binaries
  --setup               Install the official reference environment first (Python only)
  --accept-pbr-licenses Enable existing nvdiffrast/MeshFix dependencies for native builds
  -h, --help            Show this help

Both paths run full mesh cleanup, UV, 100-view observations, 2500-step texture
baking and GLB export. They also save the base-color PNG and pose JSON.
GGML defaults to strict SS attention, matching the raw regression settings.
The Python path streams the official modules with the documented mixed
precision policy to fit a 12 GiB GPU. Checkpoints and GGUF models must exist.
EOF
}
die() { echo "error: $*" >&2; exit 2; }
image="${SAM3D_IMAGE:-${ROOT}/notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png}"
mask="${SAM3D_MASK:-${ROOT}/notebook/images/shutterstock_stylish_kidsroom_1640806567/14.png}"
out="${SAM3D_WORK_DIR:-}"
seed="${SAM3D_SEED:-42}"
threads="${SAM3D_BUILD_JOBS:-6}"
backend=cuda
dtype="${SAM3D_DTYPE:-q8_0}"
models="${SAM3D_MODELS_DIR:-${CPP}/models/gguf}"
moge="${SAM3D_MOGE_MODEL:-}"
python="${SAM3D_PYTHON:-}"
config="${ROOT}/checkpoints/hf/pipeline.yaml"
skip_build=0
setup=0
licenses="${SAM3D_ACCEPT_PBR_LICENSES:-0}"
while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --skip-build) skip_build=1; shift ;;
        --setup) setup=1; shift ;;
        --accept-pbr-licenses) licenses=1; shift ;;
        --image|--mask|--out-dir|--seed|--threads|--backend|--dtype|--models-dir|--moge-model|--python|--config)
            (( $# >= 2 )) || die "missing value for $1"
            [[ -n "$2" && "$2" != --* ]] || die "missing value for $1"
            case "$1" in
                --image) image="$2" ;; --mask) mask="$2" ;; --out-dir) out="$2" ;;
                --seed) seed="$2" ;; --threads) threads="$2" ;; --backend) backend="$2" ;;
                --dtype) dtype="$2" ;; --models-dir) models="$2" ;; --moge-model) moge="$2" ;;
                --python) python="$2" ;; --config) config="$2" ;;
            esac
            shift 2 ;;
        *) die "unknown option: $1" ;;
    esac
done
[[ "$seed" =~ ^[0-9]+$ ]] || die "seed must be a non-negative integer"
[[ ${#seed} -le 10 ]] && (( 10#$seed <= 2147483647 )) || die "seed exceeds the native signed 32-bit CLI range"
seed="$((10#$seed))"
[[ "$threads" =~ ^[1-6]$ ]] || die "threads must be in 1..6"
[[ "$backend" == cuda || "$backend" == vulkan ]] || die "backend must be cuda or vulkan"
[[ "$dtype" == f16 || "$dtype" == q8_0 || "$dtype" == q4_0 ]] || die "dtype must be f16, q8_0 or q4_0"
[[ -f "$image" && -f "$mask" ]] || die "image or mask is missing"
# Resolve user-relative paths before the inference process changes directory.
image="$(realpath -- "$image")"
mask="$(realpath -- "$mask")"
out="$(realpath -m -- "${out:-${ROOT}/output/${runner}-${backend}-$(date +%Y%m%d-%H%M%S)}")"
[[ ! -e "$out" || ( -d "$out" && -z "$(find "$out" -mindepth 1 -print -quit)" ) ]] ||
    die "output directory must be new or empty: $out"

if [[ "$runner" == python ]]; then
    [[ -f "$config" ]] || die "missing checkpoint configuration: $config"
    config="$(realpath -- "$config")"
    if (( setup )); then bash "${CPP}/scripts/setup_pt_ref_env.sh"; fi
    conda_executable="${SAM3D_CONDA_EXE:-$(command -v conda || true)}"
    if [[ -z "$python" && -n "$conda_executable" ]]; then
        python="$("$conda_executable" info --base)/envs/sam3d-objects/bin/python"
    fi
    if [[ -z "$python" && -n "${CONDA_PREFIX:-}" ]]; then python="${CONDA_PREFIX}/bin/python"; fi
    [[ -x "$python" ]] || die "set --python or SAM3D_PYTHON to the official environment; use --setup to install it"
    exec "$python" "${CPP}/scripts/bench_full_e2e_pipeline.py" \
        --image "$image" --mask "$mask" --config "$config" --out-dir "$out" \
        --seed "$seed" --warmup 0 --iters 1 --stream-weights
fi
[[ "$runner" == ggml ]] || die "unsupported runner: $runner"
(( setup == 0 )) || die "--setup applies only to the Python environment"
models="$(realpath -m -- "$models")"
moge="$(realpath -m -- "${moge:-${models}/moge_vitl-f16.gguf}")"
for stage in ss_generator ss_decoder slat_generator slat_decoder_gs slat_decoder_mesh; do
    [[ -s "${models}/${stage}-${dtype}.gguf" ]] || die "missing GGUF: ${models}/${stage}-${dtype}.gguf"
done
[[ -s "$moge" ]] || die "missing MoGe GGUF: $moge"
cuda_binary="${CPP}/build-cuda-pbr/bin/sam3d-cli"
vulkan_binary="${CPP}/build-vulkan/bin/sam3d-cli"
if (( ! skip_build )); then
    [[ "$licenses" == 1 ]] || die "build requires --accept-pbr-licenses (nvdiffrast non-commercial, MeshFix GPL/commercial)"
    cmake -S "$CPP" -B "${CPP}/build-cuda-pbr" -DCMAKE_BUILD_TYPE=Release \
        -DSAM3D_GGML_CUDA=ON -DSAM3D_GGML_VULKAN=OFF -DSAM3D_GGML_NATIVE_PBR=ON \
        -DSAM3D_GGML_NVDIFFRAST_NONCOMMERCIAL=ON -DSAM3D_GGML_MESHFIX_GPL=ON
    cmake --build "${CPP}/build-cuda-pbr" --target sam3d-cli --parallel "$threads"
    if [[ "$backend" == vulkan ]]; then
        cmake -S "$CPP" -B "${CPP}/build-vulkan" -DCMAKE_BUILD_TYPE=Release \
            -DSAM3D_GGML_CUDA=OFF -DSAM3D_GGML_VULKAN=ON -DSAM3D_GGML_NATIVE_PBR=OFF \
            -DSAM3D_GGML_NVDIFFRAST_NONCOMMERCIAL=OFF -DSAM3D_GGML_MESHFIX_GPL=OFF
        cmake --build "${CPP}/build-vulkan" --target sam3d-cli --parallel "$threads"
    fi
fi
[[ -x "$cuda_binary" ]] || die "missing CUDA PBR executable: $cuda_binary"
[[ "$backend" != vulkan || -x "$vulkan_binary" ]] || die "missing Vulkan executable: $vulkan_binary"
mkdir -p -- "$out"
exec > >(tee "${out}/run.log") 2>&1
trap 'echo "Reconstruction failed at line ${LINENO}; partial artifacts: ${out}" >&2' ERR
command=(image-to-3d --model "$models" --moge-model "$moge" --image "$image" --mask "$mask"
    --backend "$backend" --dtype "$dtype" --seed "$seed" --threads "$threads"
    --ss-attention strict --out "${out}/output.ply" --pose-out "${out}/pose.json"
    --dtype-contract-out "${out}/dtype_contract.json")
if [[ "$backend" == cuda ]]; then
    "$cuda_binary" "${command[@]}" --pbr-out "${out}/output.glb"
else
    "$cuda_binary" rng-dump --implementation cuda --seed "$seed" --sizes 1 \
        --out-dir "${out}/rng_contract"
    blocks="$(sed -nE 's/.*"distribution_blocks":[[:space:]]*([0-9]+).*/\1/p' "${out}/rng_contract/rng_contract.json")"
    [[ "$blocks" =~ ^[1-9][0-9]*$ ]] || die "invalid CUDA Philox contract"
    SAM3D_PYTORCH_PHILOX_DISTRIBUTION_BLOCKS="$blocks" "$vulkan_binary" "${command[@]}" \
        --mesh-vertices-out "${out}/vertices.samt" --mesh-faces-out "${out}/faces.samt"
    (cd "$out" && sha256sum output.ply vertices.samt faces.samt pose.json dtype_contract.json > handoff.sha256)
    "$cuda_binary" pbr-assemble --vertices "${out}/vertices.samt" --faces "${out}/faces.samt" \
        --ply "${out}/output.ply" --out "${out}/output.glb" --seed "$seed"
    (cd "$out" && sha256sum --check handoff.sha256)
fi
[[ -s "${out}/output.glb" && -s "${out}/output.base_color.png" ]] || die "missing GLB or base-color texture"
echo "Textured model: ${out}/output.glb"
echo "Base-color texture: ${out}/output.base_color.png"
