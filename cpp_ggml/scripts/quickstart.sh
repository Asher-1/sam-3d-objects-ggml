#!/usr/bin/env bash
# One entry point for a clean clone. It keeps measured E2E artifacts in a
# caller-selected work directory rather than overwriting checked-in evidence.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CPP_ROOT}/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash cpp_ggml/scripts/quickstart.sh bootstrap
  bash cpp_ggml/scripts/quickstart.sh <cpu|cuda|vulkan> <configure|build|test|e2e|matrix|verify-patch>

Environment:
  SAM3D_BUILD_JOBS   Build parallelism (default 6; capped at 6)
  SAM3D_PYTHON       Python with official SAM 3D Objects dependencies
  SAM3D_DTYPE        GGUF dtype for e2e (default q8_0)
  SAM3D_IMAGE        Input image (default checked benchmark image)
  SAM3D_MASK_DIR     Input mask directory (default checked benchmark masks)
  SAM3D_MASK_INDEX   Mask index (default 14)
  SAM3D_WORK_DIR     New/empty output directory for e2e or matrix
  SAM3D_MODELS_DIR   GGUF directory (default cpp_ggml/models/gguf)

bootstrap initializes submodules, replays/verifies the one ggml patch, and
installs the official reference environment when SAM3D_PYTHON is unset. e2e
requires an exclusive GPU for CUDA/Vulkan timing. matrix configures and builds
both CUDA and Vulkan, then requires an exclusive GPU for the complete matrix.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

jobs="${SAM3D_BUILD_JOBS:-6}"
[[ "${jobs}" =~ ^[1-9][0-9]*$ ]] || die "SAM3D_BUILD_JOBS must be a positive integer"
if (( jobs > 6 )); then
    echo "limiting SAM3D_BUILD_JOBS=${jobs} to 6 for reproducible local builds" >&2
    jobs=6
fi

build_dir_for() {
    case "$1" in
        cpu) echo "${CPP_ROOT}/build-cpu" ;;
        cuda) echo "${CPP_ROOT}/build-cuda" ;;
        vulkan) echo "${CPP_ROOT}/build-vulkan" ;;
        *) die "unsupported backend: $1" ;;
    esac
}

configure() {
    local backend="$1"
    local build_dir
    build_dir="$(build_dir_for "${backend}")"
    local flags=(-DCMAKE_BUILD_TYPE=Release)
    case "${backend}" in
        cuda) flags+=(-DSAM3D_GGML_CUDA=ON) ;;
        vulkan) flags+=(-DSAM3D_GGML_VULKAN=ON) ;;
    esac
    cmake -S "${CPP_ROOT}" -B "${build_dir}" "${flags[@]}"
}

official_python() {
    if [[ -n "${SAM3D_PYTHON:-}" ]]; then
        [[ -x "${SAM3D_PYTHON}" ]] || die "SAM3D_PYTHON is not executable: ${SAM3D_PYTHON}"
        printf '%s\n' "${SAM3D_PYTHON}"
        return
    fi
    local candidate="${CONDA_PREFIX:-}/bin/python"
    if [[ -n "${CONDA_PREFIX:-}" && -x "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return
    fi
    die "set SAM3D_PYTHON or run 'quickstart.sh bootstrap' first"
}

run_e2e() {
    local backend="$1"
    local py
    py="$(official_python)"
    local work_dir="${SAM3D_WORK_DIR:-$(mktemp -d /tmp/sam3d-e2e-${backend}.XXXXXX)}"
    [[ ! -e "${work_dir}" || -z "$(find "${work_dir}" -mindepth 1 -print -quit)" ]] ||
        die "SAM3D_WORK_DIR must be new or empty: ${work_dir}"
    mkdir -p "${work_dir}"
    "${py}" "${SCRIPT_DIR}/run_image_to_3d.py" \
        --image "${SAM3D_IMAGE:-${REPO_ROOT}/notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png}" \
        --mask-dir "${SAM3D_MASK_DIR:-${REPO_ROOT}/notebook/images/shutterstock_stylish_kidsroom_1640806567}" \
        --mask-index "${SAM3D_MASK_INDEX:-14}" --backend "${backend}" \
        --dtype "${SAM3D_DTYPE:-q8_0}" --models-dir "${SAM3D_MODELS_DIR:-${CPP_ROOT}/models/gguf}" \
        --conditions-dir "${CPP_ROOT}/benchmarks/data/e2e" --replay-noise \
        --require-exclusive-gpu --render-frames 60 --render-resolution 512 \
        --out-dir "${work_dir}"
    echo "E2E artifacts: ${work_dir}"
}

run_matrix() {
    local backend="$1"
    [[ "${backend}" == "cuda" || "${backend}" == "vulkan" ]] ||
        die "matrix is a CUDA/Vulkan release matrix; choose cuda or vulkan"
    local py
    py="$(official_python)"
    local work_dir="${SAM3D_WORK_DIR:-$(mktemp -d /tmp/sam3d-e2e-matrix.XXXXXX)}"
    [[ ! -e "${work_dir}" || -z "$(find "${work_dir}" -mindepth 1 -print -quit)" ]] ||
        die "SAM3D_WORK_DIR must be new or empty: ${work_dir}"
    mkdir -p "${work_dir}"
    "${py}" "${SCRIPT_DIR}/run_e2e_matrix.py" \
        --official-python "${py}" \
        --image "${SAM3D_IMAGE:-${REPO_ROOT}/notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png}" \
        --mask-dir "${SAM3D_MASK_DIR:-${REPO_ROOT}/notebook/images/shutterstock_stylish_kidsroom_1640806567}" \
        --mask-index "${SAM3D_MASK_INDEX:-14}" \
        --models-dir "${SAM3D_MODELS_DIR:-${CPP_ROOT}/models/gguf}" \
        --conditions-dir "${CPP_ROOT}/benchmarks/data/e2e" --work-dir "${work_dir}" \
        --output-report "${work_dir}/e2e_latency.json"
}

[[ $# -ge 1 ]] || { usage; exit 2; }
if [[ "$1" == "bootstrap" ]]; then
    [[ $# -eq 1 ]] || { usage; exit 2; }
    git -C "${REPO_ROOT}" submodule update --init --recursive
    bash "${SCRIPT_DIR}/apply_ggml_patches.sh"
    if [[ -z "${SAM3D_PYTHON:-}" ]]; then
        bash "${SCRIPT_DIR}/setup_pt_ref_env.sh"
    fi
    echo "bootstrap complete"
    exit 0
fi

[[ $# -eq 2 ]] || { usage; exit 2; }
backend="$1"
action="$2"
build_dir="$(build_dir_for "${backend}")"
case "${action}" in
    configure) configure "${backend}" ;;
    build) configure "${backend}"; cmake --build "${build_dir}" --parallel "${jobs}" ;;
    test) configure "${backend}"; cmake --build "${build_dir}" --parallel "${jobs}"; ctest --test-dir "${build_dir}" --output-on-failure ;;
    e2e) configure "${backend}"; cmake --build "${build_dir}" --parallel "${jobs}"; run_e2e "${backend}" ;;
    matrix)
        [[ "${backend}" == "cuda" || "${backend}" == "vulkan" ]] ||
            die "matrix requires cuda or vulkan as the backend selector"
        configure cuda
        cmake --build "$(build_dir_for cuda)" --parallel "${jobs}"
        configure vulkan
        cmake --build "$(build_dir_for vulkan)" --parallel "${jobs}"
        run_matrix "${backend}"
        ;;
    verify-patch) bash "${SCRIPT_DIR}/apply_ggml_patches.sh" ;;
    *) usage; exit 2 ;;
esac
