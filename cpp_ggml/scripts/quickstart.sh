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
  bash cpp_ggml/scripts/quickstart.sh <cpu|cuda|vulkan> <configure|build|test|e2e|native-e2e|matrix|full-e2e|verify-patch>

Environment:
  SAM3D_BUILD_JOBS   Build parallelism (default 6; capped at 6)
  SAM3D_PYTHON       Python with official SAM 3D Objects dependencies
  SAM3D_DTYPE        GGUF dtype for e2e (default q8_0)
  SAM3D_SEED         Sampling seed (default 42)
  SAM3D_IMAGE        Input image (default checked benchmark image)
  SAM3D_MASK_DIR     Input mask directory (default checked benchmark masks)
  SAM3D_MASK_INDEX   Mask index (default 14)
  SAM3D_WORK_DIR             New/empty output directory for E2E commands
  SAM3D_MODELS_DIR           GGUF directory (default cpp_ggml/models/gguf)
  SAM3D_MOGE_MODEL           Native MoGe GGUF (default models/gguf/moge_vitl-f16.gguf)
  SAM3D_NATIVE_PBR           Set to 1 to request --pbr-out (CUDA licensed build only)
  SAM3D_ACCEPT_PBR_LICENSES  Set to 1 to enable nvdiffrast non-commercial and MeshFix GPL code
  SAM3D_MEASURE_OFFICIAL_FULL_GLB  Set to 1 to record official hot full-GLB timing
  SAM3D_DIAGNOSTIC_NOISE_REPLAY    Set to 1 for non-release official-noise diagnosis
  SAM3D_OPERATOR_ORACLE            Set to 1 to capture official SS/SLat block boundaries
  SAM3D_OPERATOR_REFERENCE_WEIGHTS same-gguf (default) or checkpoint reference for the operator oracle
  SAM3D_OPERATOR_MAX_STAGE_MAE     Optional calibrated per-boundary MAE limit
  SAM3D_OPERATOR_MAX_STAGE_ABS     Optional calibrated per-boundary maximum absolute-error limit
  SAM3D_OPERATOR_TRAJECTORY        Set to 1 to compare all SS/SLat Euler states
  SAM3D_OPERATOR_TRAJECTORY_SS_BASELINE  Optional same-scope SS trajectory baseline report
  SAM3D_OPERATOR_TRAJECTORY_SLAT_MAX_TERMINAL_MAE  Optional calibrated SLat final-latent limit
  SAM3D_OPERATOR_ORACLE_VULKAN     Set to 1 to repeat the selected oracle on Vulkan
  SAM3D_MAX_GLB_RGB_MAE_LINEAR     Predeclared final-GLB linear RGB MAE budget
  SAM3D_MIN_GLB_MASK_IOU           Predeclared final-GLB silhouette IoU budget
  SAM3D_MAX_GLB_NORMAL_ANGLE_DEG   Predeclared final-GLB maximum normal-angle budget
  SAM3D_MAX_GLB_DEPTH_MAE_NDC      Predeclared final-GLB depth MAE budget
  SAM3D_MAX_TEXTURE_MAE_U8        Controlled atlas MAE budget (0..255 scale)
  SAM3D_MAX_TEXTURE_RMSE_U8       Controlled atlas RMSE budget
  SAM3D_MAX_TEXTURE_ABS_U8        Controlled atlas maximum-error budget
  SAM3D_MAX_NORMAL_ANGLE_DEG      Controlled matching-mesh normal-angle budget

bootstrap initializes submodules, replays/verifies the one ggml patch, and
installs the official reference environment when SAM3D_PYTHON is unset. `e2e`
is the frozen-condition diagnostic wrapper. `native-e2e` calls
`sam3d-cli image-to-3d` directly and does not load Python or Torch. `matrix`
measures raw native image input for CUDA/Vulkan. `full-e2e` creates a fresh
official oracle, validates CUDA PBR directly and through a Vulkan-inference to
CUDA-PBR artifact handoff, then runs the complete raw CUDA/Vulkan matrix; it
requires explicit PBR license acceptance.
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

native_pbr_build_dir() {
    printf '%s\n' "${CPP_ROOT}/build-cuda-pbr"
}

configure() {
    local backend="$1"
    local build_dir
    build_dir="$(build_dir_for "${backend}")"
    local flags=(-DCMAKE_BUILD_TYPE=Release -DSAM3D_GGML_CUDA=OFF -DSAM3D_GGML_VULKAN=OFF)
    if [[ -n "${SAM3D_PYTHON:-}" ]]; then flags+=("-DPython3_EXECUTABLE=$(official_python)"); fi
    case "${backend}" in
        cuda) flags+=(-DSAM3D_GGML_CUDA=ON) ;;
        vulkan) flags+=(-DSAM3D_GGML_VULKAN=ON) ;;
    esac
    cmake -S "${CPP_ROOT}" -B "${build_dir}" "${flags[@]}"
}

configure_native_pbr() {
    [[ "${SAM3D_ACCEPT_PBR_LICENSES:-0}" == "1" ]] || die \
        "native PBR requires SAM3D_ACCEPT_PBR_LICENSES=1 (nvdiffrast non-commercial and MeshFix GPL/commercial terms)"
    local -a python_args=()
    if [[ -n "${SAM3D_PYTHON:-}" ]]; then python_args+=("-DPython3_EXECUTABLE=$(official_python)"); fi
    cmake -S "${CPP_ROOT}" -B "$(native_pbr_build_dir)" -DCMAKE_BUILD_TYPE=Release "${python_args[@]}" \
        -DSAM3D_GGML_CUDA=ON -DSAM3D_GGML_VULKAN=OFF -DSAM3D_GGML_NATIVE_PBR=ON \
        -DSAM3D_GGML_NVDIFFRAST_NONCOMMERCIAL=ON -DSAM3D_GGML_MESHFIX_GPL=ON
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

run_native_e2e() {
    local backend="$1"
    local build_dir="${2:-$(build_dir_for "${backend}")}"
    local executable="${build_dir}/bin/sam3d-cli"
    [[ -x "${executable}" ]] || die "missing native executable: ${executable}"
    local work_dir="${SAM3D_WORK_DIR:-$(mktemp -d /tmp/sam3d-native-e2e-${backend}.XXXXXX)}"
    [[ ! -e "${work_dir}" || -z "$(find "${work_dir}" -mindepth 1 -print -quit)" ]] ||
        die "SAM3D_WORK_DIR must be new or empty: ${work_dir}"
    mkdir -p "${work_dir}"
    local models_dir="${SAM3D_MODELS_DIR:-${CPP_ROOT}/models/gguf}"
    local dtype="${SAM3D_DTYPE:-f16}"
    local image="${SAM3D_IMAGE:-${REPO_ROOT}/notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png}"
    local mask_dir="${SAM3D_MASK_DIR:-${REPO_ROOT}/notebook/images/shutterstock_stylish_kidsroom_1640806567}"
    local mask="${mask_dir}/${SAM3D_MASK_INDEX:-14}.png"
    local moge_model="${SAM3D_MOGE_MODEL:-${models_dir}/moge_vitl-f16.gguf}"
    local seed="${SAM3D_SEED:-42}"
    local rng_blocks="${SAM3D_PYTORCH_PHILOX_DISTRIBUTION_BLOCKS:-}"
    if [[ "${backend}" != "cuda" && -z "${rng_blocks}" ]]; then
        local cuda_rng_executable="$(build_dir_for cuda)/bin/sam3d-cli"
        [[ -x "${cuda_rng_executable}" ]] || die \
            "${backend} native-seed sampling needs build-cuda/bin/sam3d-cli to record the PyTorch Philox contract"
        local rng_dir="${work_dir}/pytorch_philox_contract"
        "${cuda_rng_executable}" rng-dump --implementation cuda --seed "${seed}" \
            --sizes 1 --out-dir "${rng_dir}" >/dev/null
        rng_blocks="$(sed -nE 's/.*"distribution_blocks":[[:space:]]*([0-9]+).*/\1/p' \
            "${rng_dir}/rng_contract.json")"
        [[ "${rng_blocks}" =~ ^[1-9][0-9]*$ ]] || die \
            "CUDA sampler did not record a valid PyTorch Philox distribution-block contract"
    fi
    local command=("${executable}" image-to-3d --model "${models_dir}"
        --moge-model "${moge_model}" --image "${image}" --mask "${mask}"
        --dtype "${dtype}" --backend "${backend}"
        --seed "${seed}" --out "${work_dir}/output.ply")
    if [[ "${SAM3D_NATIVE_PBR:-0}" == "1" ]]; then
        [[ "${backend}" == "cuda" ]] || die "SAM3D_NATIVE_PBR=1 requires the CUDA backend"
        command+=(--pbr-out "${work_dir}/output.glb")
    fi
    if [[ -n "${rng_blocks}" ]]; then
        SAM3D_PYTORCH_PHILOX_DISTRIBUTION_BLOCKS="${rng_blocks}" "${command[@]}"
    else
        "${command[@]}"
    fi
    echo "Native E2E artifacts: ${work_dir}"
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
        --moge-model "${SAM3D_MOGE_MODEL:-${CPP_ROOT}/models/gguf/moge_vitl-f16.gguf}" \
        --conditions-dir "${CPP_ROOT}/benchmarks/data/e2e" --work-dir "${work_dir}" \
        --output-report "${work_dir}/e2e_latency.json"
}

run_full_e2e() {
    [[ "$1" == "cuda" ]] || die "full-e2e uses CUDA for native PBR; invoke it as 'cuda full-e2e'"
    local py
    py="$(official_python)"
    local work_dir="${SAM3D_WORK_DIR:-$(mktemp -d /tmp/sam3d-full-e2e.XXXXXX)}"
    [[ ! -e "${work_dir}" || -z "$(find "${work_dir}" -mindepth 1 -print -quit)" ]] ||
        die "SAM3D_WORK_DIR must be new or empty: ${work_dir}"
    mkdir -p "${work_dir}"
    local -a full_timing_arg=()
    if [[ "${SAM3D_MEASURE_OFFICIAL_FULL_GLB:-0}" == "1" ]]; then
        full_timing_arg=(--measure-official-full-glb)
    fi
    local -a diagnostic_noise_arg=()
    if [[ "${SAM3D_DIAGNOSTIC_NOISE_REPLAY:-0}" == "1" ]]; then
        diagnostic_noise_arg=(--diagnostic-noise-replay)
    fi
    local -a operator_oracle_arg=()
    if [[ "${SAM3D_OPERATOR_ORACLE:-0}" == "1" ]]; then
        operator_oracle_arg=(--operator-oracle)
        case "${SAM3D_OPERATOR_REFERENCE_WEIGHTS:-same-gguf}" in
            same-gguf) operator_oracle_arg+=(--operator-oracle-reference-weight-scope same-gguf-dequantized) ;;
            checkpoint) operator_oracle_arg+=(--operator-oracle-reference-weight-scope official-checkpoint) ;;
            *) die "SAM3D_OPERATOR_REFERENCE_WEIGHTS must be same-gguf or checkpoint" ;;
        esac
        if [[ -n "${SAM3D_OPERATOR_MAX_STAGE_MAE:-}" ]]; then
            operator_oracle_arg+=(--operator-oracle-max-stage-mae "${SAM3D_OPERATOR_MAX_STAGE_MAE}")
        fi
        if [[ -n "${SAM3D_OPERATOR_MAX_STAGE_ABS:-}" ]]; then
            operator_oracle_arg+=(--operator-oracle-max-stage-abs "${SAM3D_OPERATOR_MAX_STAGE_ABS}")
        fi
        if [[ "${SAM3D_OPERATOR_TRAJECTORY:-0}" == "1" ]]; then
            operator_oracle_arg+=(--operator-oracle-trajectory)
        fi
        if [[ -n "${SAM3D_OPERATOR_TRAJECTORY_SS_BASELINE:-}" ]]; then
            operator_oracle_arg+=(--operator-oracle-trajectory-ss-baseline
                                  "${SAM3D_OPERATOR_TRAJECTORY_SS_BASELINE}")
        fi
        if [[ -n "${SAM3D_OPERATOR_TRAJECTORY_SLAT_MAX_TERMINAL_MAE:-}" ]]; then
            operator_oracle_arg+=(--operator-oracle-trajectory-slat-max-terminal-mae
                                  "${SAM3D_OPERATOR_TRAJECTORY_SLAT_MAX_TERMINAL_MAE}")
        fi
        if [[ "${SAM3D_OPERATOR_ORACLE_VULKAN:-0}" == "1" ]]; then
            operator_oracle_arg+=(--operator-oracle-vulkan)
        fi
    fi
    local -a quality_args=()
    if [[ "${SAM3D_SKIP_NEURAL_DIAGNOSTIC_MATRIX:-0}" == "1" ]]; then
        quality_args+=(--skip-neural-diagnostic-matrix)
    fi
    local setting option
    for setting in MAX_GLB_RGB_MAE_LINEAR MIN_GLB_MASK_IOU MAX_GLB_NORMAL_ANGLE_DEG \
                   MAX_GLB_DEPTH_MAE_NDC MAX_TEXTURE_MAE_U8 MAX_TEXTURE_RMSE_U8 \
                   MAX_TEXTURE_ABS_U8 MAX_NORMAL_ANGLE_DEG; do
        option="--$(tr '[:upper:]_' '[:lower:]-' <<< "$setting")"
        setting="SAM3D_${setting}"
        if [[ -n "${!setting:-}" ]]; then quality_args+=("$option" "${!setting}"); fi
    done
    local result=0
    "${py}" "${SCRIPT_DIR}/run_full_e2e_parity.py" \
        --official-python "${py}" \
        --cuda-binary "$(build_dir_for cuda)/bin/sam3d-cli" \
        --vulkan-binary "$(build_dir_for vulkan)/bin/sam3d-cli" \
        --native-pbr-binary "$(native_pbr_build_dir)/bin/sam3d-cli" \
        --models-dir "${SAM3D_MODELS_DIR:-${CPP_ROOT}/models/gguf}" \
        --moge-model "${SAM3D_MOGE_MODEL:-${CPP_ROOT}/models/gguf/moge_vitl-f16.gguf}" \
        --image "${SAM3D_IMAGE:-${REPO_ROOT}/notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png}" \
        --mask-dir "${SAM3D_MASK_DIR:-${REPO_ROOT}/notebook/images/shutterstock_stylish_kidsroom_1640806567}" \
        --mask-index "${SAM3D_MASK_INDEX:-14}" --work-dir "${work_dir}" --threads "${jobs}" \
        --seed "${SAM3D_SEED:-42}" \
        "${full_timing_arg[@]}" "${diagnostic_noise_arg[@]}" "${operator_oracle_arg[@]}" "${quality_args[@]}" || result=$?
    echo "Full E2E artifacts: ${work_dir}"
    return "$result"
}

[[ $# -ge 1 ]] || { usage; exit 2; }
if [[ "$1" == "bootstrap" ]]; then
    [[ $# -eq 1 ]] || { usage; exit 2; }
    git -C "${REPO_ROOT}" submodule update --init --recursive
    bash "${SCRIPT_DIR}/apply_ggml_patches.sh"
    if [[ -z "${SAM3D_PYTHON:-}" ]]; then
        bash "${SCRIPT_DIR}/setup_pt_ref_env.sh"
    fi
    # Weights are large downloads: point at the exact next commands instead of
    # failing deep inside a later e2e run.
    models_dir="${CPP_ROOT}/models/gguf"
    missing=""
    for stage in ss_generator ss_decoder slat_generator slat_decoder_gs slat_decoder_mesh; do
        [[ -s "${models_dir}/${stage}-f16.gguf" ]] || missing="${missing} ${stage}-f16.gguf"
    done
    [[ -s "${models_dir}/moge_vitl-f16.gguf" ]] || missing="${missing} moge_vitl-f16.gguf"
    if [[ -n "$missing" ]]; then
        echo "NOTE: model weights not found:${missing}"
        echo "  bash cpp_ggml/scripts/download_gguf.sh --dtype f16   # generative GGUF weights"
        echo "  bash cpp_ggml/scripts/prepare_moge_gguf.sh --dtype f16  # MoGe GGUF"
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
    e2e)
        if [[ "${backend}" != "cuda" ]]; then
            configure cuda
            cmake --build "$(build_dir_for cuda)" --parallel "${jobs}"
        fi
        configure "${backend}"; cmake --build "${build_dir}" --parallel "${jobs}"; run_e2e "${backend}" ;;
    native-e2e)
        if [[ "${SAM3D_NATIVE_PBR:-0}" == "1" ]]; then
            [[ "${backend}" == "cuda" ]] || die "SAM3D_NATIVE_PBR=1 requires the CUDA backend"
            configure_native_pbr
            cmake --build "$(native_pbr_build_dir)" --parallel "${jobs}"
            run_native_e2e "${backend}" "$(native_pbr_build_dir)"
        else
            if [[ "${backend}" != "cuda" ]]; then
                configure cuda
                cmake --build "$(build_dir_for cuda)" --parallel "${jobs}"
            fi
            configure "${backend}"
            cmake --build "${build_dir}" --parallel "${jobs}"
            run_native_e2e "${backend}"
        fi
        ;;
    matrix)
        [[ "${backend}" == "cuda" || "${backend}" == "vulkan" ]] ||
            die "matrix requires cuda or vulkan as the backend selector"
        configure cuda
        cmake --build "$(build_dir_for cuda)" --parallel "${jobs}"
        configure vulkan
        cmake --build "$(build_dir_for vulkan)" --parallel "${jobs}"
        run_matrix "${backend}"
        ;;
    full-e2e)
        [[ "${backend}" == "cuda" ]] || die "full-e2e requires cuda as the backend selector"
        configure cuda
        cmake --build "$(build_dir_for cuda)" --parallel "${jobs}"
        configure vulkan
        cmake --build "$(build_dir_for vulkan)" --parallel "${jobs}"
        configure_native_pbr
        cmake --build "$(native_pbr_build_dir)" --parallel "${jobs}"
        run_full_e2e "${backend}"
        ;;
    verify-patch) bash "${SCRIPT_DIR}/apply_ggml_patches.sh" ;;
    *) usage; exit 2 ;;
esac
