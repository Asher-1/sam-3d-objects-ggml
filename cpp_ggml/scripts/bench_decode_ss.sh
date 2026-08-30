#!/usr/bin/env bash
#
# bench_decode_ss.sh — run the SS-decoder diagnostic matrix and append one
# JSONL row per (backend, dtype, threads) config to benchmarks/diagnostics/.
#
# The matrix matches the protocols used by the sibling integrations
# (ultralytics-ggml / General-Keypoint-Detection-GGML): a fixed input, one
# warmup pass per config, then repeated timed graph computations.
#
# Usage:
#   bash cpp_ggml/scripts/bench_decode_ss.sh                 # full matrix
#   bash cpp_ggml/scripts/bench_decode_ss.sh --backends cuda # subset
#   bash cpp_ggml/scripts/bench_decode_ss.sh --fresh         # truncate output
#
# Rows carry load/build/alloc/upload times plus graph_ms mean/min/p50/max. This
# is a module diagnostic only; release charts are generated from the E2E report.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_DIR="${CPP_DIR}/benchmarks"
DIAGNOSTIC_DIR="${BENCH_DIR}/diagnostics/ss_decoder"
DATA_DIR="${BENCH_DIR}/data"
LATENT="${DATA_DIR}/ss_latent_synthetic.bin"
JSONL="${DIAGNOSTIC_DIR}/latency.jsonl"

CPU_BUILD="${CPP_DIR}/build-cpu"
CUDA_BUILD="${CPP_DIR}/build-cuda"
VULKAN_BUILD="${CPP_DIR}/build-vulkan"
DTYPES="f32 f16 q4_0 q8_0"
BACKENDS="cpu cuda"
ITERS_CPU=10
ITERS_GPU=30
WARMUP=2
THREADS_CPU="8 32"
FRESH=0
SKIP_OUTPUTS=0
ORIGINAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backends) BACKENDS="$2"; shift 2 ;;
        --dtypes)   DTYPES="$2";   shift 2 ;;
        --iters-cpu) ITERS_CPU="$2"; shift 2 ;;
        --iters-gpu) ITERS_GPU="$2"; shift 2 ;;
        --warmup)   WARMUP="$2";   shift 2 ;;
        --threads)  THREADS_CPU="$2"; shift 2 ;;
        --jsonl)    JSONL="$2";    shift 2 ;;
        --fresh)    FRESH=1;       shift ;;
        --skip-outputs) SKIP_OUTPUTS=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ "${FRESH}" == "1" && -f "${JSONL}" ]]; then
    rm "${JSONL}"
fi

# Deterministic synthetic input (the gated official checkpoints prevent
# re-dumping a pipeline latent; see gen_ss_latent.py).
if [[ ! -f "${LATENT}" ]]; then
    python3 "${SCRIPT_DIR}/gen_ss_latent.py" --out "${LATENT}"
fi

mkdir -p "$(dirname "${JSONL}")"
run_cfg() {  # cli iters jsonl
    local cli="$1" iters="$2"
    shift 2
    echo ">>> ${cli} $* (iters=${iters})"
    local build_lib
    build_lib="$(cd "$(dirname "${cli}")/../lib" && pwd)"
    LD_LIBRARY_PATH="${build_lib}:${ORIGINAL_LD_LIBRARY_PATH}" "${cli}" decode-ss --input "${LATENT}" \
        --warmup "${WARMUP}" --iters "${iters}" --json "${JSONL}" "$@" | tail -2
}

for backend in ${BACKENDS}; do
    if [[ "${backend}" == "cpu" ]]; then
        [[ -x "${CPU_BUILD}/bin/sam3d-cli" ]] || { echo "missing ${CPU_BUILD}/bin/sam3d-cli" >&2; exit 1; }
        for th in ${THREADS_CPU}; do
            for dt in ${DTYPES}; do
                run_cfg "${CPU_BUILD}/bin/sam3d-cli" "${ITERS_CPU}" \
                    --model "${CPP_DIR}/models/gguf/ss_decoder-${dt}.gguf" \
                    --backend cpu --threads "${th}"
            done
        done
    elif [[ "${backend}" == "cuda" || "${backend}" == "vulkan" ]]; then
        build_dir="${CUDA_BUILD}"
        [[ "${backend}" == "vulkan" ]] && build_dir="${VULKAN_BUILD}"
        [[ -x "${build_dir}/bin/sam3d-cli" ]] || { echo "missing ${build_dir}/bin/sam3d-cli (configure the ${backend} backend first)" >&2; exit 1; }
        for dt in ${DTYPES}; do
            run_cfg "${build_dir}/bin/sam3d-cli" "${ITERS_GPU}" \
                --model "${CPP_DIR}/models/gguf/ss_decoder-${dt}.gguf" \
                --backend "${backend}"
            done
    else
        echo "unsupported backend: ${backend}" >&2; exit 1
    fi
done

# Quality dumps: one occupancy output per dtype for the visual parity
# charts (cpu f32 = parity reference, cuda f16 = GPU deployment format,
# cpu q4_0/q8_0 = compact formats). --skip-outputs omits this pass.
if [[ "${SKIP_OUTPUTS}" != "1" ]]; then
    for cfg in "cpu 8 f32" "cuda 8 f16" "cuda 8 q4_0" "cpu 8 q8_0"; do
        set -- ${cfg}
        local_backend=$1; local_th=$2; local_dt=$3
        if [[ "${local_backend}" == "cuda" && ! -x "${CUDA_BUILD}/bin/sam3d-cli" ]]; then
            continue
        fi
        if [[ "${local_backend}" == "vulkan" && ! -x "${VULKAN_BUILD}/bin/sam3d-cli" ]]; then
            continue
        fi
        cli_bin="${CPU_BUILD}/bin/sam3d-cli"
        [[ "${local_backend}" == "cuda" ]] && cli_bin="${CUDA_BUILD}/bin/sam3d-cli"
        [[ "${local_backend}" == "vulkan" ]] && cli_bin="${VULKAN_BUILD}/bin/sam3d-cli"
        echo ">>> quality dump ${local_backend} ${local_dt}"
        build_lib="$(cd "$(dirname "${cli_bin}")/../lib" && pwd)"
        LD_LIBRARY_PATH="${build_lib}:${ORIGINAL_LD_LIBRARY_PATH}" "${cli_bin}" decode-ss \
            --model "${CPP_DIR}/models/gguf/ss_decoder-${local_dt}.gguf" \
            --input "${LATENT}" --backend "${local_backend}" --threads "${local_th}" \
            --out "${DATA_DIR}/ss_occ_${local_dt}.samt" | tail -2
    done
fi

echo "done -> ${JSONL} ($(wc -l < "${JSONL}") rows)"
