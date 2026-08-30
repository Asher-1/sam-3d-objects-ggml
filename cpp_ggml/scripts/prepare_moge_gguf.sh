#!/usr/bin/env bash
# Download and convert the official MoGe checkpoint for the native C++ path.
# This is an offline conversion utility; sam3d-cli reads only its GGUF output.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CPP_ROOT}/.." && pwd)"
PYTHON_BIN="${SAM3D_PYTHON:-python3}"
OUTPUT_DIR="${CPP_ROOT}/models/gguf"
DTYPE="f16"
CHECKPOINT="${SAM3D_MOGE_CHECKPOINT:-}"

usage() {
    cat <<'EOF'
usage: prepare_moge_gguf.sh [--checkpoint PATH] [--dtype f32|f16|q8_0|q4_0|q4_1|q4_k] [--output DIR]

Downloads Ruicheng/moge-vitl:model.pt when --checkpoint is omitted, then
converts it into moge_vitl-<dtype>.gguf. The C++ runtime has no Python/Torch
dependency after this conversion completes.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${CHECKPOINT}" ]]; then
    if ! command -v huggingface-cli >/dev/null 2>&1; then
        echo "error: huggingface-cli is required when --checkpoint is omitted" >&2
        exit 1
    fi
    CACHE_DIR="${HF_HOME:-${HOME}/.cache/huggingface}/hub"
    huggingface-cli download Ruicheng/moge-vitl model.pt >/dev/null
    CHECKPOINT="${CACHE_DIR}/models--Ruicheng--moge-vitl/snapshots/$(cat "${CACHE_DIR}/models--Ruicheng--moge-vitl/refs/main")/model.pt"
fi

mkdir -p "${OUTPUT_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/convert_sam3d_to_gguf.py" \
    --checkpoint-dir "${REPO_ROOT}/checkpoints/hf" \
    --model moge_vitl --moge-checkpoint "${CHECKPOINT}" \
    --dtype "${DTYPE}" --output "${OUTPUT_DIR}"
