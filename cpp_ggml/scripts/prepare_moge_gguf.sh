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
FORCE=0

usage() {
    cat <<'EOF'
usage: prepare_moge_gguf.sh [--checkpoint PATH] [--dtype f32|f16|q8_0|q4_0|q4_1|q4_k] [--output DIR] [--force]

Downloads Ruicheng/moge-vitl:model.pt when --checkpoint is omitted, then
converts it into moge_vitl-<dtype>.gguf. The C++ runtime has no Python/Torch
dependency after this conversion completes.
Use SAM3D_PYTHON to select the conversion environment. Existing GGUF files
are preserved; use --force only when intentionally replacing a baseline.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint|--dtype|--output)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || {
                echo "error: $1 needs a value" >&2; exit 2;
            } ;;
    esac
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
case "$DTYPE" in
    f32|f16|q8_0|q4_0|q4_1|q4_k) ;;
    *) echo "error: unsupported dtype: $DTYPE" >&2; exit 2 ;;
esac
OUTPUT="${OUTPUT_DIR}/moge_vitl-${DTYPE}.gguf"
if [[ -e "$OUTPUT" && "$FORCE" == 0 ]]; then
    [[ -s "$OUTPUT" && "$(head -c 4 "$OUTPUT")" == GGUF ]] || {
        echo "error: existing output has an invalid GGUF header; preserved: $OUTPUT" >&2; exit 1;
    }
    echo "preserve existing model (not reconverted or numerically revalidated): $OUTPUT"
    exit 0
fi

if [[ -z "${CHECKPOINT}" ]]; then
    CHECKPOINT="$("$PYTHON_BIN" -c 'from huggingface_hub import hf_hub_download; print(hf_hub_download("Ruicheng/moge-vitl", "model.pt"))')"
fi
[[ -s "$CHECKPOINT" ]] || { echo "error: missing/empty checkpoint: $CHECKPOINT" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}"
CONVERSION_DIR="$(mktemp -d "${OUTPUT_DIR}/.moge-conversion.XXXXXX")"
trap 'rm -f "${CONVERSION_DIR}/moge_vitl-${DTYPE}.gguf"; rmdir "$CONVERSION_DIR"' EXIT
"${PYTHON_BIN}" "${SCRIPT_DIR}/convert_sam3d_to_gguf.py" \
    --checkpoint-dir "${REPO_ROOT}/checkpoints/hf" \
    --model moge_vitl --moge-checkpoint "${CHECKPOINT}" \
    --dtype "${DTYPE}" --output "${CONVERSION_DIR}"
[[ "$(head -c 4 "${CONVERSION_DIR}/moge_vitl-${DTYPE}.gguf")" == GGUF ]] || {
    echo "error: conversion did not produce a GGUF model" >&2; exit 1;
}
# Verify the fresh GGUF against the requested checkpoint before publishing it.
"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_moge_gguf.py" \
    --model "${CONVERSION_DIR}/moge_vitl-${DTYPE}.gguf" \
    --checkpoint "${CHECKPOINT}"
mv "${CONVERSION_DIR}/moge_vitl-${DTYPE}.gguf" "$OUTPUT"
echo "prepared: $OUTPUT"
