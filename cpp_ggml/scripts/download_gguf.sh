#!/usr/bin/env bash
#
# download_gguf.sh — fetch pre-converted SAM 3D Objects GGUF weights from
# https://huggingface.co/Asher-1/SAM_3D_OBJECTS_GGUF into cpp_ggml/models/gguf.
#
# Usage:
#   bash cpp_ggml/scripts/download_gguf.sh                       # recommended f16 set
#   bash cpp_ggml/scripts/download_gguf.sh --dtype f32           # whole f32 set
#   bash cpp_ggml/scripts/download_gguf.sh --dtype q4_0         # whole q4_0 set
#   bash cpp_ggml/scripts/download_gguf.sh --dtype q4_k         # whole q4_k set
#   bash cpp_ggml/scripts/download_gguf.sh --model ss_decoder    # f16 of one model
#   bash cpp_ggml/scripts/download_gguf.sh --model ss_decoder --dtype f16
#   bash cpp_ggml/scripts/download_gguf.sh --out /some/dir       # custom target
#   bash cpp_ggml/scripts/download_gguf.sh --force               # replace a different local revision
#
# Requires curl. Files already present (byte size matches the manifest) are
# skipped, so the script is safe to re-run. A same-name file with a different
# size is a different model revision and is never replaced without --force.

set -euo pipefail

REPO="Asher-1/SAM_3D_OBJECTS_GGUF"
BASE="https://huggingface.co/${REPO}/resolve/main"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${SCRIPT_DIR}/../models/gguf"

MODEL=""
DTYPE="f16"
FORCE=0
usage() {
    cat <<'EOF'
Usage: download_gguf.sh [--dtype TYPE] [--model NAME] [--out DIR] [--force]

TYPE: f16 (default), f32, q8_0, q4_0, q4_1, q4_k.
NAME: ss_generator, ss_decoder, slat_generator, slat_decoder_gs,
      slat_decoder_gs_4, slat_decoder_mesh. Omit to fetch all six.
Existing files from a different revision are preserved unless --force is used.
Downloads are checked against the published byte sizes and GGUF header.
MoGe is separate: run scripts/prepare_moge_gguf.sh for full raw inference.
EOF
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|--dtype|--out)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || {
                echo "error: $1 needs a value" >&2; exit 2;
            } ;;
    esac
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --out)   OUT="$2";   shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done
case "$DTYPE" in
    f16|f32|q8_0|q4_0|q4_1|q4_k) ;;
    *) echo "error: unsupported dtype: $DTYPE" >&2; exit 2 ;;
esac
case "$MODEL" in
    ""|ss_generator|ss_decoder|slat_generator|slat_decoder_gs|slat_decoder_gs_4|slat_decoder_mesh) ;;
    *) echo "error: unknown model: $MODEL" >&2; exit 2 ;;
esac
command -v curl >/dev/null || { echo "error: curl is required" >&2; exit 1; }

# model | f16 bytes | f32 bytes | q8_0 bytes | q4_0 bytes | q4_1 bytes | q4_k bytes
MANIFEST=(
    "ss_generator|3200717760|6394703808|1704687360|906804480|1006539840|917838720"
    "ss_decoder|295934016|443244288|226882336|190054784|194658208|190056000"
    "slat_generator|2455619616|4906037280|1308116256|696114464|772614688|708061376"
    "slat_decoder_gs|170914240|341484160|90965440|48326080|53656000|48326080"
    "slat_decoder_gs_4|170310400|340278048|90643840|48155008|53466112|48155008"
    "slat_decoder_mesh|182036640|363715680|96880352|51463680|57140768|51477600"
)

mkdir -p "${OUT}"
downloaded=0
skipped=0

for row in "${MANIFEST[@]}"; do
    IFS='|' read -r name s16 s32 sq80 sq40 sq41 sq4k <<< "${row}"
    if [[ -n "${MODEL}" && "${name}" != "${MODEL}" ]]; then
        continue
    fi
    case "${DTYPE}" in
        f16)  size="${s16}" ;;
        f32)  size="${s32}" ;;
        q8_0) size="${sq80}" ;;
        q4_0) size="${sq40}" ;;
        q4_1) size="${sq41}" ;;
        q4_k) size="${sq4k}" ;;
        *) echo "unsupported dtype: ${DTYPE} (use f16 | f32 | q4_0 | q4_1 | q4_k | q8_0)" >&2; exit 1 ;;
    esac
    file="${OUT}/${name}-${DTYPE}.gguf"
    url="${BASE}/${name}-${DTYPE}.gguf"
    if [[ -f "${file}" ]]; then
        actual_size="$(stat -c%s "${file}")"
        if [[ "${actual_size}" == "${size}" && "$(head -c 4 "$file")" == GGUF ]]; then
            echo "skip  ${name}-${DTYPE}.gguf (already present)"
            skipped=$((skipped + 1))
            continue
        fi
        if (( ! FORCE )); then
            echo "error: ${file} has ${actual_size} bytes; published revision has ${size}." >&2
            echo "       preserve the local model or pass --force to replace it explicitly." >&2
            exit 1
        fi
        rm -f "${file}.part"
    fi
    echo "fetch ${name}-${DTYPE}.gguf ($((size / 1048576)) MiB)"
    curl -L --fail --retry 5 --connect-timeout 30 -C - -o "${file}.part" "${url}"
    actual_size="$(stat -c%s "${file}.part")"
    if [[ "$actual_size" != "$size" || "$(head -c 4 "${file}.part")" != GGUF ]]; then
        echo "error: downloaded file failed size/header validation: ${file}.part ($actual_size bytes; expected $size)" >&2
        echo "       existing model was not replaced; check the published revision before retrying." >&2
        exit 1
    fi
    mv "${file}.part" "${file}"
    downloaded=$((downloaded + 1))
done

echo "done: ${downloaded} downloaded, ${skipped} skipped -> ${OUT}"
