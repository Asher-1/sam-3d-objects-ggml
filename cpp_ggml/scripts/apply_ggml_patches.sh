#!/usr/bin/env bash
#
# apply_ggml_patches.sh
#
# Apply the in-tree ggml patches to the ggml submodule (cpp_ggml/third_party/ggml).
# Idempotent: re-running is a no-op once everything is applied. Never edit the
# submodule by hand — every local change lives in cpp_ggml/third_party/ggml-patches/ and is
# applied (or verified) by this script, so other developers get the same source
# state after `git submodule update --init`.
#
# The one combined patch lives in cpp_ggml/third_party/ggml-patches/.
#
# Usage:
#   bash cpp_ggml/scripts/apply_ggml_patches.sh
#
# Exits 0 on success, non-zero on any failure. Designed to be called by CMake
# during configure but also runnable standalone for debugging.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_GGML_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
GGML_DIR="${CPP_GGML_DIR}/third_party/ggml"
PATCH_DIR="${CPP_GGML_DIR}/third_party/ggml-patches"

if [[ ! -d "${GGML_DIR}" ]]; then
    echo "error: ggml submodule not found at ${GGML_DIR}" >&2
    echo "       did you forget 'git submodule update --init --recursive'?" >&2
    exit 1
fi

if [[ ! -d "${GGML_DIR}/.git" && ! -f "${GGML_DIR}/.git" ]]; then
    echo "error: ${GGML_DIR} is not a git repository" >&2
    exit 1
fi

if [[ ! -d "${PATCH_DIR}" ]]; then
    echo "error: patch directory not found at ${PATCH_DIR}" >&2
    exit 1
fi

PATCH="${PATCH_DIR}/0001-sam3d-ggml-combined.patch"
if [[ ! -f "${PATCH}" ]]; then
    echo "error: required combined ggml patch not found: ${PATCH}" >&2
    exit 1
fi
PATCHES=("${PATCH}")

applied=0
skipped=0

cd "${GGML_DIR}"

# Serialise concurrent invocations against the shared submodule tree.
if [[ -z "${SAM3D_PATCH_FLOCK_HELD:-}" ]] && command -v flock >/dev/null 2>&1; then
    LOCK_FILE="${CPP_GGML_DIR}/.ggml-patch.lock"
    : > "${LOCK_FILE}" 2>/dev/null || true
    if [[ -e "${LOCK_FILE}" ]]; then
        export SAM3D_PATCH_FLOCK_HELD=1
        SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
        exec flock "${LOCK_FILE}" bash "${SCRIPT_PATH}" "$@"
    fi
fi

# Compare complete tree states first: replay the combined patch into a temporary
# index, then hash the current worktree via a second temporary index. This does
# not modify the user's real Git index.
expected_index="$(mktemp "${TMPDIR:-/tmp}/sam3d-ggml-expected-index.XXXXXX")"
actual_index="$(mktemp "${TMPDIR:-/tmp}/sam3d-ggml-actual-index.XXXXXX")"
rm -f "${expected_index}" "${actual_index}"
trap 'rm -f "${expected_index}" "${actual_index}"' EXIT

GIT_INDEX_FILE="${expected_index}" git read-tree HEAD
for patch in "${PATCHES[@]}"; do
    GIT_INDEX_FILE="${expected_index}" git apply --unidiff-zero --cached "${patch}"
done
expected_tree="$(GIT_INDEX_FILE="${expected_index}" git write-tree)"

GIT_INDEX_FILE="${actual_index}" git read-tree HEAD
GIT_INDEX_FILE="${actual_index}" git add -A -- .
actual_tree="$(GIT_INDEX_FILE="${actual_index}" git write-tree)"

if [[ "${actual_tree}" == "${expected_tree}" ]]; then
    echo "ggml patches: applied 0, skipped ${#PATCHES[@]} (final tree verified)"
    exit 0
fi

for patch in "${PATCHES[@]}"; do
    name="$(basename "${patch}")"

    # Already applied? `git apply --check --reverse` succeeds iff every
    # hunk is currently present in the tree (i.e. we *could* roll it back).
    if git apply --unidiff-zero --check --reverse "${patch}" >/dev/null 2>&1; then
        echo "ggml patches: skipping ${name} (already applied)"
        skipped=$((skipped + 1))
        continue
    fi

    # Otherwise it must apply cleanly forward.
    if git apply --unidiff-zero --check "${patch}" >/dev/null 2>&1; then
        if ! git apply --unidiff-zero "${patch}"; then
            echo "error: failed to apply ${name} after --check succeeded" >&2
            echo "       this should not happen; the submodule tree may be dirty" >&2
            exit 1
        fi
        echo "ggml patches: applied ${name}"
        applied=$((applied + 1))
        continue
    fi

    # Neither forward-applicable nor already-applied: bail with diagnostics.
    echo "error: cannot apply ${name}" >&2
    echo "       'git apply --check' output (forward):" >&2
    git apply --unidiff-zero --check "${patch}" 2>&1 | sed 's/^/         /' >&2 || true
    echo "       'git apply --check --reverse' output:" >&2
    git apply --unidiff-zero --check --reverse "${patch}" 2>&1 | sed 's/^/         /' >&2 || true
    echo "       submodule HEAD: $(git rev-parse HEAD)" >&2
    echo "       try: cd ${GGML_DIR} && git status" >&2
    exit 1
done

echo "ggml patches: applied ${applied}, skipped ${skipped}"
