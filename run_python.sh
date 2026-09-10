#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${ROOT}/cpp_ggml/scripts/launch_reconstruction.sh" python "$@"
