#!/usr/bin/env bash
# Retain the historical command as an alias to the complete reference setup.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "setup_env.sh now uses the shared sam3d-objects Conda environment installer."
exec bash "${SCRIPT_DIR}/setup_pt_ref_env.sh" "$@"
