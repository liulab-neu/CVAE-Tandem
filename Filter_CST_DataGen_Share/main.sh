#!/usr/bin/env bash
set -euo pipefail

# Start a virtual display for CST and force Matplotlib to use a headless backend.
# TANDEM_PYTHON can be used when the desired Python environment is not activated.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${TANDEM_PYTHON:-python}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

exec xvfb-run -a "$PYTHON_BIN" -u "$SCRIPT_DIR/main.py" "$@"
