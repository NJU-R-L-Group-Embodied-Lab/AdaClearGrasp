#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/clutter_bench_mcp.yaml}"

source "${CONDA_PREFIX:-$HOME/miniforge3}/etc/profile.d/conda.sh" 2>/dev/null \
  || source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate clear
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec python -m clutter_bench_mcp --config "$CONFIG_PATH"
