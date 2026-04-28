#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${1:-$ROOT_DIR/.venv_csci1470_smoke}"
REQ_FILE="$ROOT_DIR/requirements-smoke.txt"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT_DIR/.pip_cache}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found in PATH"
  exit 2
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "Using python: $(which python)"
python --version

python -m pip install --upgrade pip setuptools wheel
python -m pip install --cache-dir "$PIP_CACHE_DIR" -r "$REQ_FILE"

echo "Venv ready: $VENV_DIR"
