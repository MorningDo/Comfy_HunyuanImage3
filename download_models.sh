#!/usr/bin/env bash
#
# download_models.sh — fetch HunyuanImage-3.0 model weights from Hugging
# Face, in a small dedicated venv (separate from the ComfyUI/inference venv
# install.sh sets up — this one only needs huggingface_hub, not torch).
#
#   ./download_models.sh list                          # show available models
#   ./download_models.sh nf4 instruct-distil-int8       # download specific models
#   ./download_models.sh all --skip-legacy              # download everything current (no v1 dupes)
#
# All arguments are passed straight through to scripts/download_models.py
# (run `./download_models.sh --help` for its full option list).
#
# Config is via environment variables (all optional):
#
#   HUNYUAN_MODELS_DIR   Where models are downloaded, as
#                          <dir>/hunyuan/<name> and <dir>/hunyuan_instruct/<name>.
#                          Default: /workspace/models — matches the base_path
#                          in extra_model_paths.yaml.example.
#   DOWNLOAD_VENV_DIR    Default: <this repo>/.venv-download
#   PYTHON_BIN            Default: python3
#
# Safe to re-run: the venv is reused if it already exists, and downloads
# resume/skip already-matching files (huggingface_hub's normal behavior).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_VENV_DIR="${DOWNLOAD_VENV_DIR:-$REPO_DIR/.venv-download}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^#//; s/^ //'
    echo
    echo "--- scripts/download_models.py --help ---"
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "ERROR: $PYTHON_BIN not found on PATH" >&2; exit 1; }

if [ ! -f "$DOWNLOAD_VENV_DIR/bin/activate" ]; then
    echo "==> Creating download venv at $DOWNLOAD_VENV_DIR"
    "$PYTHON_BIN" -m venv "$DOWNLOAD_VENV_DIR"
    "$DOWNLOAD_VENV_DIR/bin/python" -m pip install --upgrade pip -q
    "$DOWNLOAD_VENV_DIR/bin/python" -m pip install -q -r "$REPO_DIR/scripts/requirements-download.txt"
else
    # Cheap idempotent check — installs anything missing (e.g. after a
    # requirements-download.txt update) without reinstalling every time.
    "$DOWNLOAD_VENV_DIR/bin/python" -m pip install -q -r "$REPO_DIR/scripts/requirements-download.txt"
fi

exec "$DOWNLOAD_VENV_DIR/bin/python" "$REPO_DIR/scripts/download_models.py" "$@"
