#!/usr/bin/env bash
# Sets up a ComfyUI + Comfy_HunyuanImage3 (dev branch) environment from scratch:
# clones ComfyUI, links this checkout into custom_nodes/, installs
# extra_model_paths.yaml so ComfyUI also scans /workspace for models (the
# RunPod network volume, so weights survive container restarts — see
# extra_model_paths.yaml.example and download_models.sh), creates one shared
# venv, and installs both requirement sets pinned to Tencent's tested
# versions via constraints.txt (see that file for why).
#
# Usage:
#   ./install_dev_env.sh [options]
#
# Options:
#   -c, --comfyui-dir DIR   Where to clone/find ComfyUI (default: $HOME/ComfyUI)
#   -v, --venv-dir DIR      Venv location (default: <comfyui-dir>/venv)
#       --copy              Copy this checkout into custom_nodes/ instead of
#                            symlinking (default: symlink, so local edits show
#                            up in ComfyUI immediately)
#       --with-flashinfer   Also install flashinfer-python==0.5.0 (Tencent
#                            lists this as optional; skipped by default since
#                            it needs a matching build toolchain)
#       --no-model-paths    Don't install extra_model_paths.yaml into
#                            --comfyui-dir (default: install it if that file
#                            doesn't already exist there)
#       --fresh-venv        Delete and recreate the venv if it already exists
#       --allow-any-branch  Don't require the local checkout to be on `dev`
#   -h, --help              Show this help
#
# Env var overrides: COMFYUI_DIR, VENV_DIR, COMFYUI_REPO, TORCH_INDEX_URL,
# PYTHON_BIN (defaults: unset, unset, https://github.com/comfyanonymous/ComfyUI.git,
# https://download.pytorch.org/whl/cu128, python3)

set -euo pipefail

NODE_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_NAME="Comfy_HunyuanImage3"
REQUIRED_BRANCH="dev"

COMFYUI_REPO="${COMFYUI_REPO:-https://github.com/comfyanonymous/ComfyUI.git}"
COMFYUI_DIR="${COMFYUI_DIR:-$HOME/ComfyUI}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LINK_MODE="symlink"
WITH_FLASHINFER=0
INSTALL_MODEL_PATHS=1
FRESH_VENV=0
ALLOW_ANY_BRANCH=0
VENV_DIR="${VENV_DIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--comfyui-dir) COMFYUI_DIR="$2"; shift 2 ;;
    -v|--venv-dir) VENV_DIR="$2"; shift 2 ;;
    --copy) LINK_MODE="copy"; shift ;;
    --with-flashinfer) WITH_FLASHINFER=1; shift ;;
    --no-model-paths) INSTALL_MODEL_PATHS=0; shift ;;
    --fresh-venv) FRESH_VENV=1; shift ;;
    --allow-any-branch) ALLOW_ANY_BRANCH=1; shift ;;
    -h|--help) sed -n '2,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

VENV_DIR="${VENV_DIR:-$COMFYUI_DIR/venv}"

log() { printf '\n==> %s\n' "$1"; }

# --- sanity checks -----------------------------------------------------

if [[ ! -f "$NODE_SOURCE_DIR/pyproject.toml" ]]; then
  echo "error: expected to find pyproject.toml next to this script (in $NODE_SOURCE_DIR)" >&2
  exit 1
fi

CURRENT_BRANCH="$(git -C "$NODE_SOURCE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"
if [[ "$CURRENT_BRANCH" != "$REQUIRED_BRANCH" && "$ALLOW_ANY_BRANCH" -ne 1 ]]; then
  echo "error: this checkout is on branch '$CURRENT_BRANCH', not '$REQUIRED_BRANCH'." >&2
  echo "       run 'git -C \"$NODE_SOURCE_DIR\" checkout $REQUIRED_BRANCH' first," >&2
  echo "       or pass --allow-any-branch to install whatever is currently checked out." >&2
  exit 1
fi

if [[ -n "$(git -C "$NODE_SOURCE_DIR" status --porcelain 2>/dev/null)" ]]; then
  echo "note: $NODE_SOURCE_DIR has uncommitted changes - they'll be included as-is."
fi

# --- ComfyUI -------------------------------------------------------------

if [[ -d "$COMFYUI_DIR/.git" ]]; then
  log "ComfyUI already present at $COMFYUI_DIR, leaving it as-is (re-run with a different --comfyui-dir for a clean clone)"
else
  log "Cloning ComfyUI into $COMFYUI_DIR"
  git clone "$COMFYUI_REPO" "$COMFYUI_DIR"
fi

# --- model search path ----------------------------------------------

MODEL_PATHS_FILE="$COMFYUI_DIR/extra_model_paths.yaml"
if [[ "$INSTALL_MODEL_PATHS" -eq 1 ]]; then
  if [[ -f "$MODEL_PATHS_FILE" ]]; then
    log "extra_model_paths.yaml already present at $MODEL_PATHS_FILE, leaving it as-is"
  else
    log "Installing extra_model_paths.yaml -> $MODEL_PATHS_FILE (points model search at /workspace)"
    cp "$NODE_SOURCE_DIR/extra_model_paths.yaml.example" "$MODEL_PATHS_FILE"
  fi
fi

CUSTOM_NODES_DIR="$COMFYUI_DIR/custom_nodes"
mkdir -p "$CUSTOM_NODES_DIR"
NODE_TARGET="$CUSTOM_NODES_DIR/$NODE_NAME"

if [[ -e "$NODE_TARGET" || -L "$NODE_TARGET" ]]; then
  log "Removing existing $NODE_TARGET"
  rm -rf "$NODE_TARGET"
fi

if [[ "$LINK_MODE" == "symlink" ]]; then
  log "Symlinking $NODE_TARGET -> $NODE_SOURCE_DIR (branch: $CURRENT_BRANCH)"
  ln -s "$NODE_SOURCE_DIR" "$NODE_TARGET"
else
  log "Copying $NODE_SOURCE_DIR -> $NODE_TARGET (branch: $CURRENT_BRANCH)"
  mkdir -p "$NODE_TARGET"
  git -C "$NODE_SOURCE_DIR" archive HEAD | tar -x -C "$NODE_TARGET"
fi

# --- venv ------------------------------------------------------------

if [[ "$FRESH_VENV" -eq 1 && -d "$VENV_DIR" ]]; then
  log "Removing existing venv at $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating venv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# --- torch stack first, from the CUDA-specific index, exact Tencent pins --

log "Installing torch==2.8.0 / torchvision==0.23.0 / torchaudio==2.8.0 from $TORCH_INDEX_URL"
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url "$TORCH_INDEX_URL"

# --- everything else, constrained to Tencent's tested versions -----------

log "Installing ComfyUI requirements (constrained)"
pip install -r "$COMFYUI_DIR/requirements.txt" -c "$NODE_SOURCE_DIR/constraints.txt"

log "Installing $NODE_NAME requirements (constrained)"
pip install -r "$NODE_SOURCE_DIR/requirements.txt" -c "$NODE_SOURCE_DIR/constraints.txt"

if [[ "$WITH_FLASHINFER" -eq 1 ]]; then
  log "Installing flashinfer-python==0.5.0 (optional MoE speedup)"
  pip install flashinfer-python==0.5.0
fi

# --- report ------------------------------------------------------------

log "Installed versions"
python - <<'PY'
import importlib
for mod, attr in [
    ("torch", "__version__"), ("torchvision", "__version__"),
    ("transformers", "__version__"), ("diffusers", "__version__"),
    ("tokenizers", "__version__"), ("safetensors", "__version__"),
    ("numpy", "__version__"), ("PIL", "__version__"),
    ("einops", "__version__"), ("bitsandbytes", "__version__"),
    ("accelerate", "__version__"),
]:
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:<14} {getattr(m, attr, '?')}")
    except ImportError as e:
        print(f"  {mod:<14} NOT INSTALLED ({e})")
PY

log "Done. Activate with: source \"$VENV_DIR/bin/activate\""
echo "     Launch ComfyUI with: python \"$COMFYUI_DIR/main.py\""
