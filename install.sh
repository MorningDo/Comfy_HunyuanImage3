#!/usr/bin/env bash
#
# install.sh — set up ComfyUI + this custom node (Comfy_HunyuanImage3) in a
# fresh venv on a GPU test machine, with a reproducible, verified-working
# set of ML-stack dependency versions.
#
# See INSTALL.md for the full writeup. Quick version:
#
#   ./install.sh
#
# Config is via environment variables (all optional, sensible defaults):
#
#   COMFYUI_DIR             Where to put/find ComfyUI.
#                            Default: auto-detected if this repo already
#                            sits inside <something>/custom_nodes/, else
#                            $HOME/ComfyUI
#   VENV_DIR                Default: $COMFYUI_DIR/venv
#   PYTHON_BIN               Default: python3
#   COMFYUI_REPO_URL        Default: official ComfyUI GitHub repo
#   COMFYUI_REF             Branch/tag to clone. Default: master
#   TORCH_CUDA_INDEX        pytorch.org wheel index. Default: cu128
#                            (covers RTX Ada; Blackwell *requires* cu128+)
#   TORCH_VERSION           Default: 2.13.0 (see constraints-tested.txt).
#                            Falls back to an unpinned install from the
#                            same index if this exact version+index combo
#                            isn't available.
#   HUNYUAN_TEST_MODELS_DIR If set, the full GPU test suite (not just the
#                            fast no-GPU one) is also run at the end,
#                            pointed at this directory.
#   SKIP_COMFYUI            If "1", don't clone/install ComfyUI at all —
#                            just set up this project's own venv deps.
#                            Useful if ComfyUI is already fully set up.
#
# Safe to re-run: existing ComfyUI checkout / venv / symlink are detected
# and reused rather than recreated or overwritten.
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_NAME="$(basename "$REPO_DIR")"

_auto_comfyui_dir() {
    # Mirror tests/conftest.py's _guess_comfyui_dir(): is this repo already
    # sitting at .../ComfyUI/custom_nodes/<this repo>?
    local candidate
    candidate="$(dirname "$(dirname "$REPO_DIR")")"
    if [ -f "$candidate/folder_paths.py" ]; then
        echo "$candidate"
    else
        echo "$HOME/ComfyUI"
    fi
}

COMFYUI_DIR="${COMFYUI_DIR:-$(_auto_comfyui_dir)}"
VENV_DIR="${VENV_DIR:-$COMFYUI_DIR/venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
COMFYUI_REPO_URL="${COMFYUI_REPO_URL:-https://github.com/comfyanonymous/ComfyUI.git}"
COMFYUI_REF="${COMFYUI_REF:-master}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0}"
HUNYUAN_TEST_MODELS_DIR="${HUNYUAN_TEST_MODELS_DIR:-}"
SKIP_COMFYUI="${SKIP_COMFYUI:-0}"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

_c_bold=$'\033[1m'; _c_green=$'\033[32m'; _c_yellow=$'\033[33m'; _c_red=$'\033[31m'; _c_reset=$'\033[0m'
section() { printf '\n%s==> %s%s\n' "$_c_bold$_c_green" "$1" "$_c_reset"; }
log()     { printf '%s\n' "$1"; }
warn()    { printf '%sWARNING: %s%s\n' "$_c_yellow" "$1" "$_c_reset" >&2; }
die()     { printf '%sERROR: %s%s\n' "$_c_red" "$1" "$_c_reset" >&2; exit 1; }

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^#//; s/^ //'
    exit 0
fi

section "Configuration"
log "Repo:              $REPO_DIR"
log "ComfyUI dir:       $COMFYUI_DIR"
log "Venv:              $VENV_DIR"
log "Torch:             ${TORCH_VERSION} from ${TORCH_CUDA_INDEX}"
log "ComfyUI ref:       $COMFYUI_REF"
[ -n "$HUNYUAN_TEST_MODELS_DIR" ] && log "Models dir:        $HUNYUAN_TEST_MODELS_DIR (full GPU suite will run at the end)"

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------

section "Checking prerequisites"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "$PYTHON_BIN not found on PATH"
PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
log "Python: $PYTHON_VERSION ($(command -v "$PYTHON_BIN"))"
"$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' \
    || warn "Python $PYTHON_VERSION detected — this project and current ComfyUI generally expect 3.10+. Continuing anyway."

command -v git >/dev/null 2>&1 || die "git not found on PATH"

if command -v nvidia-smi >/dev/null 2>&1; then
    log "GPU(s) detected via nvidia-smi:"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
else
    warn "nvidia-smi not found — no NVIDIA driver detected. Installing a CUDA torch build anyway" \
         "(per TORCH_CUDA_INDEX), but GPU tests will not work until a driver is present."
fi

# ---------------------------------------------------------------------------
# ComfyUI checkout
# ---------------------------------------------------------------------------

if [ "$SKIP_COMFYUI" = "1" ]; then
    section "Skipping ComfyUI setup (SKIP_COMFYUI=1)"
    [ -f "$COMFYUI_DIR/folder_paths.py" ] || die "SKIP_COMFYUI=1 but $COMFYUI_DIR doesn't look like a ComfyUI checkout (no folder_paths.py)"
elif [ -f "$COMFYUI_DIR/folder_paths.py" ]; then
    section "ComfyUI already present"
    log "Found existing ComfyUI at $COMFYUI_DIR — not re-cloning."
    log "(delete it, or set COMFYUI_DIR to a fresh path, to get a clean checkout)"
else
    section "Cloning ComfyUI"
    mkdir -p "$(dirname "$COMFYUI_DIR")"
    git clone --depth 1 --branch "$COMFYUI_REF" "$COMFYUI_REPO_URL" "$COMFYUI_DIR"
fi

# ---------------------------------------------------------------------------
# Virtualenv
# ---------------------------------------------------------------------------

section "Setting up virtualenv"

if [ -f "$VENV_DIR/bin/activate" ]; then
    log "Reusing existing venv at $VENV_DIR"
else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    log "Created venv at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python3 -m pip install --upgrade pip setuptools wheel -q
log "pip: $(python3 -m pip --version)"

# ---------------------------------------------------------------------------
# PyTorch (CUDA build) — installed on its own, before everything else, so
# nothing downstream (ComfyUI's requirements.txt, this project's
# requirements.txt) accidentally pulls in a CPU-only or mismatched-CUDA
# build first. Pinned to the version verified in constraints-tested.txt,
# falling back to an unpinned install from the same CUDA index if that
# exact version isn't available on it by the time this runs.
# ---------------------------------------------------------------------------

section "Installing PyTorch ${TORCH_VERSION} (CUDA build: ${TORCH_CUDA_INDEX})"

if ! python3 -m pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_CUDA_INDEX"; then
    warn "torch==${TORCH_VERSION} not available on ${TORCH_CUDA_INDEX} — falling back to the latest" \
         "torch on that index (unpinned). Note this in your report; it means the GPU run isn't" \
         "using the exact version this project's fixes were verified against."
    python3 -m pip install torch --index-url "$TORCH_CUDA_INDEX"
fi

# ---------------------------------------------------------------------------
# ComfyUI's own requirements
# ---------------------------------------------------------------------------

if [ "$SKIP_COMFYUI" != "1" ]; then
    section "Installing ComfyUI requirements"
    python3 -m pip install -r "$COMFYUI_DIR/requirements.txt"
fi

# ---------------------------------------------------------------------------
# Symlink this repo into ComfyUI/custom_nodes/
# ---------------------------------------------------------------------------

if [ "$SKIP_COMFYUI" != "1" ]; then
    section "Linking $REPO_NAME into custom_nodes/"
    mkdir -p "$COMFYUI_DIR/custom_nodes"
    LINK_PATH="$COMFYUI_DIR/custom_nodes/$REPO_NAME"

    if [ "$REPO_DIR" = "$LINK_PATH" ]; then
        log "This repo already sits directly inside custom_nodes/ — nothing to link."
    elif [ -L "$LINK_PATH" ]; then
        EXISTING_TARGET="$(cd "$(dirname "$LINK_PATH")" && readlink -f "$(basename "$LINK_PATH")" 2>/dev/null || true)"
        if [ "$EXISTING_TARGET" = "$REPO_DIR" ]; then
            log "Symlink already correct: $LINK_PATH -> $REPO_DIR"
        else
            warn "$LINK_PATH is a symlink to something else ($EXISTING_TARGET) — leaving it alone."
        fi
    elif [ -e "$LINK_PATH" ]; then
        warn "$LINK_PATH already exists and isn't a symlink — leaving it alone (not overwriting)."
    else
        ln -s "$REPO_DIR" "$LINK_PATH"
        log "Linked $LINK_PATH -> $REPO_DIR"
    fi
fi

# ---------------------------------------------------------------------------
# This project's requirements, pinned to the verified-working versions
# ---------------------------------------------------------------------------

section "Installing Comfy_HunyuanImage3 + test requirements (constrained by constraints-tested.txt)"
python3 -m pip install \
    -r "$REPO_DIR/requirements.txt" \
    -r "$REPO_DIR/requirements-test.txt" \
    -c "$REPO_DIR/constraints-tested.txt"

# ---------------------------------------------------------------------------
# Verify imports
# ---------------------------------------------------------------------------

section "Verifying imports"

COMFYUI_DIR="$COMFYUI_DIR" REPO_DIR="$REPO_DIR" python3 - <<'PYEOF'
import os
import sys

comfyui_dir = os.environ.get("COMFYUI_DIR", "")
repo_dir = os.environ["REPO_DIR"]
if comfyui_dir:
    sys.path.insert(0, comfyui_dir)
sys.path.insert(0, repo_dir)

import torch
print(f"torch {torch.__version__} — CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
else:
    print("  (no CUDA device visible to torch — check driver/CUDA install if this is unexpected)")

import transformers
import accelerate
import bitsandbytes
import numpy
import PIL

print(f"transformers {transformers.__version__}")
print(f"accelerate {accelerate.__version__}")
print(f"bitsandbytes {bitsandbytes.__version__}")
print(f"numpy {numpy.__version__}")
print(f"pillow {PIL.__version__}")

if comfyui_dir:
    import folder_paths  # noqa: F401
    import comfy.utils  # noqa: F401
    print("folder_paths, comfy.utils: OK (real ComfyUI)")

import hunyuan_shared  # noqa: F401
print("hunyuan_shared: OK")
PYEOF

log "All imports OK."

# ---------------------------------------------------------------------------
# Fast (no-GPU) test suite — confirms the install itself is sound
# ---------------------------------------------------------------------------

section "Running fast no-GPU regression tests"
# Deliberately NOT passing --comfyui-dir here: these tests are designed to
# be genuinely GPU/ComfyUI-independent (they use the no-op stub in
# tests/_stubs/, which is enough for every INPUT_TYPES()/schema check —
# the stub's empty model dropdown doesn't matter, nothing here inspects
# its contents). Real ComfyUI's own comfy.model_management module calls
# torch.cuda.current_device() unconditionally in a few places, which
# would needlessly couple this "no GPU needed" suite to CUDA actually
# being present. --comfyui-dir is used for the full GPU suite below,
# where real model loading genuinely needs it.

# $COMFYUI_DIR is set (and exported to child processes) earlier in this
# script — explicitly clear it for this one call so run_tests.sh's own
# "fall back to $COMFYUI_DIR env var when --comfyui-dir isn't passed"
# logic doesn't undo the "use the stub" intent above.
FAST_TESTS_PASSED=1
if COMFYUI_DIR= "$REPO_DIR/tests/run_tests.sh" \
    "$REPO_DIR/tests/test_regressions_static.py" \
    "$REPO_DIR/tests/test_schema_consistency.py" \
    "$REPO_DIR/tests/test_memory_budget.py" \
    "$REPO_DIR/tests/test_block_swap_config.py" \
    -q; then
    log "Fast test suite: PASSED"
else
    FAST_TESTS_PASSED=0
    warn "Fast test suite reported failures — see output above. Include it in your report."
fi

# ---------------------------------------------------------------------------
# Optional: full GPU suite, if a models dir was provided
# ---------------------------------------------------------------------------

FULL_SUITE_RAN=0
FULL_SUITE_PASSED=1
if [ -n "$HUNYUAN_TEST_MODELS_DIR" ]; then
    section "Running full GPU test suite against $HUNYUAN_TEST_MODELS_DIR"
    FULL_SUITE_RAN=1
    if COMFYUI_DIR="$COMFYUI_DIR" "$REPO_DIR/tests/run_tests.sh" \
        --comfyui-dir "$COMFYUI_DIR" \
        --hunyuan-models-dir "$HUNYUAN_TEST_MODELS_DIR" \
        -v; then
        log "Full GPU test suite: PASSED"
    else
        FULL_SUITE_PASSED=0
        warn "Full GPU test suite reported failures — see output above. Include it in your report."
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

section "Summary"
cat <<SUMMARY
Repo:            $REPO_DIR
ComfyUI:         $COMFYUI_DIR
Venv:            $VENV_DIR  (activate with: source $VENV_DIR/bin/activate)
Fast test suite: $([ "$FAST_TESTS_PASSED" = 1 ] && echo PASSED || echo "FAILED — see above")

Need model weights? This script only sets up code — see
$REPO_DIR/download_models.sh (own venv, downloads by key or 'all' from
Hugging Face) and $REPO_DIR/extra_model_paths.yaml.example if ComfyUI and
the models live on separate volumes.
SUMMARY
if [ "$FULL_SUITE_RAN" = 1 ]; then
    echo "Full GPU suite:  $([ "$FULL_SUITE_PASSED" = 1 ] && echo PASSED || echo "FAILED — see above")"
else
    cat <<'NEXTSTEPS'

Full GPU suite not run (no HUNYUAN_TEST_MODELS_DIR set). To run it:

    source VENV_DIR_PLACEHOLDER/bin/activate
    REPO_DIR_PLACEHOLDER/tests/run_tests.sh \
        --comfyui-dir COMFYUI_DIR_PLACEHOLDER \
        --hunyuan-models-dir /path/to/your/hunyuan/models \
        -v

See REPO_DIR_PLACEHOLDER/tests/README.md for per-variant model-directory
overrides (HUNYUAN_TEST_MODEL_NF4, etc.) if your folder names don't match
auto-detection, and --run-slow to include the BF16/multi-image tests.
NEXTSTEPS
fi | sed -e "s#VENV_DIR_PLACEHOLDER#$VENV_DIR#g" -e "s#REPO_DIR_PLACEHOLDER#$REPO_DIR#g" -e "s#COMFYUI_DIR_PLACEHOLDER#$COMFYUI_DIR#g"

if [ "$FAST_TESTS_PASSED" != 1 ] || [ "$FULL_SUITE_PASSED" != 1 ]; then
    exit 1
fi
