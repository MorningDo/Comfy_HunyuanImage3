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
#   TORCH_CUDA_INDEX        pytorch.org wheel index. Default: cu130
#                            (flashinfer's officially documented CUDA
#                            support list is 12.6/12.8/13.0/13.1 — no
#                            12.9 — so cu130 is the best-aligned "current"
#                            choice; Blackwell *requires* cu128+ regardless)
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
#   INSTALL_FLASHINFER      If "1", also installs flashinfer-python (for
#                            HunyuanInstructLoader's moe_impl=flashinfer)
#                            and attempts a system-level CUDA Toolkit
#                            install via apt (Ubuntu only — flashinfer
#                            JIT-compiles kernels at first use and needs a
#                            real nvcc matching the installed torch's CUDA
#                            version, not just the CUDA runtime bundled in
#                            the pip torch wheel). Off by default: this is
#                            a heavier, system-level step than anything
#                            else this script does. See INSTALL.md.
#   FLASHINFER_VERSION      Default: 0.5.0 (matches Tencent's own tested
#                            baseline for HunyuanImage-3.0). Only used
#                            when INSTALL_FLASHINFER=1.
#
# Safe to re-run: existing ComfyUI checkout / venv / symlink are detected
# and reused rather than recreated or overwritten.
#
# NOTE on flash-attn: this script deliberately does NOT install flash-attn
# (attention_impl=flash_attention_2). It has no prebuilt PyPI wheel (a
# from-scratch build can take hours) and, as of this writing, mainline
# flash-attn has no confirmed support for consumer/workstation Blackwell
# GPUs (SM120 — e.g. RTX PRO 6000, RTX 5090). See INSTALL.md for the
# manual/unofficial path if you want to try it anyway.
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

_index_cuda_version() {
    # "https://download.pytorch.org/whl/cu130" -> "13.0". Empty output for
    # non-CUDA indexes (e.g. .../cpu) — first 2 digits are the major
    # version, the rest is the minor version, matching pytorch.org's own
    # cuMAJORMINOR index-tag convention (cu121 -> 12.1, cu128 -> 12.8,
    # cu130 -> 13.0).
    local tag digits
    tag="$(basename "$1")"
    if [[ "$tag" =~ ^cu([0-9]{2,3})$ ]]; then
        digits="${BASH_REMATCH[1]}"
        echo "${digits:0:2}.${digits:2}"
    fi
}

COMFYUI_DIR="${COMFYUI_DIR:-$(_auto_comfyui_dir)}"
VENV_DIR="${VENV_DIR:-$COMFYUI_DIR/venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
COMFYUI_REPO_URL="${COMFYUI_REPO_URL:-https://github.com/comfyanonymous/ComfyUI.git}"
COMFYUI_REF="${COMFYUI_REF:-master}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu130}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0}"
HUNYUAN_TEST_MODELS_DIR="${HUNYUAN_TEST_MODELS_DIR:-}"
SKIP_COMFYUI="${SKIP_COMFYUI:-0}"
INSTALL_FLASHINFER="${INSTALL_FLASHINFER:-0}"
FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.5.0}"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

_c_bold=$'\033[1m'; _c_green=$'\033[32m'; _c_yellow=$'\033[33m'; _c_red=$'\033[31m'; _c_reset=$'\033[0m'
section() { printf '\n%s==> %s%s\n' "$_c_bold$_c_green" "$1" "$_c_reset"; }
# "$*" (not "$1"): several call sites pass a warning/log message as multiple
# backslash-continued string arguments (one sentence fragment per line) —
# "$*" joins them with a space into the single intended message instead of
# silently dropping everything after the first fragment.
log()     { printf '%s\n' "$*"; }
warn()    { printf '%sWARNING: %s%s\n' "$_c_yellow" "$*" "$_c_reset" >&2; }
die()     { printf '%sERROR: %s%s\n' "$_c_red" "$*" "$_c_reset" >&2; exit 1; }

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    sed -n '2,58p' "${BASH_SOURCE[0]}" | sed 's/^#//; s/^ //'
    exit 0
fi

section "Configuration"
log "Repo:              $REPO_DIR"
log "ComfyUI dir:       $COMFYUI_DIR"
log "Venv:              $VENV_DIR"
log "Torch:             ${TORCH_VERSION} from ${TORCH_CUDA_INDEX}"
log "ComfyUI ref:       $COMFYUI_REF"
[ "$INSTALL_FLASHINFER" = "1" ] && log "FlashInfer:        ${FLASHINFER_VERSION} + system CUDA Toolkit (opt-in, INSTALL_FLASHINFER=1)"
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

    # The HOST driver's max-supported CUDA version — read straight from the
    # driver (nvidia-smi), independent of anything installed in THIS
    # container (no CUDA toolkit needs to be present for this to report
    # correctly). Checked against the target now, before any install
    # happens below: this container may currently have an older CUDA
    # toolkit than TORCH_CUDA_INDEX targets (that's the whole point of
    # this script), but the driver's ceiling is fixed regardless — on
    # RunPod, set at pod-creation time by which host CUDA version you
    # selected, not by this script.
    TARGET_CUDA="$(_index_cuda_version "$TORCH_CUDA_INDEX")"
    DRIVER_CUDA="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9]*\.[0-9]*' | grep -o '[0-9.]*$')"
    if [ -n "$TARGET_CUDA" ] && [ -n "$DRIVER_CUDA" ]; then
        if awk -v a="$DRIVER_CUDA" -v b="$TARGET_CUDA" 'BEGIN{exit !(a<b)}'; then
            warn "Host GPU driver's max supported CUDA is ${DRIVER_CUDA} (per nvidia-smi), but" \
                 "TORCH_CUDA_INDEX targets CUDA ${TARGET_CUDA}. This is the HOST driver — RunPod-managed," \
                 "not something this script can change. torch/flashinfer will likely fail with" \
                 "'CUDA driver version is insufficient for CUDA runtime version' once installed." \
                 "If on RunPod, recreate the pod selecting a host with CUDA ${TARGET_CUDA}+ available," \
                 "or set TORCH_CUDA_INDEX to an index at or below cu${DRIVER_CUDA//./} to match this host."
        else
            log "Host GPU driver supports up to CUDA ${DRIVER_CUDA} (nvidia-smi) — OK for target ${TARGET_CUDA}."
        fi
    fi
else
    warn "nvidia-smi not found — no NVIDIA driver detected. Installing a CUDA torch build anyway" \
         "(per TORCH_CUDA_INDEX), but GPU tests will not work until a driver is present."
fi

# System CUDA Toolkit (nvcc) — separate from the pip-installed CUDA-enabled
# torch wheel below. Not required for sdpa/eager, but FlashInfer JIT-compiles
# kernels at first use and needs a real nvcc + headers matching torch's CUDA
# version (Tencent's own README: "It is critical that the CUDA version used
# by PyTorch matches the system's CUDA version"). Warn-only here — this is
# just a status check; the INSTALL_FLASHINFER=1 block below is what actually
# tries to install one if it's missing.
if command -v nvcc >/dev/null 2>&1; then
    NVCC_VERSION_LINE="$(nvcc --version | grep -o 'release [0-9.]*' || true)"
    log "System CUDA Toolkit: nvcc found (${NVCC_VERSION_LINE:-version unknown})"
else
    if [ "$INSTALL_FLASHINFER" = "1" ]; then
        log "System CUDA Toolkit: nvcc not found yet — will attempt to install one (INSTALL_FLASHINFER=1)."
    else
        warn "No system CUDA Toolkit (nvcc) detected. Fine for sdpa/eager (the pip torch wheel bundles" \
             "enough CUDA runtime for its own kernels), but moe_impl=flashinfer needs a real nvcc to" \
             "JIT-compile kernels at first use. See INSTALL.md before setting INSTALL_FLASHINFER=1."
    fi
fi

GCC_VERSION="$(gcc -dumpversion 2>/dev/null | cut -d. -f1 || true)"
if [ -n "$GCC_VERSION" ]; then
    if [ "$GCC_VERSION" -lt 9 ] 2>/dev/null; then
        warn "gcc ${GCC_VERSION}.x detected — Tencent's README recommends gcc >= 9 for compiling" \
             "FlashAttention/FlashInfer. Continuing anyway."
    else
        log "gcc: $(gcc -dumpversion) (>= 9, OK)"
    fi
else
    warn "gcc not found — only matters if you enable INSTALL_FLASHINFER=1 or build flash-attn manually."
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
    # --system-site-packages: on images that already ship a CUDA-matched
    # torch at the system level (e.g. RunPod's runpod/pytorch:* base
    # images), this lets the "already suitable?" check below actually see
    # it. A pip install inside the venv (below) still shadows/overrides
    # whatever's inherited, so this is never a downgrade in control.
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
    log "Created venv at $VENV_DIR (--system-site-packages)"
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
#
# Skipped entirely if a CUDA-enabled torch matching TORCH_VERSION exactly
# is already importable (e.g. inherited from the base image via
# --system-site-packages above) — avoids throwing away an
# already-integration-tested torch/CUDA pairing for no reason. Any
# TORCH_VERSION/TORCH_CUDA_INDEX override that doesn't match what's already
# there falls through to the explicit install below, same as always.
# ---------------------------------------------------------------------------

section "Installing PyTorch ${TORCH_VERSION} (CUDA build: ${TORCH_CUDA_INDEX})"

EXISTING_TORCH="$(python3 -c "
import torch, sys
if torch.__version__.split('+')[0] == '${TORCH_VERSION}' and torch.cuda.is_available():
    print(f'{torch.__version__} (CUDA {torch.version.cuda})')
" 2>/dev/null || true)"

if [ -n "$EXISTING_TORCH" ]; then
    log "torch ${TORCH_VERSION} already present and CUDA-enabled ($EXISTING_TORCH) — skipping reinstall."
elif ! python3 -m pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_CUDA_INDEX"; then
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
# Optional: FlashInfer (moe_impl=flashinfer) + the system CUDA Toolkit its
# JIT compilation needs. Opt-in (INSTALL_FLASHINFER=1) — heavier and more
# invasive than anything else in this script (apt/root, not just pip), and
# not needed at all for sdpa/eager. Every step here warns and continues on
# failure rather than aborting the rest of the install.
# ---------------------------------------------------------------------------

if [ "$INSTALL_FLASHINFER" = "1" ]; then
    section "Installing system CUDA Toolkit (for FlashInfer JIT compilation)"

    if command -v nvcc >/dev/null 2>&1 && nvcc --version | grep -q "release 13\."; then
        log "nvcc already reports CUDA 13.x — skipping Toolkit install."
    elif [ "$(id -u)" != "0" ] && ! command -v sudo >/dev/null 2>&1; then
        warn "Not running as root and no sudo available — skipping system CUDA Toolkit install." \
             "moe_impl=flashinfer's JIT compile will likely fail without it. Install" \
             "cuda-toolkit-13-0 yourself (see INSTALL.md) and re-run, or continue with sdpa/eager."
    elif ! command -v apt-get >/dev/null 2>&1; then
        warn "apt-get not found (non-Debian/Ubuntu system?) — skipping system CUDA Toolkit install." \
             "See INSTALL.md for manual instructions for your distro."
    elif ! command -v wget >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
        warn "Neither wget nor curl found — skipping system CUDA Toolkit install." \
             "See INSTALL.md for manual instructions."
    else
        _sudo=""
        [ "$(id -u)" != "0" ] && _sudo="sudo"
        log "Downloading and installing cuda-toolkit-13-0 via apt (Ubuntu 24.04) — this can take a" \
            "while and download several GB. Failure here is non-fatal to the rest of this script."
        _fetch="wget -q -O /tmp/cuda-keyring.deb"
        command -v wget >/dev/null 2>&1 || _fetch="curl -fsSL -o /tmp/cuda-keyring.deb"
        if $_sudo bash -c "
            set -e
            ${_fetch} https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
            dpkg -i /tmp/cuda-keyring.deb
            apt-get update -qq
            apt-get -y install cuda-toolkit-13-0
        "; then
            export PATH="/usr/local/cuda-13.0/bin:$PATH"
            export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"
            log "CUDA Toolkit 13.0 installed. Add these to your shell profile for future sessions:"
            log "  export PATH=/usr/local/cuda-13.0/bin:\$PATH"
            log "  export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:\$LD_LIBRARY_PATH"
        else
            warn "System CUDA Toolkit install failed (see output above) — continuing without it." \
                 "moe_impl=flashinfer's JIT compile will likely fail. This step is Ubuntu-specific;" \
                 "see INSTALL.md if you're on a different distro."
        fi
    fi

    section "Installing flashinfer-python ${FLASHINFER_VERSION}"
    if python3 -m pip install "flashinfer-python==${FLASHINFER_VERSION}"; then
        log "flashinfer-python ${FLASHINFER_VERSION} installed. Note: the FIRST inference using" \
            "moe_impl=flashinfer will take several extra minutes (JIT kernel compile, cached after)."
    else
        warn "flashinfer-python==${FLASHINFER_VERSION} failed to install — continuing without it." \
             "moe_impl=flashinfer will be unavailable in ComfyUI (sdpa/eager still work normally)."
    fi
fi

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

try:
    import flashinfer
    print(f"flashinfer {flashinfer.__version__}: importable (JIT-compiles kernels on first use)")
except ImportError:
    print("flashinfer: not installed (moe_impl=flashinfer will be unavailable in ComfyUI)")

try:
    import flash_attn
    print(f"flash_attn {flash_attn.__version__}: importable (no confirmed SM120/Blackwell support upstream — see INSTALL.md)")
except ImportError:
    print("flash_attn: not installed (expected — not installed by this script, see INSTALL.md)")
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
FlashInfer:      $([ "$INSTALL_FLASHINFER" = "1" ] && echo "install attempted — see 'Verifying imports' output above for whether it actually imports" || echo "not installed (set INSTALL_FLASHINFER=1 to enable — see INSTALL.md)")
FlashAttention:  not installed by this script — see INSTALL.md (no confirmed SM120/Blackwell support upstream as of this writing)

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
