#!/usr/bin/env bash
# Convenience wrapper for running this test suite. See tests/README.md.
#
# Why this exists: two files in this package hard-import `folder_paths`
# (a real ComfyUI module) at the very top of the file, and pytest's own
# conftest-loading mechanism needs to import this package successfully
# before it can even load conftest.py — a step earlier than any in-Python
# sys.path fix-up can reach. The only reliable point to make
# `folder_paths` importable is via $PYTHONPATH, set before the Python
# process starts at all.
#
# This script decides — in bash, before Python starts — whether to put a
# real ComfyUI checkout or the no-op stub (tests/_stubs/) on $PYTHONPATH.
# That decision deliberately does NOT happen inside conftest.py: pytest
# parses --comfyui-dir in two passes (it has to load conftest.py to even
# learn the flag exists, which is the same chicken-and-egg problem this
# script solves for folder_paths itself), and empirically, a real
# ComfyUI's `comfy` package can end up correctly used for `folder_paths`
# but still get shadowed by the already-imported stub `comfy` package for
# `folder_paths.py`'s own `from comfy.cli_args import args` — because
# Python caches `comfy` in sys.modules the first time anything imports it,
# and doesn't reconsider sys.path ordering afterward. Resolving the real
# vs. stub choice here, before Python's import system is involved at all,
# avoids that entirely: only one of them ever goes on PYTHONPATH.
#
# Usage:
#   tests/run_tests.sh                                    # fast, no-GPU tests
#   tests/run_tests.sh --hunyuan-models-dir /path/to/models -v
#   COMFYUI_DIR=/path/to/ComfyUI tests/run_tests.sh --hunyuan-models-dir /path -v --run-slow
#   tests/run_tests.sh --comfyui-dir /path/to/ComfyUI --hunyuan-models-dir /path -v
#
# All arguments are passed straight through to pytest (--comfyui-dir and
# --hunyuan-models-dir are real pytest options registered by conftest.py;
# this script only *additionally* inspects --comfyui-dir to make the
# PYTHONPATH decision, it doesn't consume/hide it from pytest).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Look for --comfyui-dir / --comfyui-dir=X among the passed args, without
# disturbing "$@" for the actual pytest invocation below.
COMFYUI_DIR_ARG=""
_prev=""
for _arg in "$@"; do
    if [ "$_prev" = "--comfyui-dir" ]; then
        COMFYUI_DIR_ARG="$_arg"
    elif [[ "$_arg" == --comfyui-dir=* ]]; then
        COMFYUI_DIR_ARG="${_arg#--comfyui-dir=}"
    fi
    _prev="$_arg"
done

RESOLVED_COMFYUI_DIR="${COMFYUI_DIR_ARG:-${COMFYUI_DIR:-}}"

if [ -n "$RESOLVED_COMFYUI_DIR" ] && [ -f "$RESOLVED_COMFYUI_DIR/folder_paths.py" ]; then
    export PYTHONPATH="${RESOLVED_COMFYUI_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
else
    if [ -n "$RESOLVED_COMFYUI_DIR" ]; then
        echo "run_tests.sh: warning: '$RESOLVED_COMFYUI_DIR' doesn't look like a ComfyUI checkout" \
             "(no folder_paths.py) — using the no-op stub instead" >&2
    fi
    export PYTHONPATH="${REPO_ROOT}/tests/_stubs${PYTHONPATH:+:${PYTHONPATH}}"
fi

# If the caller didn't pass an explicit file/dir to collect, add one
# ourselves (tests/) rather than relying on pytest's testpaths-driven
# implicit-cwd collection. With zero positional args, pytest's first
# argument-parsing pass — before it has loaded conftest.py and therefore
# before it knows --comfyui-dir/--hunyuan-models-dir take a following
# value — can misparse "--comfyui-dir /some/path" and treat /some/path as
# an implicit collection target instead, failing with "unrecognized
# arguments". An explicit path sidesteps that ambiguity entirely.
_skip_next=0
_has_path_arg=0
for _arg in "$@"; do
    if [ "$_skip_next" = 1 ]; then
        _skip_next=0
        continue
    fi
    case "$_arg" in
        --comfyui-dir|--hunyuan-models-dir) _skip_next=1 ;;
        --comfyui-dir=*|--hunyuan-models-dir=*|-*) ;;
        *) _has_path_arg=1 ;;
    esac
done
EXTRA_ARGS=()
[ "$_has_path_arg" = 0 ] && EXTRA_ARGS+=("${REPO_ROOT}/tests")

# -c and --rootdir are both needed, not just one: pytest's automatic
# rootdir/inifile discovery can otherwise latch onto a DIFFERENT
# pytest.ini/pyproject.toml it finds first — e.g. ComfyUI ships its own
# pytest.ini, and when --comfyui-dir's value ends up on $PYTHONPATH, or
# when custom options like --comfyui-dir/--hunyuan-models-dir haven't been
# recognized yet in pytest's first argument-parsing pass, pytest can treat
# ComfyUI's directory as a collection target and pick up ITS config
# instead of this project's (confusing test-ID paths at best; silently
# different marker/timeout settings at worst). Being explicit here removes
# all of that ambiguity regardless of what other arguments are passed.
exec python3 -m pytest -c "${REPO_ROOT}/pyproject.toml" --rootdir="${REPO_ROOT}" "${EXTRA_ARGS[@]}" "$@"
