#!/usr/bin/env bash
# Stop the ComfyUI tmux session started by deploy/comfy_start.sh.
# Safe to run when it's not running.
#
# Usage: deploy/comfy_stop.sh

set -euo pipefail

COMFY_SESSION_NAME="${HY3_COMFY_SESSION_NAME:-hy3-comfy}"

if ! tmux has-session -t "$COMFY_SESSION_NAME" 2>/dev/null; then
  echo "ComfyUI is not running (no tmux session '$COMFY_SESSION_NAME')."
  exit 0
fi

tmux kill-session -t "$COMFY_SESSION_NAME"
echo "Stopped tmux session '$COMFY_SESSION_NAME'."
