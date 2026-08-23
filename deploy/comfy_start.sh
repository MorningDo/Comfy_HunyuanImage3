#!/usr/bin/env bash
# Start ComfyUI under tmux so it survives SSH disconnection. Idempotent:
# if the session is already running, does nothing but report status.
# ComfyUI binds 127.0.0.1 only — reached via deploy/vast/tunnel.sh, never
# exposed directly (only port 22 is open on the instance).
#
# Usage: deploy/comfy_start.sh

set -euo pipefail

COMFY_SESSION_NAME="${HY3_COMFY_SESSION_NAME:-hy3-comfy}"
COMFY_PORT="${HY3_REMOTE_COMFY_PORT:-8188}"
VENV_DIR="${HY3_VENV_DIR:-/opt/hy3/venv}"
COMFYUI_DIR="${HY3_COMFYUI_DIR:-/opt/hy3/ComfyUI}"
COMFY_LOG_FILE="${HY3_COMFY_LOG_FILE:-/var/log/hy3-comfy.log}"

if tmux has-session -t "$COMFY_SESSION_NAME" 2>/dev/null; then
  echo "ComfyUI is already running in tmux session '$COMFY_SESSION_NAME'."
  echo "Logs: deploy/comfy_logs.sh   Stop: deploy/comfy_stop.sh"
  exit 0
fi

[[ -d "$COMFYUI_DIR" ]] || { echo "error: ComfyUI dir not found: $COMFYUI_DIR (run provision.sh first)" >&2; exit 1; }
[[ -x "$VENV_DIR/bin/python" ]] || { echo "error: venv not found: $VENV_DIR (run provision.sh first)" >&2; exit 1; }

mkdir -p "$(dirname "$COMFY_LOG_FILE")" 2>/dev/null || true

cmd="cd '$COMFYUI_DIR' && source '$VENV_DIR/bin/activate' && exec python main.py --listen 127.0.0.1 --port $COMFY_PORT >>'$COMFY_LOG_FILE' 2>&1"
tmux new-session -d -s "$COMFY_SESSION_NAME" "$cmd"

sleep 1
if tmux has-session -t "$COMFY_SESSION_NAME" 2>/dev/null; then
  echo "Started ComfyUI in tmux session '$COMFY_SESSION_NAME' on 127.0.0.1:$COMFY_PORT"
  echo "Logs: deploy/comfy_logs.sh   Stop: deploy/comfy_stop.sh"
else
  echo "error: tmux session exited immediately — check $COMFY_LOG_FILE" >&2
  exit 1
fi
