#!/usr/bin/env bash
# Show ComfyUI logs. By default, tails the log file. --attach instead
# attaches to the live tmux session (Ctrl-B D to detach without
# stopping ComfyUI).
#
# Usage: deploy/comfy_logs.sh [--attach] [-n LINES]

set -euo pipefail

COMFY_SESSION_NAME="${HY3_COMFY_SESSION_NAME:-hy3-comfy}"
COMFY_LOG_FILE="${HY3_COMFY_LOG_FILE:-/var/log/hy3-comfy.log}"
LINES=200
ATTACH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --attach) ATTACH=1; shift ;;
    -n) LINES="${2:?}"; shift 2 ;;
    --help|-h) echo "Usage: $(basename "$0") [--attach] [-n LINES]"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$ATTACH" == "1" ]]; then
  if ! tmux has-session -t "$COMFY_SESSION_NAME" 2>/dev/null; then
    echo "error: no running tmux session '$COMFY_SESSION_NAME' (run deploy/comfy_start.sh)" >&2
    exit 1
  fi
  exec tmux attach-session -t "$COMFY_SESSION_NAME"
fi

if [[ ! -f "$COMFY_LOG_FILE" ]]; then
  echo "error: log file not found: $COMFY_LOG_FILE (run deploy/comfy_start.sh)" >&2
  exit 1
fi

tail -n "$LINES" -f "$COMFY_LOG_FILE"
