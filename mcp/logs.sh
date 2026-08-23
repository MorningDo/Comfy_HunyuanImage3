#!/usr/bin/env bash
# Show comfyui-mcp-server logs. By default, tails the log file.
# --attach instead attaches to the live tmux session (Ctrl-B D to
# detach without stopping the server).
#
# Usage: mcp/logs.sh [--attach] [-n LINES]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mcp/config.sh
source "$SCRIPT_DIR/config.sh"

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
  if ! tmux has-session -t "$MCP_SESSION_NAME" 2>/dev/null; then
    echo "error: no running tmux session '$MCP_SESSION_NAME' (run mcp/start.sh)" >&2
    exit 1
  fi
  exec tmux attach-session -t "$MCP_SESSION_NAME"
fi

if [[ ! -f "$MCP_LOG_FILE" ]]; then
  echo "error: log file not found: $MCP_LOG_FILE (run mcp/start.sh)" >&2
  exit 1
fi

tail -n "$LINES" -f "$MCP_LOG_FILE"
