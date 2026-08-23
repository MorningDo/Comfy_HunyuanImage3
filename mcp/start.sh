#!/usr/bin/env bash
# Start the comfyui-mcp-server in a detached tmux session (same
# survive-disconnection rationale as deploy/comfy_start.sh, just local
# instead of over SSH). Idempotent — reports status if already running.
#
# Requires: mcp/setup.sh already run, and ComfyUI reachable at
# http://localhost:8188 (deploy/vast/tunnel.sh running against the
# vast.ai instance, or a local ComfyUI).
#
# Usage: mcp/start.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mcp/config.sh
source "$SCRIPT_DIR/config.sh"

if tmux has-session -t "$MCP_SESSION_NAME" 2>/dev/null; then
  echo "comfyui-mcp-server is already running in tmux session '$MCP_SESSION_NAME'."
  echo "Logs: mcp/logs.sh   Stop: mcp/stop.sh   URL: $MCP_URL"
  exit 0
fi

[[ -x "$MCP_SERVER_VENV_DIR/bin/python" ]] || { echo "error: venv not found: $MCP_SERVER_VENV_DIR (run mcp/setup.sh first)" >&2; exit 1; }

if ! curl -sS -o /dev/null --max-time 5 "http://localhost:8188/"; then
  echo "warn: ComfyUI not reachable at http://localhost:8188 yet." >&2
  echo "Start deploy/vast/tunnel.sh (and deploy/comfy_start.sh on the instance) first — the MCP server retries a few times on startup but then gives up." >&2
fi

mkdir -p "$(dirname "$MCP_LOG_FILE")"
cmd="cd '$MCP_SERVER_DIR' && source '$MCP_SERVER_VENV_DIR/bin/activate' && exec python server.py >>'$MCP_LOG_FILE' 2>&1"
tmux new-session -d -s "$MCP_SESSION_NAME" "$cmd"

sleep 1
if tmux has-session -t "$MCP_SESSION_NAME" 2>/dev/null; then
  echo "Started comfyui-mcp-server in tmux session '$MCP_SESSION_NAME'"
  echo "  URL:  $MCP_URL"
  echo "  Logs: mcp/logs.sh   Stop: mcp/stop.sh"
else
  echo "error: tmux session exited immediately — check $MCP_LOG_FILE" >&2
  exit 1
fi
