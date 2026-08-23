#!/usr/bin/env bash
# Stop the comfyui-mcp-server tmux session started by mcp/start.sh.
# Safe to run when it's not running.
#
# Usage: mcp/stop.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mcp/config.sh
source "$SCRIPT_DIR/config.sh"

if ! tmux has-session -t "$MCP_SESSION_NAME" 2>/dev/null; then
  echo "comfyui-mcp-server is not running (no tmux session '$MCP_SESSION_NAME')."
  exit 0
fi

tmux kill-session -t "$MCP_SESSION_NAME"
echo "Stopped tmux session '$MCP_SESSION_NAME'."
