#!/usr/bin/env bash
# Shared configuration for mcp/*.sh. Sourced, not executed directly.
#
# shellcheck disable=SC2034  # used only by sourcing scripts

MCP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_REPO_ROOT="$(cd "$MCP_SCRIPT_DIR/.." && pwd)"

MCP_SERVER_DIR="${HY3_COMFY_MCP_DIR:-$HOME/.local/share/comfyui-mcp-server}"
MCP_SERVER_REPO_URL="${HY3_COMFY_MCP_REPO:-https://github.com/joenorton/comfyui-mcp-server.git}"
MCP_SERVER_VENV_DIR="$MCP_SERVER_DIR/.venv"
MCP_SESSION_NAME="${HY3_COMFY_MCP_SESSION_NAME:-hy3-comfy-mcp}"
MCP_LOG_FILE="${HY3_COMFY_MCP_LOG_FILE:-$HOME/.local/share/comfyui-mcp-server.log}"

# The MCP server's own defaults already match our setup with zero
# overrides needed: COMFYUI_URL defaults to http://localhost:8188
# (confirmed in server.py), which is exactly what
# deploy/vast/tunnel.sh forwards. Listen port is hardcoded to 9000
# server-side (also confirmed in server.py), not configurable via env.
MCP_URL="http://127.0.0.1:9000/mcp"

mcp_pinned_sha() {
  grep -E '^[0-9a-f]{40}$' "$MCP_REPO_ROOT/mcp/pins/comfyui-mcp-server-commit.txt"
}
