#!/usr/bin/env bash
# Clones joenorton/comfyui-mcp-server (pinned commit, see
# mcp/pins/comfyui-mcp-server-commit.txt) and installs its dependencies
# into a dedicated venv, so an MCP client (Claude Code or similar) can
# drive ComfyUI via tool calls instead of the web UI.
#
# Runs on the HOST/sandbox, not the vast.ai instance — this server
# talks to ComfyUI's HTTP API at http://localhost:8188, which
# deploy/vast/tunnel.sh already forwards from the remote instance.
# Deliberately installed outside this repo (default
# $HOME/.local/share/comfyui-mcp-server) since it's third-party
# tooling, not project source.
#
# Usage: mcp/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mcp/config.sh
source "$SCRIPT_DIR/config.sh"

sha="$(mcp_pinned_sha)"

if [[ -d "$MCP_SERVER_DIR/.git" ]]; then
  echo "Updating existing checkout at $MCP_SERVER_DIR"
  git -C "$MCP_SERVER_DIR" fetch origin "$sha"
else
  echo "Cloning $MCP_SERVER_REPO_URL to $MCP_SERVER_DIR"
  mkdir -p "$(dirname "$MCP_SERVER_DIR")"
  git clone "$MCP_SERVER_REPO_URL" "$MCP_SERVER_DIR"
fi
git -C "$MCP_SERVER_DIR" checkout "$sha"

if [[ ! -x "$MCP_SERVER_VENV_DIR/bin/python" ]]; then
  echo "Creating venv at $MCP_SERVER_VENV_DIR"
  python3 -m venv "$MCP_SERVER_VENV_DIR"
fi

# shellcheck disable=SC1091
source "$MCP_SERVER_VENV_DIR/bin/activate"
pip install --upgrade pip >/dev/null
pip install -r "$MCP_SERVER_DIR/requirements.txt"

# requirements.txt pins only `mcp>=0.9.0` (no upper bound), which
# resolves to whatever's newest — currently mcp 2.0.0 (released
# 2026-07-28, months after this repo's pinned commit). That's a real
# breaking change: `from mcp.server.fastmcp import FastMCP` in
# server.py fails outright (ModuleNotFoundError), confirmed live.
# Force the last 1.x release that predates the pinned commit
# (mcp-1.26.0, 2026-01-24 — confirmed its wheel actually contains
# mcp/server/fastmcp/ before relying on it) instead of trusting the
# unbounded floor.
pip install "mcp==1.26.0"

# Install our project-specific workflows (auto-discovered by the server
# from its workflows/ dir — this is how it learns about the
# HunyuanInstructLoader/HunyuanInstructGenerate nodes, which its own
# generic bundled workflows have no idea exist). Copied, not symlinked,
# so a stale link surviving a repo move doesn't silently break it.
cp "$MCP_REPO_ROOT"/mcp/workflows/*.json "$MCP_SERVER_DIR/workflows/"
installed_names=()
for f in "$MCP_REPO_ROOT"/mcp/workflows/*.json; do
  installed_names+=("$(basename "$f")")
done
echo "Installed project workflows: ${installed_names[*]}"

echo
echo "comfyui-mcp-server installed at $MCP_SERVER_DIR"
echo "  venv:    $MCP_SERVER_VENV_DIR"
echo "  pinned:  $sha"
echo
echo "Next: mcp/start.sh (needs deploy/vast/tunnel.sh already running,"
echo "since this server expects ComfyUI reachable at http://localhost:8188)"
