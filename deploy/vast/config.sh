#!/usr/bin/env bash
# Shared configuration for deploy/vast/*.sh. Sourced, not executed
# directly. All values overridable via env for testing.
#
# shellcheck disable=SC2034  # many of these are used only by sourcing scripts

VAST_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAST_REPO_ROOT="$(cd "$VAST_SCRIPT_DIR/../.." && pwd)"

VAST_API_BASE="${HY3_VAST_API_BASE:-https://console.vast.ai/api/v0}"

# Pinned per deploy/pins/torch-cuda-notes.md — devel (not runtime) tag,
# no pre-baked torch, confirmed to exist on Docker Hub 2026-08-23.
VAST_BASE_IMAGE="${HY3_VAST_BASE_IMAGE:-vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py311-2026-08-21}"

VAST_DEFAULT_DISK_GB="${HY3_VAST_DISK_GB:-200}"
VAST_DEFAULT_LABEL="${HY3_VAST_LABEL:-hy3-deploy}"
VAST_REMOTE_REPO_DIR="${HY3_REMOTE_REPO_DIR:-/root/Comfy_HunyuanImage3}"
VAST_INSTANCE_INFO_REMOTE_PATH="${HY3_INSTANCE_INFO_FILE:-/etc/hy3-instance-info.json}"

VAST_SSH_KEY="${HY3_VAST_SSH_KEY:-$VAST_SCRIPT_DIR/keys/id_ed25519}"
VAST_CURRENT_INSTANCE_FILE="${HY3_CURRENT_INSTANCE_FILE:-$VAST_SCRIPT_DIR/.current-instance}"

VAST_LOCAL_TUNNEL_PORT="${HY3_LOCAL_TUNNEL_PORT:-8188}"
VAST_REMOTE_COMFY_PORT="${HY3_REMOTE_COMFY_PORT:-8188}"

require_api_key() {
  if [[ -z "${VAST_API_KEY:-}" ]]; then
    if [[ -f "$VAST_REPO_ROOT/.env" ]]; then
      set -a
      # shellcheck disable=SC1091
      source "$VAST_REPO_ROOT/.env"
      set +a
    fi
  fi
  if [[ -z "${VAST_API_KEY:-}" ]]; then
    echo "error: VAST_API_KEY not set and not found in $VAST_REPO_ROOT/.env" >&2
    exit 2
  fi
}

# Reads a top-level string/number field from a JSON file. Usage:
#   json_field FILE field_name
json_field() {
  local file="$1" field="$2"
  [[ -f "$file" ]] || return 1
  python3 -c '
import json, sys
path, field = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
except (OSError, json.JSONDecodeError):
    sys.exit(1)
v = d.get(field)
if v is None:
    sys.exit(1)
print(v)
' "$file" "$field"
}
