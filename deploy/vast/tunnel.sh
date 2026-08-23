#!/usr/bin/env bash
# Establish an SSH local port forward to ComfyUI on the instance.
# ComfyUI binds 127.0.0.1 on the instance and only port 22 is exposed —
# this tunnel is the only way to reach the web UI. Runs in the
# foreground, reconnects on drop, with clear status output.
#
# Usage:
#   deploy/vast/tunnel.sh [--local-port N] [--remote-port N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=deploy/vast/config.sh
source "$SCRIPT_DIR/config.sh"

LOCAL_PORT="$VAST_LOCAL_TUNNEL_PORT"
REMOTE_PORT="$VAST_REMOTE_COMFY_PORT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local-port) LOCAL_PORT="${2:?}"; shift 2 ;;
    --remote-port) REMOTE_PORT="${2:?}"; shift 2 ;;
    --help|-h) echo "Usage: $(basename "$0") [--local-port N] [--remote-port N]"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

ssh_host="$(json_field "$VAST_CURRENT_INSTANCE_FILE" ssh_host || true)"
ssh_port="$(json_field "$VAST_CURRENT_INSTANCE_FILE" ssh_port || true)"

if [[ -z "$ssh_host" || -z "$ssh_port" ]]; then
  echo "error: no current instance connection info in $VAST_CURRENT_INSTANCE_FILE" >&2
  echo "Run deploy/vast/up.sh first." >&2
  exit 2
fi

echo "Tunnelling 127.0.0.1:$LOCAL_PORT -> instance 127.0.0.1:$REMOTE_PORT (via root@$ssh_host:$ssh_port)"
echo "Open http://127.0.0.1:$LOCAL_PORT once ComfyUI is started (deploy/comfy_start.sh)."
echo "Ctrl-C to stop. Reconnects automatically if the tunnel drops."

while true; do
  ssh -i "$VAST_SSH_KEY" -p "$ssh_port" -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -N -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
    "root@$ssh_host" \
    && echo "tunnel closed cleanly" \
    || echo "tunnel dropped (exit $?)"
  echo "reconnecting in 5s... (Ctrl-C to stop)"
  sleep 5
done
