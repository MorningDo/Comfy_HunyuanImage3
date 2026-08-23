#!/usr/bin/env bash
# SSH wrapper that reads connection details from
# deploy/vast/.current-instance so nobody hand-types IPs/ports.
#
# Usage:
#   deploy/vast/ssh.sh                 # interactive shell
#   deploy/vast/ssh.sh -- some command # run a remote command

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=deploy/vast/config.sh
source "$SCRIPT_DIR/config.sh"

ssh_host="$(json_field "$VAST_CURRENT_INSTANCE_FILE" ssh_host || true)"
ssh_port="$(json_field "$VAST_CURRENT_INSTANCE_FILE" ssh_port || true)"

if [[ -z "$ssh_host" || -z "$ssh_port" ]]; then
  echo "error: no current instance connection info in $VAST_CURRENT_INSTANCE_FILE" >&2
  echo "Run deploy/vast/up.sh first." >&2
  exit 2
fi

exec ssh -i "$VAST_SSH_KEY" -p "$ssh_port" -o StrictHostKeyChecking=accept-new "root@$ssh_host" "$@"
