#!/usr/bin/env bash
# Push the local repo state to the instance over rsync/SSH. This is how
# iteration happens: edit locally, sync, re-run the relevant provision
# stage over SSH. Never edit the remote instance directly and leave the
# fix only there — sync it back by editing locally and re-running this.
#
# Usage:
#   deploy/vast/sync.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=deploy/vast/config.sh
source "$SCRIPT_DIR/config.sh"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

ssh_host="$(json_field "$VAST_CURRENT_INSTANCE_FILE" ssh_host || true)"
ssh_port="$(json_field "$VAST_CURRENT_INSTANCE_FILE" ssh_port || true)"

if [[ -z "$ssh_host" || -z "$ssh_port" ]]; then
  echo "error: no current instance connection info in $VAST_CURRENT_INSTANCE_FILE" >&2
  echo "Run deploy/vast/up.sh first." >&2
  exit 2
fi

rsync_args=(
  -avz --delete
  -e "ssh -i $VAST_SSH_KEY -p $ssh_port -o StrictHostKeyChecking=accept-new"
  --exclude ".git/"
  --exclude "models/"
  --exclude "venv/"
  --exclude ".venv/"
  --exclude "__pycache__/"
  --exclude "*.pyc"
  --exclude ".env"
  --exclude "deploy/vast/keys/"
  --exclude "deploy/vast/.current-instance"
)
[[ "$DRY_RUN" == "1" ]] && rsync_args+=(--dry-run)

echo "Syncing $VAST_REPO_ROOT/ -> root@$ssh_host:$VAST_REMOTE_REPO_DIR/ ${DRY_RUN:+[dry-run]}"
rsync "${rsync_args[@]}" "$VAST_REPO_ROOT/" "root@$ssh_host:$VAST_REMOTE_REPO_DIR/"
