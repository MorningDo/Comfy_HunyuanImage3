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

dry_run_label=""
[[ "$DRY_RUN" == "1" ]] && dry_run_label=" [dry-run]"
echo "Syncing $VAST_REPO_ROOT/ -> root@$ssh_host:$VAST_REMOTE_REPO_DIR/${dry_run_label}"
rsync "${rsync_args[@]}" "$VAST_REPO_ROOT/" "root@$ssh_host:$VAST_REMOTE_REPO_DIR/"

if [[ "$DRY_RUN" != "1" ]]; then
  # .git/ is deliberately excluded above (no point shipping the whole
  # history/objects on every sync), which means write_manifest.py can't
  # `git rev-parse` on the instance — confirmed live: node_pack_sha and
  # branch both came back null in the first real manifest. Carry just
  # the two values that matter instead.
  git_sha="$(git -C "$VAST_REPO_ROOT" rev-parse HEAD)"
  git_branch="$(git -C "$VAST_REPO_ROOT" rev-parse --abbrev-ref HEAD)"
  python3 -c '
import json, sys
print(json.dumps({"node_pack_sha": sys.argv[1], "branch": sys.argv[2]}))
' "$git_sha" "$git_branch" | \
    ssh -i "$VAST_SSH_KEY" -p "$ssh_port" -o StrictHostKeyChecking=accept-new \
      "root@$ssh_host" "cat > $VAST_REMOTE_REPO_DIR/.git-info.json"
fi
