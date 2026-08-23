#!/usr/bin/env bash
# Destroy a vast.ai instance (irreversible — deletes any data not on
# persistent/model storage). Requires interactive confirmation unless
# --yes is given. Always prints a final billing note.
#
# Usage:
#   deploy/vast/down.sh [--instance-id ID] [--yes]
#
# With no --instance-id, reads deploy/vast/.current-instance.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=deploy/vast/config.sh
source "$SCRIPT_DIR/config.sh"

INSTANCE_ID=""
ASSUME_YES=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--instance-id ID] [--yes]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --instance-id) INSTANCE_ID="${2:?}"; shift 2 ;;
    --yes) ASSUME_YES=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$INSTANCE_ID" ]]; then
  INSTANCE_ID="$(json_field "$VAST_CURRENT_INSTANCE_FILE" instance_id || true)"
fi
[[ -n "$INSTANCE_ID" ]] || { echo "error: no --instance-id given and none found in $VAST_CURRENT_INSTANCE_FILE" >&2; exit 2; }

require_api_key

created_at="$(json_field "$VAST_CURRENT_INSTANCE_FILE" created_at_utc || echo "unknown")"
echo "About to DESTROY instance $INSTANCE_ID (created: $created_at)."
echo "This is irreversible and stops billing for this instance."

if [[ "$ASSUME_YES" != "1" ]]; then
  read -r -p "Type 'destroy' to confirm: " confirm
  if [[ "$confirm" != "destroy" ]]; then
    echo "Confirmation did not match — aborting, instance NOT destroyed." >&2
    exit 1
  fi
fi

response="$(curl -sS -X DELETE "$VAST_API_BASE/instances/$INSTANCE_ID/" \
  -H "Authorization: Bearer $VAST_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{}')"

ok="$(echo "$response" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("1" if d.get("success") else "0")' 2>/dev/null || echo "0")"

if [[ "$ok" == "1" ]]; then
  echo "Destroyed instance $INSTANCE_ID."
  if [[ -f "$VAST_CURRENT_INSTANCE_FILE" ]]; then
    mv "$VAST_CURRENT_INSTANCE_FILE" "$VAST_CURRENT_INSTANCE_FILE.destroyed-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  echo "BILLING NOTE: instance $INSTANCE_ID destroyed at $(date -u +%FT%TZ). Verify in the vast.ai console that billing has actually stopped."
else
  echo "error: destroy request did not report success: $response" >&2
  exit 1
fi
