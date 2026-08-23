#!/usr/bin/env bash
# Create a vast.ai instance from a chosen offer (from deploy/vast/search.py),
# wait for SSH to come up, and record connection details in
# deploy/vast/.current-instance (gitignored — this is per-instance state,
# not source of truth).
#
# NEVER creates an instance without an explicit interactive confirmation
# (or --yes, for scripted/CI use where the caller has already confirmed
# out of band) — see the briefing's "ask before spending" rule.
#
# Usage:
#   deploy/vast/up.sh --offer-id ID [--disk-gb N] [--label NAME]
#                      [--yes] [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=deploy/vast/config.sh
source "$SCRIPT_DIR/config.sh"

OFFER_ID=""
DISK_GB="$VAST_DEFAULT_DISK_GB"
LABEL="$VAST_DEFAULT_LABEL"
ASSUME_YES=0
DRY_RUN=0
POLL_TIMEOUT_S="${HY3_UP_POLL_TIMEOUT_S:-600}"
POLL_INTERVAL_S="${HY3_UP_POLL_INTERVAL_S:-10}"

usage() {
  cat <<EOF
Usage: $(basename "$0") --offer-id ID [--disk-gb N] [--label NAME] [--yes] [--dry-run]

Run deploy/vast/search.py first to find an offer id. This script rents
real, billed hardware unless --dry-run is given.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --offer-id) OFFER_ID="${2:?}"; shift 2 ;;
    --disk-gb) DISK_GB="${2:?}"; shift 2 ;;
    --label) LABEL="${2:?}"; shift 2 ;;
    --yes) ASSUME_YES=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$OFFER_ID" ]] || { echo "error: --offer-id is required (see deploy/vast/search.py)" >&2; exit 2; }

require_api_key

request_body="$(python3 -c '
import json, sys
print(json.dumps({
    "client_id": "me",
    "image": sys.argv[1],
    "disk": float(sys.argv[2]),
    "label": sys.argv[3],
    "runtype": "ssh",
    "env": {},
    "onstart": None,
}))
' "$VAST_BASE_IMAGE" "$DISK_GB" "$LABEL")"

echo "About to create a vast.ai instance:"
echo "  offer id: $OFFER_ID"
echo "  image:    $VAST_BASE_IMAGE"
echo "  disk:     ${DISK_GB} GB"
echo "  label:    $LABEL"
echo "This will incur real, billed cost until deploy/vast/down.sh is run."

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] would PUT $VAST_API_BASE/asks/$OFFER_ID/ with body:"
  echo "$request_body"
  exit 0
fi

if [[ "$ASSUME_YES" != "1" ]]; then
  read -r -p "Type the offer id ($OFFER_ID) again to confirm: " confirm
  if [[ "$confirm" != "$OFFER_ID" ]]; then
    echo "Confirmation did not match offer id — aborting, nothing created." >&2
    exit 1
  fi
fi

response="$(curl -sS -X PUT "$VAST_API_BASE/asks/$OFFER_ID/" \
  -H "Authorization: Bearer $VAST_API_KEY" \
  -H "Content-Type: application/json" \
  -d "$request_body")"

instance_id="$(python3 -c '
import json, sys
d = json.loads(sys.argv[1])
if d.get("success") is False:
    print("error: vast.ai rejected the request: " + str(d.get("msg")), file=sys.stderr)
    sys.exit(1)
iid = d.get("new_contract") or d.get("instance_id") or d.get("id")
if iid is None:
    print("error: could not find an instance id in response: " + sys.argv[1], file=sys.stderr)
    sys.exit(1)
print(iid)
' "$response")"

echo "Created instance $instance_id. Waiting for SSH (timeout ${POLL_TIMEOUT_S}s)..."

deadline=$((SECONDS + POLL_TIMEOUT_S))
ssh_host=""
ssh_port=""
status=""
status_msg=""
ssh_ready=0
while (( SECONDS < deadline )); do
  info="$(curl -sS "$VAST_API_BASE/instances/$instance_id/?owner=me" -H "Authorization: Bearer $VAST_API_KEY")"
  status="$(echo "$info" | python3 -c 'import json,sys; d=json.load(sys.stdin).get("instances") or {}; print(d.get("actual_status") or "")')"
  status_msg="$(echo "$info" | python3 -c 'import json,sys; d=json.load(sys.stdin).get("instances") or {}; print(d.get("status_msg") or "")')"
  ssh_host="$(echo "$info" | python3 -c 'import json,sys; d=json.load(sys.stdin).get("instances") or {}; print(d.get("ssh_host") or "")')"
  ssh_port="$(echo "$info" | python3 -c 'import json,sys; d=json.load(sys.stdin).get("instances") or {}; print(d.get("ssh_port") or "")')"
  echo "  status=${status:-unknown} ssh_host=${ssh_host:-<pending>}"
  if [[ -n "$ssh_host" && -n "$ssh_port" && "$status" == "running" ]]; then
    ssh_ready=1
    break
  fi
  sleep "$POLL_INTERVAL_S"
done

# Deliberately check the flag set only by the break above, not just
# ssh_host/ssh_port being non-empty — vast.ai's API populates the SSH
# proxy host/port well before the container is actually running (e.g.
# while still "loading" or even stuck failing to pull its image), so
# checking only for non-empty host/port here previously declared
# success on a dead/loading instance. Caught by a real run 2026-08-23:
# the instance never got past status=loading (bad image tag), yet the
# old check called it "SSH ready" and the next command's SSH connection
# was immediately reset by the remote host.
if [[ "$ssh_ready" != "1" ]]; then
  echo "error: instance $instance_id did not reach status=running within ${POLL_TIMEOUT_S}s (last status: ${status:-unknown})." >&2
  [[ -n "$status_msg" ]] && echo "status_msg from vast.ai: $status_msg" >&2
  echo "It is still being billed. Check the vast.ai console, or run:" >&2
  echo "  deploy/vast/down.sh --instance-id $instance_id" >&2
  exit 1
fi

python3 -c '
import json, sys, time
path = sys.argv[1]
data = {
    "instance_id": sys.argv[2],
    "offer_id": sys.argv[3],
    "ssh_host": sys.argv[4],
    "ssh_port": int(sys.argv[5]),
    "image": sys.argv[6],
    "label": sys.argv[7],
    "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
json.dump(data, open(path, "w"), indent=2)
print()
' "$VAST_CURRENT_INSTANCE_FILE" "$instance_id" "$OFFER_ID" "$ssh_host" "$ssh_port" "$VAST_BASE_IMAGE" "$LABEL"

echo "SSH ready: $ssh_host:$ssh_port"
echo "Wrote $VAST_CURRENT_INSTANCE_FILE"

echo "Pushing instance metadata to $VAST_INSTANCE_INFO_REMOTE_PATH on the instance..."
python3 -c '
import json, sys
print(json.dumps({
    "offer_id": sys.argv[1],
    "instance_id": sys.argv[2],
    "image": sys.argv[3],
}))
' "$OFFER_ID" "$instance_id" "$VAST_BASE_IMAGE" | \
  ssh -i "$VAST_SSH_KEY" -p "$ssh_port" -o StrictHostKeyChecking=accept-new \
    "root@$ssh_host" "cat > $VAST_INSTANCE_INFO_REMOTE_PATH" \
  || echo "warn: could not write instance info on the instance yet (may still be booting) — provision.sh's manifest step will show nulls for vast.* fields until this is retried"

echo "Next: deploy/vast/sync.sh, then ssh (deploy/vast/ssh.sh) and run deploy/provision.sh"
