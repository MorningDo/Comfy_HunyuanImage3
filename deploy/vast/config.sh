#!/usr/bin/env bash
# Shared configuration for deploy/vast/*.sh. Sourced, not executed
# directly. All values overridable via env for testing.
#
# shellcheck disable=SC2034  # many of these are used only by sourcing scripts

VAST_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAST_REPO_ROOT="$(cd "$VAST_SCRIPT_DIR/../.." && pwd)"

VAST_API_BASE="${HY3_VAST_API_BASE:-https://console.vast.ai/api/v0}"

# Pinned per deploy/pins/torch-cuda-notes.md — devel (not runtime) tag,
# no pre-baked torch. Confirmed to exist via a direct Docker Hub API
# call (not WebFetch, which fabricated a plausible-looking but wrong
# date-suffixed variant of this tag during phase-1 planning and caused
# a real failed instance: "manifest unknown" from the container
# runtime, discovered on the first live up.sh run 2026-08-23).
VAST_BASE_IMAGE="${HY3_VAST_BASE_IMAGE:-vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py311}"

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

# Runs a command on the instance inside a detached tmux session, so it
# survives an SSH disconnection, then polls for completion via short
# reconnecting SSH calls (never one long-lived connection). Confirmed
# live 2026-08-23: a 51GB model download over a single foreground SSH
# session got cut by "Connection closed by remote host" partway
# through — the instance itself was fine (nvidia-smi/tmux both still
# reachable seconds later), the long connection itself was the
# fragile part. Any remote step that might run more than a couple
# minutes (model download, model load + generation, big pip installs)
# should go through this rather than a bare `ssh.sh "..."` call.
#
# Usage: run_remote_resilient SESSION_NAME REMOTE_LOG_PATH COMMAND
# Echoes the remote command's tail as it polls; returns the remote
# command's real exit code (parsed from a sentinel line it appends).
run_remote_resilient() {
  local session_name="$1" remote_log="$2" cmd="$3"
  local ssh_host ssh_port
  ssh_host="$(json_field "$VAST_CURRENT_INSTANCE_FILE" ssh_host || true)"
  ssh_port="$(json_field "$VAST_CURRENT_INSTANCE_FILE" ssh_port || true)"
  if [[ -z "$ssh_host" || -z "$ssh_port" ]]; then
    echo "error: no current instance connection info in $VAST_CURRENT_INSTANCE_FILE" >&2
    return 2
  fi

  # -q: vast.ai's per-connection login banner goes to stderr and
  # bypasses the stdout capture below entirely, so it can't be
  # filtered the same way as the sentinel lines — confirmed live it
  # was printing on every single poll iteration. Quiet mode suppresses
  # it; troubleshooting an actual connection failure here still shows
  # a non-zero exit / empty output, just without the banner noise.
  local ssh_opts=(-i "$VAST_SSH_KEY" -p "$ssh_port" -q -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)

  # Write the actual command to a small script file on the instance
  # first, rather than nesting it as an escaped string inside the ssh
  # command inside the tmux new-session argument. That three-layer
  # nesting was tried first and broke in confusing ways live (an `&&`
  # in cmd meant a trailing redirect only applied to the last command
  # in the chain; a stray unescaped character somewhere left bash
  # trying to execute the log path itself as a command). A script file
  # collapses this to one simple, unambiguous remote command line.
  local remote_script="/tmp/hy3-run-${session_name}.sh"
  # shellcheck disable=SC2029  # intentional client-side expansion
  printf '#!/usr/bin/env bash\nset -o pipefail\n%s\n' "$cmd" | \
    ssh "${ssh_opts[@]}" "root@$ssh_host" "cat > $remote_script && chmod +x $remote_script"

  local remote_start="if tmux has-session -t $session_name 2>/dev/null; then echo ALREADY_RUNNING; else tmux new-session -d -s $session_name \"$remote_script > $remote_log 2>&1; echo __HY3_EXIT_CODE:\\\$?__ >> $remote_log\"; echo STARTED; fi"
  # shellcheck disable=SC2029  # intentional client-side expansion
  ssh "${ssh_opts[@]}" "root@$ssh_host" "$remote_start"

  local last_size=0
  while true; do
    local poll_out
    # Prefixed sentinels (__HY3_SIZE:.../ __HY3_STILL_RUNNING__ /
    # __HY3_ENDED__), filtered by grep -v rather than positional
    # "last N lines" — the latter was tried first and broke live:
    # sed '$d;$d' only deletes the last line ONCE (its `d` ends the
    # cycle before the second $d command can run), so the byte-count
    # sentinel leaked into the visible output. Prefix-based filtering
    # is also immune to vast.ai's per-connection SSH login banner,
    # which was leaking through the same way.
    # shellcheck disable=SC2029  # intentional client-side expansion
    # wc -c FILE (not `wc -c < FILE`) deliberately: with the redirect
    # form, bash sets up `<FILE` before `2>/dev/null` takes effect, so
    # a missing-file error prints to the ORIGINAL stderr regardless of
    # the trailing redirect — confirmed live, this leaked "bash: line
    # 1: FILE: No such file or directory" on every poll before the
    # remote script had created its log file yet. The file-argument
    # form lets wc handle a missing file itself, where 2>/dev/null
    # actually suppresses it; cat pipes it to plain "0" either way.
    poll_out="$(ssh "${ssh_opts[@]}" "root@$ssh_host" "tail -c +$((last_size + 1)) $remote_log 2>/dev/null; echo __HY3_SIZE:\$(cat $remote_log 2>/dev/null | wc -c)__; tmux has-session -t $session_name 2>/dev/null && echo __HY3_STILL_RUNNING__ || echo __HY3_ENDED__")"
    local new_output
    new_output="$(echo "$poll_out" | grep -v '^Welcome to vast\.ai\|^__HY3_\|shell access\|Have fun!')"
    [[ -n "$new_output" ]] && echo "$new_output"
    if [[ "$poll_out" =~ __HY3_SIZE:([0-9]+)__ ]]; then
      last_size="${BASH_REMATCH[1]}"
    fi
    [[ "$poll_out" == *"__HY3_ENDED__"* ]] && break
    sleep 15
  done

  local final_log
  # shellcheck disable=SC2029  # intentional client-side expansion
  final_log="$(ssh "${ssh_opts[@]}" "root@$ssh_host" "cat $remote_log" 2>/dev/null)"
  if [[ "$final_log" =~ __HY3_EXIT_CODE:([0-9]+)__ ]]; then
    return "${BASH_REMATCH[1]}"
  fi
  echo "warn: could not determine remote exit code from $remote_log" >&2
  return 1
}
