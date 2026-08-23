#!/usr/bin/env bash
# Orchestrates the full validation pipeline: sync -> smoke checks (run
# on the instance, over SSH, cheapest first) -> pull the generated
# image back -> host-side vision-judge (tests/validate_image.py) ->
# one combined report written to deploy/manifests/.
#
# Smoke checks run on the instance because they need the GPU/model.
# validate_image.py runs on the HOST because OPENROUTER_API_KEY must
# never reach the instance (see CLAUDE_CODE_BRIEFING.md's credentials
# table). This split is why this script exists rather than just
# calling the smoke scripts directly over SSH.
#
# Usage:
#   tests/run_all.sh [--skip-sync] [--skip-judge] [--strict]
#
# --strict: also fail (nonzero exit) if the advisory vision-judge step
#   fails an assertion, not just if a smoke check fails.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=deploy/vast/config.sh
source "$REPO_ROOT/deploy/vast/config.sh"

SKIP_SYNC=0
SKIP_JUDGE=0
STRICT=0
CASE_ID="${HY3_SMOKE_CASE_ID:-red-apple-01}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-sync) SKIP_SYNC=1; shift ;;
    --skip-judge) SKIP_JUDGE=1; shift ;;
    --strict) STRICT=1; shift ;;
    --help|-h) echo "Usage: $(basename "$0") [--skip-sync] [--skip-judge] [--strict]"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPORT_DIR="$REPO_ROOT/deploy/manifests"
mkdir -p "$REPORT_DIR"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_IMAGE_PATH="$REPO_ROOT/tests/smoke/.last-generation.png"
REPORT_FILE="$REPORT_DIR/test-report-$TIMESTAMP.json"

declare -A STAGE_STATUS
STAGE_STATUS=()
OVERALL_PASS=1
JUDGE_SUMMARY=""

REMOTE_VENV_DIR="${HY3_VENV_DIR:-/opt/hy3/venv}"

run_remote() {
  # Runs a command on the instance inside the activated venv, inside a
  # tmux session so it survives an SSH drop — check_generation.py loads
  # a ~50GB quantized model and runs a real generation, easily long
  # enough to hit the same "Connection closed by remote host" a plain
  # foreground SSH session hit for the model download in this same
  # session. session_name doubles as the log filename so concurrent
  # stages (there aren't any here, but future callers) don't collide.
  local session_name="$1" cmd="$2"
  run_remote_resilient "$session_name" "/root/${session_name}.log" \
    "cd $VAST_REMOTE_REPO_DIR && source $REMOTE_VENV_DIR/bin/activate && $cmd"
}

record() {
  # record NAME STATUS [affects_result=1]
  # affects_result=0 is for the advisory vision-judge stage: its FAIL
  # is recorded but doesn't flip the overall result unless --strict
  # (handled separately where it's called).
  local name="$1" status="$2" affects_result="${3:-1}"
  STAGE_STATUS["$name"]="$status"
  echo "=== $name: $status ==="
  # SKIPPED never flips the result on its own: either an upstream stage
  # already FAILed (which already flipped it), or skipping was
  # explicitly requested (--skip-sync/--skip-judge), which isn't a
  # failure. Only an actual FAIL, for a stage where it matters, counts.
  if [[ "$status" == "FAIL" && "$affects_result" == "1" ]]; then
    OVERALL_PASS=0
  fi
  return 0
}

if [[ "$SKIP_SYNC" != "1" ]]; then
  echo "--- syncing repo to instance ---"
  "$REPO_ROOT/deploy/vast/sync.sh"
fi

echo "--- smoke: imports ---"
if run_remote hy3-check-imports "python3 tests/smoke/check_imports.py"; then
  record smoke_imports PASS
else
  record smoke_imports FAIL
fi

if [[ "${STAGE_STATUS[smoke_imports]}" == "PASS" ]]; then
  echo "--- smoke: model quantization ---"
  if run_remote hy3-check-model-quant "python3 tests/smoke/check_model_quant.py"; then
    record smoke_model_quant PASS
  else
    record smoke_model_quant FAIL
  fi
else
  record smoke_model_quant SKIPPED
fi

if [[ "${STAGE_STATUS[smoke_model_quant]}" == "PASS" ]]; then
  echo "--- smoke: minimal generation ---"
  if run_remote hy3-check-generation "HY3_SMOKE_PROMPT='A red apple on a wooden table, studio lighting' python3 tests/smoke/check_generation.py"; then
    record smoke_generation PASS
  else
    record smoke_generation FAIL
  fi
else
  record smoke_generation SKIPPED
fi

if [[ "${STAGE_STATUS[smoke_generation]}" == "PASS" && "$SKIP_JUDGE" != "1" ]]; then
  echo "--- pulling generated image back to host ---"
  ssh_host="$(json_field "$VAST_CURRENT_INSTANCE_FILE" ssh_host || true)"
  ssh_port="$(json_field "$VAST_CURRENT_INSTANCE_FILE" ssh_port || true)"
  if scp -i "$VAST_SSH_KEY" -P "$ssh_port" -o StrictHostKeyChecking=accept-new \
      "root@$ssh_host:$VAST_REMOTE_REPO_DIR/tests/smoke/.last-generation.png" "$LOCAL_IMAGE_PATH"; then
    record pull_image PASS
  else
    record pull_image FAIL
  fi
else
  record pull_image SKIPPED
fi

if [[ "${STAGE_STATUS[pull_image]}" == "PASS" ]]; then
  echo "--- host-side vision judge (advisory) ---"
  # Always pass --strict to validate_image.py itself so its exit code
  # actually reflects whether any assertion failed; run_all.sh's own
  # --strict (via record()'s affects_result arg below) then decides
  # whether that failure propagates to the overall pipeline result.
  judge_output="$(python3 "$REPO_ROOT/tests/validate_image.py" --image "$LOCAL_IMAGE_PATH" --case-id "$CASE_ID" --strict 2>&1)" \
    && judge_exit=0 || judge_exit=$?
  echo "$judge_output"
  if [[ "$judge_exit" == "0" ]]; then
    record vision_judge PASS
  else
    record vision_judge FAIL "$STRICT"
  fi
  JUDGE_SUMMARY="$judge_output"
else
  record vision_judge SKIPPED
fi

RESULT="PASS"
[[ "$OVERALL_PASS" == "1" ]] || RESULT="FAIL"

STAGES_FILE="$(mktemp)"
trap 'rm -f "$STAGES_FILE"' EXIT
for name in "${!STAGE_STATUS[@]}"; do
  echo "$name=${STAGE_STATUS[$name]}"
done > "$STAGES_FILE"

python3 -c '
import json, sys, subprocess

stages_file, result, case_id, report_file, repo_root, judge_summary = sys.argv[1:7]

stages = {}
with open(stages_file) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        name, status = line.split("=", 1)
        stages[name] = status

git_sha = subprocess.run(
    ["git", "-C", repo_root, "rev-parse", "HEAD"],
    capture_output=True, text=True, check=True,
).stdout.strip()

report = {
    "schema_version": 1,
    "timestamp_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
    "git_sha": git_sha,
    "case_id": case_id,
    "stages": stages,
    "judge_output": judge_summary or None,
    "result": result,
}
with open(report_file, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nWrote report: {report_file}")
' "$STAGES_FILE" "$RESULT" "$CASE_ID" "$REPORT_FILE" "$REPO_ROOT" "$JUDGE_SUMMARY"

echo "=== tests/run_all.sh: $RESULT ==="
[[ "$RESULT" == "PASS" ]]
