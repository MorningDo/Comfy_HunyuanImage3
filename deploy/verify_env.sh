#!/usr/bin/env bash
# Diffs the active venv's installed packages against
# deploy/pins/tencent-requirements.txt (the authoritative baseline) and
# deploy/pins/DEVIATIONS.md (approved, logged exceptions). Any
# undocumented drift is a build failure, not a warning — pip will
# silently upgrade a pinned package as a transitive dependency of a
# later install, and this is the check that catches it.
#
# Run at the end of provision.sh, and again before any test run.
#
# Env overrides (for testing off-instance, without a real venv):
#   HY3_PIP_FREEZE_CMD  — command to produce `pip freeze`-style output
#                          (default: "pip freeze")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PIN_FILE="$REPO_ROOT/deploy/pins/tencent-requirements.txt"
DEVIATIONS_FILE="$REPO_ROOT/deploy/pins/DEVIATIONS.md"
FREEZE_CMD="${HY3_PIP_FREEZE_CMD:-pip freeze}"

freeze_file="$(mktemp)"
trap 'rm -f "$freeze_file"' EXIT
eval "$FREEZE_CMD" > "$freeze_file"

freeze_version() {
  local pkg_norm
  pkg_norm="$(echo "$1" | tr 'A-Z_' 'a-z-')"
  # `|| true`: no match means "not installed", which the caller handles
  # via the empty-string check — not a script-level error under -e/pipefail.
  grep -iE "^${pkg_norm}==" "$freeze_file" | tr 'A-Z_' 'a-z-' | cut -d= -f3 || true
}

approved_deviation() {
  local pkg="$1" version="$2"
  [[ -f "$DEVIATIONS_FILE" ]] || return 1
  grep -iE "^\| *${pkg} *\|.*\| *${version} *\|.*\| *yes *\|" "$DEVIATIONS_FILE" >/dev/null 2>&1
}

fail=0

while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  pkg_raw="${line%%==*}"
  pinned="${line##*==}"
  pkg="${pkg_raw%%\[*}" # strip extras, e.g. transformers[accelerate,tiktoken]
  installed="$(freeze_version "$pkg")"
  if [[ -z "$installed" ]]; then
    echo "MISSING: $pkg (pinned $pinned) not found in installed environment"
    fail=1
  elif [[ "$installed" != "$pinned" ]]; then
    if approved_deviation "$pkg" "$installed"; then
      echo "OK (approved deviation): $pkg installed=$installed pinned=$pinned — see DEVIATIONS.md"
    else
      echo "DRIFT: $pkg pinned=$pinned installed=$installed — undocumented, treating as failure"
      fail=1
    fi
  else
    echo "OK: $pkg==$pinned"
  fi
done < <(grep -E '^[A-Za-z0-9_.-]+(\[[^]]*\])?==' "$PIN_FILE")

if [[ "$fail" -eq 0 ]]; then
  echo "verify_env: PASS — no undocumented drift from Tencent pins"
else
  echo "verify_env: FAIL — see DRIFT/MISSING lines above" >&2
fi

exit "$fail"
