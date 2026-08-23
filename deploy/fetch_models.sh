#!/usr/bin/env bash
# Download the pinned model checkpoint (see deploy/pins/model-revision.txt)
# to persistent storage, idempotently. Separate from provision.sh and
# separately re-runnable, per the briefing — these are 45-160GB
# artifacts and vast bandwidth is billed, so this never blindly
# re-downloads: it verifies size before AND after pulling, and skips
# entirely if a prior run already left a complete, matching copy.
#
# Usage:
#   deploy/fetch_models.sh [--dry-run] [--force]
#
# Env overrides:
#   HY3_MODEL_REPO, HY3_MODEL_REVISION  — override the pin
#   HY3_MODELS_DIR                      — default: $REPO_ROOT/models
#   HY3_HF_API_BASE                     — default: https://huggingface.co

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PIN_FILE="$SCRIPT_DIR/pins/model-revision.txt"

pinned_lines() { grep -vE '^\s*#|^\s*$' "$PIN_FILE"; }

DEFAULT_REPO="$(pinned_lines | sed -n '1p')"
DEFAULT_REVISION="$(pinned_lines | sed -n '2p')"

MODEL_REPO="${HY3_MODEL_REPO:-$DEFAULT_REPO}"
MODEL_REVISION="${HY3_MODEL_REVISION:-$DEFAULT_REVISION}"
MODELS_DIR="${HY3_MODELS_DIR:-$REPO_ROOT/models}"
HF_API_BASE="${HY3_HF_API_BASE:-https://huggingface.co}"

DRY_RUN=0
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    --help|-h) echo "Usage: $(basename "$0") [--dry-run] [--force]"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

TARGET_DIR="$MODELS_DIR/$(basename "$MODEL_REPO")"
MANIFEST_FILE="$MODELS_DIR/.manifest-$(basename "$MODEL_REPO")-$MODEL_REVISION.json"

echo "Model:    $MODEL_REPO"
echo "Revision: $MODEL_REVISION"
echo "Target:   $TARGET_DIR"

mkdir -p "$MODELS_DIR"

echo "Fetching expected file manifest from $HF_API_BASE ..."
curl -sS "$HF_API_BASE/api/models/$MODEL_REPO/tree/$MODEL_REVISION?recursive=true" \
  --max-time 30 -o "$MANIFEST_FILE.raw"

python3 -c '
import json, sys
raw = json.load(open(sys.argv[1]))
out = [{"path": e["path"], "size": e.get("size")} for e in raw if e.get("type") == "file"]
json.dump(out, open(sys.argv[2], "w"), indent=2)
total_gb = sum(e["size"] or 0 for e in out) / 1e9
print(f"{len(out)} files in manifest, {total_gb:.1f} GB total")
' "$MANIFEST_FILE.raw" "$MANIFEST_FILE"
rm -f "$MANIFEST_FILE.raw"

already_complete() {
  [[ -d "$TARGET_DIR" ]] || return 1
  python3 "$SCRIPT_DIR/lib/verify_download.py" --manifest "$MANIFEST_FILE" --target-dir "$TARGET_DIR" >/dev/null 2>&1
}

if [[ "$FORCE" != "1" ]] && already_complete; then
  echo "Already complete and verified at $TARGET_DIR — nothing to download."
  echo "(use --force to re-download anyway)"
  exit 0
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] would run: huggingface-cli download $MODEL_REPO --revision $MODEL_REVISION --local-dir $TARGET_DIR"
  python3 "$SCRIPT_DIR/lib/verify_download.py" --manifest "$MANIFEST_FILE" --target-dir "$TARGET_DIR" || true
  exit 0
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "error: huggingface-cli not found. It ships with huggingface_hub[cli]," >&2
  echo "which deploy/pins/tencent-requirements.txt pins — activate the venv first." >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"
echo "Downloading (idempotent — huggingface-cli resumes partial files)..."
huggingface-cli download "$MODEL_REPO" --revision "$MODEL_REVISION" --local-dir "$TARGET_DIR"

echo "Verifying download against manifest..."
if ! python3 "$SCRIPT_DIR/lib/verify_download.py" --manifest "$MANIFEST_FILE" --target-dir "$TARGET_DIR"; then
  echo "error: download completed but verification failed — see problems above." >&2
  exit 1
fi

echo "fetch_models: PASS — $TARGET_DIR is complete and verified."
