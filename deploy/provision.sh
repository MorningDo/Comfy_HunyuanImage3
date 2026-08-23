#!/usr/bin/env bash
# Idempotent, staged provisioning for a vast.ai instance running
# Comfy_HunyuanImage3. Runs ON the instance. Safe to re-run: each stage
# skips if a prior run's marker matches the stage's current logic AND a
# real-state verification check passes; otherwise it re-runs.
#
# Usage:
#   deploy/provision.sh                # run all stages
#   deploy/provision.sh --stage torch  # run/verify a single stage
#   deploy/provision.sh --dry-run      # log intended actions, mutate nothing
#
# Env overrides (mainly for local/dry-run testing off-instance):
#   HY3_STATE_DIR, HY3_VENV_DIR, HY3_COMFYUI_DIR, HY3_LOG_FILE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

STAGES=(system_packages venv torch comfyui node_install reconcile manifest)

DRY_RUN=0
ONLY_STAGE=""
STATE_DIR="${HY3_STATE_DIR:-/var/lib/hy3-provision/state}"
VENV_DIR="${HY3_VENV_DIR:-/opt/hy3/venv}"
COMFYUI_DIR="${HY3_COMFYUI_DIR:-/opt/hy3/ComfyUI}"
LOG_FILE="${HY3_LOG_FILE:-/var/log/hy3-provision.log}"
OVERALL_STATUS="PASS"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run] [--stage NAME] [--help]

Stages (in order): ${STAGES[*]}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --stage) ONLY_STAGE="${2:?--stage requires a NAME}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log() {
  local msg
  msg="[$(date -u +%FT%TZ)] $*"
  mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
  if ! echo "$msg" | tee -a "$LOG_FILE" 2>/dev/null; then
    echo "$msg"
  fi
}

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[dry-run] $*"
  else
    log "+ $*"
    "$@"
  fi
}

activate_venv() {
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  local resolved
  resolved="$(command -v python)"
  case "$resolved" in
    "$VENV_DIR"/*) ;;
    *)
      log "FATAL: 'which python' resolved to $resolved, outside venv $VENV_DIR"
      log "Refusing to continue: system Python must never receive installs."
      exit 1
      ;;
  esac
}

comfyui_pinned_sha() {
  grep -E '^[0-9a-f]{40}$' "$REPO_ROOT/deploy/pins/comfyui-commit.txt"
}

# ---- stage marker / idempotency machinery ----------------------------

stage_marker() { echo "$STATE_DIR/$1.done"; }

stage_hash() {
  # Hash the stage's own function body, so editing a stage's logic
  # automatically invalidates its marker on the next run. Known gap:
  # this does NOT cover edits to shared helpers a stage calls into
  # (e.g. torch_index_url_for_driver, activate_venv) — a helper-only
  # change requires manually removing the affected marker file(s)
  # from $STATE_DIR before the next run.
  declare -f "stage_$1" | sha256sum | awk '{print $1}'
}

stage_marker_matches() {
  local name="$1" marker recorded
  marker="$(stage_marker "$name")"
  [[ -f "$marker" ]] || return 1
  recorded="$(awk -F= '/^hash=/{print $2}' "$marker")"
  [[ "$recorded" == "$(stage_hash "$name")" ]]
}

mark_stage_done() {
  local name="$1"
  mkdir -p "$STATE_DIR" 2>/dev/null || true
  {
    echo "timestamp=$(date -u +%FT%TZ)"
    echo "hash=$(stage_hash "$name")"
  } > "$(stage_marker "$name")" 2>/dev/null || log "warn: could not write marker for $name (state dir not writable)"
}

run_stage() {
  local name="$1"
  if [[ -n "$ONLY_STAGE" && "$ONLY_STAGE" != "$name" ]]; then
    return 0
  fi

  log "=== stage: $name ==="

  if stage_marker_matches "$name" && "verify_stage_$name"; then
    log "skip: $name already complete and verified"
    return 0
  fi

  if stage_marker_matches "$name"; then
    log "warn: $name marker present but live verification failed — re-running"
  fi

  if ! "stage_$name"; then
    log "FAIL: $name errored"
    OVERALL_STATUS="FAIL"
    return 1
  fi

  if ! "verify_stage_$name"; then
    log "FAIL: $name did not pass verification after running"
    OVERALL_STATUS="FAIL"
    return 1
  fi

  mark_stage_done "$name"
  log "pass: $name"
}

# ---- stage 1: system packages + driver check --------------------------

stage_system_packages() {
  run_cmd apt-get update
  run_cmd apt-get install -y --no-install-recommends \
    git curl build-essential python3-venv python3-pip \
    libgl1 libglib2.0-0 ffmpeg
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[dry-run] would check nvidia-smi and nvcc --version"
  else
    nvidia-smi || { log "FATAL: nvidia-smi failed — no NVIDIA driver visible"; exit 1; }
    nvcc --version || { log "FATAL: nvcc not found — base image must be a 'devel' tag"; exit 1; }
  fi
}

verify_stage_system_packages() {
  [[ "$DRY_RUN" == "1" ]] && return 0
  command -v git >/dev/null && command -v nvcc >/dev/null && nvidia-smi >/dev/null 2>&1
}

# ---- stage 2: venv, no system site packages ---------------------------

stage_venv() {
  run_cmd mkdir -p "$(dirname "$VENV_DIR")"
  run_cmd python3 -m venv "$VENV_DIR"
  [[ "$DRY_RUN" == "1" ]] || activate_venv
}

verify_stage_venv() {
  [[ "$DRY_RUN" == "1" ]] && return 0
  [[ -x "$VENV_DIR/bin/python" ]] || return 1
  local prefix
  prefix="$("$VENV_DIR/bin/python" -c 'import sys; print(sys.prefix)')"
  [[ "$prefix" == "$VENV_DIR" ]]
}

# ---- stage 3: torch matched to detected GPU/driver ---------------------

torch_index_url_for_driver() {
  # Driver-first: read the host driver, pick the CUDA family it supports.
  # See deploy/pins/torch-cuda-notes.md for the policy this encodes.
  local driver_major
  driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
  if [[ "$driver_major" -ge 525 ]]; then
    echo "https://download.pytorch.org/whl/cu128"
  else
    log "FATAL: driver major version $driver_major is too old for the CUDA 12.8 pin"
    log "Pick a different vast.ai offer with a newer driver — do not downgrade the torch pin."
    exit 1
  fi
}

stage_torch() {
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[dry-run] would read driver version and pip install torch==2.8.0 matched to it"
    return 0
  fi
  activate_venv
  local index_url
  index_url="$(torch_index_url_for_driver)"
  run_cmd pip install --no-deps torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url "$index_url"
}

verify_stage_torch() {
  [[ "$DRY_RUN" == "1" ]] && return 0
  activate_venv
  python -c 'import torch; assert torch.__version__.startswith("2.8.0")'
}

# ---- stage 4: ComfyUI at pinned SHA ------------------------------------

stage_comfyui() {
  local sha
  sha="$(comfyui_pinned_sha)"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[dry-run] would clone ComfyUI and checkout $sha"
    return 0
  fi
  if [[ ! -d "$COMFYUI_DIR/.git" ]]; then
    run_cmd git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
  fi
  run_cmd git -C "$COMFYUI_DIR" fetch origin "$sha"
  run_cmd git -C "$COMFYUI_DIR" checkout "$sha"
}

verify_stage_comfyui() {
  [[ "$DRY_RUN" == "1" ]] && return 0
  [[ "$(git -C "$COMFYUI_DIR" rev-parse HEAD)" == "$(comfyui_pinned_sha)" ]]
}

# ---- stage 5: install this fork's node pack ----------------------------

stage_node_install() {
  local target="$COMFYUI_DIR/custom_nodes/Comfy_HunyuanImage3"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[dry-run] would symlink $REPO_ROOT into $target"
    return 0
  fi
  run_cmd mkdir -p "$(dirname "$target")"
  run_cmd ln -sfn "$REPO_ROOT" "$target"
}

verify_stage_node_install() {
  [[ "$DRY_RUN" == "1" ]] && return 0
  local target="$COMFYUI_DIR/custom_nodes/Comfy_HunyuanImage3"
  [[ -L "$target" ]] && [[ "$(readlink -f "$target")" == "$REPO_ROOT" ]]
}

# ---- stage 6: dependency reconciliation --------------------------------
# Order: Tencent pins > ComfyUI needs > node-pack needs > convenience.
# After each layer, confirm the previous layer's pins weren't silently
# upgraded as a transitive dependency (pip will do this and say nothing).

freeze_snapshot() {
  pip freeze > "$STATE_DIR/freeze-after-$1.txt" 2>/dev/null \
    || pip freeze > "/tmp/hy3-freeze-after-$1.txt"
}

check_no_drift() {
  local baseline="$STATE_DIR/freeze-after-$1.txt"
  [[ -f "$baseline" ]] || return 0
  local now
  now="$(pip freeze)"
  local pkg pinned current
  while IFS='=' read -r pkg pinned; do
    [[ -z "$pkg" ]] && continue
    current="$(echo "$now" | grep -i "^${pkg}==" || true)"
    if [[ -n "$current" && "$current" != "${pkg}==${pinned}" ]]; then
      log "DRIFT: $pkg pinned at $pinned after '$1' layer, now $current"
      return 1
    fi
  done < <(sed 's/==/=/' "$baseline" | awk -F= '{print $1"="$2}')
  return 0
}

stage_reconcile() {
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[dry-run] would install Tencent pins, then ComfyUI reqs, then node-pack reqs, checking drift after each, then pip check"
    return 0
  fi
  activate_venv
  mkdir -p "$STATE_DIR" 2>/dev/null || true

  run_cmd pip install --no-deps -r "$REPO_ROOT/deploy/pins/tencent-requirements.txt"
  freeze_snapshot tencent

  run_cmd pip install -r "$COMFYUI_DIR/requirements.txt"
  check_no_drift tencent || { log "FATAL: ComfyUI requirements drifted a Tencent pin"; exit 1; }

  run_cmd pip install --no-deps -r "$REPO_ROOT/requirements.txt"
  check_no_drift tencent || { log "FATAL: node-pack requirements drifted a Tencent pin"; exit 1; }

  run_cmd pip check
}

verify_stage_reconcile() {
  [[ "$DRY_RUN" == "1" ]] && return 0
  activate_venv
  pip check >/dev/null 2>&1
}

# ---- stage 7: verify_env + manifest ------------------------------------

stage_manifest() {
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[dry-run] would run verify_env.sh and write a manifest to deploy/manifests/"
    return 0
  fi
  activate_venv
  if [[ -x "$SCRIPT_DIR/verify_env.sh" ]]; then
    run_cmd "$SCRIPT_DIR/verify_env.sh"
  else
    log "warn: verify_env.sh not present yet — skipping pin-drift verification"
  fi
  if [[ -f "$SCRIPT_DIR/lib/write_manifest.py" ]]; then
    run_cmd python "$SCRIPT_DIR/lib/write_manifest.py" --repo-root "$REPO_ROOT" --comfyui-dir "$COMFYUI_DIR"
  else
    log "warn: manifest writer not present yet — skipping manifest"
  fi
}

verify_stage_manifest() {
  [[ "$DRY_RUN" == "1" ]] && return 0
  return 0
}

# ---- main ---------------------------------------------------------------

for stage in "${STAGES[@]}"; do
  run_stage "$stage" || true
done

log "=== provision.sh summary: $OVERALL_STATUS ==="
[[ "$OVERALL_STATUS" == "PASS" ]]
