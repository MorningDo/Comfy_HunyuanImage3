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

STAGES=(system_packages venv torch comfyui comfy_model_paths node_install reconcile manifest)

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
  # Always echo exactly once to stdout, then best-effort append to
  # LOG_FILE separately. Previously piped through `tee -a LOG_FILE`
  # and echoed again if that pipeline's exit status was nonzero — but
  # tee still writes to stdout even when it can't open LOG_FILE (e.g.
  # unwritable /var/log, confirmed live when testing locally as a
  # non-root user), so a nonzero exit there meant "the file write
  # failed," not "stdout wasn't written," and every line was doubled.
  local msg
  msg="[$(date -u +%FT%TZ)] $*"
  echo "$msg"
  mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
  # 2>/dev/null must come BEFORE the >>LOG_FILE redirect: bash sets up
  # redirections left-to-right, and if >>LOG_FILE fails to open (e.g.
  # unwritable /var/log), the error prints to whatever stderr was
  # active at that point — which is only /dev/null if that redirect
  # was already applied first. Confirmed live: reversed order leaked
  # "Permission denied" past the suppression on every single log line.
  { echo "$msg" >> "$LOG_FILE"; } 2>/dev/null || true
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
  # No --no-deps here deliberately: torch is the first thing installed
  # into a fresh venv, so there's nothing yet for its own transitive
  # deps (typing_extensions, sympy, networkx, jinja2, filelock, fsspec)
  # to clobber — --no-deps at this stage just breaks the torch import
  # outright (confirmed live: ModuleNotFoundError: typing_extensions).
  # The --no-deps protection that matters is in stage_reconcile, where
  # later layers must not silently override earlier-pinned exact
  # versions; pip install of an exact pin there still forces the
  # correct version regardless of what an earlier layer pulled in.
  run_cmd pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url "$index_url"
  mkdir -p "$STATE_DIR" 2>/dev/null || true
  echo "$index_url" > "$STATE_DIR/torch-index-url.txt" 2>/dev/null || true
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

# ---- stage 4b: register model search path for hunyuan/hunyuan_instruct --
# hunyuan_shared.py's _register_hunyuan_model_paths() only ever registers
# ComfyUI's OWN models_dir (folder_paths.models_dir) as a search location
# for the "hunyuan"/"hunyuan_instruct" categories — it has no idea
# deploy/fetch_models.sh puts checkpoints under $REPO_ROOT/models instead.
# Confirmed live (2026-08-23): the Instruct loader node's dropdown was
# empty in the actual ComfyUI web UI even though check_generation.py
# could load the checkpoint fine — that script bypasses folder_paths
# entirely by passing an absolute path, which a real user clicking
# through the UI cannot do. ComfyUI's own extra_model_paths.yaml
# mechanism (loaded at startup, before custom_nodes/ import) is the
# supported way to add a second search root for a category; the node
# pack's own docstring documents this exact key format.

stage_comfy_model_paths() {
  local yaml_file="$COMFYUI_DIR/extra_model_paths.yaml"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[dry-run] would write $yaml_file pointing hunyuan/hunyuan_instruct at $REPO_ROOT/models"
    return 0
  fi
  log "+ write $yaml_file"
  cat > "$yaml_file" <<YAMLEOF
# Generated by deploy/provision.sh (stage_comfy_model_paths) — do not
# hand-edit, this is regenerated on every provision.sh run. Lets the
# Hunyuan loader nodes find checkpoints under this repo's models/
# (where deploy/fetch_models.sh puts them) in addition to ComfyUI's
# own models/ dir.
comfy_hunyuan_image3:
    base_path: $REPO_ROOT
    hunyuan: models/
    hunyuan_instruct: models/
YAMLEOF
}

verify_stage_comfy_model_paths() {
  [[ "$DRY_RUN" == "1" ]] && return 0
  local yaml_file="$COMFYUI_DIR/extra_model_paths.yaml"
  [[ -f "$yaml_file" ]] && grep -q "$REPO_ROOT" "$yaml_file"
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

# Only checks packages Tencent's requirements.txt actually pins with
# an exact ==, mirroring deploy/verify_env.sh's own pin parsing.
# Deliberately NOT a full pip-freeze diff against everything installed
# after the Tencent layer: Tencent's file also lists genuinely
# unconstrained deps (e.g. `huggingface_hub[cli]`, no version) that are
# meant to be negotiated by later layers — treating those as frozen
# pins too was a real bug, caught live: ComfyUI's requirements.txt
# legitimately wants a different huggingface_hub than whatever version
# pip happened to resolve for the unpinned Tencent line, and the old
# full-freeze-diff check treated that as a fatal drift.
check_no_drift() {
  local layer="$1" pin_file="$2"
  local now problems=0
  now="$(pip freeze)"
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local pkg_raw="${line%%==*}" pinned="${line##*==}"
    local pkg="${pkg_raw%%\[*}" # strip extras, e.g. transformers[accelerate,tiktoken]
    local current
    current="$(echo "$now" | grep -i "^${pkg}==" || true)"
    if [[ -n "$current" && "$current" != "${pkg}==${pinned}" ]]; then
      log "DRIFT: $pkg pinned at $pinned (Tencent) after '$layer' layer, now $current"
      problems=1
    fi
  done < <(grep -E '^[A-Za-z0-9_.-]+(\[[^]]*\])?==' "$pin_file")
  return "$problems"
}

stage_reconcile() {
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[dry-run] would install Tencent pins, then ComfyUI reqs, then node-pack reqs, checking drift after each, then pip check"
    return 0
  fi
  activate_venv
  mkdir -p "$STATE_DIR" 2>/dev/null || true
  local pin_file="$REPO_ROOT/deploy/pins/tencent-requirements.txt"

  # Deliberately skip gradio: it's Tencent's own optional interactive
  # demo UI, which this pipeline never runs (ComfyUI is the only UI
  # here) — installing it with --no-deps (correct for the actual
  # exact-pinned core packages in this file) breaks gradio's own large,
  # loosely-bound dependency tree and fails `pip check` for no benefit.
  # See deploy/pins/DEVIATIONS.md. Also uninstall it if a prior partial
  # run already left it in this state — `pip install` never removes a
  # package just because it's no longer in the requirements list, so a
  # stale broken install here would otherwise survive across re-runs of
  # this idempotent stage indefinitely (confirmed live: exactly this
  # happened while developing this exclusion).
  pip uninstall -y gradio >/dev/null 2>&1 || true
  grep -viE '^gradio' "$pin_file" > "$STATE_DIR/tencent-requirements.filtered.txt" 2>/dev/null \
    || grep -viE '^gradio' "$pin_file" > "/tmp/hy3-tencent-requirements.filtered.txt"
  local filtered_pin_file="$STATE_DIR/tencent-requirements.filtered.txt"
  [[ -f "$filtered_pin_file" ]] || filtered_pin_file="/tmp/hy3-tencent-requirements.filtered.txt"

  # Two-step install: --no-deps first locks every package in this file
  # to exactly the pinned version, regardless of what pip's resolver
  # would otherwise pick. A plain follow-up install (no --no-deps)
  # then backfills genuine transitive dependencies these packages need
  # (e.g. diffusers==0.35.2 needs importlib-metadata, which --no-deps
  # dropped, breaking `pip check` for no reason — confirmed live).
  # Safe: pip leaves an already-satisfied exact pin alone and only
  # installs what's still missing underneath it.
  run_cmd pip install --no-deps -r "$filtered_pin_file"
  run_cmd pip install -r "$filtered_pin_file"

  run_cmd pip install -r "$COMFYUI_DIR/requirements.txt"
  check_no_drift comfyui "$pin_file" || { log "FATAL: ComfyUI requirements drifted an exact Tencent pin"; exit 1; }

  run_cmd pip install --no-deps -r "$REPO_ROOT/requirements.txt"
  check_no_drift node-pack "$pin_file" || { log "FATAL: node-pack requirements drifted an exact Tencent pin"; exit 1; }

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
