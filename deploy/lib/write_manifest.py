#!/usr/bin/env python3
"""Write a deploy manifest to deploy/manifests/<UTC-timestamp>_<git-sha>.json.

Called by deploy/provision.sh's final stage. Every field is best-effort:
on a real instance most fields resolve; run standalone (e.g. locally
during development, no GPU/venv/instance) most fields come back null
rather than raising, since a manifest documenting "we don't know yet" is
still useful provenance for a partial/failed run.

Env overrides (mainly for local testing without pip/nvidia-smi/etc.):
  HY3_PIP_FREEZE_CMD     — command producing `pip freeze` output
  HY3_PIP_CHECK_CMD      — command producing `pip check` output
  HY3_STATE_DIR          — provision.sh's state dir (for stage markers,
                            torch-index-url.txt)
  HY3_VENV_DIR           — venv path recorded in the manifest
  HY3_INSTANCE_INFO_FILE — JSON file with vast instance metadata
                            (written by deploy/vast/up.sh on the
                            instance); defaults to /etc/hy3-instance-info.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1


def run(cmd: str) -> str | None:
    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
    except Exception:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_sha(repo_dir: Path) -> str | None:
    return run(f"git -C {repo_dir} rev-parse HEAD")


def git_branch(repo_dir: Path) -> str | None:
    return run(f"git -C {repo_dir} rev-parse --abbrev-ref HEAD")


def synced_git_info(repo_dir: Path) -> dict:
    # deploy/vast/sync.sh deliberately excludes .git/ from the rsync
    # payload, so `git rev-parse` on the instance returns nothing —
    # confirmed live: node_pack_sha and branch both came back null in
    # the first real manifest. sync.sh writes this small file instead
    # with just the two values that matter.
    info_file = repo_dir / ".git-info.json"
    if not info_file.is_file():
        return {}
    try:
        return json.loads(info_file.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def sha256_of(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def read_comfyui_pin(repo_root: Path) -> str | None:
    pin_file = repo_root / "deploy" / "pins" / "comfyui-commit.txt"
    if not pin_file.is_file():
        return None
    for line in pin_file.read_text().splitlines():
        line = line.strip()
        if len(line) == 40 and all(c in "0123456789abcdef" for c in line):
            return line
    return None


def count_deviations(repo_root: Path) -> int:
    dev_file = repo_root / "deploy" / "pins" / "DEVIATIONS.md"
    if not dev_file.is_file():
        return 0
    count = 0
    for line in dev_file.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells or cells[0] in ("Package", "---") or cells[0].startswith("---"):
            continue
        if cells[0].startswith("_(") or cells[0] == "":
            continue
        count += 1
    return count


def stage_statuses(state_dir: Path) -> dict:
    stages = {}
    if not state_dir.is_dir():
        return stages
    for marker in sorted(state_dir.glob("*.done")):
        name = marker.stem
        info = {"status": "pass"}
        for line in marker.read_text().splitlines():
            if line.startswith("timestamp="):
                info["timestamp_utc"] = line.split("=", 1)[1]
        stages[name] = info
    return stages


def nvcc_version() -> str | None:
    out = run("nvcc --version")
    if not out:
        return None
    for line in out.splitlines():
        if "release" in line:
            return line.strip()
    return out.splitlines()[-1] if out else None


def torch_cuda_version() -> str | None:
    return run(
        'python -c "import torch; print(torch.version.cuda)"'
    ) or None


def nvidia_smi_field(query: str) -> str | None:
    return run(f"nvidia-smi --query-gpu={query} --format=csv,noheader")


def load_instance_info() -> dict:
    info_file = Path(
        os.environ.get("HY3_INSTANCE_INFO_FILE", "/etc/hy3-instance-info.json")
    )
    if not info_file.is_file():
        return {}
    try:
        return json.loads(info_file.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def pip_freeze_lines() -> list[str]:
    cmd = os.environ.get("HY3_PIP_FREEZE_CMD", "pip freeze")
    out = run(cmd)
    return out.splitlines() if out else []


def pip_check_result() -> dict:
    cmd = os.environ.get("HY3_PIP_CHECK_CMD", "pip check")
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    except Exception as exc:
        return {"status": "unknown", "output": str(exc)}
    combined = (out.stdout + out.stderr).strip()
    return {"status": "ok" if out.returncode == 0 else "conflict", "output": combined}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--comfyui-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root: Path = args.repo_root.resolve()
    comfyui_dir: Path = args.comfyui_dir
    out_dir = args.out_dir or (repo_root / "deploy" / "manifests")
    out_dir.mkdir(parents=True, exist_ok=True)

    state_dir = Path(os.environ.get("HY3_STATE_DIR", "/var/lib/hy3-provision/state"))
    venv_dir = os.environ.get("HY3_VENV_DIR", "/opt/hy3/venv")

    git_fallback = synced_git_info(repo_root)
    node_pack_sha = git_sha(repo_root) or git_fallback.get("node_pack_sha")
    node_pack_branch = git_branch(repo_root) or git_fallback.get("branch")
    instance_info = load_instance_info()
    pip_check = pip_check_result()
    freeze = pip_freeze_lines()

    torch_index_url = None
    idx_file = state_dir / "torch-index-url.txt"
    if idx_file.is_file():
        torch_index_url = idx_file.read_text().strip() or None

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": {
            "node_pack_sha": node_pack_sha,
            "branch": node_pack_branch,
            "comfyui_sha": git_sha(comfyui_dir) if comfyui_dir.is_dir() else None,
        },
        "vast": {
            "offer_id": instance_info.get("offer_id"),
            "instance_id": instance_info.get("instance_id"),
            "gpu_model": instance_info.get("gpu_model") or nvidia_smi_field("name"),
            "vram_gb": instance_info.get("vram_gb"),
            "driver_version": instance_info.get("driver_version")
            or nvidia_smi_field("driver_version"),
            "image": instance_info.get("image"),
            "image_digest": instance_info.get("image_digest"),
        },
        "cuda": {
            "nvcc_version": nvcc_version(),
            "torch_cuda_version": torch_cuda_version(),
            "wheel_index_url": torch_index_url,
        },
        "python": {
            "venv_path": venv_dir,
            "python_version": run("python --version") or sys.version.split()[0],
        },
        "pins": {
            "tencent_requirements_sha256": sha256_of(
                repo_root / "deploy" / "pins" / "tencent-requirements.txt"
            ),
            "comfyui_commit_sha": read_comfyui_pin(repo_root),
            "deviation_count": count_deviations(repo_root),
        },
        "pip_freeze": freeze,
        "pip_check": pip_check,
        "stages": stage_statuses(state_dir),
        "result": "PASS" if pip_check["status"] == "ok" else "FAIL",
        "test_report_ref": None,
    }

    sha_short = (node_pack_sha or "unknown")[:12]
    out_path = out_dir / f"{manifest['timestamp_utc'].replace(':', '')}_{sha_short}.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote manifest: {out_path}")
    print(f"result: {manifest['result']}")
    return 0 if manifest["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
