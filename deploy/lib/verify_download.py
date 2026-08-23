#!/usr/bin/env python3
"""Verify a local directory against an expected file manifest (path +
size in bytes), used by deploy/fetch_models.sh to decide whether a
model checkpoint needs (re-)downloading, and to confirm a download
actually completed. Size-only by design (sha256 over 50GB+ safetensors
shards on every idempotency check would be slow and mostly redundant
with HF Hub's own transfer integrity checking) — this catches the
common failure modes (partial download, truncated file, wrong
revision) cheaply.

Usage:
  verify_download.py --manifest MANIFEST.json --target-dir DIR
    MANIFEST.json: [{"path": "...", "size": 123}, ...]
  Prints one line per problem (MISSING/MISMATCH), and a final summary.
  Exit 0 if everything matches, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def verify(manifest: list[dict], target_dir: Path) -> list[str]:
    problems = []
    for entry in manifest:
        rel = entry["path"]
        expected_size = entry.get("size")
        if expected_size is None:
            continue  # directories / entries with no size (e.g. LFS pointers pre-resolve) are skipped
        local_path = target_dir / rel
        if not local_path.is_file():
            problems.append(f"MISSING: {rel}")
            continue
        actual_size = local_path.stat().st_size
        if actual_size != expected_size:
            problems.append(f"MISMATCH: {rel} expected={expected_size} actual={actual_size}")
    return problems


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--target-dir", required=True, type=Path)
    args = p.parse_args(argv)

    manifest = load_manifest(args.manifest)
    problems = verify(manifest, args.target_dir)

    for line in problems:
        print(line)

    total = len(manifest)
    ok = total - len(problems)  # each manifest entry contributes at most one problem
    if problems:
        print(f"verify_download: FAIL — {len(problems)} problem(s) out of {total} files")
        return 1
    print(f"verify_download: PASS — {ok}/{total} files present with matching size")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
