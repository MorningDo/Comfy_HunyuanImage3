#!/usr/bin/env python3
"""Smoke check 1/3: does the node pack import cleanly and register
nodes, the way ComfyUI itself would load it from custom_nodes/?

Cheapest possible check — no model, no GPU compute, just "does the
Python actually import." Run this first; if it fails, nothing later
in tests/run_all.sh is worth attempting.

Usage: tests/smoke/check_imports.py
Env:   HY3_COMFYUI_DIR (default /opt/hy3/ComfyUI) — the node pack is
       expected to be symlinked at $HY3_COMFYUI_DIR/custom_nodes/Comfy_HunyuanImage3
       (see deploy/provision.sh's node_install stage).
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def main() -> int:
    comfyui_dir = Path(os.environ.get("HY3_COMFYUI_DIR", "/opt/hy3/ComfyUI"))
    custom_nodes_dir = comfyui_dir / "custom_nodes"
    package_name = "Comfy_HunyuanImage3"

    if not (custom_nodes_dir / package_name).exists():
        print(f"FAIL: {custom_nodes_dir / package_name} does not exist — "
              f"run deploy/provision.sh's node_install stage first.", file=sys.stderr)
        return 1

    # ComfyUI's own root must be on sys.path too, not just custom_nodes/
    # — the node pack does a hard `import folder_paths` at module level
    # (a ComfyUI-internal module living at $COMFYUI_DIR/folder_paths.py,
    # only normally importable because ComfyUI's own main.py puts its
    # root on sys.path before loading custom nodes). Confirmed live:
    # ModuleNotFoundError: No module named 'folder_paths' without this.
    sys.path.insert(0, str(comfyui_dir))
    sys.path.insert(0, str(custom_nodes_dir))
    try:
        module = importlib.import_module(package_name)
    except ModuleNotFoundError as exc:
        print(f"FAIL: import raised {exc!r}.", file=sys.stderr)
        print("If this is a missing third-party package (torch, transformers, ...), "
              "this must run inside the provisioned venv, not a bare system Python — "
              "activate $HY3_VENV_DIR/bin/activate first.", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - smoke test: any import-time failure is a finding
        print(f"FAIL: import raised {exc!r}.", file=sys.stderr)
        return 1

    mappings = getattr(module, "NODE_CLASS_MAPPINGS", None)
    if not mappings:
        print("FAIL: NODE_CLASS_MAPPINGS is missing or empty after import.", file=sys.stderr)
        return 1

    print(f"PASS: imported {package_name}, {len(mappings)} node(s) registered:")
    for name in sorted(mappings):
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
