#!/usr/bin/env python3
"""Smoke check 2/3: does the downloaded checkpoint load and report the
expected quantization type?

Deliberately loads only the config (AutoConfig), not the full model
weights — cheap and fast (smoke tests should be, per the definition of
done), and sufficient to catch the two failure modes that matter here:
wrong/corrupted checkpoint dir, or a quantization config that doesn't
match what deploy/fetch_models.sh actually pinned. A full weight load
happens implicitly in check_generation.py anyway.

Needs the provisioned venv (transformers with trust_remote_code
support) — see tests/smoke/check_imports.py's note on activating it.

Usage: tests/smoke/check_model_quant.py
Env:   HY3_MODELS_DIR (default /opt/hy3/models or $REPO_ROOT/models)
       HY3_MODEL_NAME (default HunyuanImage-3.0-Instruct-Distil-NF4-v2,
         must match the basename deploy/fetch_models.sh downloaded to)
       HY3_EXPECTED_QUANT (default nf4)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    model_name = os.environ.get("HY3_MODEL_NAME", "HunyuanImage-3.0-Instruct-Distil-NF4-v2")
    default_models_dir = Path(__file__).resolve().parents[2] / "models"
    models_dir = Path(os.environ.get("HY3_MODELS_DIR", str(default_models_dir)))
    model_dir = models_dir / model_name
    expected_quant = os.environ.get("HY3_EXPECTED_QUANT", "nf4")

    if not model_dir.is_dir():
        print(f"FAIL: {model_dir} does not exist — run deploy/fetch_models.sh first.", file=sys.stderr)
        return 1

    try:
        from transformers import AutoConfig
    except ModuleNotFoundError as exc:
        print(f"FAIL: {exc!r} — run inside the provisioned venv.", file=sys.stderr)
        return 1

    try:
        config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 - smoke test: any load-time failure is a finding
        print(f"FAIL: AutoConfig.from_pretrained raised {exc!r}", file=sys.stderr)
        return 1

    quant_cfg = getattr(config, "quantization_config", None)
    if quant_cfg is None:
        print("FAIL: config has no quantization_config attribute.", file=sys.stderr)
        return 1

    quant_dict = quant_cfg if isinstance(quant_cfg, dict) else vars(quant_cfg)
    is_4bit = bool(quant_dict.get("load_in_4bit"))
    is_8bit = bool(quant_dict.get("load_in_8bit"))
    actual = "nf4" if is_4bit else ("int8" if is_8bit else "unknown")

    if actual != expected_quant:
        print(f"FAIL: expected quantization '{expected_quant}', got '{actual}' "
              f"(load_in_4bit={is_4bit}, load_in_8bit={is_8bit})", file=sys.stderr)
        return 1

    print(f"PASS: {model_dir} config loads, quantization={actual} as expected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
