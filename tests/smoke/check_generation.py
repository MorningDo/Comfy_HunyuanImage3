#!/usr/bin/env python3
"""Smoke check 3/3: run one minimal generation at the smallest
supported resolution (1.0MP class — HunyuanImage3Generate has no
smaller preset) with a low step count, and confirm the output is a
non-degenerate image before anything spends money on a vision-model
judge (tests/validate_image.py).

Calls the node pack's loader/generate node classes directly — the
same classes ComfyUI's graph executor would call — rather than driving
a full ComfyUI server, so this can run headless over SSH as part of
tests/run_all.sh. Passes the model directory as an absolute path
directly to the loader, which resolve_hunyuan_model_path() in
hunyuan_shared.py supports as a fallback for exactly this reason (no
folder_paths registration needed outside a running ComfyUI process).

This cannot be exercised without the real GPU/model, so treat the
node-calling-convention details here as best-effort against the code
as read, not as verified — the first real run on the RTX 6000 is the
actual test of this script.

Usage: tests/smoke/check_generation.py
Env:   HY3_COMFYUI_DIR, HY3_MODELS_DIR, HY3_MODEL_NAME (see
         check_model_quant.py for defaults)
       HY3_SMOKE_STEPS (default 8 — low, this is a smoke test not a
         quality check)
       HY3_SMOKE_RESOLUTION (default "1:1 (1.0MP)")
       HY3_SMOKE_PROMPT (default below)
       HY3_SMOKE_OUTPUT (default tests/smoke/.last-generation.png,
         useful for eyeballing what the noise/blank checks saw)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "smoke" / "lib"))


def main() -> int:
    comfyui_dir = Path(os.environ.get("HY3_COMFYUI_DIR", "/opt/hy3/ComfyUI"))
    model_name = os.environ.get("HY3_MODEL_NAME", "HunyuanImage-3.0-Instruct-Distil-NF4-v2")
    default_models_dir = REPO_ROOT / "models"
    models_dir = Path(os.environ.get("HY3_MODELS_DIR", str(default_models_dir)))
    model_dir = models_dir / model_name
    steps = int(os.environ.get("HY3_SMOKE_STEPS", "8"))
    resolution = os.environ.get("HY3_SMOKE_RESOLUTION", "1:1 (1.0MP)")
    prompt = os.environ.get("HY3_SMOKE_PROMPT", "A red apple on a wooden table, studio lighting")
    output_path = Path(os.environ.get("HY3_SMOKE_OUTPUT", str(Path(__file__).parent / ".last-generation.png")))

    if not model_dir.is_dir():
        print(f"FAIL: {model_dir} does not exist — run deploy/fetch_models.sh first.", file=sys.stderr)
        return 1

    custom_nodes_dir = comfyui_dir / "custom_nodes"
    sys.path.insert(0, str(custom_nodes_dir))
    try:
        import numpy as np
        from PIL import Image

        from Comfy_HunyuanImage3.hunyuan_quantized_nodes import (
            HunyuanImage3Generate,
            HunyuanImage3QuantizedLoader,
        )
    except ModuleNotFoundError as exc:
        print(f"FAIL: {exc!r} — run inside the provisioned venv, with node_install stage complete.", file=sys.stderr)
        return 1

    import image_checks

    print(f"Loading model from {model_dir} ...")
    try:
        loader = HunyuanImage3QuantizedLoader()
        (model,) = loader.load_model(model_name=str(model_dir), force_reload=False, reserve_memory_gb=6.0)
    except Exception as exc:  # noqa: BLE001 - smoke test: any load failure is a finding
        print(f"FAIL: model load raised {exc!r}", file=sys.stderr)
        return 1

    print(f"Generating at {resolution}, {steps} steps: {prompt!r}")
    try:
        generator = HunyuanImage3Generate()
        image_tensor, _rewritten_prompt, status, _trigger = generator.generate(
            model=model,
            prompt=prompt,
            seed=0,
            steps=steps,
            resolution=resolution,
            guidance_scale=6.0,
            post_action="full_unload",
        )
    except Exception as exc:  # noqa: BLE001 - smoke test: any generation failure is a finding
        print(f"FAIL: generation raised {exc!r}", file=sys.stderr)
        return 1

    print(f"Generation status: {status}")

    # ComfyUI IMAGE convention: float tensor, shape [B, H, W, C], range [0, 1].
    arr = image_tensor[0].detach().cpu().numpy()
    arr_uint8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(arr_uint8)
    pil_image.save(output_path)
    print(f"Saved generated image to {output_path}")

    ok, stats = image_checks.is_non_degenerate(pil_image)
    print(f"Image stats: {stats}")
    if not ok:
        print(f"FAIL: generated image looks degenerate: {stats['reasons']}", file=sys.stderr)
        return 1

    print("PASS: generation produced a non-degenerate image.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
