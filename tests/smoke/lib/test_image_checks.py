#!/usr/bin/env python3
"""Regression test for image_checks.is_non_degenerate — no GPU, no
ComfyUI, no model needed. Run directly: python3 test_image_checks.py

Guards against silently regressing the noise vs. real-image
discriminator, which is the non-obvious part of this check (variance
and entropy alone can't tell pure noise from a real image — both are
high for both).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import image_checks as ic  # noqa: E402


def make_blank() -> Image.Image:
    return Image.fromarray(np.full((256, 256, 3), 128, dtype=np.uint8))


def make_noise(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))


def make_realish(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.linspace(0, 255, 256), np.linspace(0, 255, 256))
    base = 0.5 * x + 0.5 * y
    base[64:128, 64:160] += 60
    base[150:220, 30:100] -= 40
    base = base + rng.normal(0, 8, base.shape)
    arr = np.clip(base, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([arr] * 3, axis=-1))


def main() -> int:
    ok, stats = ic.is_non_degenerate(make_blank())
    assert ok is False, f"blank image should be flagged degenerate, got {stats}"
    assert "variance" in stats["reasons"][0]

    ok, stats = ic.is_non_degenerate(make_noise())
    assert ok is False, f"pure iid noise should be flagged degenerate, got {stats}"
    assert any("noise_ratio" in r for r in stats["reasons"]), stats

    ok, stats = ic.is_non_degenerate(make_realish())
    assert ok is True, f"realistic image should pass, got {stats}"

    # A handful of different seeds shouldn't flip the verdict — guards
    # against thresholds that are borderline/flaky rather than robust.
    for seed in range(1, 6):
        assert ic.is_non_degenerate(make_noise(seed))[0] is False, f"noise seed {seed} should fail"
        assert ic.is_non_degenerate(make_realish(seed))[0] is True, f"realish seed {seed} should pass"

    print("ALL image_checks TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
