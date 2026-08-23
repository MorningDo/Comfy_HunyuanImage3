"""Cheap, GPU-free sanity checks on a generated image, run before
spending money on a vision-model judge (tests/validate_image.py).

Catches two distinct cheap failure modes, not just one:
  - blank / near-uniform output (decoder produced a constant image) —
    caught by pixel variance and entropy both being near zero.
  - raw, uncorrelated noise (e.g. an uninitialized or un-decoded
    tensor rendered directly) — caught separately, since pure noise
    actually has HIGH variance/entropy and would pass a variance-only
    check. Real images have spatially correlated neighboring pixels;
    iid noise does not, so the ratio of adjacent-pixel-difference
    variance to overall pixel variance is a cheap discriminator
    (near 2.0 for true iid noise, well under 1.0 for real photos).

Needs PIL and numpy, which are already required by the served
environment (Tencent/ComfyUI pins) — no extra dependency for code that
runs on the instance.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

DEFAULT_MIN_VARIANCE = 10.0
DEFAULT_MIN_ENTROPY_BITS = 1.0
DEFAULT_MAX_NOISE_RATIO = 1.2


def compute_stats(image: Image.Image) -> dict:
    gray = np.asarray(image.convert("L"), dtype=np.float64)

    variance = float(gray.var())

    counts, _ = np.histogram(gray, bins=256, range=(0, 256))
    probs = counts.astype(np.float64) / counts.sum()
    probs = probs[probs > 0]
    entropy_bits = float(-(probs * np.log2(probs)).sum())

    if gray.shape[1] > 1:
        h_diffs = np.diff(gray, axis=1)
        diff_variance = float(h_diffs.var())
    else:
        diff_variance = 0.0
    noise_ratio = (diff_variance / variance) if variance > 0 else 0.0

    return {
        "variance": variance,
        "entropy_bits": entropy_bits,
        "noise_ratio": noise_ratio,
    }


def is_non_degenerate(
    image: Image.Image,
    min_variance: float = DEFAULT_MIN_VARIANCE,
    min_entropy_bits: float = DEFAULT_MIN_ENTROPY_BITS,
    max_noise_ratio: float = DEFAULT_MAX_NOISE_RATIO,
) -> tuple[bool, dict]:
    stats = compute_stats(image)
    reasons = []
    if stats["variance"] < min_variance:
        reasons.append(f"variance {stats['variance']:.2f} < {min_variance} (looks blank/uniform)")
    if stats["entropy_bits"] < min_entropy_bits:
        reasons.append(f"entropy {stats['entropy_bits']:.2f} bits < {min_entropy_bits} (looks blank/uniform)")
    if stats["noise_ratio"] > max_noise_ratio:
        reasons.append(f"noise_ratio {stats['noise_ratio']:.2f} > {max_noise_ratio} (looks like uncorrelated noise)")
    stats["reasons"] = reasons
    return (len(reasons) == 0), stats
