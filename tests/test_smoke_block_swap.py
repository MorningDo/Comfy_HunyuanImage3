"""
GPU smoke test for block swap, specifically exercising the cross-CUDA-stream
release/prefetch race fix in hunyuan_block_swap.py.

IMPORTANT: this must run against INT8 (or BF16), not NF4. NF4 explicitly
skips the async prefetch path entirely (a *different*, unrelated
constraint — bitsandbytes Params4bit crashes on non-blocking transfers, see
hunyuan_block_swap.py's `if self._has_nf4_layers: continue` in
_prefetch_upcoming), so the release/prefetch race this test targets cannot
occur for NF4 at all. Only INT8/BF16 block swap goes through the prefetch
stream where the fix applies.

IMPORTANT LIMITATION: this cannot deterministically *prove* the race is
fixed — it was a timing-dependent hazard (release write vs. prefetch read
of the same pinned buffer on different streams), and a passing run doesn't
guarantee the hazard can't occur on different hardware/driver timing. What
it does verify: block swap with prefetching enabled produces numerically
sane output (no NaN/Inf, non-degenerate variance) across enough diffusion
steps to exercise every swapped block's release+prefetch cycle multiple
times, and that this matches a blocks_to_swap=0 (no swap) baseline closely
enough to be plausible. Report any run that fails this — even
intermittently — since that's exactly the failure signature the race
would produce.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.model]

# The model has no sub-1MP resolution presets — "1024x1024 (1:1 Square)" is
# the smallest entry in RESOLUTION_PRESETS (hunyuan_unified_v2.py).
_RESOLUTION = "1024x1024 (1:1 Square)"
_PROMPT = "a red apple on a white table, studio lighting"


def test_int8_block_swap_produces_finite_output(cuda_device, torch_module, int8_dir, require_vram):
    require_vram(90, "INT8 model")
    hunyuan_unified_v2 = pytest.importorskip("hunyuan_unified_v2")
    node = hunyuan_unified_v2.HunyuanUnifiedV2()

    images, _prompt = node.generate(
        model_name=int8_dir,
        prompt=_PROMPT,
        resolution=_RESOLUTION,
        # More steps than the other smoke tests: every extra step is
        # another full release/prefetch cycle for every swapped block,
        # which is what actually exercises the fixed race window.
        num_inference_steps=12,
        guidance_scale=3.0,
        seed=1,
        # Force a meaningful chunk of blocks through the CPU<->GPU swap
        # path with async prefetch active (the default) — 10 of 32 blocks;
        # each is ~2.5GB, so this needs ~25GB of pinned system RAM for the
        # buffer store on top of the ~85GB resident on GPU.
        blocks_to_swap=10,
        post_action="full_unload",
    )

    assert torch_module.isfinite(images).all(), (
        "block-swapped generation produced NaN/Inf pixels — this is the "
        "failure signature of the release/prefetch cross-stream race "
        "(corrupted block weights read mid-transfer)"
    )
    assert images.float().std().item() > 1e-4, (
        "block-swapped generation produced suspiciously flat/uniform "
        "output — could indicate corrupted weights rather than a real image"
    )


@pytest.mark.slow
def test_int8_block_swap_output_resembles_no_swap_baseline(
    cuda_device, torch_module, int8_dir, require_vram
):
    """Same seed/prompt with and without block swap should produce visually
    similar output (not bit-identical — block swap changes execution order
    slightly — but not wildly different). A large divergence is a strong
    signal that swapped blocks are computing with wrong/stale weights.
    Marked slow: this is two full INT8 loads back to back."""
    require_vram(90, "INT8 model (loaded twice)")
    hunyuan_unified_v2 = pytest.importorskip("hunyuan_unified_v2")

    kwargs = dict(
        model_name=int8_dir, prompt=_PROMPT, resolution=_RESOLUTION,
        num_inference_steps=8, guidance_scale=3.0, seed=42,
    )

    node = hunyuan_unified_v2.HunyuanUnifiedV2()
    baseline_images, _ = node.generate(blocks_to_swap=0, post_action="full_unload", **kwargs)

    node2 = hunyuan_unified_v2.HunyuanUnifiedV2()
    swapped_images, _ = node2.generate(blocks_to_swap=10, post_action="full_unload", **kwargs)

    assert baseline_images.shape == swapped_images.shape
    diff = (baseline_images.float() - swapped_images.float()).abs().mean().item()
    # 0.15 is a generous, uncalibrated heuristic (pixels are in [0,1]) — it
    # was not tuned against real hardware since this suite was written
    # without GPU access. If this fails on a run that otherwise looks fine
    # when you inspect the two images, the threshold is probably just too
    # strict for legitimate step-order numerical variation; loosen it rather
    # than treating a near-miss as a confirmed bug.
    assert diff < 0.15, (
        f"mean absolute pixel difference between blocks_to_swap=0 and "
        f"blocks_to_swap=10 is {diff:.3f} (0=identical, 1=max) — this is "
        f"much higher than expected numerical variation and suggests "
        f"block swap is computing with corrupted weights"
    )
