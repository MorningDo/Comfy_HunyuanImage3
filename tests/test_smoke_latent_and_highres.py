"""
GPU smoke tests for the latent-control nodes (HunyuanEmptyLatent,
HunyuanLatentNoise, HunyuanGenerateWithLatent) and the HighRes efficient
generation node.

HunyuanGenerateWithLatent subclasses HunyuanUnifiedV2 and was the source of
issue #46 (moe_drop_tokens/vae_dtype not threaded through) — the pure
schema check for that lives in test_regressions_static.py /
test_schema_consistency.py; this file exercises the actual runtime path
with a real model, including the image/latent-injection branch that #46's
signature mismatch would have crashed before even reaching.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.model]

# The model has no sub-1MP resolution presets — "1024x1024 (1:1 Square)" is
# the smallest entry in RESOLUTION_PRESETS (hunyuan_unified_v2.py).
_RESOLUTION = "1024x1024 (1:1 Square)"
_PROMPT = "a red apple on a white table, studio lighting"


class TestLatentNodesNoModel:
    """HunyuanEmptyLatent / HunyuanLatentNoise are pure tensor ops — no
    model load needed, just torch + the resolution preset table."""

    def test_empty_latent_shape(self, torch_module):
        hunyuan_latent_nodes = pytest.importorskip("hunyuan_latent_nodes")
        node = hunyuan_latent_nodes.HunyuanEmptyLatent()
        (latent,) = node.generate(resolution=_RESOLUTION, seed=1, batch_size=1)
        assert isinstance(latent, dict)
        assert "latent" in latent and "height" in latent and "width" in latent
        assert latent["latent"].shape[0] == 1

    def test_latent_noise_low_pass_filter(self, torch_module):
        hunyuan_latent_nodes = pytest.importorskip("hunyuan_latent_nodes")
        empty_node = hunyuan_latent_nodes.HunyuanEmptyLatent()
        (latent,) = empty_node.generate(resolution=_RESOLUTION, seed=1, batch_size=1)

        noise_node = hunyuan_latent_nodes.HunyuanLatentNoise()
        (shaped,) = noise_node.apply(latent=latent, operation="low_pass_filter", strength=0.5, seed=1)
        assert isinstance(shaped, dict)
        assert shaped["latent"].shape == latent["latent"].shape


def test_generate_with_latent_text_to_image_delegates_to_parent(
    cuda_device, torch_module, nf4_dir, require_vram
):
    """No image/latent connected -> should fully delegate to
    HunyuanUnifiedV2.generate() (case 1 in the node's own docstring)."""
    require_vram(30, "NF4 model")
    hunyuan_latent_nodes = pytest.importorskip("hunyuan_latent_nodes")
    node = hunyuan_latent_nodes.HunyuanGenerateWithLatent()

    images, final_prompt = node.generate(
        model_name=nf4_dir, prompt=_PROMPT, resolution=_RESOLUTION,
        num_inference_steps=4, guidance_scale=3.0, seed=1,
        blocks_to_swap=0, post_action="full_unload",
        moe_drop_tokens=True, vae_dtype="bfloat16",
    )
    assert images is not None
    assert images.shape[-1] == 3
    assert isinstance(final_prompt, str)


def test_generate_with_latent_composition_mode(cuda_device, torch_module, nf4_dir, require_vram):
    """Exercises the image/latent-injection branch specifically — this is
    the code path issue #46 crashed before ever reaching, since the
    TypeError happened at the ComfyUI kwarg-dispatch level before generate()
    could even inspect `image`/`latent`."""
    require_vram(30, "NF4 model")
    hunyuan_latent_nodes = pytest.importorskip("hunyuan_latent_nodes")

    # Build a tiny fake input image directly (bypasses needing a LoadImage
    # node / real file — a ComfyUI IMAGE tensor is just (B,H,W,3) float32 in [0,1]).
    torch = torch_module
    fake_image = torch.rand(1, 512, 512, 3, dtype=torch.float32)

    node = hunyuan_latent_nodes.HunyuanGenerateWithLatent()
    images, final_prompt = node.generate(
        model_name=nf4_dir, prompt=_PROMPT, resolution=_RESOLUTION,
        num_inference_steps=4, guidance_scale=3.0, seed=1,
        blocks_to_swap=0, post_action="full_unload",
        moe_drop_tokens=True, vae_dtype="bfloat16",
        image=fake_image, image_mode="composition", denoise_strength=0.6,
    )
    assert images is not None
    assert images.shape[-1] == 3
    assert isinstance(final_prompt, str)


def test_highres_generation(cuda_device, torch_module, nf4_dir, require_vram):
    require_vram(30, "NF4 model")
    hunyuan_quantized_nodes = pytest.importorskip("hunyuan_quantized_nodes")
    hunyuan_highres_nodes = pytest.importorskip("hunyuan_highres_nodes")

    loader = hunyuan_quantized_nodes.HunyuanImage3QuantizedLoader()
    (model,) = loader.load_model(model_name=nf4_dir, force_reload=True)

    node = hunyuan_highres_nodes.HunyuanImage3GenerateHighRes()
    result = node.generate_highres(
        model=model, prompt=_PROMPT, seed=1, steps=4,
        resolution=_RESOLUTION, guidance_scale=3.0,
        offload_mode="smart", post_action="full_unload",
    )
    image = result[0]
    assert image is not None
    assert image.shape[-1] == 3
