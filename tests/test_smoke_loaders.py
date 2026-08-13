"""
GPU smoke tests: load each available model variant through its primary
node family, run one tiny generation to confirm the weights actually work
(not just that from_pretrained() didn't throw), then unload and confirm
VRAM comes back down.

Needs --hunyuan-models-dir (or per-role env var overrides — see README.md)
and a CUDA GPU. Each test independently skips if its model role isn't
found or won't fit in this GPU's VRAM.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.model]

# The model has no sub-1MP resolution presets — "1024x1024 (1:1 Square)" is
# the smallest entry in RESOLUTION_PRESETS (hunyuan_unified_v2.py).
_TINY_RESOLUTION = "1024x1024 (1:1 Square)"
_PROMPT = "a red apple on a white table, studio lighting"


def _vram_allocated_gb(torch) -> float:
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated(0) / 1024**3


class TestV2UnifiedLoader:
    """The primary/most-promoted loading path: HunyuanUnifiedV2 backed by
    CleanModelLoader. Exercises the repair_unquantized_bnb_modules /
    apply_nf4_transformers_compat ordering fix for NF4, and general
    load -> generate -> full_unload -> VRAM-freed round trip."""

    def _load_generate_unload(self, torch_module, model_path):
        hunyuan_unified_v2 = pytest.importorskip("hunyuan_unified_v2")
        node = hunyuan_unified_v2.HunyuanUnifiedV2()
        before_gb = _vram_allocated_gb(torch_module)

        images, final_prompt = node.generate(
            model_name=model_path,
            prompt=_PROMPT,
            resolution=_TINY_RESOLUTION,
            num_inference_steps=4,
            guidance_scale=3.0,
            seed=1,
            blocks_to_swap=0,
            post_action="full_unload",
        )

        assert images is not None
        assert images.ndim == 4  # (batch, H, W, C)
        assert images.shape[-1] == 3
        assert isinstance(final_prompt, str)
        # A real generation should not be a degenerate flat image (e.g. all
        # zeros from a crashed/garbage forward pass slipping through).
        assert images.float().std().item() > 1e-4

        after_gb = _vram_allocated_gb(torch_module)
        assert after_gb < before_gb + 3.0, (
            f"VRAM after full_unload ({after_gb:.1f}GB) is not close to "
            f"baseline ({before_gb:.1f}GB) — post_action may not have run "
            f"(see the post_action-skipped-on-exception fix; this path is "
            f"the success path, so a regression here is a different bug)"
        )

    def test_nf4(self, cuda_device, torch_module, nf4_dir, require_vram):
        require_vram(30, "NF4 model")
        self._load_generate_unload(torch_module, nf4_dir)

    def test_nf4_v2(self, cuda_device, torch_module, nf4_v2_dir, require_vram):
        require_vram(30, "NF4-v2 model")
        self._load_generate_unload(torch_module, nf4_v2_dir)

    def test_int8(self, cuda_device, torch_module, int8_dir, require_vram):
        require_vram(90, "INT8 model")
        self._load_generate_unload(torch_module, int8_dir)

    def test_int8_v2(self, cuda_device, torch_module, int8_v2_dir, require_vram):
        require_vram(90, "INT8-v2 model")
        self._load_generate_unload(torch_module, int8_v2_dir)

    @pytest.mark.slow
    def test_bf16_with_block_swap(self, cuda_device, torch_module, bf16_dir, require_vram):
        # Full BF16 is ~160GB — needs block swap even on a 96GB card.
        require_vram(60, "BF16 model (block-swapped)")
        hunyuan_unified_v2 = pytest.importorskip("hunyuan_unified_v2")
        node = hunyuan_unified_v2.HunyuanUnifiedV2()
        images, final_prompt = node.generate(
            model_name=bf16_dir,
            prompt=_PROMPT,
            resolution=_TINY_RESOLUTION,
            num_inference_steps=4,
            guidance_scale=3.0,
            seed=1,
            blocks_to_swap=24,
            post_action="full_unload",
        )
        assert images is not None
        assert images.shape[-1] == 3


class TestLegacyQuantizedLoaders:
    """hunyuan_quantized_nodes.py's standalone NF4/INT8 loader + generate
    nodes — the original/legacy node family, still shipped alongside V2."""

    def test_nf4_loader_and_generate(self, cuda_device, torch_module, nf4_dir, require_vram):
        require_vram(30, "NF4 model")
        hunyuan_quantized_nodes = pytest.importorskip("hunyuan_quantized_nodes")
        loader = hunyuan_quantized_nodes.HunyuanImage3QuantizedLoader()
        (model,) = loader.load_model(model_name=nf4_dir, force_reload=True)
        assert model is not None
        assert any(p.device.type == "cuda" for p in model.parameters())

        gen = hunyuan_quantized_nodes.HunyuanImage3Generate()
        result = gen.generate(
            model=model, prompt=_PROMPT, seed=1, steps=4,
            resolution=_TINY_RESOLUTION, guidance_scale=3.0,
            post_action="full_unload",
        )
        images = result[0]
        assert images is not None
        assert images.shape[-1] == 3

    def test_int8_loader_and_generate(self, cuda_device, torch_module, int8_dir, require_vram):
        require_vram(90, "INT8 model")
        hunyuan_quantized_nodes = pytest.importorskip("hunyuan_quantized_nodes")
        loader = hunyuan_quantized_nodes.HunyuanImage3Int8Loader()
        (model,) = loader.load_model(model_name=int8_dir, force_reload=True)
        assert model is not None
        assert any(p.device.type == "cuda" for p in model.parameters())

        gen = hunyuan_quantized_nodes.HunyuanImage3Generate()
        result = gen.generate(
            model=model, prompt=_PROMPT, seed=1, steps=4,
            resolution=_TINY_RESOLUTION, guidance_scale=3.0,
            post_action="full_unload",
        )
        images = result[0]
        assert images is not None
        assert images.shape[-1] == 3


class TestInstructLoader:
    """HunyuanInstructLoader + HunyuanInstructGenerate. Prefers the Distil
    variants when available since they're much faster (8 steps vs 40+)."""

    def _load_and_generate(self, torch_module, model_path, bot_task="image"):
        hunyuan_instruct_nodes = pytest.importorskip("hunyuan_instruct_nodes")
        loader = hunyuan_instruct_nodes.HunyuanInstructLoader()
        (model,) = loader.load_model(model_name=model_path, force_reload=True, blocks_to_swap=0)
        assert model is not None

        gen = hunyuan_instruct_nodes.HunyuanInstructGenerate()
        images, cot_reasoning, status = gen.generate(
            model=model,
            prompt=_PROMPT,
            bot_task=bot_task,
            resolution=_TINY_RESOLUTION,
            seed=1,
            steps=4,
        )
        assert images is not None
        assert images.shape[-1] == 3
        assert isinstance(cot_reasoning, str)
        assert isinstance(status, str)

        unload = hunyuan_instruct_nodes.HunyuanInstructUnload()
        unload.unload(enabled=True)
        assert hunyuan_instruct_nodes._instruct_cache.model is None

    def test_instruct_distil_nf4_v2(self, cuda_device, torch_module, instruct_distil_nf4_v2_dir, require_vram):
        require_vram(30, "Instruct-Distil-NF4-v2 model")
        self._load_and_generate(torch_module, instruct_distil_nf4_v2_dir)

    def test_instruct_distil_int8_v2(self, cuda_device, torch_module, instruct_distil_int8_v2_dir, require_vram):
        require_vram(90, "Instruct-Distil-INT8-v2 model")
        self._load_and_generate(torch_module, instruct_distil_int8_v2_dir)
