"""
GPU regression test for the single highest-impact bug found in the paranoid
audit: three independent, mutually-unaware model caches (ModelCacheV2,
InstructModelCache, HunyuanModelCache) meant every "unload" node only ever
freed its own family's model, silently leaving models loaded via a
different node family fully resident in VRAM.

Each test here loads a real model through ONE node family (keeping it
cached, not unloading), then calls a DIFFERENT family's unload node, and
asserts the first model's cache is now ALSO empty. This is deliberately
light on GPU load (only one real model loaded at a time) while still
exercising the actual cross-cache coordination path end to end.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.model]

# The model has no sub-1MP resolution presets — "1024x1024 (1:1 Square)" is
# the smallest entry in RESOLUTION_PRESETS (hunyuan_unified_v2.py).
_TINY_RESOLUTION = "1024x1024 (1:1 Square)"
_PROMPT = "a red apple on a white table"


def test_instruct_unload_also_clears_v2_cache(cuda_device, torch_module, nf4_dir, require_vram):
    require_vram(30, "NF4 model")
    hunyuan_unified_v2 = pytest.importorskip("hunyuan_unified_v2")
    hunyuan_instruct_nodes = pytest.importorskip("hunyuan_instruct_nodes")
    hunyuan_cache_v2 = pytest.importorskip("hunyuan_cache_v2")

    # 1. Load via V2, keep it cached (don't unload).
    node = hunyuan_unified_v2.HunyuanUnifiedV2()
    node.generate(
        model_name=nf4_dir, prompt=_PROMPT, resolution=_TINY_RESOLUTION,
        num_inference_steps=2, guidance_scale=3.0, seed=1,
        blocks_to_swap=0, post_action="keep_loaded",
    )
    assert hunyuan_cache_v2.get_cache().get_status()["cached"] is True, (
        "setup failed: V2 cache should hold the just-loaded model"
    )

    # 2. Call the Instruct family's unload node — nothing is loaded there,
    #    but it must ALSO reach into ModelCacheV2 and clear it.
    instruct_unload = hunyuan_instruct_nodes.HunyuanInstructUnload()
    instruct_unload.unload(enabled=True)

    assert hunyuan_cache_v2.get_cache().get_status()["cached"] is False, (
        "HunyuanInstructUnload did not clear the V2 cache — the "
        "cross-cache fix regressed (fragmented-cache bug is back)"
    )


def test_v2_unload_also_clears_instruct_cache(
    cuda_device, torch_module, instruct_distil_nf4_v2_dir, require_vram
):
    require_vram(30, "Instruct-Distil-NF4-v2 model")
    hunyuan_unified_v2 = pytest.importorskip("hunyuan_unified_v2")
    hunyuan_instruct_nodes = pytest.importorskip("hunyuan_instruct_nodes")

    # 1. Load via the Instruct loader, keep it cached.
    loader = hunyuan_instruct_nodes.HunyuanInstructLoader()
    loader.load_model(model_name=instruct_distil_nf4_v2_dir, force_reload=True, blocks_to_swap=0)
    assert hunyuan_instruct_nodes._instruct_cache.model is not None, (
        "setup failed: Instruct cache should hold the just-loaded model"
    )

    # 2. Call the V2 family's unload node.
    v2_unload = hunyuan_unified_v2.HunyuanUnloadV2()
    v2_unload.unload(unload_type="full_unload")

    assert hunyuan_instruct_nodes._instruct_cache.model is None, (
        "HunyuanUnloadV2 did not clear the Instruct cache — the "
        "cross-cache fix regressed (fragmented-cache bug is back)"
    )


def test_force_unload_clears_all_three_caches(
    cuda_device, torch_module, nf4_dir, require_vram
):
    """HunyuanImage3ForceUnload ("Nuclear") is the last-resort tool — it
    must clear every cache regardless of which family loaded a model."""
    require_vram(30, "NF4 model")
    hunyuan_unified_v2 = pytest.importorskip("hunyuan_unified_v2")
    hunyuan_shared = pytest.importorskip("hunyuan_shared")
    hunyuan_cache_v2 = pytest.importorskip("hunyuan_cache_v2")

    node = hunyuan_unified_v2.HunyuanUnifiedV2()
    node.generate(
        model_name=nf4_dir, prompt=_PROMPT, resolution=_TINY_RESOLUTION,
        num_inference_steps=2, guidance_scale=3.0, seed=1,
        blocks_to_swap=0, post_action="keep_loaded",
    )
    assert hunyuan_cache_v2.get_cache().get_status()["cached"] is True

    force_unload = hunyuan_shared.HunyuanImage3ForceUnload()
    force_unload.force_unload(
        first_run_only=False, clear_all_models=True, aggressive_gc=True,
        reset_cuda_allocator=True, clear_comfy_cache=False,
        nuke_orphaned_tensors=False, nuke_ram_tensors=False,
    )

    assert hunyuan_cache_v2.get_cache().get_status()["cached"] is False, (
        "HunyuanImage3ForceUnload did not clear the V2 cache"
    )
