"""
GPU smoke tests for the optional attention_impl=flash_attention_2 /
moe_impl=flashinfer backends on HunyuanInstructLoader (see
validate_attention_moe_impl() in hunyuan_shared.py and INSTALL.md's
"FlashInfer / FlashAttention setup" section).

Both tests skip outright if their package isn't importable — install with
requirements-flash.txt (flashinfer) or manually (flash_attn — see
INSTALL.md, no confirmed upstream support for consumer/workstation
Blackwell/SM120 as of this writing).

test_flash_attention_2_loads_and_generates is EXPECTED to legitimately
fail on SM120 GPUs (RTX PRO 6000, RTX 5090) today — that's a correct,
informative result (confirms the known upstream gap), not a bug in this
repo to chase. See INSTALL.md for how to tell that failure apart from an
actual regression.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.model]

_TINY_RESOLUTION = "1024x1024 (1:1 Square)"
_PROMPT = "a red apple on a white table, studio lighting"


def _load_and_generate(model_path, *, attention_impl, moe_impl):
    hunyuan_instruct_nodes = pytest.importorskip("hunyuan_instruct_nodes")
    loader = hunyuan_instruct_nodes.HunyuanInstructLoader()
    (model,) = loader.load_model(
        model_name=model_path,
        force_reload=True,
        blocks_to_swap=0,
        attention_impl=attention_impl,
        moe_impl=moe_impl,
    )
    assert model is not None

    gen = hunyuan_instruct_nodes.HunyuanInstructGenerate()
    images, cot_reasoning, status = gen.generate(
        model=model,
        prompt=_PROMPT,
        bot_task="image",
        resolution=_TINY_RESOLUTION,
        seed=1,
        steps=4,
    )
    assert images is not None
    assert images.shape[-1] == 3
    assert images.float().std().item() > 1e-4

    unload = hunyuan_instruct_nodes.HunyuanInstructUnload()
    unload.unload(enabled=True)
    assert hunyuan_instruct_nodes._instruct_cache.model is None


def test_flashinfer_moe_loads_and_generates(
    cuda_device, torch_module, instruct_distil_nf4_v2_dir, require_vram
):
    pytest.importorskip("flashinfer")
    require_vram(30, "Instruct-Distil-NF4-v2 model")
    # First run JIT-compiles flashinfer's kernels for this GPU (per
    # Tencent's README, ~10 minutes) — pyproject.toml's 900s pytest timeout
    # may need --timeout override on a cold cache; a slow first pass here
    # is expected, not a bug.
    _load_and_generate(instruct_distil_nf4_v2_dir, attention_impl="sdpa", moe_impl="flashinfer")


def test_flash_attention_2_loads_and_generates(
    cuda_device, torch_module, instruct_distil_nf4_v2_dir, require_vram
):
    pytest.importorskip("flash_attn")
    require_vram(30, "Instruct-Distil-NF4-v2 model")
    _load_and_generate(instruct_distil_nf4_v2_dir, attention_impl="flash_attention_2", moe_impl="eager")
