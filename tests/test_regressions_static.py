"""
Fast, dependency-light regression tests for the specific bugs fixed in the
2026 paranoid-audit pass (fragmented caches, block-swap stream race, bnb
repair, silent failures, etc).

None of these need a GPU or model weights — they check that the fix is
still structurally present (right method called, right parameter threaded
through, right exception behaviour) via signature/source inspection. They
are a tripwire against accidental regressions, not a substitute for the
GPU smoke tests in test_smoke_*.py, which exercise the real runtime paths.

Run with: pytest tests/test_regressions_static.py -v
"""
from __future__ import annotations

import inspect

import pytest


def _import_or_skip(module_name: str):
    return pytest.importorskip(
        module_name,
        reason=f"{module_name} not importable in this environment "
        f"(needs ComfyUI on sys.path — see --comfyui-dir in README.md)",
    )


# ---------------------------------------------------------------------------
# Issue #46 — moe_drop_tokens/vae_dtype not threaded through
# HunyuanGenerateWithLatent.generate()
# ---------------------------------------------------------------------------

def test_generate_with_latent_accepts_moe_drop_tokens_and_vae_dtype():
    hunyuan_latent_nodes = _import_or_skip("hunyuan_latent_nodes")
    sig = inspect.signature(hunyuan_latent_nodes.HunyuanGenerateWithLatent.generate)
    assert "moe_drop_tokens" in sig.parameters
    assert "vae_dtype" in sig.parameters
    assert sig.parameters["moe_drop_tokens"].default is True
    assert sig.parameters["vae_dtype"].default == "bfloat16"


def test_generate_with_latent_input_types_match_generate_signature():
    """The class the exact bug shape (#46) came from: INPUT_TYPES() advertises a
    widget that generate() can't accept. Verify every optional INPUT_TYPES key
    that isn't consumed via **kwargs has a matching generate() parameter."""
    hunyuan_latent_nodes = _import_or_skip("hunyuan_latent_nodes")
    cls = hunyuan_latent_nodes.HunyuanGenerateWithLatent
    schema = cls.INPUT_TYPES()
    sig = inspect.signature(cls.generate)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        pytest.skip("generate() accepts **kwargs, widget/signature mismatch can't occur")
    param_names = set(sig.parameters) - {"self"}
    for section in ("required", "optional"):
        for key in schema.get(section, {}):
            assert key in param_names, (
                f"INPUT_TYPES()['{section}']['{key}'] has no matching parameter "
                f"in generate() — this is exactly the #46 bug shape"
            )


# ---------------------------------------------------------------------------
# Issues #36, #41 — shared_mlp/mlp.gate wrongly quantized
# ---------------------------------------------------------------------------

def test_repair_unquantized_bnb_modules_exists():
    hunyuan_shared = _import_or_skip("hunyuan_shared")
    assert callable(hunyuan_shared.repair_unquantized_bnb_modules)
    assert callable(hunyuan_shared._get_bnb_skip_fragments)


def test_get_bnb_skip_fragments_reads_model_quantization_config():
    hunyuan_shared = _import_or_skip("hunyuan_shared")

    class FakeConfig:
        quantization_config = {"llm_int8_skip_modules": ["shared_mlp", "mlp.gate", "vae"]}

    class FakeModel:
        config = FakeConfig()

    fragments = hunyuan_shared._get_bnb_skip_fragments(FakeModel())
    assert "shared_mlp" in fragments
    assert "mlp.gate" in fragments


def test_get_bnb_skip_fragments_falls_back_without_config():
    hunyuan_shared = _import_or_skip("hunyuan_shared")

    class FakeModel:
        pass

    fragments = hunyuan_shared._get_bnb_skip_fragments(FakeModel())
    assert "shared_mlp" in fragments
    assert "mlp.gate" in fragments


def test_repair_unquantized_bnb_modules_demotes_fake_quantized_linear():
    """End-to-end of the repair logic using a plain nn.Module tree — no
    bitsandbytes/GPU required, exercises the detection + rebuild logic."""
    torch = pytest.importorskip("torch")
    hunyuan_shared = _import_or_skip("hunyuan_shared")

    try:
        from bitsandbytes.nn import Linear4bit
    except ImportError:
        pytest.skip("bitsandbytes not installed")

    class FakeConfig:
        quantization_config = {"llm_int8_skip_modules": ["shared_mlp"]}

    class Mlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # A Linear4bit that was constructed but never had .cuda()/.to()
            # called on it — exactly the "wrapped but never quantized" state
            # (weight.quant_state is None, weight.dtype is still floating).
            self.shared_mlp = torch.nn.Module()
            self.shared_mlp.gate_and_up_proj = Linear4bit(8, 8, bias=False)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = FakeConfig()
            self.mlp = Mlp()

    model = FakeModel()
    fixed = hunyuan_shared.repair_unquantized_bnb_modules(model)
    assert fixed == 1
    assert isinstance(model.mlp.shared_mlp.gate_and_up_proj, torch.nn.Linear)
    assert not isinstance(model.mlp.shared_mlp.gate_and_up_proj, Linear4bit)


# ---------------------------------------------------------------------------
# Issue #39 — cache_position not migrated in HunyuanStaticCache device patch
# ---------------------------------------------------------------------------

def test_static_cache_device_patch_migrates_cache_position():
    """patch_hunyuan_static_cache_device looks the target class up via
    type(model).__module__ first, falling back to scanning sys.modules for
    anything exposing HunyuanStaticCache. A plain test-local FakeModel class
    lives in *this* test module (not a fake "remote code" module), so the
    primary lookup deliberately misses and this exercises the fallback scan
    — which is realistic, since that's the same path real Hunyuan models hit
    (their class's __module__ is the dynamically-imported remote-code
    module, which is what actually gets registered in sys.modules)."""
    hunyuan_shared = _import_or_skip("hunyuan_shared")
    torch = pytest.importorskip("torch")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    stale_device = "cpu" if device == "cuda:0" else device  # only truly "stale" with a real GPU

    class FakeLayer:
        keys = torch.zeros(1, device=stale_device)
        values = torch.zeros(1, device=stale_device)

    class FakeCache:
        layers = [FakeLayer()]
        calls = []

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            # What the patch is responsible for: by the time the *original*
            # update() runs, cache_position must already be on the same
            # device as key_states (this is what index_copy_ requires).
            cp = cache_kwargs.get("cache_position") if cache_kwargs else None
            self.calls.append(cp.device if cp is not None else None)
            assert cp is None or cp.device == key_states.device
            return key_states, value_states

    import types
    import sys
    fake_module = types.ModuleType("fake_hunyuan_remote_module")
    fake_module.HunyuanStaticCache = FakeCache
    FakeCache.__module__ = "fake_hunyuan_remote_module"
    sys.modules["fake_hunyuan_remote_module"] = fake_module

    class FakeModel:
        pass

    try:
        patched = hunyuan_shared.patch_hunyuan_static_cache_device(FakeModel())
        assert patched is True

        cache = FakeCache()
        key_states = torch.zeros(1, 1, 1, 1, device=device)
        stale_cache_position = torch.tensor([0], device=stale_device)
        cache.update(key_states, key_states, 0, cache_kwargs={"cache_position": stale_cache_position})
        assert cache.calls[-1] == torch.device(device), (
            "cache_position was not migrated to key_states.device before "
            "reaching the original update() — the index_copy_ device "
            "mismatch (issue #39) would reproduce here"
        )
    finally:
        del sys.modules["fake_hunyuan_remote_module"]


# ---------------------------------------------------------------------------
# Issue #40 (residual) — flag stored on unpatchable bound-method object
# ---------------------------------------------------------------------------

def test_apply_nf4_transformers_compat_image_processor_patch_flag_survives():
    """Regression for: setting an attribute on a types.MethodType object
    raises AttributeError on every Python version — the flag must live on
    the image_processor object, not the bound method."""
    hunyuan_shared = _import_or_skip("hunyuan_shared")
    torch = pytest.importorskip("torch")
    if not hunyuan_shared._TRANSFORMERS_GTE_5:
        pytest.skip("this shim is a no-op below transformers 5.0")

    class FakeImageProcessor:
        def vit_process_image(self, *a, **kw):
            raise AttributeError("'list' object has no attribute 'squeeze'")

    class FakeModel:
        image_processor = FakeImageProcessor()

        def modules(self):
            return iter(())

    model = FakeModel()
    # Must not raise — this crashed before the fix (bound-method attribute set).
    hunyuan_shared.apply_nf4_transformers_compat(model)
    assert getattr(model.image_processor, "_hunyuan_t5_compat_patched", False) is True
    # Idempotent second call must also not raise.
    hunyuan_shared.apply_nf4_transformers_compat(model)


# ---------------------------------------------------------------------------
# CleanModelLoader.SKIP_MODULES missing shared_mlp/mlp.gate
# ---------------------------------------------------------------------------

def test_clean_model_loader_skip_modules_includes_moe_critical_layers():
    hunyuan_loader_clean = _import_or_skip("hunyuan_loader_clean")
    skip = hunyuan_loader_clean.CleanModelLoader.SKIP_MODULES
    assert "shared_mlp" in skip
    assert "mlp.gate" in skip


# ---------------------------------------------------------------------------
# Fragmented model caches
# ---------------------------------------------------------------------------

def test_clear_all_hunyuan_caches_exists_and_is_callable():
    hunyuan_shared = _import_or_skip("hunyuan_shared")
    assert callable(hunyuan_shared.clear_all_hunyuan_caches)


def test_clear_all_hunyuan_caches_calls_all_three_by_default(monkeypatch):
    hunyuan_shared = _import_or_skip("hunyuan_shared")
    _import_or_skip("hunyuan_cache_v2")
    _import_or_skip("hunyuan_instruct_nodes")

    calls = []

    monkeypatch.setattr(
        hunyuan_shared.HunyuanModelCache, "clear",
        classmethod(lambda cls: (calls.append("legacy"), True)[1]),
    )

    import hunyuan_cache_v2

    class FakeV2Cache:
        def full_unload(self, *a, **kw):
            calls.append("v2")

    monkeypatch.setattr(hunyuan_cache_v2, "get_cache", lambda: FakeV2Cache())

    import hunyuan_instruct_nodes

    class FakeInstructCache:
        def clear(self):
            calls.append("instruct")

    monkeypatch.setattr(hunyuan_instruct_nodes, "_instruct_cache", FakeInstructCache())

    results = hunyuan_shared.clear_all_hunyuan_caches()
    assert set(calls) == {"legacy", "v2", "instruct"}
    assert {name for name, ok in results} == {"legacy", "v2", "instruct"}
    assert all(ok for _name, ok in results)


def test_clear_all_hunyuan_caches_exclude_skips_the_caller(monkeypatch):
    hunyuan_shared = _import_or_skip("hunyuan_shared")
    _import_or_skip("hunyuan_cache_v2")
    _import_or_skip("hunyuan_instruct_nodes")

    calls = []
    monkeypatch.setattr(
        hunyuan_shared.HunyuanModelCache, "clear",
        classmethod(lambda cls: (calls.append("legacy"), True)[1]),
    )

    import hunyuan_cache_v2

    class FakeV2Cache:
        def full_unload(self, *a, **kw):
            calls.append("v2")

    monkeypatch.setattr(hunyuan_cache_v2, "get_cache", lambda: FakeV2Cache())

    import hunyuan_instruct_nodes

    class FakeInstructCache:
        def clear(self):
            calls.append("instruct")

    monkeypatch.setattr(hunyuan_instruct_nodes, "_instruct_cache", FakeInstructCache())

    hunyuan_shared.clear_all_hunyuan_caches(exclude="v2")
    assert "v2" not in calls
    assert set(calls) == {"legacy", "instruct"}


def test_clear_all_hunyuan_caches_reentrancy_guard(monkeypatch):
    """If one cache's clear() itself calls clear_all_hunyuan_caches() (as a
    naive fix might), the nested call must no-op instead of recursing."""
    hunyuan_shared = _import_or_skip("hunyuan_shared")
    _import_or_skip("hunyuan_cache_v2")
    _import_or_skip("hunyuan_instruct_nodes")

    nested_result = {}

    def naive_legacy_clear(cls):
        nested_result["value"] = hunyuan_shared.clear_all_hunyuan_caches(exclude="legacy")
        return True

    monkeypatch.setattr(hunyuan_shared.HunyuanModelCache, "clear", classmethod(naive_legacy_clear))

    import hunyuan_cache_v2
    monkeypatch.setattr(hunyuan_cache_v2, "get_cache", lambda: type("C", (), {"full_unload": lambda self: None})())

    import hunyuan_instruct_nodes
    monkeypatch.setattr(hunyuan_instruct_nodes, "_instruct_cache", type("C", (), {"clear": lambda self: None})())

    hunyuan_shared.clear_all_hunyuan_caches()
    # The re-entrant call made from inside HunyuanModelCache.clear() must
    # have been suppressed by the guard and returned immediately empty.
    assert nested_result["value"] == []


def test_clear_all_hunyuan_caches_tolerates_one_cache_failing(monkeypatch):
    hunyuan_shared = _import_or_skip("hunyuan_shared")
    _import_or_skip("hunyuan_cache_v2")
    _import_or_skip("hunyuan_instruct_nodes")

    def boom(cls):
        raise RuntimeError("legacy cache is on fire")

    monkeypatch.setattr(hunyuan_shared.HunyuanModelCache, "clear", classmethod(boom))

    calls = []
    import hunyuan_cache_v2
    monkeypatch.setattr(
        hunyuan_cache_v2, "get_cache",
        lambda: type("C", (), {"full_unload": lambda self: calls.append("v2")})(),
    )
    import hunyuan_instruct_nodes
    monkeypatch.setattr(
        hunyuan_instruct_nodes, "_instruct_cache",
        type("C", (), {"clear": lambda self: calls.append("instruct")})(),
    )

    results = hunyuan_shared.clear_all_hunyuan_caches()
    # v2 and instruct must still have been attempted despite legacy raising.
    assert "v2" in calls
    assert "instruct" in calls
    result_map = dict(results)
    assert result_map["legacy"] is False
    assert result_map["v2"] is True
    assert result_map["instruct"] is True


def test_unload_nodes_call_clear_all_hunyuan_caches():
    """Source-level tripwire: each user-facing 'unload everything' node must
    reference the cross-cache coordinator, not just its own cache."""
    hunyuan_unified_v2 = _import_or_skip("hunyuan_unified_v2")
    hunyuan_shared = _import_or_skip("hunyuan_shared")

    unload_src = inspect.getsource(hunyuan_unified_v2.HunyuanUnloadV2.unload)
    assert "clear_all_hunyuan_caches" in unload_src

    cleanup_src = inspect.getsource(hunyuan_unified_v2.HunyuanEmergencyCleanup.cleanup)
    assert "clear_all_hunyuan_caches" in cleanup_src

    force_unload_src = inspect.getsource(hunyuan_shared.HunyuanImage3ForceUnload.force_unload)
    assert "clear_all_hunyuan_caches" in force_unload_src

    unload_node_src = inspect.getsource(hunyuan_shared.HunyuanImage3Unload.unload)
    assert "clear_all_hunyuan_caches" in unload_node_src

    instruct_nodes = _import_or_skip("hunyuan_instruct_nodes")
    instruct_unload_src = inspect.getsource(instruct_nodes.HunyuanInstructUnload.unload)
    assert "clear_all_hunyuan_caches" in instruct_unload_src


# ---------------------------------------------------------------------------
# clear_generation_cache() didn't clear the real KV-cache location
# ---------------------------------------------------------------------------

def test_clear_generation_cache_clears_pipeline_model_kwargs():
    hunyuan_shared = _import_or_skip("hunyuan_shared")
    torch = pytest.importorskip("torch")

    class FakePipeline:
        model_kwargs = {"past_key_values": object(), "output_hidden_states": True, "keep_me": 1}

    class FakeModel:
        _pipeline = FakePipeline()

        def named_modules(self):
            return iter(())

    model = FakeModel()
    hunyuan_shared.clear_generation_cache(model)
    assert "past_key_values" not in model._pipeline.model_kwargs
    assert "output_hidden_states" not in model._pipeline.model_kwargs
    assert model._pipeline.model_kwargs["keep_me"] == 1


# ---------------------------------------------------------------------------
# Silent failure returned a fake-success image instead of raising
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method_name",
    ["generate", "edit", "fuse"],
)
def test_instruct_nodes_raise_on_failure_instead_of_returning_placeholder(method_name):
    hunyuan_instruct_nodes = _import_or_skip("hunyuan_instruct_nodes")
    cls_by_method = {
        "generate": hunyuan_instruct_nodes.HunyuanInstructGenerate,
        "edit": hunyuan_instruct_nodes.HunyuanInstructImageEdit,
        "fuse": hunyuan_instruct_nodes.HunyuanInstructMultiFusion,
    }
    src = inspect.getsource(getattr(cls_by_method[method_name], method_name))
    # The old bug shape: `return (empty_image, "", f"Error: ...")` inside an
    # except block. The fix replaces both except blocks with `raise`.
    assert "return (empty_image" not in src
    assert src.count("raise") >= 2  # OOM branch + generic Exception branch


def test_highres_node_already_raises_on_oom():
    """HunyuanImage3GenerateHighRes was the one node that already did this
    right — used as the reference pattern for the Instruct fix above."""
    hunyuan_highres_nodes = _import_or_skip("hunyuan_highres_nodes")
    src = inspect.getsource(hunyuan_highres_nodes.HunyuanImage3GenerateHighRes.generate_highres)
    assert "raise" in src
    assert "return (empty" not in src.replace(" ", "")


# ---------------------------------------------------------------------------
# post_action skipped on generation failure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "module_name, class_name, method_name",
    [
        ("hunyuan_unified_v2", "HunyuanUnifiedV2", "generate"),
        ("hunyuan_latent_nodes", "HunyuanGenerateWithLatent", "generate"),
    ],
)
def test_post_action_runs_on_exception_paths(module_name, class_name, method_name):
    module = _import_or_skip(module_name)
    cls = getattr(module, class_name)
    src = inspect.getsource(getattr(cls, method_name))
    # Must reference _handle_post_action at least twice: once on the success
    # path, and at least once more inside the except blocks.
    assert src.count("_handle_post_action") >= 2


def test_highres_post_action_runs_on_oom():
    hunyuan_highres_nodes = _import_or_skip("hunyuan_highres_nodes")
    src = inspect.getsource(hunyuan_highres_nodes.HunyuanImage3GenerateHighRes.generate_highres)
    # There are two "except RuntimeError" blocks in this method (an early
    # CUDA-param validation check, and the actual generation OOM handler) —
    # target the generation one specifically via its "as e" binding, which
    # only the latter has.
    assert "except RuntimeError as e:" in src
    except_block = src.split("except RuntimeError as e:", 1)[1].split("finally:", 1)[0]
    assert "HunyuanModelCache" in except_block


# ---------------------------------------------------------------------------
# Missing IS_CHANGED on Instruct generation nodes (seed=-1 caching bug)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "class_name",
    ["HunyuanInstructGenerate", "HunyuanInstructImageEdit", "HunyuanInstructMultiFusion"],
)
def test_instruct_nodes_have_is_changed_for_random_seed(class_name):
    hunyuan_instruct_nodes = _import_or_skip("hunyuan_instruct_nodes")
    cls = getattr(hunyuan_instruct_nodes, class_name)
    assert hasattr(cls, "IS_CHANGED"), f"{class_name} is missing IS_CHANGED"
    import math
    result = cls.IS_CHANGED(seed=-1)
    assert isinstance(result, float) and math.isnan(result)
    result2 = cls.IS_CHANGED(seed=12345)
    assert not (isinstance(result2, float) and result2 != result2)  # not NaN


# ---------------------------------------------------------------------------
# HunyuanModelCache.full_unload() doesn't exist
# ---------------------------------------------------------------------------

def test_hunyuan_model_cache_has_no_full_unload_method():
    hunyuan_shared = _import_or_skip("hunyuan_shared")
    assert not hasattr(hunyuan_shared.HunyuanModelCache, "full_unload")
    assert hasattr(hunyuan_shared.HunyuanModelCache, "clear")


def test_moe_test_node_full_unload_action_calls_clear_not_full_unload():
    hunyuan_moe_test_node = _import_or_skip("hunyuan_moe_test_node")
    src = inspect.getsource(hunyuan_moe_test_node.HunyuanImage3MoETest.generate)
    assert "HunyuanModelCache.full_unload()" not in src
    assert "HunyuanModelCache.clear()" in src


# ---------------------------------------------------------------------------
# HunyuanEmergencyCleanup's hook-cleanup loop was dead code (sys.modules
# instead of nn.Module instances)
# ---------------------------------------------------------------------------

def test_emergency_cleanup_hook_scan_uses_gc_not_sys_modules():
    hunyuan_unified_v2 = _import_or_skip("hunyuan_unified_v2")
    src = inspect.getsource(hunyuan_unified_v2.HunyuanEmergencyCleanup.cleanup)
    assert "gc.get_objects()" in src
    assert "for name, module in list(sys.modules.items())" not in src


# ---------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES leaked on the normal (2+ GPU) DualGPULoader path
# ---------------------------------------------------------------------------

def test_dual_gpu_loader_restores_cuda_visible_devices_unconditionally():
    hunyuan_full_bf16_nodes = _import_or_skip("hunyuan_full_bf16_nodes")
    src = inspect.getsource(hunyuan_full_bf16_nodes.HunyuanImage3DualGPULoader.load_model)
    assert "finally:" in src
    finally_block = src.rsplit("finally:", 1)[1]
    assert "CUDA_VISIBLE_DEVICES" in finally_block


# ---------------------------------------------------------------------------
# HunyuanAPIConfig.load_config() doesn't exist
# ---------------------------------------------------------------------------

def test_highres_prompt_rewrite_uses_real_api_config_function():
    hunyuan_highres_nodes = _import_or_skip("hunyuan_highres_nodes")
    hunyuan_api_config = _import_or_skip("hunyuan_api_config")
    assert not hasattr(hunyuan_api_config, "HunyuanAPIConfig")
    assert callable(hunyuan_api_config.get_api_config)
    src = inspect.getsource(hunyuan_highres_nodes.HunyuanImage3GenerateHighRes.generate_highres)
    assert "HunyuanAPIConfig" not in src
    assert "get_api_config" in src


# ---------------------------------------------------------------------------
# Pre-quantized INT8 loading allowed CPU offload (silent corruption)
# ---------------------------------------------------------------------------

def test_load_int8_fallback_does_not_set_cpu_max_memory():
    hunyuan_loader_clean = _import_or_skip("hunyuan_loader_clean")
    src = inspect.getsource(hunyuan_loader_clean.CleanModelLoader._load_int8)
    # Check for an actual assignment, not just the substring — the fix's
    # own explanatory comment ("Deliberately NOT setting
    # max_memory[\"cpu\"]: ...") legitimately contains this text without
    # being the bug.
    assert 'max_memory["cpu"] =' not in src
    assert 'max_memory = {"cpu": "100GiB"}' not in src


def test_int8_loader_does_not_set_cpu_max_memory():
    hunyuan_quantized_nodes = _import_or_skip("hunyuan_quantized_nodes")
    src = inspect.getsource(hunyuan_quantized_nodes.HunyuanImage3Int8Loader.load_model)
    assert '"cpu": "100GiB"' not in src


# ---------------------------------------------------------------------------
# Cross-CUDA-stream race in block swap release/prefetch
# ---------------------------------------------------------------------------

def test_block_swap_manager_has_release_events_tracking():
    hunyuan_block_swap = _import_or_skip("hunyuan_block_swap")
    torch = pytest.importorskip("torch")

    src_init = inspect.getsource(hunyuan_block_swap.BlockSwapManager.__init__)
    assert "_release_events" in src_init

    move_src = inspect.getsource(hunyuan_block_swap.BlockSwapManager._move_block_to_device)
    assert "_release_events" in move_src
    assert "torch.cuda.Event()" in move_src

    prefetch_src = inspect.getsource(hunyuan_block_swap.BlockSwapManager._prefetch_upcoming)
    assert "_release_events" in prefetch_src
    assert "wait_event" in prefetch_src


# ---------------------------------------------------------------------------
# validate_attention_moe_impl() — flash_attn/flashinfer availability guard
# ---------------------------------------------------------------------------

def test_validate_attention_moe_impl_raises_when_backend_unavailable(monkeypatch):
    hunyuan_shared = _import_or_skip("hunyuan_shared")

    monkeypatch.setattr(hunyuan_shared, "_FLASH_ATTN_AVAILABLE", False)
    monkeypatch.setattr(hunyuan_shared, "_FLASHINFER_AVAILABLE", False)

    with pytest.raises(ValueError, match="flash_attention_2"):
        hunyuan_shared.validate_attention_moe_impl("flash_attention_2", "eager")

    with pytest.raises(ValueError, match="flashinfer"):
        hunyuan_shared.validate_attention_moe_impl("sdpa", "flashinfer")

    # Doesn't raise for the always-available defaults, regardless of
    # whether flash_attn/flashinfer happen to be installed.
    hunyuan_shared.validate_attention_moe_impl("sdpa", "eager")


def test_validate_attention_moe_impl_allows_backend_when_available(monkeypatch):
    hunyuan_shared = _import_or_skip("hunyuan_shared")

    monkeypatch.setattr(hunyuan_shared, "_FLASH_ATTN_AVAILABLE", True)
    monkeypatch.setattr(hunyuan_shared, "_FLASHINFER_AVAILABLE", True)

    # Should not raise now that both sentinels report "available" — even
    # without real torch CUDA (get_device_capability is only consulted
    # inside an `if torch.cuda.is_available()` guard).
    hunyuan_shared.validate_attention_moe_impl("flash_attention_2", "flashinfer")


def test_instruct_loader_calls_validate_attention_moe_impl():
    """The choke-point wiring itself: load_model() must call the guard
    before dispatching to from_pretrained, not just have it importable."""
    hunyuan_instruct_nodes = _import_or_skip("hunyuan_instruct_nodes")
    src = inspect.getsource(hunyuan_instruct_nodes.HunyuanInstructLoader.load_model)
    assert "validate_attention_moe_impl(attention_impl, moe_impl)" in src
