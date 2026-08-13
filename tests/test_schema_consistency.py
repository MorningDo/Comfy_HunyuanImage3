"""
Structural validation of every registered ComfyUI node in this package.

No GPU or model weights needed. This is the general form of the bug found
in issue #46 (HunyuanGenerateWithLatent.generate() didn't accept
moe_drop_tokens/vae_dtype even though INPUT_TYPES() advertised them as
widgets) — applied to every node so a similar mismatch anywhere else in the
package fails a test instead of a user's queue.

Needs this package importable (conftest.py handles that) and, for the
legacy loader nodes, a real ComfyUI checkout on sys.path (--comfyui-dir) —
those modules are skipped individually if unavailable rather than failing
collection.
"""
from __future__ import annotations

import inspect

import pytest


def _try_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:
        return None


# Each entry: (module_name, [ClassName, ...]). Only classes that are actual
# ComfyUI nodes (have INPUT_TYPES/FUNCTION) belong here.
_NODE_MODULES = [
    ("hunyuan_unified_v2", [
        "HunyuanUnifiedV2", "HunyuanUnloadV2", "HunyuanCacheStatusV2",
        "HunyuanVRAMCalculatorV2", "HunyuanEmergencyCleanup",
    ]),
    ("hunyuan_latent_nodes", [
        "HunyuanEmptyLatent", "HunyuanLatentNoise", "HunyuanGenerateWithLatent",
    ]),
    ("hunyuan_highres_nodes", ["HunyuanImage3GenerateHighRes"]),
    ("hunyuan_instruct_nodes", [
        "HunyuanInstructLoader", "HunyuanInstructGenerate", "HunyuanInstructImageEdit",
        "HunyuanInstructMultiFusion", "HunyuanInstructUnload",
    ]),
    ("hunyuan_quantized_nodes", [
        "HunyuanImage3QuantizedLoader", "HunyuanImage3Int8Loader",
        "HunyuanImage3NF4LoaderLowVRAMBudget", "HunyuanImage3Int8LoaderBudget",
        "HunyuanImage3Generate", "HunyuanImage3GenerateLarge",
        "HunyuanImage3GenerateLargeBudget", "HunyuanImage3GenerateLowVRAM",
        "HunyuanImage3GenerateLowVRAMBudget",
    ]),
    ("hunyuan_full_bf16_nodes", [
        "HunyuanImage3FullLoader", "HunyuanImage3FullGPULoader",
        "HunyuanImage3DualGPULoader", "HunyuanImage3GPUInfo",
        "HunyuanImage3SingleGPU88GB",
    ]),
    ("hunyuan_shared", [
        "HunyuanImage3Unload", "HunyuanImage3SoftUnload", "HunyuanImage3ForceUnload",
        "HunyuanImage3ClearDownstream", "HunyuanRAMDiagnostic",
    ]),
    ("hunyuan_api_nodes", ["HunyuanPromptRewriter"]),
    ("hunyuan_moe_test_node", ["HunyuanImage3MoETest"]),
]


def _collect_node_classes():
    """(id, cls) pairs for every node class above that's actually importable
    in this environment. Import failures are skipped (module or file just
    isn't on this machine's path), not raised."""
    out = []
    for module_name, class_names in _NODE_MODULES:
        module = _try_import(module_name)
        if module is None:
            continue
        for class_name in class_names:
            cls = getattr(module, class_name, None)
            if cls is not None:
                out.append((f"{module_name}.{class_name}", cls))
    return out


_NODE_CLASSES = _collect_node_classes()

if not _NODE_CLASSES:
    pytest.skip(
        "no node modules importable in this environment — run inside a "
        "ComfyUI environment (see README.md)",
        allow_module_level=True,
    )


@pytest.mark.parametrize("node_id, cls", _NODE_CLASSES, ids=[n[0] for n in _NODE_CLASSES])
class TestNodeSchema:
    def test_input_types_is_callable_classmethod(self, node_id, cls):
        assert hasattr(cls, "INPUT_TYPES"), f"{node_id} has no INPUT_TYPES"
        schema = cls.INPUT_TYPES()
        assert isinstance(schema, dict)
        assert "required" in schema or "optional" in schema

    def test_function_attribute_names_a_real_method(self, node_id, cls):
        assert hasattr(cls, "FUNCTION"), f"{node_id} has no FUNCTION"
        func_name = cls.FUNCTION
        assert hasattr(cls, func_name), (
            f"{node_id}.FUNCTION = {func_name!r} but the class has no such method"
        )
        assert callable(getattr(cls, func_name))

    def test_input_types_keys_are_accepted_by_function(self, node_id, cls):
        """The core #46-class check: every INPUT_TYPES() key (required and
        optional; skipping 'hidden', which ComfyUI injects separately) must
        be a real parameter of the FUNCTION method, unless that method
        accepts **kwargs."""
        schema = cls.INPUT_TYPES()
        method = getattr(cls, cls.FUNCTION)
        sig = inspect.signature(method)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            pytest.skip(f"{node_id}.{cls.FUNCTION} accepts **kwargs — mismatch can't manifest")
        param_names = {
            name for name, p in sig.parameters.items()
            if p.kind not in (inspect.Parameter.VAR_POSITIONAL,)
        } - {"self"}

        missing = []
        for section in ("required", "optional"):
            for key in schema.get(section, {}):
                if key not in param_names:
                    missing.append(f"{section}.{key}")

        assert not missing, (
            f"{node_id}: INPUT_TYPES() advertises {missing} but "
            f"{cls.FUNCTION}{sig} doesn't accept them — ComfyUI will pass these "
            f"as kwargs and the node will crash with TypeError on every run "
            f"(this is the exact shape of issue #46)"
        )

    def test_return_types_and_names_length_match(self, node_id, cls):
        return_types = getattr(cls, "RETURN_TYPES", None)
        return_names = getattr(cls, "RETURN_NAMES", None)
        if return_types is None or return_names is None:
            pytest.skip(f"{node_id} doesn't define both RETURN_TYPES and RETURN_NAMES")
        assert len(return_types) == len(return_names), (
            f"{node_id}: RETURN_TYPES has {len(return_types)} entries but "
            f"RETURN_NAMES has {len(return_names)}"
        )

    def test_is_changed_seed_minus_one_is_nan_when_present(self, node_id, cls):
        """Convention used throughout this codebase: seed=-1 means random,
        and IS_CHANGED must return NaN for it or ComfyUI's cache will replay
        stale output on re-queue (issue found in HunyuanInstructGenerate/
        ImageEdit/MultiFusion — this guards every node using the convention,
        not just those three)."""
        if not hasattr(cls, "IS_CHANGED"):
            pytest.skip(f"{node_id} has no IS_CHANGED")
        sig = inspect.signature(cls.IS_CHANGED)
        if "seed" not in sig.parameters:
            pytest.skip(f"{node_id}.IS_CHANGED doesn't take a seed parameter")
        import math
        try:
            result = cls.IS_CHANGED(seed=-1)
        except TypeError:
            pytest.skip(f"{node_id}.IS_CHANGED(seed=-1) needs other required args to call directly")
            return
        assert isinstance(result, float) and math.isnan(result), (
            f"{node_id}.IS_CHANGED(seed=-1) returned {result!r}, expected NaN — "
            f"ComfyUI will treat seed=-1 as a stable cache key and replay the "
            f"same output on re-queue instead of generating a new random result"
        )
