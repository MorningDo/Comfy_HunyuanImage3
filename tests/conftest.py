"""
Fixtures for the Comfy_HunyuanImage3 test suite. sys.path setup and pytest
hooks (pytest_addoption/pytest_configure) live in the root-level
conftest.py (../conftest.py) — see that file's docstring for why. pytest
merges fixtures from both files automatically; nothing here needs to
import from there.

WHERE TO POINT THIS AT YOUR MODELS
-----------------------------------
Every model-dependent test is skipped unless it can find the model it needs.
Point the suite at the top-level directory that *contains* your model
subfolders (e.g. the directory holding "HunyuanImage-3-NF4/",
"HunyuanImage-3-NF4-v2/", "HunyuanImage-3.0-Instruct-Distil-INT8-v2/", ...)
via EITHER:

    --hunyuan-models-dir /path/to/models      (pytest CLI flag)
    HUNYUAN_TEST_MODELS_DIR=/path/to/models   (env var, lower priority than the flag)

The suite scans that directory's immediate subfolders and classifies each
one by name (nf4 / nf4_v2 / int8 / int8_v2 / bf16 / instruct_* / distil_*)
using the same naming convention as the published HuggingFace repos. If
your folder names don't match, override any single role explicitly with
its own env var instead of (or in addition to) --hunyuan-models-dir, e.g.:

    HUNYUAN_TEST_MODEL_NF4_V2=/mnt/models/my-renamed-nf4-v2-folder

See README.md in this directory for the full list of role env vars and
how to run the suite on a fresh machine (e.g. the RTX 6000 test box).

WHERE TO POINT THIS AT YOUR COMFYUI INSTALL
--------------------------------------------
Two files in this package (hunyuan_quantized_nodes.py, hunyuan_full_bf16_nodes.py)
import `folder_paths` and `comfy.utils` unconditionally. A real ComfyUI
checkout on sys.path (auto-detected, or via --comfyui-dir/$COMFYUI_DIR)
makes those genuinely functional; otherwise a no-op stub (tests/_stubs/)
keeps them merely *importable* so the rest of the suite still collects and
runs — see ../conftest.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pytest

# ---------------------------------------------------------------------------
# GPU fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def torch_module():
    torch = pytest.importorskip("torch", reason="torch is a hard dependency of this package")
    return torch


@pytest.fixture(scope="session")
def gpu_available(torch_module) -> bool:
    return bool(torch_module.cuda.is_available())


@pytest.fixture(scope="session")
def cuda_device(gpu_available, torch_module) -> str:
    if not gpu_available:
        pytest.skip("no CUDA GPU available")
    return "cuda:0"


@pytest.fixture(scope="session")
def gpu_total_vram_gb(gpu_available, torch_module) -> float:
    if not gpu_available:
        return 0.0
    _free, total = torch_module.cuda.mem_get_info(0)
    return total / 1024**3


@pytest.fixture()
def require_vram(gpu_total_vram_gb: float):
    """require_vram(needed_gb, label) -> skips the test if the GPU doesn't
    have enough VRAM, with a clear reason. A fixture (not a plain helper
    function) so it works reliably regardless of pytest's import mode —
    avoid `from conftest import ...` in test files, request this instead."""

    def _require(needed_gb: float, label: str) -> None:
        if gpu_total_vram_gb < needed_gb:
            pytest.skip(
                f"{label} needs ~{needed_gb:.0f}GB VRAM, only "
                f"{gpu_total_vram_gb:.1f}GB detected on this GPU"
            )

    return _require


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

# role -> (must-contain-all-of, must-contain-none-of), matched against the
# lowercased directory name. Order matters: first match wins.
_ROLE_RULES = [
    ("instruct_distil_nf4_v2", ["instruct", "distil", "nf4"], []),
    ("instruct_distil_int8_v2", ["instruct", "distil", "int8"], []),
    ("instruct_nf4_v2", ["instruct", "nf4", "v2"], ["distil"]),
    ("instruct_nf4", ["instruct", "nf4"], ["distil", "v2"]),
    ("instruct_int8_v2", ["instruct", "int8", "v2"], ["distil"]),
    ("instruct_int8", ["instruct", "int8"], ["distil", "v2"]),
    ("instruct_bf16", ["instruct"], ["nf4", "int8"]),
    ("nf4_v2", ["nf4", "v2"], ["instruct"]),
    ("nf4", ["nf4"], ["instruct", "v2"]),
    ("int8_v2", ["int8", "v2"], ["instruct"]),
    ("int8", ["int8"], ["instruct", "v2"]),
    ("bf16", [], ["instruct", "nf4", "int8"]),  # catch-all: full-precision folder
]

# Per-role override env vars, e.g. HUNYUAN_TEST_MODEL_NF4_V2=/path/to/dir
_ROLE_ENV_PREFIX = "HUNYUAN_TEST_MODEL_"


def _classify(dirname: str) -> Optional[str]:
    name = dirname.lower()
    for role, must_have, must_not_have in _ROLE_RULES:
        if all(tok in name for tok in must_have) and not any(tok in name for tok in must_not_have):
            return role
    return None


@dataclass
class ModelRegistry:
    models_dir: Optional[Path]
    roles: Dict[str, Path] = field(default_factory=dict)

    def path_for(self, role: str) -> Optional[Path]:
        return self.roles.get(role)

    def require(self, role: str) -> Path:
        path = self.roles.get(role)
        if path is None:
            hint = f"{_ROLE_ENV_PREFIX}{role.upper()}"
            if self.models_dir is None:
                pytest.skip(
                    f"no model directory configured — pass --hunyuan-models-dir, "
                    f"set $HUNYUAN_TEST_MODELS_DIR, or set ${hint} directly to skip "
                    f"auto-detection for just this role"
                )
            pytest.skip(
                f"could not find a '{role}' model under {self.models_dir} "
                f"(auto-detection didn't match any subfolder name) — set ${hint} "
                f"to the exact path if your folder is named differently"
            )
        return path


def _discover(models_dir: Optional[Path]) -> ModelRegistry:
    registry = ModelRegistry(models_dir=models_dir)

    if models_dir is not None and models_dir.is_dir():
        for entry in sorted(models_dir.iterdir()):
            if not entry.is_dir():
                continue
            role = _classify(entry.name)
            if role and role not in registry.roles:
                registry.roles[role] = entry

    # Explicit per-role overrides always win, and work even without
    # --hunyuan-models-dir/HUNYUAN_TEST_MODELS_DIR set at all.
    for role, _rules, _excl in _ROLE_RULES:
        env_name = f"{_ROLE_ENV_PREFIX}{role.upper()}"
        override = os.environ.get(env_name)
        if override:
            path = Path(override).expanduser().resolve()
            if not path.is_dir():
                raise pytest.UsageError(f"${env_name}={override!r} is not a directory")
            registry.roles[role] = path

    return registry


@pytest.fixture(scope="session")
def hunyuan_models(request: pytest.FixtureRequest) -> ModelRegistry:
    models_dir_opt = request.config.getoption("--hunyuan-models-dir") or os.environ.get(
        "HUNYUAN_TEST_MODELS_DIR"
    )
    models_dir = Path(models_dir_opt).expanduser().resolve() if models_dir_opt else None
    if models_dir is not None and not models_dir.is_dir():
        raise pytest.UsageError(f"--hunyuan-models-dir {models_dir} is not a directory")
    registry = _discover(models_dir)
    if registry.roles:
        found = ", ".join(f"{r}={p.name}" for r, p in sorted(registry.roles.items()))
        print(f"\n[hunyuan_models] discovered: {found}")
    else:
        print(
            "\n[hunyuan_models] no models discovered — model-dependent tests will "
            "be skipped. Pass --hunyuan-models-dir or set env vars (see README.md)."
        )
    return registry


# Convenience per-role fixtures used by most smoke tests. Each is just
# `hunyuan_models.require(role)` with a readable name at the call site.
def _role_fixture(role: str):
    @pytest.fixture()
    def _fixture(hunyuan_models: ModelRegistry) -> str:
        return str(hunyuan_models.require(role))

    _fixture.__name__ = f"{role}_dir"
    return _fixture


nf4_dir = _role_fixture("nf4")
nf4_v2_dir = _role_fixture("nf4_v2")
int8_dir = _role_fixture("int8")
int8_v2_dir = _role_fixture("int8_v2")
bf16_dir = _role_fixture("bf16")
instruct_nf4_v2_dir = _role_fixture("instruct_nf4_v2")
instruct_int8_v2_dir = _role_fixture("instruct_int8_v2")
instruct_distil_nf4_v2_dir = _role_fixture("instruct_distil_nf4_v2")
instruct_distil_int8_v2_dir = _role_fixture("instruct_distil_int8_v2")
instruct_bf16_dir = _role_fixture("instruct_bf16")


# ---------------------------------------------------------------------------
# Cache hygiene — this suite loads several multi-GB-to-160GB models across a
# session; leaking one into the next test is the fastest way to OOM the
# whole run. Force every cache empty before AND after every test that's
# plausibly touched one.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_hunyuan_caches_around_test():
    def _clear_all():
        try:
            import hunyuan_shared
            hunyuan_shared.clear_all_hunyuan_caches()
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[cache cleanup] clear_all_hunyuan_caches failed: {exc}")
        try:
            import torch
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
        except Exception:
            pass

    _clear_all()
    yield
    _clear_all()
