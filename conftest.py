"""
Root-level pytest bootstrap for the Comfy_HunyuanImage3 test suite (test
fixtures live in tests/conftest.py — pytest merges both automatically).

WHY THIS FILE IS AT THE REPO ROOT AND NOT INSIDE tests/
---------------------------------------------------------
This package's own __init__.py hard-imports `folder_paths` (a real
ComfyUI module) at module scope, via hunyuan_quantized_nodes.py /
hunyuan_full_bf16_nodes.py. pytest's collection algorithm wraps any
rootdir that itself contains __init__.py in a "Package" collector and
imports that __init__.py as its very first collection step — before it
descends into tests/, and therefore before a conftest.py living inside
tests/ would get a chance to run any sys.path setup. Since this repo's
root IS such a package (it has to be, to work as a ComfyUI custom node),
that import happens unconditionally on every pytest invocation in this
repo, regardless of --import-mode.

So: the sys.path setup that makes `folder_paths`/`comfy` importable (a
real ComfyUI checkout via --comfyui-dir/$COMFYUI_DIR, or the no-op stub
in tests/_stubs/ as a fallback) has to live in a conftest.py that pytest
is guaranteed to load before that Package-collector import — which is
this file, loaded during pytest's plugin-discovery phase, strictly
before collection starts. pytest_addoption/pytest_configure also live
here for the same reason (CLI-flag-provided --comfyui-dir needs to win
over the stub before collection imports __init__.py; see the ordering
comment on _add_comfyui_to_syspath below).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    # This package's modules use `from .hunyuan_shared import X` with an
    # `except ImportError: from hunyuan_shared import X` fallback everywhere
    # — that fallback is exactly the flat-import style this enables, so we
    # don't need to import this package as `Comfy_HunyuanImage3.hunyuan_shared`,
    # plain `import hunyuan_shared` works.
    sys.path.insert(0, str(REPO_ROOT))


def _guess_comfyui_dir() -> Optional[Path]:
    """Best-effort: is this repo sitting inside ComfyUI/custom_nodes/<repo>?"""
    candidate = REPO_ROOT.parent.parent
    if (candidate / "folder_paths.py").exists():
        return candidate
    return None


def _add_comfyui_to_syspath(comfyui_dir: Optional[str]) -> Optional[Path]:
    """Insert a real ComfyUI checkout at sys.path[0] (highest priority).
    Safe to call multiple times (e.g. once from $COMFYUI_DIR at module-import
    time, again from --comfyui-dir once CLI args are parsed) — inserting at
    index 0 each time means whichever call provides a real, valid path wins
    over the tests/_stubs/ fallback below, even if the fallback was already
    appended to sys.path earlier, AS LONG AS this runs before the actual
    `import folder_paths` statement executes (i.e. before collection)."""
    path = Path(comfyui_dir).expanduser().resolve() if comfyui_dir else _guess_comfyui_dir()
    if path is None:
        return None
    if not (path / "folder_paths.py").exists():
        return None
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def _folder_paths_importable() -> bool:
    try:
        import folder_paths  # noqa: F401
        return True
    except Exception:
        return False


def _ensure_folder_paths_importable() -> None:
    """Real ComfyUI (env var, checked here at module-import time) first; the
    no-op stub only if that didn't work. --comfyui-dir (a CLI flag, not yet
    parsed at this point) gets its own chance to win in pytest_configure()
    below, which — because _add_comfyui_to_syspath always inserts at index 0
    — still takes priority over the stub even if the stub was already
    appended to sys.path here, since actual folder_paths import doesn't
    happen until collection, which runs after pytest_configure."""
    _add_comfyui_to_syspath(os.environ.get("COMFYUI_DIR"))
    stubs_dir = REPO_ROOT / "tests" / "_stubs"
    if not _folder_paths_importable() and str(stubs_dir) not in sys.path:
        sys.path.append(str(stubs_dir))


_ensure_folder_paths_importable()


def _alias_flat_module_names() -> None:
    """Four files in this package (hunyuan_quantized_nodes.py,
    hunyuan_full_bf16_nodes.py, hunyuan_highres_nodes.py,
    hunyuan_api_nodes.py) use an UNGUARDED `from .hunyuan_shared import X`
    relative import at module scope — unlike most other files here, they
    have no `except ImportError: from hunyuan_shared import X` fallback.
    A relative import only resolves when the module is imported as part of
    a real package (i.e. Python knows its __package__), so these four can
    never be imported as flat top-level modules (`import
    hunyuan_highres_nodes`) the way test files expect — that raises
    "attempted relative import with no known parent package".

    But pytest's own collection mechanism already imports this repo as a
    real package (Comfy_HunyuanImage3 or whatever this directory is named)
    before test collection ever starts — see this file's module docstring
    — which means Python has ALREADY correctly resolved
    `<pkg>.hunyuan_highres_nodes` etc. with working relative imports. Alias
    each into sys.modules under its flat name too, so a later `import
    hunyuan_highres_nodes` in a test file finds the already-resolved
    module instead of attempting (and failing) a second, context-free one.
    """
    package_name = REPO_ROOT.name
    parent_dir = str(REPO_ROOT.parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    try:
        import importlib
        importlib.import_module(package_name)
    except Exception as exc:
        print(f"[conftest] could not import {package_name!r} as a package "
              f"for flat-name aliasing (some tests will skip): {exc}")
        return

    prefix = f"{package_name}."
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith(prefix):
            continue
        flat_name = mod_name[len(prefix):]
        if "." in flat_name:
            continue  # only alias direct submodules, not sub-submodules
        sys.modules.setdefault(flat_name, mod)


_alias_flat_module_names()


# ---------------------------------------------------------------------------
# pytest options / markers — registered here (not tests/conftest.py) so
# --comfyui-dir is available before collection imports this repo's
# __init__.py; see module docstring.
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--hunyuan-models-dir",
        action="store",
        default=None,
        help="Top-level directory containing HunyuanImage-3 model subfolders "
             "(NF4/INT8/BF16, v1 and v2, Instruct variants). "
             "Falls back to $HUNYUAN_TEST_MODELS_DIR if not given.",
    )
    parser.addoption(
        "--comfyui-dir",
        action="store",
        default=None,
        help="Path to a ComfyUI checkout, needed for tests that exercise "
             "the legacy loader nodes (which hard-import folder_paths/comfy). "
             "Auto-detected if this repo sits under ComfyUI/custom_nodes/. "
             "Falls back to $COMFYUI_DIR. Without either, a no-op stub is "
             "used so imports still succeed (see tests/_stubs/).",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Also run slow tests (full image generation, multi-image fusion, "
             "BF16 full-precision loads). Off by default — smoke tests use "
             "small resolutions/step counts and are fast enough to always run.",
    )


def pytest_configure(config: pytest.Config) -> None:
    for name, desc in [
        ("gpu", "requires a CUDA GPU"),
        ("model", "requires real model weights on disk"),
        ("slow", "expensive (full generation, BF16, multi-image); skipped unless --run-slow"),
    ]:
        config.addinivalue_line("markers", f"{name}: {desc}")

    # A real --comfyui-dir always wins over the no-op stub, as long as this
    # runs before collection actually imports folder_paths — it does (see
    # module docstring).
    comfyui_dir_opt = config.getoption("--comfyui-dir")
    if comfyui_dir_opt:
        _add_comfyui_to_syspath(comfyui_dir_opt)


def pytest_collection_modifyitems(config: pytest.Config, items) -> None:
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test — pass --run-slow to include it")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
