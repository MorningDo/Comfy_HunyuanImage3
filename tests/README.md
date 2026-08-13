# Comfy_HunyuanImage3 test suite

Smoke tests + targeted regression tests. The no-GPU tests
(`test_regressions_static.py`, `test_schema_consistency.py`,
`test_memory_budget.py`, `test_block_swap_config.py` — 194 tests) have
actually been run, in a from-scratch venv with no ComfyUI installed, and
pass cleanly; that run also found and fixed 3 real bugs along the way (see
CHANGELOG.md's "Fixed (found by running the test suite)" entry) — this is
a codebase where "should work" and "does work" have repeatedly diverged,
so treat the smoke tests (`test_smoke_*.py`, GPU + real models required)
as unverified until they've actually been run on real hardware. Please
report anything that fails there too, including a failure that looks like
it should obviously pass — that's more likely a wrong assumption in the
test than a real product bug. Each smoke test file's module docstring
explains what it's actually proving (and, where relevant, what it can't
prove — e.g. the block-swap race test).

## What's in here

| File | Needs GPU? | Needs real models? | What it checks |
|---|---|---|---|
| `test_regressions_static.py` | no | no | Structural/source checks that each of the ~15 bugs fixed in the 2026 audit pass is still fixed (right method called, right param threaded through, right exception behavior). Fast, run these first. |
| `test_schema_consistency.py` | no | no | Every registered node's `INPUT_TYPES()` keys are accepted by its `FUNCTION` method — the general form of issue #46. |
| `test_memory_budget.py` | no | no | `MemoryBudget`'s VRAM/block-swap math, with a mocked VRAM report. |
| `test_block_swap_config.py` | no | no | `BlockSwapConfig` validation. |
| `test_smoke_loaders.py` | **yes** | **yes** | Load each available quant variant (NF4/NF4-v2/INT8/INT8-v2, legacy + V2 + Instruct loaders), generate one tiny image, unload, confirm VRAM comes back. |
| `test_smoke_cross_cache_unload.py` | **yes** | **yes** | The key regression test for the fragmented-cache fix: load via one node family, unload via a *different* family, confirm the first model's cache is also cleared. |
| `test_smoke_block_swap.py` | **yes** | **yes** | Block swap with prefetching enabled produces finite, non-degenerate output — exercises the cross-CUDA-stream race fix (can't prove a timing race is gone, see the file's docstring). |
| `test_smoke_latent_and_highres.py` | **yes** (some) | **yes** (some) | Latent-control nodes (some run on CPU-only tensors) and the HighRes node. |

The no-GPU/no-model files run anywhere this package + torch are importable.
The GPU files need real weights and are individually skipped per-model-role
if that role isn't found — you don't need every variant present for the
suite to be useful, but per the ask, having v1 *and* v2 of NF4/INT8 present
gets you the most coverage (the shared_mlp/bnb-repair fix specifically
targets v2 pre-quantized checkpoints).

## Quick start

```bash
# From inside your ComfyUI's Python environment:
pip install -r requirements-test.txt   # or: pip install -e ".[test]"

# Fast, no-GPU tests first — always use tests/run_tests.sh, not `pytest`
# directly (see "Why run_tests.sh, not plain pytest" below for why):
tests/run_tests.sh tests/test_regressions_static.py tests/test_schema_consistency.py \
                   tests/test_memory_budget.py tests/test_block_swap_config.py -v

# Then the GPU suite, pointed at wherever your model folders live:
tests/run_tests.sh --hunyuan-models-dir /path/to/your/hunyuan/models -v
```

### Why `run_tests.sh`, not plain `pytest`

Two files in this package (`hunyuan_quantized_nodes.py`,
`hunyuan_full_bf16_nodes.py`) hard-import `folder_paths`/`comfy.utils` (real
ComfyUI modules) at the top of the file, and this repo's own `__init__.py`
imports both unconditionally. pytest's collection mechanism needs to
import `__init__.py` as its very first step whenever it collects anything
in this repo **including just loading conftest.py** — a step earlier than
any fix living in a conftest.py can reach. So `folder_paths` has to
already be importable before the `pytest` process even starts, which only
`$PYTHONPATH` (set by the shell, before Python starts) can guarantee.
`run_tests.sh` sets `PYTHONPATH` to a no-op stub (`tests/_stubs/` — just
enough of the API surface for imports to succeed, not a functional
replacement) and execs `pytest` with your arguments. A real ComfyUI
checkout via `--comfyui-dir`/`$COMFYUI_DIR` still correctly takes priority
over the stub if you provide one (needed for
`test_smoke_loaders.py::TestLegacyQuantizedLoaders`'s "does the real
dropdown/discovery machinery work" checks — everything else only needs
`folder_paths` to be *importable*, not functional).

If you'd rather not use the script: `PYTHONPATH=path/to/tests/_stubs pytest ...`
does the same thing (or point `PYTHONPATH` at a real ComfyUI checkout
directly instead of the stub).

`--hunyuan-models-dir` should point at the **parent** directory containing
your model subfolders, e.g. if you have:

```
/mnt/models/hunyuan/
├── HunyuanImage-3-NF4/
├── HunyuanImage-3-NF4-v2/
├── HunyuanImage-3-INT8/
├── HunyuanImage-3-INT8-v2/
├── HunyuanImage-3.0-Instruct-Distil-NF4-v2/
└── HunyuanImage-3.0-Instruct-Distil-INT8-v2/
```

then run with `--hunyuan-models-dir /mnt/models/hunyuan`.

You can also set `HUNYUAN_TEST_MODELS_DIR=/mnt/models/hunyuan` in your
environment instead of passing the flag every time — the flag wins if both
are set.

## If your folder names don't match auto-detection

The suite classifies subfolders by matching substrings in the (lowercased)
folder name — `"nf4"`, `"int8"`, `"instruct"`, `"distil"`, `"v2"` — against
the same convention the published HuggingFace repos use. If you renamed a
folder and auto-detection misses it, point at it directly with the
matching role env var instead (this works with or without
`--hunyuan-models-dir` set):

```bash
export HUNYUAN_TEST_MODEL_NF4=/mnt/models/my-nf4-folder
export HUNYUAN_TEST_MODEL_NF4_V2=/mnt/models/my-renamed-nf4-v2-folder
export HUNYUAN_TEST_MODEL_INT8=/mnt/models/int8
export HUNYUAN_TEST_MODEL_INT8_V2=/mnt/models/int8-v2
export HUNYUAN_TEST_MODEL_BF16=/mnt/models/full-bf16
export HUNYUAN_TEST_MODEL_INSTRUCT_DISTIL_NF4_V2=/mnt/models/instruct-distil-nf4
export HUNYUAN_TEST_MODEL_INSTRUCT_DISTIL_INT8_V2=/mnt/models/instruct-distil-int8
export HUNYUAN_TEST_MODEL_INSTRUCT_NF4_V2=/mnt/models/instruct-nf4
export HUNYUAN_TEST_MODEL_INSTRUCT_INT8_V2=/mnt/models/instruct-int8
export HUNYUAN_TEST_MODEL_INSTRUCT_BF16=/mnt/models/instruct-bf16
```

Run `tests/run_tests.sh --hunyuan-models-dir ... --collect-only -q` and
check the `[hunyuan_models] discovered: ...` line printed at the start of
the run to see what was actually found before any test executes.

## Using a real ComfyUI instead of the stub

`test_smoke_loaders.py::TestLegacyQuantizedLoaders` and the model-dropdown
machinery exercise real `folder_paths` behavior, not just importability —
point at a real ComfyUI checkout to get that coverage (auto-detected if
this repo sits at the normal install location,
`ComfyUI/custom_nodes/Comfy_HunyuanImage3/`):

```bash
tests/run_tests.sh --comfyui-dir /path/to/ComfyUI ...
# or: export COMFYUI_DIR=/path/to/ComfyUI
```

Without a real ComfyUI, everything still runs against the no-op stub —
nothing is skipped for this reason alone (unlike earlier drafts of this
suite, `folder_paths` being merely a stub doesn't block collection or
force skips; see "Why run_tests.sh" above).

## Useful flags

- `--run-slow` — also run `@pytest.mark.slow` tests (full BF16 loads,
  multi-image fusion). Off by default; the rest of the suite already uses
  4-12 diffusion steps and 1024x1024 (the model's smallest resolution
  preset — there is no smaller one) to stay fast.
- `-k <substring>` — run a subset, e.g. `-k nf4` or `-k cross_cache`.
- `-x` — stop on first failure (recommended for a first run — GPU tests
  are slow, and one broken loader path often cascades into unrelated
  failures via leftover cache state).
- `-v` — verbose test names, worth it given how long the GPU tests take.

**Do not** use `pytest-xdist` (`-n auto`)  — the GPU tests load large
models sequentially and assume exclusive VRAM; parallel workers will OOM
each other.

## VRAM sizing

Tests skip themselves if the GPU doesn't have enough VRAM for that
model (`require_vram` fixture): ~30GB for NF4, ~90GB for INT8, and the
BF16 full-precision test is marked `slow` (needs `--run-slow`) since it's
~160GB and always needs block swap. A 48GB card gets NF4 coverage; a
96GB+ card (e.g. RTX PRO 6000 Blackwell) gets NF4 + INT8 + (with
`--run-slow`) BF16.

## Cache hygiene between tests

Every test runs with an autouse fixture that calls
`hunyuan_shared.clear_all_hunyuan_caches()` before and after — this is
itself exercising the fix from the paranoid audit, and also keeps one
test's leftover VRAM from OOMing the next one. If a test crashes hard
enough to skip its own teardown, restart the pytest process before
re-running (or run `HunyuanImage3ForceUnload`/restart ComfyUI) rather than
trusting VRAM state across a broken run.

## Reporting results

For anything that fails, the most useful things to capture are:
1. The full pytest output (`-v`, and add `--tb=long` if a traceback got
   truncated).
2. Which model role/variant it was running against (the test name usually
   says, e.g. `test_int8_v2`).
3. GPU name and total VRAM (`nvidia-smi` or `python -c "import torch;
   print(torch.cuda.get_device_properties(0))"`), torch/transformers/
   bitsandbytes/accelerate versions (`pip freeze | grep -Ei
   "torch|transformers|bitsandbytes|accelerate"`).
