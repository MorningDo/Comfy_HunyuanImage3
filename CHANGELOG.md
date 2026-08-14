# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **`tests/` pytest suite**: fast, no-GPU regression/schema-consistency tests (run anywhere this package + torch are importable) plus GPU smoke tests covering the V2/legacy/Instruct loaders, generation, block swap, and — specifically — the cross-cache unload fix from the paranoid-audit pass. Point the GPU tests at your model directory with `--hunyuan-models-dir` or `$HUNYUAN_TEST_MODELS_DIR` (per-variant overrides available too). Run via `tests/run_tests.sh` (sets `$PYTHONPATH` so `folder_paths`/`comfy` imports resolve — see `tests/README.md`'s "Why run_tests.sh" section for why plain `pytest` doesn't work here). Test deps in `requirements-test.txt` / the `test` extra in `pyproject.toml`.
- **`install.sh` / `INSTALL.md`**: one-shot setup for a clean GPU machine — clones ComfyUI, creates a venv, installs a CUDA torch build plus this project's requirements pinned to a verified-working combination (`constraints-tested.txt`), symlinks this repo into `custom_nodes/`, and runs the fast test suite as a sanity check (optionally the full GPU suite too, via `$HUNYUAN_TEST_MODELS_DIR`). See `INSTALL.md` for what's pinned and why — `requirements.txt` itself stays floor-only.
- **`download_models.sh` / `scripts/download_models.py`**: downloads any HunyuanImage-3.0 model weights (base or Instruct, v1/v2, or `all`) from Hugging Face by short key (`./download_models.sh list` to see them), in its own small venv separate from the ComfyUI/inference one (just needs `huggingface_hub`, optionally `hf_transfer` for faster parallel downloads). Defaults to `/workspace/models/{hunyuan,hunyuan_instruct}/<name>` (override with `$HUNYUAN_MODELS_DIR`/`--models-dir`), matching the layout in the new `extra_model_paths.yaml.example` — for deployments that keep ComfyUI + custom_nodes on one volume (e.g. `/opt`) and multi-hundred-GB model weights on another (e.g. `/workspace`).
- **Opt-in FlashInfer support (`moe_impl=flashinfer` on `HunyuanInstructLoader`)**: new `validate_attention_moe_impl()` in `hunyuan_shared.py` raises a clear, actionable error before an unavailable `flash_attn`/`flashinfer` package reaches `from_pretrained()`, instead of letting whatever exception `transformers`/the remote model code happens to raise surface deep in a stack trace — wired into the single choke point in `HunyuanInstructLoader.load_model()` that all 8 `from_pretrained()` call sites share. Also warns (doesn't block) when `attention_impl=flash_attention_2` is selected on a compute-capability 12.0 GPU (consumer/workstation Blackwell — RTX PRO 6000, RTX 5090): mainline flash-attn has no confirmed SM120 support as of this writing (see `INSTALL.md`). `install.sh`'s new `INSTALL_FLASHINFER=1` opt-in installs `flashinfer-python` (pinned to `0.5.0` in the new `requirements-flash.txt` / `constraints-tested.txt`, matching Tencent's own tested baseline) plus, since FlashInfer JIT-compiles kernels at first use against the system CUDA Toolkit (not just the CUDA runtime bundled in the pip torch wheel), an opt-in apt-based `cuda-toolkit-13-0` install — warn-and-continue, never fatal to the rest of the install, if root/apt aren't available. `flash-attn` itself is deliberately **not** installed by `install.sh` at all: no prebuilt PyPI wheel (multi-hour source build) and no confirmed upstream SM120 support regardless of CUDA version — see `INSTALL.md`'s "FlashInfer / FlashAttention setup" section for the full risk writeup and the unofficial community-fork path if you want to try it anyway.
- **Host GPU driver CUDA-ceiling preflight check**: `install.sh` now compares `nvidia-smi`'s reported driver CUDA ceiling against `TORCH_CUDA_INDEX`'s target before installing anything, and warns loudly if the host driver can't actually support it — this is a container/host distinction that matters specifically for cloud GPU rentals (e.g. RunPod): the container's CUDA toolkit is whatever `install.sh` installs, but the underlying host driver is fixed outside this script's control (on RunPod, set by which host CUDA version you selected at pod-creation time) and `nvidia-smi` reports that ceiling regardless of what's currently installed in the container, so the check is meaningful even pre-uplift.
- **`start_comfyui.sh` / `run_gpu_tests.sh`**: `install.sh` now (re)generates both directly in `$COMFYUI_DIR` on every run, baking in the resolved venv/repo paths. `start_comfyui.sh` launches ComfyUI (`--listen 0.0.0.0 --port $COMFY_PORT`, new env vars, default 8000; CORS left off — ComfyUI's own default — via `COMFY_EXTRA_ARGS` if you want it back on) and puts the opt-in CUDA Toolkit on `PATH`/`LD_LIBRARY_PATH` first if `INSTALL_FLASHINFER=1` installed one (needed for FlashInfer's JIT compile — doesn't otherwise persist outside `install.sh`'s own run). `run_gpu_tests.sh` runs the full GPU suite against that same ComfyUI install (`--comfyui-dir` is required for `folder_paths`/`comfy.*` to resolve for real, which is what the hunyuan/hunyuan_instruct model-folder scan actually needs to find your models).

### Changed
- **`install.sh`'s default `TORCH_CUDA_INDEX` moved from cu128 to cu130**: flashinfer's officially documented CUDA support list is 12.6/12.8/13.0/13.1 (no 12.9), so cu130 is the better-aligned "current" choice; also matches where PyTorch's own wheel index has moved. The pip torch install step is now skipped entirely if a CUDA-enabled torch already matching `TORCH_VERSION` is importable (the venv is now created with `--system-site-packages` so it can see one inherited from the base image) — avoids throwing away an already-integration-tested torch/CUDA pairing (e.g. from a `runpod/pytorch:*` base image) for no reason; any version/index override that doesn't match falls through to the explicit install as before.
- **`install.sh`'s default `COMFYUI_DIR` moved from `$HOME/ComfyUI` to `/opt/ComfyUI`**: matches this project's documented deployment convention (ComfyUI + custom_nodes under `/opt`, model weights under `/workspace` — `extra_model_paths.yaml.example` already assumed `/opt/ComfyUI` specifically). Falls back to `$HOME/ComfyUI` if `/opt` isn't writable (e.g. running as a non-root user without sudo), so this doesn't break non-root/local use.

### Fixed
- **`install.sh`'s `log()`/`warn()`/`die()` helpers silently dropped every argument after the first**: several call sites (both pre-existing and new) pass a message as multiple backslash-continued string arguments (one sentence fragment per line) — the helpers only ever referenced `$1`, so everything after the first fragment was silently discarded from the printed output. Changed to `"$*"` so the fragments are joined with a space into the intended single message.
- **`requirements.txt` was missing `diffusers` entirely — every real model load crashed with `ModuleNotFoundError: No module named 'diffusers'`**: found on the first real-hardware GPU run (RTX PRO 6000 Blackwell, torch 2.13.0+cu130 — the CUDA 13.0 uplift itself worked correctly). Both the Instruct loader and the legacy standalone NF4 loader failed identically, since it's the *remote* model code (`trust_remote_code=True`, fetched from HF at load time — `autoencoder_kl_3d.py`) that subclasses `diffusers.configuration_utils.ConfigMixin` for its VAE, not this repo's own node code — so nothing in the fast no-GPU suite (which never actually loads the remote code) could have caught it. Added `diffusers>=0.35.2` (`==0.35.2` in `constraints-tested.txt`, matching Tencent's own reference `requirements.txt` exactly) plus `einops`, `loguru`, `tiktoken` — also listed as core (not demo-only) dependencies in Tencent's reference file, added preemptively since they're plausible next `ModuleNotFoundError`s from the same remote-code import chain, rather than making real-hardware GPU time discover them one at a time.
- **`install.sh` reinstalling `torch` alone left `torchvision`/`torchaudio` stranded at a mismatched version**: found by actually running this on a RunPod `runpod/pytorch:*` base image, where torch/torchvision/torchaudio/triton are all preinstalled at the system level and inherited into the venv via `--system-site-packages`. When the base image's torch didn't match `TORCH_VERSION` and the explicit reinstall kicked in, it only reinstalled `torch` itself — `torchvision 0.23.0+cu128`/`torchaudio 2.8.0+cu128` (built against the OLD torch) were left in place, since pip won't touch a package it finds satisfied outside the venv, and nothing else in the script ever revisits them. Result: `pip`'s dependency-conflict warning (`torchvision ... requires torch==2.8.0, but you have torch 2.13.0`) and a real risk of ABI-mismatched compiled extensions at runtime. Fixed by installing `torch`+`torchvision`+`torchaudio` together (unpinned versions for the latter two, so pip's resolver picks whatever pairs correctly with `TORCH_VERSION` from the same CUDA index) in both the primary and fallback install paths.
- **`tests/run_tests.sh`'s "did the caller already pass a collection path" heuristic could misfire on `-k`/`-m`/etc.**: found while testing the new `run_gpu_tests.sh` with `-k <pattern>` — the heuristic only knew `--comfyui-dir`/`--hunyuan-models-dir` consume a following value; any other value-taking pytest flag's value (e.g. `-k some_pattern`) fell through to the wildcard case and got misread as an explicit collection path, silently skipping the default `tests/` append and reintroducing the exact "unrecognized arguments" two-pass-parsing bug this logic exists to avoid (see the entry above from the `install.sh` end-to-end testing pass). Fixed by (1) recognizing a few more common value-taking flags (`-k`, `-m`, `-n`, `--timeout`, `--maxfail`) and (2) additionally requiring a bare token to actually look like a path/node-id (contains `/`, ends in `.py`, contains `::`, or is `.`) before counting it as one — the second guard covers any other value-taking flag not explicitly listed.

### Fixed (found by actually running `install.sh` end-to-end against a real ComfyUI checkout)
- **`tests/run_tests.sh` could silently test against the wrong `comfy`/`folder_paths`**: passing `--comfyui-dir` used to still resolve `folder_paths` against the no-op test stub instead of the real ComfyUI checkout, because the stub was already on `$PYTHONPATH` (for pytest's own package-import step, needed before conftest.py can even register `--comfyui-dir`) before Python ever saw the flag — and once `comfy` is cached in `sys.modules`, later sys.path changes don't reconsider it. `run_tests.sh` now decides real-vs-stub in bash, before Python starts, so only one of them ever lands on `$PYTHONPATH`.
- **`tests/run_tests.sh --comfyui-dir ... --hunyuan-models-dir ...` (no explicit test path) picked up ComfyUI's own `pytest.ini`**: with no positional path argument, pytest's first argument-parsing pass (before conftest.py has registered these custom options) could misread the directory following `--comfyui-dir` as an implicit collection target, so it discovered ComfyUI's shipped config instead of this project's. Fixed by always passing `-c pyproject.toml --rootdir=...` explicitly and defaulting to `tests/` as the collection path when the caller didn't specify one.

### Fixed (found by actually running the new test suite)
- **`patch_hunyuan_static_cache_device`'s fallback class lookup could match `torch.ops` instead of the real `HunyuanStaticCache`, then crash**: when the primary lookup (`type(model).__module__`) fails, the fallback scans `sys.modules` for anything with a `HunyuanStaticCache` attribute. `torch.ops` has a dynamic `__getattr__` that returns a `torch._ops._OpNamespace` object for *any* attribute name — `hasattr(torch.ops, "HunyuanStaticCache")` is always `True` — and `torch.ops` is always in `sys.modules` before the real remote-code module. The scan matched it, then crashed with an uncaught `AttributeError` on `cache_cls.update` instead of falling through to "patch skipped". Fixed by requiring `inspect.isclass(candidate)` before accepting a match.
- **...and separately, that same fallback scan could abort silently on the very first module it checked**: `getattr(mod, "HunyuanStaticCache", None)` only suppresses `AttributeError` — several transformers lazy-loaded submodules (`transformers.models.*.image_processing_*_fast`) raise `ModuleNotFoundError` (missing optional deps like `torchvision`) on *any* attribute access. The `try/except` wrapped the whole scan loop instead of each iteration, so the first such module (alphabetically early, e.g. `aria`) silently ended the entire scan before it could ever reach the real target — meaning this fallback path was effectively dead whenever the primary lookup failed and transformers had any torchvision-optional fast image processors registered (i.e. most current transformers installations). Fixed by moving the `try/except` inside the loop.
- **Invalid escape sequence in a `repair_unquantized_bnb_modules()` docstring** (`\.` in a non-raw string): harmless today (`DeprecationWarning`), but Python has scheduled this to become a hard `SyntaxError` in a future version. Escaped it properly.

### Fixed
- **Issue #46 — `moe_drop_tokens`/`vae_dtype` crash on `HunyuanGenerateWithLatent`**: the node's `generate()` override predated the `moe_drop_tokens`/`vae_dtype` options added to `HunyuanUnifiedV2` in 1.4.0, so ComfyUI's widget values (inherited via `INPUT_TYPES()`) were passed to a signature that didn't accept them, crashing with `TypeError: unexpected keyword argument 'moe_drop_tokens'`. Both parameters are now threaded through to the parent-delegation and image/latent-injection code paths.
- **Issues #36, #41 — shared_mlp / mlp.gate wrongly quantized (CB/SCB errors)**: current transformers' `should_convert_module()` only skips a module when a `llm_int8_skip_modules` entry is a *prefix* or a *suffix* of its full dotted path — plain substring matching was dropped. `shared_mlp` and `mlp.gate` are interior path segments here (real leaves are `...mlp.shared_mlp.gate_and_up_proj` and `...mlp.gate.wg`), so the checkpoint's skip-list entries for them silently never match, and those layers get wrapped in `Linear8bitLt`/`Linear4bit` despite being saved full-precision on disk — leaving quant state (`weight.CB`/`SCB`/`quant_state`) uninitialized and crashing on first forward with `AttributeError: 'Parameter' object has no attribute 'CB'`. New `repair_unquantized_bnb_modules()` in `hunyuan_shared.py` detects these never-actually-quantized modules after load and rebuilds them as plain `nn.Linear` (no data loss — the underlying tensor was always full precision). Wired into the `CleanModelLoader.load()` and Instruct loader entry points for `nf4`/`int8`, and specifically **before** `apply_nf4_transformers_compat()` for NF4 — that shim calls `module.cuda()` on any uninitialized `Linear4bit`, and bitsandbytes' `Params4bit.to()` unconditionally real-quantizes whatever data it finds, so running it first would have silently NF4-quantized `shared_mlp` instead of restoring it to full precision.
- **Issue #39 — KV-cache device mismatch on newer torch/CUDA with block swap**: `HunyuanStaticCache.update()`'s device-harmonization patch moved the cache tensors (`layer.keys`/`layer.values`) but not `cache_position`, the index tensor passed to `index_copy_`. `cache_position` is built once per generation call and reused across every layer; with block swap a layer's cache and freshly-computed key/value states may currently be on CPU while `cache_position` is still on GPU, crashing with `RuntimeError: Expected all tensors to be on the same device`. The patch now also migrates `cache_position` to match `key_states.device` before delegating to the original `update()`.
- **Issue #40 (residual) — `apply_nf4_transformers_compat` crash on the image-processor compat shim**: the patch stored its "already patched" flag on the *bound method* object (`bound._hunyuan_t5_compat_patched = True`), but `types.MethodType` instances have no `__dict__` and reject attribute assignment on every Python version (verified on 3.11/3.12) — so this raised `AttributeError` instead of completing the patch. The flag now lives on `image_processor` itself.
- **`CleanModelLoader.SKIP_MODULES` missing `shared_mlp`/`mlp.gate`**: the on-the-fly NF4/INT8 quantization path (`_load_nf4`, `_load_int8` fallback) used a narrower skip list than the legacy standalone loaders in `hunyuan_quantized_nodes.py`, so it was genuinely — not just accidentally — quantizing the shared expert and MoE router even before the `should_convert_module()` issue above. Brought in line with the reference list.
- **`repair_unquantized_bnb_modules()` skip-fragment matching hardcoded to 2 entries**: now reads the loaded model's own `config.quantization_config` skip list (`llm_int8_skip_modules`/`bnb_4bit_skip_modules`/`modules_to_not_convert`) when available, falling back to the hardcoded pair only if that's absent — so a future checkpoint revision that adds a new interior-path-segment skip entry doesn't need a matching code change here.
- Removed the now-redundant post-load `repair_unquantized_bnb_modules()` call for `nf4` in `CleanModelLoader.load()` and the Instruct loader — both NF4 branches already run it earlier (before `apply_nf4_transformers_compat()`, where ordering matters); the late call is kept for `int8`, which has no equivalent early call site.
- **Fragmented model caches never fully unloaded**: three independent, mutually-unaware model-cache singletons exist (`ModelCacheV2` for HunyuanUnifiedV2/HunyuanGenerateWithLatent, `InstructModelCache` for the Instruct family, `HunyuanModelCache` for the legacy standalone loaders) — this is very likely the primary cause of the "RAM accumulation" documented above and in README.md. New `clear_all_hunyuan_caches()` in `hunyuan_shared.py` clears all three (re-entrancy guarded); wired into `HunyuanUnloadV2`, `HunyuanEmergencyCleanup`, `HunyuanImage3Unload`, `HunyuanImage3ForceUnload`, and `HunyuanInstructUnload` so every "unload" node now actually frees all cached Hunyuan models, not just its own family's. The nuclear orphaned-tensor hunt in `HunyuanImage3ForceUnload` was also scoped to zero modules for anyone using only V2 nodes (never included `ModelCacheV2` in its module-ID allowlist) — fixed the same way.
- **`clear_generation_cache()` didn't clear the model's actual KV cache**: it nulled several attributes that aren't where this model's KV cache lives; the real location (`model._pipeline.model_kwargs['past_key_values']`) was previously only cleared inline in two of several callers. The 5 Instruct node call sites and the pre-VAE-decode OOM-prevention hook (applied to every loader) relied solely on the previously-ineffective version. Fixed at the source so every caller benefits.
- **Silent failure returned a fake-success black image** in `HunyuanInstructGenerate`/`ImageEdit`/`MultiFusion`: OOM/exception handlers caught the error and returned `(empty_image, "", "Error: ...")` as a normal-looking output tuple — ComfyUI shows no error indicator on a successful-looking return, so a failed generation could be silently processed downstream as if it succeeded. Now raises (matching the existing, correct pattern already used by `HunyuanImage3GenerateHighRes`), surfacing the error in the UI and stopping the queue.
- **Missing `IS_CHANGED` on `HunyuanInstructGenerate`/`ImageEdit`/`MultiFusion`**: `seed=-1` (random) is a documented convention across this codebase, but only `HunyuanUnifiedV2` had the `IS_CHANGED` override needed to stop ComfyUI's cache from keying off the literal `-1` and replaying the same cached image on re-queue. Added to all three.
- **`post_action` (e.g. `full_unload`) silently skipped on any generation failure** in `HunyuanUnifiedV2`, `HunyuanGenerateWithLatent`, and `HunyuanImage3GenerateHighRes`: the post-action call only ran on the success path, so the exact moment a user most needs `full_unload` — right after an OOM — was precisely when it never fired, compounding VRAM pressure on repeated failed attempts. Now runs in the exception handlers too.
- **`HunyuanModelCache.full_unload()` doesn't exist**: `hunyuan_moe_test_node.py`'s `full_unload` post-action called a method that was never defined (only `.clear()` is), raising `AttributeError` instead of freeing VRAM. Fixed to call `.clear()`.
- **`HunyuanEmergencyCleanup`'s hook-cleanup step was dead code**: it walked `sys.modules` (Python module objects) checking for `_forward_hooks`, an attribute that only exists on `nn.Module` *instances* — the loop could never match anything. Rewrote to walk `gc.get_objects()` for actual `nn.Module` instances, scoped to known Hunyuan model IDs (never touches other extensions' models running in the same ComfyUI process).
- **`CUDA_VISIBLE_DEVICES` permanently leaked into the process on the normal multi-GPU path**: `HunyuanImage3DualGPULoader` cleared the env var to detect all GPUs, but the restore was nested inside the `num_gpus < 2` fallback branch — so the normal 2+ GPU path never restored it, silently defeating the user's GPU isolation for the rest of the ComfyUI process (and any subprocess it spawns). Moved the restore into a `finally`.
- **`HunyuanAPIConfig.load_config()` doesn't exist**: `hunyuan_highres_nodes.py`'s prompt-rewrite feature called a class/method that was never defined (`hunyuan_api_config.py` only exports a `get_api_config()` function); caught by a broad `except Exception`, so prompt rewriting in the HighRes node has never worked. Fixed the call.
- **Pre-quantized INT8 loading allowed CPU offload, silently corrupting the model**: both `hunyuan_loader_clean.py::_load_int8`'s fallback path and `hunyuan_quantized_nodes.py::HunyuanImage3Int8Loader` set a `"cpu"` entry in `max_memory`, but CPU-mapped layers get materialized as plain `nn.Linear` (not `Linear8bitLt`) while the on-disk data is packed int8 bytes with SCB scales — this doesn't fit, producing a corrupted model rather than a clean error. Removed the CPU entry so accelerate raises a clear "doesn't fit" error instead when the ~80GB model exceeds available GPU memory.

### Fixed (correctness, found via paranoid audit — no known issue #)
- **Cross-CUDA-stream race in block swap**: the async GPU→CPU block "release" (issued non-blocking on the current/default stream) and the async CPU→GPU "prefetch" of that same block (issued non-blocking on a separate dedicated prefetch stream) wrote to and read from the same pinned buffer with no synchronization between the two streams anywhere in the file. On a fast-enough GPU relative to PCIe transfer time, a prefetch could start reading a block's buffer before the prior release's write had actually landed — silently corrupting that block's weights with no exception raised. Fixed with a standard CUDA producer/consumer pattern: the release records a `torch.cuda.Event` after issuing its writes; the prefetch, before reading the same block's buffer, calls `stream.wait_event()` on it — an async, host-non-blocking cross-stream barrier. New `_release_events` dict is drained alongside the existing `_prefetch_events` in all teardown paths.

## [1.4.1] - 2026-04-25

### Fixed
- **E702 lint errors in `hunyuan_latent_nodes.py`**: split semicolon-chained statements onto separate lines so the file passes the Comfy registry's upcoming security/style check (multi-statement lines will soon be a hard error).

## [1.4.0] - 2026-04-25

### Added
- **`moe_drop_tokens` option** (V2 unified loader, Instruct loader): exposes the MoE token-drop toggle to the user. Default `True` matches previous behaviour. Set `False` to disable expert capacity dropping for higher fidelity at 2K+ resolutions (small speed and VRAM cost). See `Docs/QUALITY_NOTES.md`.
- **`vae_dtype` option** (V2 unified loader, Instruct loader): choose `bfloat16` (default) or `float32` for the VAE module. float32 reduces banding in dark gradients and color shifts in skin tones at ~600 MB extra VRAM. See `Docs/QUALITY_NOTES.md`.
- **`Docs/QUALITY_NOTES.md`**: developer reference covering negative-prompt findings (HunyuanImage-3 does not support them — uses a trained `<cfg>` placeholder token, not empty-string conditioning), `moe_drop_tokens` and `vae_dtype` recommendations, `flow_shift` recipes per subject type, step-count guidance, and `bot_task` recaption performance notes.
- **NF4 block-swap support**: New `_load_nf4_block_swap` path mirrors the proven INT8 block-swap loader (load to CPU → move non-block components to GPU → manage 32 transformer blocks via `BlockSwapManager`). Enables NF4 generation on 24–32 GB cards (issue #32, supersedes PR #33).
- **MoE single-token fast path**: `_efficient_moe_forward` now short-circuits for `bsz=1, seq_len=1` and only runs the `topk` selected experts instead of looping all 64 (Tencent PR #93–style optimization, ~10–20× faster autoregressive decode).
- **Explicit VAE controls on `HunyuanInstructGenerate`**: `vae_tiling` (auto/on/off) and `vae_offload` (auto/on/off) for predictable behaviour at high resolution (issue #22).
- **Resolution selector on `HunyuanInstructImageEdit`**: matches the generate node; honours `align_output_size` only when `resolution=auto` (PR #33 cosmetic improvement).

### Fixed
- **NF4 + transformers ≥5.0 compat (issues #24, #27)**: `apply_nf4_transformers_compat` walks `Linear4bit` modules after load and materializes uninitialized `quant_state` tensors so `fix_4bit_weight_quant_state_from_module` no longer fails its `weight.shape[1] == 1` assertion. No transformers pin required.
- **NF4 image processor compat (issue #34)**: Patches `image_processor.vit_process_image` to coerce `pixel_values` list → stacked tensor before `.squeeze(0)` on transformers ≥5.0.
- **NF4 block movement crash**: `Params4bit.to(device, non_blocking=True)` was raising `cudaErrorInvalidValue` on CUDA→CPU transfers. New `_move_nf4_block_params` moves each parameter synchronously and explicitly walks `quant_state` tensors. Async prefetch is disabled for NF4 (the main-stream sync path is now used everywhere).
- **VAE OOM after long generation (issue #22)**: Pre-VAE cleanup now drains pending block-swap events and releases all swapped blocks to CPU **before** measuring free VRAM, so `auto` tiling/offload decisions reflect actual headroom. New `BlockSwapManager.release_all_blocks()` helper.
- **Newer Instruct/Distil v2 generate path**: `patch_hunyuan_generate_image` now falls back to `model.generate` when `_generate` is absent.

### Changed
- `transformers >= 4.47` remains the only floor in `requirements.txt` — no upper pin. Version-specific shims live in `apply_nf4_transformers_compat` and `patch_static_cache_lazy_init`.
- **Tooltip improvements** across V2 unified node and Instruct generate / image-edit / fuse nodes:
  - `num_inference_steps`: notes that 60–80 steps reduce flow-matching artifacts at 2K+ but generation time scales linearly.
  - `flow_shift`: now lists subject-type presets (portraits 2.0–2.5, landscapes 3.5–5.0).
  - `bot_task` (Instruct): warns that `recaption` and `think_recaption` are very slow (30–120 s of LLM time before diffusion starts).

## [1.2.0] - 2026-02-11

### Added
- **33 Resolution Presets**: Instruct resolution dropdown now includes all model-native bucket resolutions (~1MP each), ordered tallest portrait (512×2048) → square (1024×1024) → widest landscape (2048×512).
- **Multi-Image Fusion 5-input support**: Added `image_4` and `image_5` optional inputs (experimental — model officially supports up to 3, pipeline accepts more).

### Fixed
- **Issue #16 — NF4 Low VRAM OOM**: Two-stage `max_memory` estimation in quantized loader replaces one-shot approach that left no headroom for inference tensors.
- **Issue #15 — Multi-GPU device mismatch**: Explicit `.to(device)` on `freqs_cis` / `image_pos_id` prevents cross-device errors during block-swap forward pass.
- **Issue #12 — Transformers 5.x compatibility**: `_lookup` dict guard in block swap, `BitsAndBytesConfig` import path, and `modeling_utils` attribute checks updated for forward compatibility.
- **Instruct Image Edit / Multi-Fusion**: Added missing `torch.cuda.OutOfMemoryError` handlers with actionable error messages.
- **Instruct Multi-Fusion**: Applied multi-GPU block-swap device patch (was missing from instruct nodes).

### Changed
- Instruct Multi-Fusion `fuse()` method refactored: image path conversion uses a loop instead of separate if-blocks for each image.
- Resolution tooltips updated across all Instruct generate nodes.
- Multi-Fusion workflow diagram updated for 3+ images with `think_recaption` recommendation.

### Removed
- Dead `gc` import from `hunyuan_highres_nodes.py`.

### Code Quality
- `hunyuan_cache_v2.py`: Added `clear_generation_cache()` helper used by all generate nodes for KV cache cleanup.
- `hunyuan_shared.py`: Centralized `_aggressive_vram_cleanup()` with stale KV-cache detection.
- `hunyuan_block_swap.py`: `_lookup` guard for INT8 `Module._apply` hook (transformers 5.x).
- `hunyuan_quantized_nodes.py`: Two-stage `max_memory` with headroom for inference VRAM.
- `hunyuan_loader_clean.py`: Multi-GPU device-mismatch fix for `freqs_cis` / `image_pos_id`.

## [1.1.0] - 2026-02-09

### Added
- **Instruct Model Nodes**: 5 new nodes for HunyuanImage-3.0-Instruct and Instruct-Distil models
  - **Hunyuan Instruct Loader**: Load any Instruct variant (BF16/INT8/NF4, Distil/Full). Auto-detects quant type from folder name.
  - **Hunyuan Instruct Generate**: Text-to-image with bot_task modes (image/recaption/think_recaption). Returns CoT reasoning text.
  - **Hunyuan Instruct Image Edit**: Edit images with natural language instructions.
  - **Hunyuan Instruct Multi-Image Fusion**: Combine 2–3 reference images with instructions.
  - **Hunyuan Instruct Unload**: Free cached Instruct model from VRAM/RAM.
- **Block Swap**: Async GPU↔CPU transformer block swapping for all loaders. Enables running BF16 (~160GB) and INT8 (~81GB) models on 48–96GB GPUs.
- **HighRes Efficient Node**: Loop-based MoE expert routing uses ~75× less VRAM than dispatch_mask. Generates 3MP–4K+ images on 96GB GPUs.
- **Unified V2 Node**: Single auto-detecting generate node with integrated block swap, VAE management, and VRAM budget.
- **Flexible Model Paths**: All loaders now use ComfyUI's `folder_paths` system. Models can be stored anywhere via `extra_model_paths.yaml` (`hunyuan` and `hunyuan_instruct` categories).
- **Pre-quantized Instruct models** on Hugging Face: INT8 and NF4 variants for both Instruct and Instruct-Distil.
- **INT8 bitsandbytes fix**: Guard hooks that fix `Module._apply` discarding `Int8Params.CB/SCB` during `.to()` calls. Enables block swap with INT8 models.
- **Soft Unload node**: Move model to CPU (keep cached) for fast restore without full reload.
- **Force Unload node**: Complete VRAM + RAM cleanup with aggressive garbage collection.
- **Clear Downstream node**: Clear other models from VRAM while preserving cached Hunyuan model.

### Changed
- Instruct Loader model discovery uses `folder_paths.get_folder_paths()` instead of hardcoded paths
- All base loaders (NF4, INT8, BF16, Multi-GPU, HighRes) migrated to centralized `get_available_hunyuan_models()` and `resolve_hunyuan_model_path()` in `hunyuan_shared.py`
- Updated README with comprehensive Instruct documentation, HuggingFace links, hardware tables, and workflow diagrams

### Known Issues
- **Instruct (full) INT8 with block swap**: OOM during inference. Distil-INT8 works fine. Under investigation.
- **RAM accumulation**: Successive model loads may leak RAM. Restart ComfyUI if needed.

## [Unreleased]

### Added
- **Rewritten Prompt Output**: Both `HunyuanImage3Generate` and `HunyuanImage3GenerateLarge` now output the rewritten prompt used for generation
  - Useful for saving to EXIF metadata
  - Can be reused for regeneration or variations
  - Contains the LLM-enhanced prompt when prompt rewriting is enabled
- **Status Output**: Both generation nodes now provide a status message indicating:
  - Whether prompt rewriting was used and which style
  - If prompt rewriting failed with error message
  - Large image mode settings (CPU offload status)

### Changed
- Generation nodes now return 3 outputs: `(image, rewritten_prompt, status)` instead of just `(image,)`
- Status messages provide better feedback about generation settings

### Fixed
- **Low VRAM NF4 Loader**: Resolved validation errors on 24GB/32GB cards by implementing a custom device map strategy that forces NF4 layers to GPU while allowing other components to offload to CPU.
- **Device Mapping**: Added logic to prevent `bitsandbytes` from seeing 4-bit layers on CPU, which was causing crashes in Low VRAM mode.

### Technical Details
- `rewritten_prompt`: STRING - The final prompt used for generation (either original or LLM-rewritten)
- `status`: STRING - Human-readable status message about the generation process

## [1.0.0] - 2024-11-18

### Initial Release
- Full BF16 and NF4 quantized model loading
- Multi-GPU support with smart memory management
- Official HunyuanImage-3.0 prompt enhancement with LLM APIs
- Large image generation with CPU offload
- Professional resolution presets with megapixel indicators

## [Low VRAM Fix] - 2024-11-19

### Fixed Low VRAM NF4 Loader
- Resolved validation errors on 24GB/32GB cards by implementing a custom device map strategy that forces NF4 layers to GPU while allowing other components to offload to CPU.

### Enhanced Device Mapping
- Added logic to prevent `bitsandbytes` from seeing 4-bit layers on CPU, which was causing crashes in Low VRAM mode.
