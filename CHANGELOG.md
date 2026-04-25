# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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
