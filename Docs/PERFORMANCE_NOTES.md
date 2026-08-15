 # Hunyuan Image 3.0 Performance Notes

This document captures performance findings, optimization attempts, and library limitations discovered during development of this ComfyUI node set.

## Hardware Context

Testing performed on:
- **GPU**: NVIDIA RTX 6000 Pro Blackwell (96GB VRAM)
- **System RAM**: 387GB DDR5
- **Storage**: NVMe SSD
- **Model**: Hunyuan Image 3.0 (13B parameters)

## Model Size Reference

| Format | Model Size | VRAM Required |
|--------|-----------|---------------|
| BF16 (full precision) | ~160GB | 160GB+ (doesn't fit in 96GB) |
| INT8 (bitsandbytes) | ~81GB | ~81GB |
| NF4 (bitsandbytes) | ~40GB | ~40GB |

## Performance Benchmarks

### Inference Times (1280x720, 38 steps)

| Configuration | Load Time | Inference Time | Total Cycle |
|--------------|-----------|----------------|-------------|
| INT8, no offload | ~1:40 | ~3:00 | ~4:40 |
| BF16 + smart offload | ~0:38 | ~11:00+ | ~12:00 |
| BF16, no offload | N/A | N/A | Requires 160GB+ VRAM |

### VRAM Requirements for Inference

Empirical formula: `VRAM_inference ≈ 12 × (megapixels)^1.4`

| Resolution | Megapixels | Inference VRAM |
|-----------|------------|----------------|
| 1024×1024 | 1.0 MP | ~12 GB |
| 1280×720 | 0.92 MP | ~11 GB |
| 1920×1080 | 2.07 MP | ~22 GB |
| 2560×1440 | 3.69 MP | ~45 GB |

## Key Findings

### 1. INT8 Pre-Quantized Checkpoint Limitation (CPU Offloading)

**Pre-quantized INT8 checkpoints CANNOT use CPU offloading via `device_map`.**

This is a fundamental limitation of the bitsandbytes/transformers INT8 loading path:

1. When transformers sees a CPU entry in `device_map`, it adds that module to `modules_to_not_convert`
2. The module stays as regular `nn.Linear` (not converted to `Linear8bitLt`)
3. Loading pre-quantized int8 weights into `nn.Linear` via `load_state_dict(assign=True)` fails because `nn.Parameter(int8_tensor)` tries to set `requires_grad=True`, but **integer tensors cannot require gradients**

**This ONLY affects pre-quantized INT8 checkpoints** (weights stored as int8 on disk with SCB column-wise absmax scales). Models that **quantize at load time** (float16/bf16 checkpoint + `load_in_8bit=True`) do NOT hit this issue because CPU-mapped layers stay in fp16/fp32 and only GPU-mapped layers get quantized to int8 on-the-fly.

**Workarounds:**
- Load entirely to GPU (`device_map="cuda:0"`) — works if GPU has enough VRAM
- Use `blocks_to_swap` parameter to load to CPU first, then manually move components to GPU (bypasses the device_map entirely)
- Use NF4 quantization instead (smaller, fits easily on single GPU)

### 2. INT8 is Optimal for 96GB GPUs

With 96GB VRAM, INT8 quantization provides the best balance:
- Model fits entirely in VRAM (81GB)
- No PCIe bandwidth bottleneck during inference
- Full GPU utilization
- 3-minute inference vs 11+ minutes with BF16 offloading

### 3. BF16 + Smart Offload is PCIe-Bound

When using `accelerate`'s device_map offloading with BF16:
- ~64GB of weights must shuttle between CPU and GPU
- PCIe 4.0 x16 = ~32 GB/s theoretical, lower in practice
- GPU utilization drops significantly (waiting for data)
- Results in 3.8x slower inference than INT8

### 4. Quality Differences

Trained eyes can notice subtle differences between INT8 and BF16:
- BF16 preserves full model precision
- INT8 has minor quantization artifacts
- For critical/production work, BF16 may be preferred despite time cost

### 5. Steps and Flow Shift Tuning (Feb 2026)

**Steps:** 40 steps produces quality very close to 50 steps across all model types
(Instruct, Instruct-Distil, and base text-to-image). Default changed from 50 to 40.
Distil models still default to 8 steps.

**Flow shift:** Text-to-image generation produces slightly better fine detail at
slightly lower flow_shift values than the model default of 3.0. Default changed to 2.8.
The relationship is roughly linear — if the default was 2.5, then 2.15 would be better.
Lower values = more fine detail, higher values = smoother/simpler output.

## Optimization Attempts

### Soft Unload (Move Model to CPU RAM)

**Goal**: Keep model in CPU RAM instead of deleting, enabling fast reload (~20-30s) vs disk reload (~1:40).

**Status (as of Nov 2025)**: ❌ Not possible with current libraries.

**Update (Dec 2025)**: implemented after all — see "Enable Soft Unload for
INT8/NF4 quantized models" (`63c17a8`). Soft Unload is now available and used
throughout the current loaders' `post_action` handling; the limitations
described below were worked around rather than being fundamental.

#### INT8/NF4 Models (bitsandbytes)
- bitsandbytes does not support `.to()` on quantized models
- Error: `".to" is not supported for "8-bit" bitsandbytes models`
- Quantization state is device-bound
- Would require dequantize → move → requantize path

#### BF16 with Offloading (accelerate)
- accelerate's `device_map` creates meta tensors
- Meta tensors are placeholders with no actual data
- Error: `Cannot copy out of meta tensor; no data!`
- Data lives on disk, loaded on-demand

### Smart Offload Override for INT8

**Goal**: Prevent OOM when generating high-resolution images with INT8.

**Status**: ✅ Implemented

When INT8 model + disabled offload + resolution > 1.2MP:
- Automatically switches to "smart" offload mode
- Prevents inference-time OOM
- Logs warning to user

### Force Unload (Nuclear Option)

**Goal**: Aggressively clear VRAM when normal unload fails.

**Status**: ✅ Implemented

- Multiple GC passes
- Clears all model references
- Resets CUDA memory allocator
- Handles post-OOM memory leaks
- **NEW**: "Nuke Orphaned Tensors" option for stuck memory after OOM

### OOM Memory Leak Issue

**Problem**: When an OOM error occurs during inference:
1. ComfyUI's OOM handler clears Python references to models
2. Our cache reference (`HunyuanModelCache._cached_model`) becomes None
3. BUT the GPU memory is not actually freed - tensors are orphaned
4. Force Unload sees "no model cached" but 70GB+ is stuck in VRAM

**Solution**: The "nuke_orphaned_tensors" option in Force Unload:
1. Scans ALL Python objects via `gc.get_objects()`
2. Finds any `torch.nn.Module` instances and clears their parameters
3. Finds any raw `torch.Tensor` on CUDA and replaces with empty CPU tensors
4. Forces multiple GC passes to release the memory

**Usage**: After an OOM error, run Force Unload with `nuke_orphaned_tensors=True`.

## Library Limitations Summary

### bitsandbytes Limitations

| Feature | Status | Impact |
|---------|--------|--------|
| `.to()` device movement | ❌ Not supported | Cannot soft unload INT8/NF4 models |
| CPU inference | ❌ Not supported | Model must stay on GPU |
| Re-quantization | ❌ Not exposed | No path to move and re-quantize |

### accelerate Limitations

| Feature | Status | Impact |
|---------|--------|--------|
| Meta tensor materialization | ❌ Not available | Cannot move offloaded models |
| Pinned memory for offload | ⚠️ Partial | Some speedup possible |
| Selective layer pinning | ❌ Not granular | All-or-nothing offloading |

## Recommendations by Use Case

### Iterative/Experimental Work
- Use **INT8** loader
- Accept 1:40 reload time between sessions
- Fastest inference (3 min)

### Quality-Critical/Production Work
- Use **BF16 Full Loader** with smart offload
- Accept 11+ min inference time
- Best quality output

### Multi-Model Workflows
- Use **Force Unload** between models
- Ensures clean VRAM state
- Prevents memory leak accumulation

### High-Resolution Generation
- Use **Budget loaders** with appropriate headroom
- Let auto-offload manage VRAM
- Or manually set offload_mode="smart"

## Future Improvements

Potential improvements pending library updates:

1. **bitsandbytes**: Device movement for quantized models
   - See: [Feature Request Link TBD]
   
2. **accelerate**: Meta tensor materialization
   - See: [Feature Request Link TBD]

3. **Alternative quantization**: GGUF-style quantization supports CPU↔GPU movement but lacks HuggingFace pipeline integration for image models.

## Modern Optimizations Not Yet Applied (Aug 2026)

Written after the August 2026 dependency upgrade (torch 2.13.0, transformers
5.15.0, bitsandbytes 0.50.0, CUDA 13.0, optional flashinfer 0.5.0 — see
`constraints-tested.txt`). A repo-wide grep for the usual markers of modern
inference-acceleration techniques (`torch.compile`, `cudagraph`, `fp8`/
`float8`, `nvfp4`, `awq`/`gptq`, `xformers`, `channels_last`, `allow_tf32`,
`teacache`/`deepcache`/`first_block_cache`, `sdpa_kernel`) came back empty —
none of these are in use yet. Already well covered: block-swap (pinned
staging buffers, async non-blocking H2D/D2H, cross-stream event sync — see
`hunyuan_block_swap.py`), a memory-efficient (not speed-efficient, per its
own docstring) per-expert-loop MoE dispatch (`_efficient_moe_forward` in
`hunyuan_shared.py`), the bnb NF4/INT8/bf16 quantization ladder, and the
optional flashinfer/flash_attn backends (see
`Docs/flashinfer_flash_attention_setup.md`).

### Revision-pinning: is the model code we run actually pinned?

Checked `scripts/download_models.py`: it calls `huggingface_hub.
snapshot_download(..., revision=revision)` where `revision` defaults to
`None` — only set via an unused-by-default `--revision` flag — and nothing in
this repo records which commit hash actually got downloaded (no lockfile, no
manifest). Since `from_pretrained` loads `trust_remote_code` files from a
local checkpoint directory rather than reaching out to the hub, any single
already-downloaded checkpoint's modeling code is effectively frozen — but
**the pin isn't reproducible or recorded**: re-running the download script on
a different day or machine can silently fetch a different upstream commit
with no diff trail here.

Checked actual upstream activity:

- `tencent/HunyuanImage-3.0`: last real code fix Sept 29, 2025 ("Fix a few
  bugs").
- `tencent/HunyuanImage-3.0-Instruct`: last modeling-relevant commit Feb 3,
  2026 — *"Fix: Implement deepseek moe for `moe_impl == 'eager'` to solve
  oom"* — directly relevant to this project's own eager-MoE memory-pressure
  work. Nothing since (6+ months quiet as of Aug 2026).
- The quantized checkpoints actually run on the RTX 6000
  (`EricRollei/HunyuanImage-3.0-Instruct-Distil-NF4-v2` et al.) are
  themselves frozen one-time re-uploads (2 commits, both Feb 22, 2026 —
  "initial commit" + one folder upload), not tracking Tencent's upstream at
  all.

Upstream is genuinely quiet. Whether Eric's Feb 22 snapshot already includes
Tencent's Feb 3 fix (19 days earlier — plausible, not confirmed) should be
checked directly against the actual downloaded checkpoint's `.py` files
rather than assumed.

### Vendoring recommendation

Worth doing, and worth doing before the step-caching/fused-MoE items below.
In practice, a vendored-but-undocumented snapshot (Eric's frozen re-upload)
is already what's running — the change is making that explicit, checked into
git, and diffable against upstream. Upstream's low churn keeps ongoing sync
burden low. This project already patches around upstream behavior at runtime
via `hunyuan_shared.py`'s `patch_*` functions (`patch_hunyuan_generate_image`,
`patch_moe_efficient_forward`, `patch_hunyuan_static_cache_device`) —
several exist specifically *because* the code isn't vendored and can't be
edited directly; step-caching and a fused MoE kernel are both substantially
easier as clean edits to vendored source than as reverse-engineered
monkey-patches over a black box. Main real cost: base vs. instruct
architectures, and different maintainers (Tencent direct for bf16, Eric's
re-quantized forks for NF4/INT8), mean the vendored copy needs reconciling
per checkpoint family.

**Complexity: Medium.** Copy the `modeling_*.py`/`configuration_*.py` files
out of a downloaded checkpoint into this repo (e.g. `vendor/hunyuan_image_3/`),
diff against a fresh Tencent checkout to see what Eric's forks changed for
quantization compat, then point loading at the vendored copy instead of the
implicit local `trust_remote_code` copy (`transformers` supports this via
`AutoModel.register()`/a local code path — exact mechanism needs a quick
check against transformers 5.15's current API). **First concrete step**:
diff the `.py` files in a downloaded model directory against a fresh clone
of `tencent/HunyuanImage-3.0-Instruct` at HEAD.

### Ranked optimizations (impact / complexity)

| # | Optimization | Impact | Complexity | Why |
|---|---|---|---|---|
| 1 | Diffusion step-caching (TeaCache/DeepCache/First-Block-Cache-style) | **High** — ~1.5-2x on the diffusion portion is typical for flow/DiT models elsewhere (Flux, SDXL, Wan), near-imperceptible quality loss when tuned | **Medium** (Low if vendored first) | Biggest untouched lever. Most benefit on the 40/50-step non-distil paths — an 8-step distil model has less redundant inter-step similarity to exploit. Needs the exact per-step transformer call in the (currently un-vendored) sampling loop. |
| 2 | Fused/grouped-GEMM MoE kernel for `moe_impl=eager` | **Medium** — narrower than it first looks: `moe_impl=flashinfer` *already* gets a fused MoE kernel, so this closes the gap for the eager/default path only | **Medium** (adapt an existing open-source Triton fused-MoE kernel) / **High** (from scratch) | `_efficient_moe_forward`'s own docstring: "Speed: Similar" — it fixed memory, not speed. Torch 2.13 ships Triton already. |
| 3 | `torch.compile` | **Medium**, scoped — typically 10-30% for compile-friendly workloads, but variable resolution (recompiles) and block-swap's dynamic device placement fight full-graph compilation | **Low** (VAE only) / **High** (full transformer) | Start with the VAE (fixed op graph); treat the full transformer as a separate, harder investigation. |
| 4 | Native hardware FP8 (torchao/TensorRT-Model-Optimizer) | **High but unvalidated** — real tensor-core matmul vs. bnb's dequant-to-bf16 (NF4)/outlier-decomposition (INT8), better quality-per-bit than INT8 at similar footprint | **High** — new dependency, re-quantization/on-the-fly cast, and real correctness risk (this model already broke bnb's skip-list on `shared_mlp`/`mlp.gate` via interior path-segment matching — see the "issues #36/#41" fix in `hunyuan_shared.py` — a new quant backend is likely to hit an analogous gap) | FP8 is 8-bit — it will **not** beat NF4 on memory, only on speed/quality. Blackwell-native NVFP4 (true 4-bit hardware) could match NF4's footprint while being faster, but tooling is newer/less mature — track, don't commit yet. |
| 5 | `channels_last` + TF32 for the VAE | **Low-Medium**, cheap | **Low** — a few lines | VAE encode/decode is a small fraction of total time relative to the 32-block transformer + MoE dispatch (+ autoregressive CoT decode for instruct). Also benefits the existing `vae_dtype=float32` option, which runs true fp32 matmuls today with TF32 off. |
| 6 | Explicit SDPA backend selection | **Low-Medium** — mostly de-risks a bad auto-heuristic on newer Blackwell rather than a big new win | **Low** — `torch.nn.attention.sdpa_kernel` / `torch.backends.cuda.enable_*_sdp` | Measurement-only, no new dependency. |
| 7 | Weight-only quant kernels (torchao int4, AWQ, GPTQ) as a bnb alternative | **Medium** — faster dequant+matmul than bitsandbytes at similar memory, incremental not transformative | **High** — new loader path, re-quantized checkpoints, same interior-module quant-skip risk as item 4 | Longer-term, not a quick patch. |
| 8 | KV-cache quantization for the CoT/recaption stage | **Low** — only `bot_task="think"/"recaption"`, buys seqlen headroom more than speed | **Medium** | Secondary. |
| 9 | Speculative decoding for the CoT/recaption stage | **Low-Medium** for that stage alone, which doesn't run at all for `bot_task="image"` | **High** — needs upstream draft-head/self-speculation support | Lowest priority, experimental — another point in favor of vendoring first. |

## Version History

- **2025-11-30**: Initial documentation based on RTX 6000 Pro testing
- Soft unload investigation completed
- Force unload implemented
- Auto-offload override for INT8 implemented
- **2026-08-15**: Corrected stale Soft Unload status; added "Modern
  Optimizations Not Yet Applied" section (revision-pinning finding, vendoring
  recommendation, ranked optimization list with impact/complexity estimates)
