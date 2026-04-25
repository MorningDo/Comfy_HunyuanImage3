# HunyuanImage-3 Quality & Tuning Notes

Findings from a code-level investigation of the upstream Tencent HunyuanImage-3
sources (`transformers_modules/HunyuanImage-3/*.py`) and recommendations baked
into the V2 unified loader and Instruct loader options.

## 1. Negative prompts (NOT SUPPORTED)

HunyuanImage-3's architecture does **not** support negative prompts in the way
that Stable Diffusion / SDXL / Flux do. This was verified end-to-end against
the upstream model code:

- `hunyuan_image_3_pipeline.py` `__call__` accepts only `batch_size`,
  `image_size`, `num_inference_steps`, `guidance_scale`, etc. There is no
  `negative_prompt` parameter, `prompt_embeds`, or `negative_prompt_embeds`
  argument anywhere in the pipeline signature.
- The CFG operator (`ClassifierFreeGuidance` class, line 504) takes only
  `pred_cond`, `pred_uncond`, and `guidance_scale` — the standard 2-pass
  formulation `pred = pred_uncond + scale * (pred_cond - pred_uncond)`.
- The unconditional pass is constructed inside `tokenizer_wrapper.py`. When
  `cfg_factor=2`, `apply_general_template` calls `make_batch` with
  `uncondition_repeat_times=cfg_factor-1`, and the **uncond text tokens are
  replaced wholesale with a single special `<cfg>` token** (see
  `encode_text`: `text_token = [self.cfg_token_id] * len(text_token)`).
- The model was trained with `uncond_p` (random `<cfg>` token dropout) so the
  trained "null" condition is literally a sequence of `<cfg>` placeholder
  tokens. It is not an empty string, not a Chinese phrase, not BOS/EOS only.

### Why a Chinese / English negative prompt won't work as a drop-in

Substituting `<cfg>` with Chinese or English text tokens at the uncond
position would produce **off-distribution** conditioning the model never saw
during training. Empirically this would either be a no-op (small effect) or a
quality regression — it cannot function as a negative prompt because the
network has no representation for "push away from this".

### What a true negative prompt would require

Three batches per step instead of two:
1. `pred_cond`   — text tokens of the user prompt
2. `pred_uncond` — `<cfg>` tokens (the trained null)
3. `pred_neg`    — text tokens of the negative prompt

Then `pred = pred_uncond + scale*(pred_cond - pred_uncond) - neg_scale*(pred_neg - pred_uncond)`
or similar. This requires patching `prepare_model_inputs` in `hunyuan.py`
and `apply_general_template` in `tokenizer_wrapper.py`, plus a 50% memory
overhead per diffusion step. Not implemented — out of scope for this round.
Tracked in `Docs/UNIFIED_V2_IMPLEMENTATION_PLAN.md` as an aspirational item.

## 2. `moe_drop_tokens` (new option in V2 + Instruct loaders)

The MoE transformer routes each token to a top-k subset of 64 experts. By
default `moe_drop_tokens=True` enables capacity-based dropping: when more
tokens are routed to one expert than its `expert_capacity`, the overflow is
silently dropped (zero output). This is an inference-speed optimization
inherited from training but on **single-batch inference** the routing
distribution is far less balanced than at train time, so dropping kicks in
much more aggressively than expected.

Setting `moe_drop_tokens=False` removes the cap and keeps every token. Cost:
slightly slower expert dispatch and marginally higher peak VRAM. Benefit:
more stable detail and fewer "empty patch" artifacts at high resolutions.

**Recommendation:** keep `True` for speed-critical work, flip to `False` for
2K+ portraits and any output where you see localized blur or detail dropout.

## 3. `vae_dtype` (new option in V2 + Instruct loaders)

The VAE is loaded in bf16 by default and decode happens under
`torch.autocast(dtype=float16, enabled=True)` inside `vae_encode`. For decode
the autocast wrapper does its own casting, so loading the VAE in bf16 vs
float32 mostly affects the **decode kernels' precision floor**, not the
encode path.

Setting `vae_dtype=float32` adds ~600 MB VRAM but reduces banding in dark
gradients and mild color shifts in skin tones. With 48 GB cards it's almost
free.

**Recommendation:** float32 for portrait / skin / dark-scene work, bf16
otherwise.

## 4. `flow_shift` recipes

Default 2.8 (slightly below model's 3.0 default — gives crisper detail).

| Subject                   | flow_shift |
|---------------------------|------------|
| Portraits / faces / skin  | 2.0 – 2.5  |
| General / mixed           | 2.8 – 3.0  |
| Landscapes / illustration | 3.5 – 5.0  |

Lower values weight early (high-noise) timesteps more strongly, which
preserves fine detail. Higher values shift weight to late timesteps, which
smooths gradients (useful for sky / clouds / soft illustration shading).

## 5. Step counts

Auto defaults: **8 steps for Distil**, **50 steps for full Instruct / base**.

Going to 60–80 steps reduces flow-matching artifacts at 2K+ resolution but
generation time scales linearly. Above ~80 the marginal quality gain is in
the noise floor and not worth the wait.

## 6. `bot_task` recaption modes (Instruct only)

`recaption` and `think_recaption` route the prompt through the model's CoT
chain to rewrite it into the model's preferred format. This is **very**
slow — easily 30–120 s of pure LLM time before any diffusion starts —
because it generates up to `max_new_tokens` (2048 default) tokens
autoregressively on the full 17 B / 80 B model.

Use only when the prompt is sparse / underspecified. For well-written
prompts, `auto` is faster and at least as good.

## 7. Legacy loader coverage

The new `moe_drop_tokens` and `vae_dtype` options are exposed on:

- `HunyuanCleanLoader` (V2 backend) and `HunyuanUnifiedV2`
- `HunyuanInstructLoader`

The legacy loaders in `hunyuan_full_bf16_nodes.py` and
`hunyuan_quantized_nodes.py` retain the previous defaults
(`moe_drop_tokens=True`, VAE in bf16). New work should use V2 — these
legacy nodes are kept for backward compatibility with existing workflows.
