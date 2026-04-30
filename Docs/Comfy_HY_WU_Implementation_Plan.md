# Comfy_HY_WU — Implementation Plan

> Working document for building ComfyUI nodes for Tencent HY-WU  
> Model repo: https://github.com/Tencent-Hunyuan/HY-WU  
> Target hardware: RTX PRO 6000 Blackwell 96GB + RTX A4000 16GB

---

## What HY-WU Is

HY-WU is a **two-model stack** built on top of `HunyuanImage-3.0-Instruct`:

- **Base model** (80B / 13B active MoE): frozen, used for CoT generation, hidden state extraction, and final image generation
- **WU / ParameterGenerator model** (8B): generates instance-conditioned LoRA adapters on the fly from the input images + prompt, injected into the base model during the final forward pass

No test-time optimization — adapters are generated dynamically per inference.

### Four-Step Inference Pipeline

```
Step 1: model.generate_cot()       → cot_text string (skippable if cached)
Step 2: model.generate_image()     → hidden states / condition tensors
Step 3: parameter_generator()      → pg_state_dict (the LoRA weights)
Step 4: model.generate_image()     → final PIL Image (with LoRA injected)
```

The WU model only runs once (Step 3). Peak simultaneous VRAM is base model + WU model + condition tensors during Step 2→3 transition.

---

## VRAM Analysis

### WU Model Fixed Cost
The `ParameterGenerator` always loads at BF16 — no quantization exposed upstream.

| WU Precision | VRAM |
|---|---|
| BF16 (as shipped) | ~16 GB |

### Combined Cost on 96GB Card

| Base Model | WU (BF16) | Total | Verdict |
|---|---|---|---|
| BF16 (~160 GB) | ~16 GB | ~176 GB | ❌ Not viable single card |
| INT8 (~81 GB) | ~16 GB | ~97 GB | ⚠️ 1 GB over — block swap needed |
| NF4 (~45 GB) | ~16 GB | ~61 GB | ✅ ~35 GB headroom |

**NF4 base + BF16 WU is the primary target.** INT8 is viable with block swap.  
BF16 base is not viable on a single card regardless of block swap.

### Key Difference From Plain Instruct Nodes

The upstream `WUPipeline.__init__` loads the base model with `torch_dtype="auto"` — no quantization config. The entire purpose of our wrapper is to inject `BitsAndBytesConfig` before `from_pretrained` is called.

---

## Architectural Decision: Subclass `WUPipeline`

Override `__init__` to inject quantization config. Inherit `generate()` unchanged.

**Rationale:**
- Upstream repo is 2 days old — API will change; inheriting `generate()` means we get fixes for free
- We own the loading logic where quantization needs to be injected
- If upstream breaks the API, we adapt the override, not the whole codebase

---

## Package Structure

```
Comfy_HY_WU/
├── __init__.py              # NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
├── nodes/
│   ├── __init__.py
│   ├── loader.py            # Eric_HYWU_Loader node
│   └── generate.py          # Eric_HYWU_Generate node
├── wu_wrapper.py            # QuantizedWUPipeline subclass
├── utils/
│   ├── __init__.py
│   ├── image_utils.py       # tensor↔PIL conversion
│   └── model_cache.py       # singleton pipeline cache
├── requirements.txt
└── README.md
```

---

## `wu_wrapper.py` — `QuantizedWUPipeline`

### Constructor Signature

```python
def __init__(
    self,
    base_model_path: str,
    pg_model_path: str,
    base_quantization: str = "nf4",   # "nf4" | "int8" | "bf16"
    pg_quantization: str = "bf16",    # stub — "int8" for future use
    device_map: str = "auto",
    moe_impl: str = "eager",
    moe_drop_tokens: bool = False,
)
```

### Implementation Steps

1. **Build `BitsAndBytesConfig`** based on `base_quantization`:
   - `"nf4"` → `load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16`
   - `"int8"` → `load_in_8bit=True`
   - `"bf16"` → no quantization config; set `torch_dtype=torch.bfloat16` explicitly

2. **Load `ParameterGenerator`** exactly as upstream:
   ```python
   torch_dtype=torch.bfloat16, device_map=device_map
   ```
   `pg_quantization` is a stub parameter only — leave INT8 PG unimplemented with a `# TODO` comment. Reason: PG runs under autocast BF16; quantizing its output projection layers needs validation.

3. **Build `model_kwargs`** matching upstream exactly, except:
   - Inject `quantization_config=bnb_config` when not BF16
   - Override `torch_dtype="auto"` → explicit `torch.bfloat16` for BF16 path

4. **Call `inject_pg(...).from_pretrained(base_model_path, **model_kwargs)`** — same as upstream

5. **Call `self.model.load_tokenizer()`, `.enable_pg()`, `.eval()`** — same as upstream

6. **Store attributes** for cache key: `self.base_model_path`, `self.pg_model_path`, `self.base_quantization`

### Block Swap Hook (Future)

```python
# TODO: Test whether HunyuanImage3ForCausalMM (WU-patched) accepts
# blocks_to_swap attr from the Instruct node block-swap implementation.
# If confirmed, add blocks_to_swap parameter here and wire it up after
# from_pretrained(), matching the pattern in Comfy_HunyuanImage3.
```

Do NOT implement block swap until confirmed compatible with the `inject_pg` patched model class.

---

## `utils/model_cache.py`

### Cache Key

```python
(base_model_path, pg_model_path, base_quantization, moe_impl, moe_drop_tokens)
```

### `get_or_load_pipeline()` Logic

1. Check module-level `_pipeline_cache: dict[tuple, QuantizedWUPipeline]`
2. If key exists → return cached pipeline
3. If a different key exists → evict: `del _pipeline_cache[old_key]`, `torch.cuda.empty_cache()`, `gc.collect()`
4. Instantiate new `QuantizedWUPipeline(...)`, store, return
5. Only one pipeline in memory at a time

---

## `utils/image_utils.py`

### `comfy_tensor_to_pil_list(tensor) -> list[Image.Image]`

- Input: `[B, H, W, C]` float32 0–1 ComfyUI IMAGE tensor
- Output: list of PIL RGB images, one per batch item
- Clamp to [0, 1] before conversion

### `pil_to_comfy_tensor(image) -> torch.Tensor`

- Input: PIL Image RGB
- Output: `[1, H, W, C]` float32 0–1 on CPU

### `build_imgs_input(img1, img2=None, img3=None) -> list[Image.Image]`

- Accepts ComfyUI tensors or `None`
- Takes only **first frame** (batch index 0) from each tensor
- Filters out `None` slots
- Returns PIL list ready for `pipeline.generate()`
- Raises `ValueError` if all three are `None`

---

## `nodes/loader.py` — `Eric_HYWU_Loader`

```
CATEGORY:      "HY-WU"
RETURN_TYPES:  ("WU_PIPELINE",)
RETURN_NAMES:  ("wu_pipeline",)
FUNCTION:      "load"
```

### Inputs

| Name | Type | Default | Notes |
|---|---|---|---|
| `base_model_path` | STRING | `"tencent/HunyuanImage-3.0-Instruct"` | HF repo ID or absolute local path |
| `pg_model_path` | STRING | `"tencent/HY-WU"` | HF repo ID or absolute local path |
| `base_quantization` | COMBO | `"nf4"` | `["nf4", "int8", "bf16"]` |
| `moe_impl` | COMBO | `"eager"` | `["eager", "sdpa"]` |
| `moe_drop_tokens` | BOOLEAN | `False` | |

### `load()` Method

1. Validate `bitsandbytes` is importable when `base_quantization != "bf16"` — raise informative error rather than letting it explode in `from_pretrained`
2. Validate `base_model_path` and `pg_model_path` are non-empty strings
3. Call `get_or_load_pipeline(...)` from cache module
4. Return `(pipeline,)` tuple
5. Wrap in try/except — surface clean error messages for OOM, missing model, missing `wu` package

---

## `nodes/generate.py` — `Eric_HYWU_Generate`

```
CATEGORY:      "HY-WU"
RETURN_TYPES:  ("IMAGE", "STRING")
RETURN_NAMES:  ("image", "cot_text")
FUNCTION:      "generate"
```

### Inputs

| Name | Type | Required | Default | Notes |
|---|---|---|---|---|
| `wu_pipeline` | WU_PIPELINE | ✅ | — | From loader node |
| `prompt` | STRING | ✅ | `""` | Multiline |
| `image_1` | IMAGE | ✅ | — | Base image (Figure 1 in paper examples) |
| `image_2` | IMAGE | Optional | — | Reference/source image (Figure 2) |
| `image_3` | IMAGE | Optional | — | Third reference image |
| `cot_text` | STRING | Optional | — | Pass cached CoT to skip Step 1 |
| `steps` | INT | ✅ | 50 | min 1, max 100 |
| `seed` | INT | ✅ | 42 | min 0, max 2³²-1 |

`image_2`, `image_3`, `cot_text` declared via `"optional"` key in `INPUT_TYPES`.

### `generate()` Method Steps

1. Call `build_imgs_input(image_1, image_2, image_3)` → PIL list
2. Resolve `cot_text`: empty string → `None` (triggers CoT generation internally)
3. **CoT capture** — to expose `cot_text` as a node output for caching:
   - If `cot_text` is `None`, call `pipeline.model.generate_cot(...)` directly with same args as Step 1 in upstream; capture the returned string
   - Pass the captured string to `pipeline.generate(cot_text=captured, ...)`
   - ⚠️ **Requires confirming `generate_cot()` signature** — see Open Questions
4. Call `pipeline.generate(prompt=..., imgs_input=..., cot_text=..., diff_infer_steps=steps, seed=seed, verbose=1)`
5. Convert returned PIL → `pil_to_comfy_tensor()`
6. Return `(output_tensor, cot_text_string)`

---

## `__init__.py` — Registration

```python
NODE_CLASS_MAPPINGS = {
    "Eric_HYWU_Loader":   Eric_HYWU_Loader,
    "Eric_HYWU_Generate": Eric_HYWU_Generate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Eric_HYWU_Loader":   "HY-WU Loader",
    "Eric_HYWU_Generate": "HY-WU Generate",
}
```

Wrap all imports in try/except with a console warning if `wu` package isn't found — nodes won't register rather than crashing all of ComfyUI.

---

## `requirements.txt`

```
# IMPORTANT: The 'wu' package must be installed manually from the HY-WU repo.
# git clone https://github.com/Tencent-Hunyuan/HY-WU
# pip install -e /path/to/HY-WU
bitsandbytes>=0.43.0
Pillow>=9.0.0
torch>=2.1.0
```

---

## Open Questions (Resolve Before Coding)

### 1. `generate_cot()` Signature  ← Most Critical
Need `models.py` or `mixin.py` source to confirm whether `model.generate_cot()` takes identical parameters to `generate_image()`. The CoT-capture design in the generate node depends on this. If the signature differs, we may need to override `generate()` in the subclass instead.

**To check:** Open `wu/models.py` or wherever `HunyuanImage3ForCausalMM` is defined and find `generate_cot`.

### 2. `inject_pg` Safety on Re-instantiation
Is `inject_pg(config, HunyuanImage3ForCausalMM)` safe to call multiple times (e.g., cache evict + re-create)? Does it mutate the class permanently or create a fresh subclass each call?

**To check:** Open `wu/mixin.py`.

### 3. Block Swap Compatibility
After `inject_pg` patches `HunyuanImage3ForCausalMM`, does the resulting model still accept the `blocks_to_swap` attribute used in your Instruct nodes?

**To check:** Runtime test — load with NF4, check `dir(pipeline.model)` and the class MRO.

### 4. `imgs_input` Accepts PIL or Paths Only?
The type hint says `list[str | Image.Image]`. Upstream `infer.py` passes file paths. We plan to pass PIL images directly. Confirm this path works before finalizing `image_utils.py`.

**To check:** Open `wu/pipeline.py`'s `generate()` and trace where `imgs_input` is first consumed (likely in `generate_cot` or `generate_image` — check if it does `Image.open()` on strings but passes PIL through directly).

---

## Implementation Order

1. `utils/image_utils.py` — no dependencies, fully testable in isolation  
2. `utils/model_cache.py` — straightforward  
3. Resolve Open Questions 1 and 2 from source inspection  
4. `wu_wrapper.py` — core logic, needs `wu` package installed  
5. `nodes/loader.py`  
6. `nodes/generate.py`  
7. `__init__.py` + `requirements.txt` + README  
8. Test: NF4 base + BF16 WU → confirm basic generation works  
9. Test: INT8 base + BF16 WU → confirm block swap covers 1 GB overflow  
10. Validate block swap compatibility (Open Question 3) → add if confirmed  

---

## License Note

HY-WU uses the Tencent Hunyuan license (same as your existing HI3 work). Not Apache/MIT. Attribution required, "Powered by Tencent Hunyuan" encouraged in products. Include license notice in README.
