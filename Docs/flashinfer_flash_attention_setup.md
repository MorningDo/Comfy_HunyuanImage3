# FlashInfer / FlashAttention setup runbook (RTX 6000 / SM120)

Condensed, copy-pasteable version of what to actually run to get
`moe_impl=flashinfer` and `attention_impl=flash_attention_2` set up on a
Blackwell workstation card (RTX PRO 6000, RTX 5090 — compute capability
12.0 / "SM120"). For the full rationale behind every decision here, see
`INSTALL.md`'s "FlashInfer / FlashAttention setup" section and the header
comments in `install.sh` / `requirements-flash.txt` — this file only has the
commands and the short version of *why*.

**Bottom line up front**: `flashinfer` is the recommended, working path for
MoE acceleration on this GPU. `flash-attn` (`attention_impl=flash_attention_2`)
is **expected to fail** on SM120 today — a prior investigation on this exact
GPU model already hit `CUDA error: invalid argument`, and upgrading CUDA
toolkit versions does not fix it (it's a missing-kernel gap in flash-attn
itself, not a CUDA-version mismatch). Use `sdpa` (the default) for attention
unless you specifically want to confirm/characterize that failure.

## 0. Confirm the GPU

```bash
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
```

Expect `compute_cap` = `12.0`. `validate_attention_moe_impl()`
(`hunyuan_shared.py`) keys its SM120 warning off exactly this value via
`torch.cuda.get_device_capability(0) == (12, 0)`.

Also worth checking before either install (both are warn-only, not blocking,
in `install.sh`):

```bash
nvcc --version        # system CUDA Toolkit, separate from the pip torch wheel
gcc -dumpversion       # Tencent's README recommends >= 9 for compiling either backend
```

## 1. FlashInfer (`moe_impl=flashinfer`) — do this first

One command, from the repo root:

```bash
INSTALL_FLASHINFER=1 ./install.sh
```

Safe to re-run (reuses your existing venv/ComfyUI checkout — see
`install.sh`'s own "safe to re-run" notes). This does two things:

1. Installs a **system-level CUDA 13.0 Toolkit via apt** (Ubuntu-specific:
   `cuda-keyring` + `cuda-toolkit-13-0`, needs root or `sudo`). This is
   separate from the CUDA-enabled torch *wheel* `install.sh` already
   installs — FlashInfer JIT-compiles its own kernels at first use and needs
   a real `nvcc` + headers matching torch's CUDA version, which a pip torch
   wheel doesn't provide on its own.
2. `pip install flashinfer-python==0.5.0` (Tencent's own tested baseline;
   override with `FLASHINFER_VERSION=x.y.z INSTALL_FLASHINFER=1 ./install.sh`
   if needed).

If the apt step succeeds, `install.sh` prints these — add them to your shell
profile if you want `nvcc` available outside of `install.sh`'s own run (the
generated `$COMFYUI_DIR/start_comfyui.sh` already exports them automatically
for ComfyUI itself, no action needed there):

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
```

**Verify:**

```bash
source "$COMFYUI_DIR/venv/bin/activate"   # or wherever your venv is
python3 -c "import flashinfer; print(flashinfer.__version__)"
```

**Note**: the *first* inference in ComfyUI with `moe_impl=flashinfer`
selected takes several extra minutes (JIT kernel compile for your specific
GPU) — cached after that, not a bug. It also uses more peak VRAM during MoE
dispatch than `eager`.

If the apt step fails or isn't available (no root/sudo, non-Ubuntu, no
network) — `install.sh` warns and continues rather than aborting; install
`cuda-toolkit-13-0` yourself for your distro and re-run.

## 2. FlashAttention (`attention_impl=flash_attention_2`) — expect this to fail here

`install.sh` deliberately does **not** install `flash-attn`:

- No prebuilt PyPI wheel — `pip install flash-attn` compiles from source,
  which can take hours and needs a matching `nvcc`/compiler.
- Mainline [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
  has **no confirmed SM120 support** as of the last check
  ([#1987](https://github.com/Dao-AILab/flash-attention/issues/1987),
  [#2307](https://github.com/Dao-AILab/flash-attention/issues/2307)). A
  reporter on the *exact same* RTX PRO 6000 Blackwell / CUDA 12.8 / torch
  2.8.0 combo hit `CUDA error (invalid argument)` invoking the attention
  kernels. Bumping to CUDA 13.0 does **not** fix this.

**Recommendation**: don't install it — use `attention_impl=sdpa` (the
default). `validate_attention_moe_impl()` will raise a clear `ValueError`
if you select `flash_attention_2` without the package installed, rather than
letting it crash deep inside `transformers`.

**If you want to try anyway** (to confirm the gap yourself, or test an
unofficial fork):

```bash
source "$COMFYUI_DIR/venv/bin/activate"
pip install flash-attn        # from-source build, can take hours
```

Or an unofficial SM120 fork instead of mainline (entirely at your own risk —
not tested or endorsed by this repo): `roy86/flash-attention_sm120` on
GitHub, or the `flash-attn-4-sm120` distribution on Hugging Face. Switch back
to upstream once real SM120 support merges there.

**Verify:**

```bash
python3 -c "import flash_attn; print(flash_attn.__version__)"
```

**How to tell which failure you're looking at**, once installed and selected
in ComfyUI (`attention_impl=flash_attention_2`):

| Symptom | Meaning |
|---|---|
| Low-level `CUDA error: invalid argument` (or similar) from inside `flash_attn_func`/`flash_attn.flash_attn_interface` | The known SM120 gap — expected, not a bug in this repo. Use `sdpa` or `moe_impl=flashinfer` instead. |
| `ImportError` | The package/build itself is broken (unrelated to SM120) — check the build log. |
| `TypeError`/`AttributeError` from this repo's own code (not `flash_attn`) | A bug in `validate_attention_moe_impl()` or its wiring in `hunyuan_instruct_nodes.py::load_model()` — worth reporting. |
| OOM | Unrelated VRAM-budget issue — see `vram_reserve_gb`/`blocks_to_swap`, not the attention backend. |

## 3. Selecting the backends in ComfyUI

On the `HunyuanInstructLoader` node:

- `attention_impl`: `sdpa` (default) or `flash_attention_2`
- `moe_impl`: `eager` (default) or `flashinfer`

Selecting a backend whose package isn't installed raises a clear error at
load time (`validate_attention_moe_impl()` in `hunyuan_shared.py`) instead of
failing deep inside `transformers`. Selecting `flash_attention_2` on this GPU
with the package present logs the SM120 warning but doesn't block you from
trying it.

## 4. Automated verification

Two GPU smoke tests exist specifically for this
(`tests/test_gpu_attention_backends.py`), each skipping outright if its
package isn't importable and needing real model weights (~30GB VRAM,
`Instruct-Distil-NF4-v2`):

```bash
"$COMFYUI_DIR/run_gpu_tests.sh" -k attention_backends -v --run-slow
```

- `test_flashinfer_moe_loads_and_generates` — expected to **pass**. First run
  is slow (JIT compile); the test's own comment notes pytest's 900s default
  timeout may need `--timeout` bumped on a cold cache.
- `test_flash_attention_2_loads_and_generates` — expected to **fail** on this
  hardware today per the test file's own docstring. That's a correct,
  informative result confirming the known upstream gap, not a regression to
  chase.

## See also

- `INSTALL.md` — full setup guide (this file's source of truth) and the
  "Troubleshooting" section for driver/CUDA-version mismatch errors.
- `install.sh --help` — same config summary as `INSTALL.md`'s table, at any
  time.
- `requirements-flash.txt` — the pinned `flashinfer-python` version (and why
  `flash-attn` is deliberately not listed there).
