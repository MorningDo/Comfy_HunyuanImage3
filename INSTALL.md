# Installing for GPU test validation

`install.sh` sets up ComfyUI + this custom node in a fresh venv, with a
reproducible, verified-working set of ML-stack dependency versions, and
runs the fast (no-GPU) regression suite as a sanity check. It's meant for
standing up a clean test machine (e.g. to validate the fixes in
`CHANGELOG.md` on real GPU hardware), not as end-user install instructions
— see the main `README.md` for that.

## Quick start

```bash
./install.sh
```

That's it for defaults: clones ComfyUI to `~/ComfyUI` (or reuses it if
this repo is already checked out at `<somewhere>/custom_nodes/<this repo>`
— then that `<somewhere>` is used instead), creates a venv at
`$COMFYUI_DIR/venv`, installs a CUDA-enabled PyTorch build, ComfyUI's own
requirements, this project's requirements (pinned to verified-working
versions — see "What gets pinned, and why" below), symlinks this repo
into `custom_nodes/`, verifies every import works, and runs the fast
no-GPU test suite.

To also run the full GPU suite (needs real model weights) as part of the
same invocation:

```bash
HUNYUAN_TEST_MODELS_DIR=/path/to/your/hunyuan/models ./install.sh
```

Or run it separately afterward — the script's summary prints the exact
command either way. See `tests/README.md` for everything about the test
suite itself (per-variant model directory overrides, `--run-slow`, etc.).

## Configuration

All via environment variables, all optional:

| Variable | Default | What it does |
|---|---|---|
| `COMFYUI_DIR` | auto-detected, else `~/ComfyUI` | Where to put/find ComfyUI |
| `VENV_DIR` | `$COMFYUI_DIR/venv` | Python venv location |
| `PYTHON_BIN` | `python3` | Interpreter used to create the venv |
| `COMFYUI_REPO_URL` | official ComfyUI GitHub repo | Only matters on first clone |
| `COMFYUI_REF` | `master` | Branch/tag to clone (shallow clone — arbitrary commit SHAs aren't supported by `--branch`) |
| `TORCH_CUDA_INDEX` | `https://download.pytorch.org/whl/cu130` | pytorch.org wheel index. flashinfer's officially documented CUDA support list is 12.6/12.8/13.0/13.1 (no 12.9), so cu130 is the best-aligned "current" choice; Blackwell *requires* cu128+ regardless |
| `TORCH_VERSION` | `2.13.0` | Falls back to an unpinned install from the same index if this exact version isn't available there by the time you run this. Skipped entirely if a CUDA-enabled torch matching this version is already importable (e.g. inherited from the base image via the venv's `--system-site-packages`) |
| `HUNYUAN_TEST_MODELS_DIR` | unset | If set, the full GPU suite also runs at the end, not just the fast one |
| `SKIP_COMFYUI` | `0` | Set to `1` to skip cloning/installing ComfyUI entirely — just sets up this project's own deps in the venv (useful if ComfyUI is already fully set up) |
| `INSTALL_FLASHINFER` | `0` | Set to `1` to also install `flashinfer-python` and attempt a system CUDA Toolkit install via apt — see "FlashInfer / FlashAttention setup" below |
| `FLASHINFER_VERSION` | `0.5.0` | Matches Tencent's own tested baseline. Only used when `INSTALL_FLASHINFER=1` |

Run `./install.sh --help` for the same summary at any time. Safe to
re-run: an existing ComfyUI checkout, venv, or `custom_nodes/` symlink is
detected and reused rather than recreated or overwritten.

## What gets pinned, and why

`requirements.txt` deliberately floors its versions (`torch>=2.8.0`,
`transformers>=4.47.0`, etc.) rather than capping them — see the comment
at the top of that file. That's the right choice for end users who want
this node to keep working alongside whatever else is in their ComfyUI
environment. It is *not* the right choice for a controlled validation
run: this session's compatibility fixes (in `hunyuan_shared.py` —
`repair_unquantized_bnb_modules`, the `should_convert_module`
prefix/suffix-matching workaround, the bitsandbytes `Linear8bitLt`/
`Linear4bit` device-restore handling, the transformers ≥5.0 `StaticCache`
shims, the block-swap cross-stream race fix, etc.) were verified by
reading the *actual source* of specific library versions, not "whatever
floats to latest at install time." A newer release could change internals
again the same way transformers 5.x did relative to 4.x, and an unpinned
install wouldn't tell you whether a test failure is a real regression or
just a new library version doing something new.

So `install.sh` installs `requirements.txt` + `requirements-test.txt`
constrained by `constraints-tested.txt` — a pip
[constraints file](https://pip.pypa.io/en/stable/user_guide/#constraints-files)
that pins the exact versions this project's fixes were checked against:

```
torch==2.13.0
transformers==5.15.0
accelerate==1.14.0
bitsandbytes==0.50.0
numpy==2.4.6
pillow==12.3.0
pytest==9.1.1
pytest-timeout==2.4.0
```

A constraints file only *constrains* versions of packages that get
installed for some other reason — it can't pull in a package on its own
and it doesn't change what `requirements.txt` itself declares for anyone
installing this project outside of `install.sh`. `torch` itself is
installed as its own separate step (see `TORCH_CUDA_INDEX`/`TORCH_VERSION`
above) rather than solely through the constraints file, since the
CUDA-build/index selection isn't something a plain version pin can
express — the constraints file entry for it mainly guards against
something else in the dependency graph pulling in a different version.

**Provenance**: these exact versions were installed together and
exercised against this project's fast no-GPU test suite in a from-scratch
venv (CPU-only PyTorch build, since that machine had no GPU) — 194 passed,
0 failed — on 2026-08-13. That run also caught and fixed 3 real bugs (2 in
`hunyuan_shared.py`'s `patch_hunyuan_static_cache_device` fallback path, 1
docstring `DeprecationWarning`) that only surfaced when the code actually
ran against this specific transformers/torch combination — see
`CHANGELOG.md`. The GPU-dependent behavior (real CUDA kernels,
bitsandbytes quant state materialization, the block-swap cross-stream fix)
has **not** been exercised — that's what running this install and the
full test suite on real hardware is for.

If you deliberately want to test against a newer combination instead of
the pinned baseline, edit or delete the relevant line(s) in
`constraints-tested.txt` before running `install.sh`, and note that
deviation when reporting results.

## FlashInfer / FlashAttention setup (optional)

`HunyuanInstructLoader`'s `attention_impl`/`moe_impl` widgets support two
alternative backends beyond the defaults (`sdpa`/`eager`). Neither is
installed by `install.sh` unless you opt in — selecting one without the
package installed raises a clear error (`validate_attention_moe_impl()` in
`hunyuan_shared.py`) rather than crashing deep inside `transformers`.

**Deployment note**: this was investigated starting from
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (CUDA 12.8.1 / torch
2.8.0 — this happens to exactly match Tencent's own officially-tested
baseline). `install.sh` reaches CUDA 13.0 by installing it on top of that
base image itself — both the pip side (`TORCH_CUDA_INDEX`/`TORCH_VERSION`
above) and, for FlashInfer, the system CUDA Toolkit side (below). It does
not assume or require a different pre-built base image.

### FlashInfer (`moe_impl=flashinfer`) — recommended path

```bash
INSTALL_FLASHINFER=1 ./install.sh
```

This does two things:

1. **Installs a system-level CUDA 13.0 Toolkit via apt** (Ubuntu-specific
   — `cuda-keyring` + `cuda-toolkit-13-0`). This is *not* the same thing
   as the CUDA-enabled torch wheel installed earlier in the script: a pip
   torch wheel bundles enough CUDA runtime for its own kernels, but
   FlashInfer JIT-compiles its own kernels at first use and needs a real
   `nvcc` + headers matching the installed torch's CUDA version (Tencent's
   own README: *"It is critical that the CUDA version used by PyTorch
   matches the system's CUDA version"*). Requires root (RunPod pods
   commonly run as root already) — skipped with a warning, never fatal to
   the rest of the install, if root/apt/network aren't available. If it
   succeeds, `install.sh` prints the `PATH`/`LD_LIBRARY_PATH` exports to
   add to your shell profile (they only apply within the script's own run
   otherwise).
2. **Installs `flashinfer-python` (default `0.5.0`, matching Tencent's own
   tested baseline)** via pip.

The **first** inference using `moe_impl=flashinfer` takes several extra
minutes (JIT kernel compile for your specific GPU) — cached after that,
not a bug. `moe_impl=flashinfer` also uses more peak VRAM during MoE
dispatch than `eager` (see the tooltip in ComfyUI, and
`hunyuan_loader_clean.py`'s comments on why the other loaders hardcode
`eager`).

### FlashAttention (`attention_impl=flash_attention_2`) — NOT recommended on consumer/workstation Blackwell (SM120)

`install.sh` deliberately does **not** install `flash-attn`. Two reasons:

1. It has no prebuilt PyPI wheel — `pip install flash-attn` compiles from
   source, which can take hours and needs a matching `nvcc`/compiler setup.
2. As of this writing, mainline
   [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
   has **no confirmed support for consumer/workstation Blackwell GPUs**
   (SM120 — RTX PRO 6000, RTX 5090). See
   [#1987](https://github.com/Dao-AILab/flash-attention/issues/1987) and
   [#2307](https://github.com/Dao-AILab/flash-attention/issues/2307). A
   reporter on the exact same GPU model (RTX PRO 6000 Blackwell, CUDA
   12.8, torch 2.8.0) hit `CUDA error (invalid argument)` invoking the
   attention kernels. **Bumping to CUDA 13.0 does not fix this** — it's a
   missing-kernel/codegen gap in flash-attn itself, independent of which
   CUDA toolkit you point it at.

If you want to try it anyway, unofficial community forks add SM120
support as a stopgap (e.g. `roy86/flash-attention_sm120` on GitHub, a
`flash-attn-4-sm120` distribution on Hugging Face) — install manually,
entirely at your own risk; this repo doesn't test or endorse them, and
recommends switching back to upstream once SM120 support actually merges
there.

**How to tell which failure you're looking at**, if you try it:

- A low-level `CUDA error: invalid argument` (or similar) raised from
  inside `flash_attn_func`/`flash_attn.flash_attn_interface` matches the
  known SM120 gap above — expected, not a bug in this repo. Use `sdpa` or
  `moe_impl=flashinfer` instead.
- An `ImportError` means the package/build itself is broken (unrelated to
  SM120) — check the build log.
- A `TypeError`/`AttributeError` from this repo's own code (not
  `flash_attn`) means a bug in `validate_attention_moe_impl()` or its
  wiring in `hunyuan_instruct_nodes.py::load_model()` — worth reporting.
- An OOM is an unrelated VRAM-budget issue — see `vram_reserve_gb`/
  `blocks_to_swap`, not the attention backend.

## Downloading model weights

`install.sh` sets up code, not model weights — the Hunyuan checkpoints are
20-160GB each (see README.md's model table) and you generally want them on
a different volume than the code/venv, especially in a split deployment
(e.g. ComfyUI + custom_nodes on a smaller `/opt` disk, models on a larger
`/workspace` volume — see `extra_model_paths.yaml.example`). Use
`download_models.sh` for that, separately from `install.sh`:

```bash
./download_models.sh list                          # see all available models, sizes, keys
./download_models.sh nf4 instruct-distil-int8       # download specific models by key
./download_models.sh all --skip-legacy              # everything current (skips v1 duplicates)
```

It creates its own small venv (`.venv-download/` by default — just needs
`huggingface_hub`, not torch) so it doesn't need `install.sh` to have run
first, and defaults to downloading into `/workspace/models/{hunyuan,hunyuan_instruct}/`
(override with `$HUNYUAN_MODELS_DIR` or `--models-dir`). Run
`./download_models.sh --help` for the full option list (`--dry-run`,
`--force`, `--token`, etc.).

If your ComfyUI install (from `install.sh`, or an existing one) doesn't
already scan wherever you downloaded models to, copy
`extra_model_paths.yaml.example` to `$COMFYUI_DIR/extra_model_paths.yaml`
(edit `base_path` first if it's not `/workspace`) and restart ComfyUI —
see the comments in that file for what it does and why it's additive
rather than a replacement for ComfyUI's own `models/` folder.

## Troubleshooting

- **`Host GPU driver's max supported CUDA is X, but TORCH_CUDA_INDEX targets Y`**:
  the *host* GPU driver (not this container) doesn't support the CUDA
  version you're targeting — `nvidia-smi`'s reported "CUDA Version" is the
  driver's fixed ceiling, read directly from the driver, independent of
  whatever CUDA toolkit is currently installed in this container (so this
  check is meaningful even before `install.sh` has done anything). On
  RunPod, this is set by which host CUDA version you selected at
  pod-creation time — recreate the pod with a host that supports your
  target, or lower `TORCH_CUDA_INDEX` to match the host you have. This is
  a warning, not a hard failure, so the script keeps going — but torch/
  flashinfer will very likely fail at runtime with `CUDA driver version is
  insufficient for CUDA runtime version` if you ignore it.
- **`torch==X not available on <index>`**: printed as a warning, not a
  failure — the script automatically falls back to installing the latest
  torch on that CUDA index instead. Note this in your report; it means
  the run isn't using the exact pinned version.
- **No `nvidia-smi`**: printed as a warning; the script still installs a
  CUDA-enabled torch build (per `TORCH_CUDA_INDEX`) since it assumes
  you're setting this up on a GPU machine, but GPU tests obviously won't
  work without a driver.
- **Fast test suite fails**: this is the one result that should be very
  surprising — it's designed to be GPU/ComfyUI-independent and passed
  cleanly in the verification run above. A failure here means either a
  real regression or an environment difference worth digging into before
  even getting to the GPU suite.
- **Re-running after a partial/failed run**: safe — just run `./install.sh`
  again. If you want a fully clean slate, delete `$COMFYUI_DIR` (or point
  `COMFYUI_DIR` at a fresh path) first.
