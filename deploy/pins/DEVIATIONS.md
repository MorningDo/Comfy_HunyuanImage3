# Deviations from Tencent's pinned environment

Per the environment policy, resolution order for this project's dependency
set is:

```
Tencent pins  >  ComfyUI needs  >  node-pack needs  >  convenience
```

`deploy/pins/tencent-requirements.txt` is the reference baseline. Whenever
`provision.sh`'s dependency-reconciliation stage has to install a version
newer or otherwise different from what Tencent pins for a given package
(to satisfy ComfyUI itself, this node pack's `requirements.txt`, or a
runtime error), that is a **finding**, not routine housekeeping. Log it
here — don't just let `pip` silently resolve around it.

Every row must be added at the time the deviation was actually needed
(i.e. observed during a real `provision.sh` run), not speculatively.

| Package | Tencent pin | Version actually installed | Reason | Necessary? |
|---|---|---|---|---|
| gradio | `>=4.21.0` (unpinned floor) | not installed at all | Tencent's own optional interactive demo UI — this pipeline only ever drives generation through ComfyUI, never Tencent's gradio demo. Installing it under `--no-deps` (correct for this file's actual exact pins) broke gradio's own large, loosely-bound dependency tree (18 missing transitive deps) and failed `pip check` for a package we don't use. Excluded entirely in `provision.sh`'s `stage_reconcile` (grep -v before install). | yes — confirmed live 2026-08-23: `pip check` was clean once gradio was excluded; installing it (even with full deps) adds real weight/risk for zero functional benefit here. |
| huggingface_hub | `[cli]` (unpinned) | 0.36.2 (from ComfyUI's `requirements.txt`) | Tencent's line has no version pin at all, so this isn't actually a pin override — ComfyUI's own requirements.txt legitimately resolves it. Listed here only because an earlier version of `provision.sh`'s drift-check incorrectly treated it as a frozen pin (a bug in the check, not a real deviation) and failed the whole reconcile stage; the check was fixed to only compare packages Tencent pins with an exact `==`. | n/a — not a real deviation, the check that flagged it was wrong |
| flashinfer-python | `0.5.0` (commented out by Tencent, not installed by default) | 0.5.0 — uncommented, installed by `deploy/provision.sh`'s `stage_acceleration` (best-effort, non-fatal). | Enables `moe_impl="flashinfer"` on `HunyuanInstructLoader` (`hunyuan_instruct_nodes.py`). Installs and imports cleanly (confirmed live 2026-08-23, instance 48500012, RTX PRO 6000 Blackwell). **Confirmed non-functional at runtime**, independent of packaging: the checkpoint's vendored `modeling_hunyuan_image_3.py` calls `flashinfer.fused_moe.cutlass_fused_moe(...)`, which raises `ValueError: Invalid shape of output: expected (4138, 6291456), got torch.Size([4138, 4096])` — an API-convention mismatch between Tencent's modeling code and flashinfer 0.5.0's actual `cutlass_fused_moe` signature, not a GPU/sm_120 support issue as initially suspected. Also confirmed: if the checkpoint's modeling module is imported into a long-running ComfyUI process *before* flashinfer is installed, its own `try: import flashinfer except Exception: flashinfer = None` (broad, silently swallowed) permanently caches `flashinfer = None` for that process's lifetime — restart ComfyUI after installing. Not fixable without patching the vendored model source (out of scope; not vendored into this repo). | no — installing it does not make `moe_impl="flashinfer"` usable; would need Tencent to fix the vendored modeling code or ship a compatible flashinfer version |
| flash-attn | not present in Tencent's file at all | 2.8.3 (`+cu12torch2.8cxx11abiFALSE`, exact GitHub-release wheel matched to the venv's actual Python minor version) — installed by `stage_acceleration`. | Enables `attention_impl="flash_attention_2"` on `HunyuanInstructLoader`. Installs and imports cleanly (confirmed live 2026-08-23). **Confirmed non-functional**, and not a packaging or GPU-support question at all: the checkpoint's vendored `Hunyuan_ATTENTION_CLASSES` dict (`modeling_hunyuan_image_3.py`) only registers `"eager"` and `"sdpa"`, both mapped to the same `HunyuanImage3SDPAAttention` class — `flash_attention_2` was never wired into this model's attention dispatch, so selecting it raises `ValueError: Unsupported attention implementation: flash_attention_2` regardless of whether flash-attn is installed. | no — the checkpoint's own code has no implementation to route to, independent of the package |

## Column guide

- **Package** — the exact pip package name.
- **Tencent pin** — the version (or absence of a pin) from
  `deploy/pins/tencent-requirements.txt`.
- **Version actually installed** — what ended up in the venv, and why (a
  ComfyUI `requirements.txt` constraint, a node-pack `requirements.txt`
  constraint, or a runtime failure that forced an upgrade).
- **Reason** — the specific error or constraint that forced the deviation.
  "It seemed newer" is not a reason; a stack trace, an import error, or a
  cited requirements line is.
- **Necessary?** — `yes` if removing the deviation reproduces a concrete
  failure; `unconfirmed` if suspected but not verified by reverting and
  retesting; never leave blank once a row exists.
