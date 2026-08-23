# Known upstream issues

Documented here, **not fixed** — this is deliberately out of scope for
the current deploy-tooling phase (see `CLAUDE_CODE_BRIEFING.md` §1 and
§7). Bug-fixing is a later, separate, tested phase. Listed here so
they're not rediscovered from scratch, and so a failure that matches
one of these during provisioning/testing is recognized as a known
upstream issue rather than a new regression in this fork's deploy
tooling.

Upstream (`EricRollei/Comfy_HunyuanImage3`) is effectively abandoned:
last release v1.3.0 (Feb 2026), issue creation now restricted. Do not
expect upstream fixes or merges — any real fix has to land in this
fork.

## 1. Quantized param state lost during device moves / block swap

**Issues:** #41, #36, and probably #39.

`shared_mlp.gate_and_up_proj`'s quantized parameter state (`CB`,
`SCB`, `quant_state` — bitsandbytes' NF4/INT8 quantization metadata)
gets lost when the tensor moves devices (e.g. CPU offload / block
swap during low-VRAM generation). The parameter silently reverts to
an unquantized or corrupted state, which manifests as either a crash
deeper in the forward pass or, worse, silently wrong output.

Relevant code: `hunyuan_block_swap.py`, and the `ensure_model_on_device`
/ `skip_quantized_params` handling referenced from
`hunyuan_quantized_nodes.py`'s loader classes (see the
`ensure_model_on_device(cached, torch.device("cuda:0"),
skip_quantized_params=True)` call pattern — the `skip_quantized_params`
flag existing at all is itself evidence this is a known, worked-around-
but-not-fixed problem).

## 2. RAM lifecycle / leak across successive model loads

**Issues:** #43, #35.

Repeated load/unload cycles (e.g. `force_reload=True`, or
`post_action="full_unload"` followed by a fresh load) don't fully
release host RAM. Over many cycles — the kind of thing
`tests/run_all.sh` or interactive ComfyUI use would do routinely —
this can exhaust host memory on a long-running instance.

Relevant code: `hunyuan_memory_manager.py`, `hunyuan_memory_budget.py`,
`HunyuanModelCache` (referenced from `hunyuan_quantized_nodes.py`).

## 3. Multi-GPU device mapping never tested by the author

**Issues:** #38, #39.

Multi-GPU support exists in the code path (device_map handling) but
the author has stated it was never actually tested on real multi-GPU
hardware. Treat any multi-GPU behavior as unverified, not just
undocumented.

This deploy-tooling phase doesn't exercise multi-GPU at all — every
script here (`deploy/vast/search.py`'s default filters,
`deploy/provision.sh`'s torch/CUDA stage) targets a single-GPU
instance. Multi-GPU is an explicit non-goal (see
`CLAUDE_CODE_BRIEFING.md` §7).

## How this list should be used

- Before filing a "new" bug during smoke testing or real generation
  runs, check whether it matches one of the three clusters above.
- If a provisioning or smoke-test failure looks like #1 or #2, that's
  useful signal that the run partially worked (far enough to hit a
  known model-code issue) rather than a deploy-tooling failure — worth
  noting in the manifest / test report, not necessarily worth
  debugging further in this phase.
- Do not attempt fixes here. If one of these is genuinely blocking
  forward progress on the deploy-tooling phase itself, stop and say so
  rather than silently starting a fix (per the briefing's explicit
  non-goals).
