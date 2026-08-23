# Deploying to vast.ai

Reproducible, pinned deployment tooling for running this fork's
ComfyUI custom nodes (Comfy_HunyuanImage3, wrapping Tencent's
HunyuanImage-3.0) on a rented vast.ai GPU instance. See
`CLAUDE_CODE_BRIEFING.md` for the full design rationale and
`KNOWN_ISSUES.md` for upstream bugs this tooling deliberately does not
fix.

## Credentials

Real values live in a git-ignored `.env` at the repo root (never
commit this) plus a git-ignored SSH keypair at `deploy/vast/keys/`.

| Variable | Used by | Notes |
|---|---|---|
| `VAST_API_KEY` | `deploy/vast/search.py`, `up.sh`, `down.sh` | Full account control — high value. |
| `OPENROUTER_API_KEY` | `tests/validate_image.py` | **Host-side only.** Never copy to the instance, never let it end up in a synced repo state or an on-instance env var. |
| `HF_TOKEN` | `deploy/fetch_models.sh` (via `huggingface-cli`) | Read-only token, injected via env at download time. |
| `GITHUB_TOKEN` | manual `git push` / PR creation | Not read by any script here. |

All four are also referenced by name (not value) in
`CLAUDE_CODE_BRIEFING.md`'s credentials table. If you ever see a raw
key/token in a diff, stop and scrub it before committing — see that
file's git history for how the original exposure was handled
(`git log --oneline -- CLAUDE_CODE_BRIEFING.md`).

## Definition-of-done pipeline

```
deploy/vast/search.py      # (human picks an offer — read-only, no cost)
deploy/vast/up.sh --offer-id <ID>   # asks for confirmation before creating
deploy/vast/sync.sh
ssh (deploy/vast/ssh.sh) → deploy/provision.sh
ssh → deploy/fetch_models.sh
deploy/vast/tunnel.sh       # foreground, -L 8188:127.0.0.1:8188
tests/run_all.sh
```

A clean run of all of the above on a **freshly created instance**,
with no manual intervention beyond picking the offer, ending in a
passing `tests/run_all.sh` report and a working browser-accessible
ComfyUI over the tunnel, is "done" for a given environment change.
Then: destroy the instance (`deploy/vast/down.sh`), create a new one,
and do it again — if the second run needs any manual step the first
didn't, the scripts aren't done yet.

Every real provisioning run writes a manifest to `deploy/manifests/`
(via `deploy/provision.sh`'s final stage) and every `tests/run_all.sh`
run writes a test report to the same directory — both are committed,
not gitignored, since they're the provenance record this whole
exercise exists to produce.

## Quick reference

| Script | Runs on | Purpose |
|---|---|---|
| `deploy/vast/search.py` | host | Find a GPU offer. Read-only. |
| `deploy/vast/up.sh` | host | Create an instance. Costs money — confirms first. |
| `deploy/vast/down.sh` | host | Destroy an instance. Confirms first. |
| `deploy/vast/ssh.sh` | host | SSH wrapper, reads `.current-instance`. |
| `deploy/vast/sync.sh` | host | rsync local → instance. |
| `deploy/vast/tunnel.sh` | host | Foreground SSH tunnel to ComfyUI. |
| `deploy/provision.sh` | instance | Staged, idempotent environment setup. |
| `deploy/verify_env.sh` | instance | Pin-drift check against Tencent's requirements. |
| `deploy/fetch_models.sh` | instance | Idempotent, revision-pinned model download. |
| `deploy/comfy_start.sh` / `_stop.sh` / `_logs.sh` | instance | tmux-based ComfyUI process control. |
| `tests/smoke/*` | instance | Cheap, fast, GPU-dependent sanity checks. |
| `tests/validate_image.py` | **host** | Vision-model judge — needs `OPENROUTER_API_KEY`. |
| `tests/run_all.sh` | host (drives instance over SSH) | Orchestrates the above into one report. |

## Environment policy (summary)

Pin resolution order: **Tencent pins > ComfyUI needs > node-pack needs
> convenience** (`deploy/pins/`). PyTorch/CUDA is the one deliberate
exception — matched to the rented GPU's driver, not blindly pinned
(`deploy/pins/torch-cuda-notes.md`). Any package version that has to
diverge from Tencent's pin to satisfy a later layer is a finding,
logged in `deploy/pins/DEVIATIONS.md`, not silently absorbed.

Base image: `vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py311`
(`deploy/vast/config.sh`) — `devel`, not `runtime`; no pre-baked torch;
no pre-built ComfyUI template. Only port 22 is ever exposed on the
instance; ComfyUI binds `127.0.0.1` and is reached exclusively through
`deploy/vast/tunnel.sh`.

## Open items

- **`main` has not yet been proven to deploy end-to-end.** Every
  script in this tooling has been built and locally verified
  (syntax/lint clean, exercised against fixtures and mocked
  ssh/scp/rsync/HTTP — see each commit's message for what was actually
  tested), but **no real vast.ai instance has been created this
  phase** — that was explicitly out of scope for this pass (no
  spending without a separate, explicit go-ahead on a specific offer
  and price). The first real run is still ahead: `search.py` → pick an
  offer → confirm the price → `up.sh` → the rest of the pipeline above
  → `down.sh`. Until that's happened once successfully, treat every
  script here as "believed correct, not yet field-proven."
- `deploy/pins/DEVIATIONS.md` is currently empty (no real provisioning
  run has happened to populate it).
- Bug-fixing the three clusters in `KNOWN_ISSUES.md` is a deliberately
  separate, later phase.
