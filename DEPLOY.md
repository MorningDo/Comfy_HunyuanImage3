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
ssh (deploy/vast/ssh.sh) → deploy/provision.sh   # all 9 stages, no --stage filter
ssh → deploy/fetch_models.sh
ssh → deploy/comfy_start.sh   # ComfyUI itself; provision.sh only sets it up
deploy/vast/tunnel.sh       # foreground, -L 8188:127.0.0.1:8188
tests/run_all.sh            # smoke checks bypass the ComfyUI server entirely
                             # (direct node calls), so this alone doesn't
                             # prove comfy_start.sh/tunnel.sh work — see below
mcp/setup.sh                # host-side: clone/install comfyui-mcp-server + workflows
mcp/start.sh                # host-side: needs the tunnel already up
```

A clean run of all of the above on a **freshly created instance**,
with no manual intervention beyond picking the offer, ending in a
passing `tests/run_all.sh` report, a working browser-accessible
ComfyUI over the tunnel, AND a real generation driven through the MCP
server (`mcp__comfyui-mcp-server__run_workflow` with
`workflow_id="hunyuan_instruct_generate"`, which exercises
`comfy_start.sh` + `tunnel.sh` + the `extra_model_paths.yaml`
registration together — none of which `tests/run_all.sh` alone
proves, since its smoke checks call the node classes directly and
never touch the running ComfyUI server) — is "done" for a given
environment change. Then: destroy the instance (`deploy/vast/down.sh`),
create a new one, and do it again — if the second run needs any manual
step the first didn't, the scripts aren't done yet. First proven
2026-08-23 (instance 48487646); second-instance repeatability proof
tracked in `deploy/manifests/`.

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
| `deploy/provision.sh` | instance | Staged, idempotent environment setup (includes registering `extra_model_paths.yaml` so Hunyuan checkpoints under this repo's `models/` are visible in ComfyUI's own loader-node dropdowns, not just to scripts that bypass folder_paths). |
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

## Driving ComfyUI via MCP

`mcp/` sets up [joenorton/comfyui-mcp-server](https://github.com/joenorton/comfyui-mcp-server)
(pinned commit, `mcp/pins/comfyui-mcp-server-commit.txt`) so an MCP
client (Claude Code or similar) can call `generate_image`, `list_models`,
`run_workflow`, etc. directly instead of driving the ComfyUI web UI by
hand. Runs on the **host**, not the instance — it talks to ComfyUI's
HTTP API at `http://localhost:8188`, which `deploy/vast/tunnel.sh`
already forwards.

```
deploy/vast/tunnel.sh   # must be running first — MCP server needs 127.0.0.1:8188
mcp/setup.sh            # clone (pinned commit) + venv + deps, one-time / idempotent
mcp/start.sh            # tmux session, survives disconnection
```

`.mcp.json` at the repo root registers the server (`http://127.0.0.1:9000/mcp`,
streamable-http) for any MCP client opened against this repo.
**Claude Code loads MCP servers at session start** — if you add/start
the server mid-session, tools won't appear until you restart or
reconnect.

`requirements.txt` pins only `mcp>=0.9.0` (no upper bound); `mcp/setup.sh`
force-pins `mcp==1.26.0` on top of it — `mcp 2.0.0` (released months
after this repo's pinned commit) is a breaking major version that
removes `mcp.server.fastmcp`, which `server.py` imports directly.
Confirmed live: the unpinned install crashed on startup with
`ModuleNotFoundError`.

## Open items

- `deploy/pins/DEVIATIONS.md` has two real entries from the first
  live run (see below) — check it before assuming Tencent's pins hold
  exactly.
- Bug-fixing the three clusters in `KNOWN_ISSUES.md` is a deliberately
  separate, later phase.
- The full "destroy, recreate, repeat with zero manual steps" second
  proof from the original definition of done hasn't happened yet —
  only one successful run so far (2026-08-23, instance 48487646).
