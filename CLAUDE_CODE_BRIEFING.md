# Briefing: Comfy_HunyuanImage3 fork — reproducible vast.ai deployment

## 0. Read this first

You are working in a local sandbox on a fork of a ComfyUI custom-node pack.
Your job **in this phase is not to fix bugs**. It is to build a deterministic,
repeatable installer that produces a *working, conservative* environment on a
rented vast.ai GPU instance, plus the tooling to drive that instance and prove
an image was actually generated.

Bug fixing comes later, as a separate, tested phase. Resist scope creep
aggressively — a previous attempt at this project failed precisely because too
many changes landed too fast without isolating what worked.

### Non-negotiable working rules

1. **One concern per commit.** Every commit must be independently revertible
   and must state in its message what was verified and how.
2. **Never edit the remote instance without syncing back.** If you fix
   something interactively over SSH, the fix is not done until the equivalent
   change is in the local deploy scripts and committed. Treat the remote box as
   *disposable*; the scripts are the source of truth.
3. **Work on branches, PR into `main`.** `main` must always be a state that
   provably deployed successfully at least once.
4. **No version drift.** Pins are the point of this exercise (see §2).
5. **Record provenance.** Every successful deploy writes a manifest
   (`pip freeze`, driver/CUDA version, GPU model, git SHAs, vast offer ID) into
   the repo under `deploy/manifests/`. Commit it.
6. **Ask before spending.** Do not create a vast.ai instance without explicit
   confirmation of the instance type and hourly price. Always destroy instances
   when done and confirm destruction.

---

## 1. Repository context

- Fork: `MorningDo/Comfy_HunyuanImage3`
- Upstream: `EricRollei/Comfy_HunyuanImage3` — effectively abandoned. Last
  release v1.3.0 (Feb 2026); six open bug reports; issue creation now
  restricted. Do not expect upstream fixes or merges.
- Upstream ships a `requirements.txt` with **upgraded** dependencies relative to
  Tencent's original `HunyuanImage-3.0/requirements.txt`. Those upgrades are a
  primary source of breakage. We are deliberately reverting to Tencent's
  environment as the baseline.
- Known upstream bug clusters (for later, **not now**):
  - Quantized param state (`CB`/`SCB`/`quant_state`) lost on
    `shared_mlp.gate_and_up_proj` during device moves and block swap
    (issues #41, #36, and probably #39).
  - RAM lifecycle / leak across successive loads (#43, #35).
  - Multi-GPU device mapping (#38, #39) — never tested by the author.
  Note these in a `KNOWN_ISSUES.md` and leave them alone for now.

---

## 2. Environment policy

The environment is built in layers, most-authoritative first:

1. **Tencent's `requirements.txt` for HunyuanImage-3.0** is the reference. Fetch
   it from the official repo, vendor a copy into
   `deploy/pins/tencent-requirements.txt`, and record the commit SHA it came
   from.
2. **PyTorch + CUDA** are chosen to match the rented GPU, not pinned blindly.
   Record the exact wheel index URL used. This is the one place where a
   deviation from Tencent's pins is expected and acceptable — document it.
3. **ComfyUI** is pinned to a specific commit SHA, not a branch.
4. **Node requirements** (`Comfy_HunyuanImage3/requirements.txt`) are
   *reconciled against* Tencent's pins, not merged over them. Where the node
   pack demands something newer, that is a finding: log it in
   `deploy/pins/DEVIATIONS.md` with the reason and whether it was necessary.
5. `bitsandbytes`, `transformers`, `accelerate`, `huggingface-hub` are the
   historical troublemakers. Pin all four explicitly and loudly.

Install with `pip install --no-deps` where a pinned set is already resolved, so
pip cannot silently upgrade a pinned package as a transitive dependency. Run a
`pip check` afterwards and treat conflicts as build failures, not warnings.

---

## 2a. Instance image selection

Pick the image at instance-creation time; pin it in `deploy/vast/config.sh` and
treat it as part of the environment spec. A changed base image is an
environment change and must be recorded in a manifest.

**Default choice: `vastai/base-image`, a CUDA *devel* + cuDNN tag on Ubuntu
22.04 or 24.04** — e.g. `vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04`.
Verify the exact tag exists before use rather than trusting this string.

Rationale:

- Vast hosts cache layers of common base images (`nvidia/cuda`, `vastai/*`), so
  cold boots are fast despite the size. A random third-party image will be
  pulled cold and you'll pay for the wait.
- **`devel`, not `runtime`.** You need `nvcc` and the CUDA headers: bitsandbytes,
  and anything later involving custom attention kernels, will not build against
  a runtime-only image. Choosing `runtime` to save disk is a false economy that
  surfaces as a confusing compile error an hour in.
- It ships a clean venv at `/venv/main` and does **not** pre-install torch,
  which matters — see §2b. An image with a pre-baked torch (`vastai/pytorch`,
  ComfyUI-branded templates) forces you to either accept their version or fight
  it, and both are worse than starting empty.
- CUDA minor-version compatibility means a 12.x image runs on any host with a
  12.x driver (>= 525). The 12.x and 13.x families are *not* interchangeable, so
  the search filter in `deploy/vast/search.py` must filter on host driver
  version, not just GPU model. If you later target Blackwell-class cards with
  PTX-compiled kernels, the compatibility guarantee weakens — note it and
  re-verify rather than assume.

**Do not use a pre-built ComfyUI template.** They install their own ComfyUI,
their own Python deps, and their own launch supervision, all of which we then
have to detect and undo. We want an empty box.

Explicitly record in the manifest: image name + tag + digest, host driver
version, `nvcc --version`, GPU model and VRAM.

## 2b. System vs repository package versions

This is where the previous attempt most likely went subtly wrong, so treat it
as a first-class hazard rather than housekeeping.

**The core rule: nothing installs into system Python. Ever.** Create or use a
dedicated venv, activate it in every script, and make `provision.sh` fail loudly
if `which python` resolves outside it. A silent fallback to system Python
produces an environment that half-works and is nearly impossible to diagnose
later, because two interpreters each hold a partial, different dependency set.

Specific hazards and how to handle them:

1. **Distro-packaged Python libs shadowing pip ones.** Ubuntu ships
   `python3-numpy`, `python3-yaml` and friends. If the venv is created with
   `--system-site-packages`, these leak in and can win over your pinned
   versions. **Create the venv without system site packages.** If something
   genuinely needs a system library, install the *native* library
   (`libgl1`, `libglib2.0-0`, `ffmpeg`) — never the Python binding.

2. **CUDA toolkit: system vs pip wheel.** The image provides a system CUDA
   toolkit; torch wheels bundle their own CUDA runtime; `nvidia-*` pip packages
   provide yet another. These can coexist but the resolution order matters.
   Prefer torch's bundled runtime for torch itself, and only rely on the system
   toolkit for *compilation* (`nvcc`). If you find yourself setting
   `LD_LIBRARY_PATH` to make something work, that is a finding — write it down
   in `DEVIATIONS.md`, don't just bury it in a script.

3. **The driver is the one thing you cannot change.** Everything else is
   negotiable; the host's NVIDIA driver is fixed. Read it first with
   `nvidia-smi`, and derive the CUDA/torch choice from it — never the other way
   round. If the driver is too old for the pinned torch, the correct action is
   to pick a different vast offer, not to downgrade the pins.

4. **Tencent pins vs what the image already contains.** Tencent's
   `requirements.txt` will specify versions older than whatever the image or
   ComfyUI's own `requirements.txt` wants. Resolution order:
   Tencent pins > ComfyUI needs > node-pack needs > convenience.
   Install in that order, and after each layer run a check that the previously
   pinned versions are still installed. `pip` will happily upgrade a pinned
   package as a transitive dependency of a later install and say nothing about
   it — this is the single most likely cause of "it worked yesterday."

5. **Verify, don't assume.** After the full install, dump `pip freeze` and diff
   it against the intended pin set programmatically. Any package that moved is
   a build failure. Add this as `deploy/verify_env.sh` and run it at the end of
   `provision.sh` and again before any test run.

6. **ComfyUI's own dependency churn.** ComfyUI pulls its requirements at a
   pinned SHA, but its `requirements.txt` is unpinned in places and will resolve
   to whatever is newest that day. Snapshot the resolved set on the first
   successful build and pin from the snapshot thereafter, so a rebuild next
   month gets the same environment.

7. **Interactive fixes on the instance.** When you fix a dependency by hand over
   SSH, immediately: record the exact command, add it to the correct layer of
   `provision.sh`, note *why* in `DEVIATIONS.md`, and re-verify with
   `verify_env.sh`. The instance is disposable; an undocumented hand-fix is a
   guaranteed future mystery.

---

## 3. Deliverables for this phase

Create these under `deploy/` in the fork. Bash preferred for anything that must
run on a bare instance; Python is fine for the host-side tooling.

### 3.1 Provisioning

- `deploy/provision.sh` — runs *on the instance*. Idempotent. Stages:
  1. system packages, verify NVIDIA driver + `nvidia-smi`
  2. Python env creation (venv; do not rely on the image's system Python)
  3. torch install matched to detected GPU/CUDA
  4. ComfyUI clone at pinned SHA
  5. custom node install from *this fork* at the current SHA
  6. dependency reconciliation per §2, then `pip check`
  7. write manifest, print a clear PASS/FAIL summary

  Every stage must be independently re-runnable and must skip cleanly if already
  complete. Log to `/var/log/hy3-provision.log` **and** stdout.

- `deploy/fetch_models.sh` — model download, separate from provisioning and
  separately re-runnable. Downloads must be idempotent (verify size/hash before
  re-pulling; these are 45–160 GB artifacts and vast bandwidth is billed).
  Default target: the pre-quantized NF4 v2 Instruct-Distil variant unless told
  otherwise — smallest thing that exercises the instruct path.
  Models go on persistent storage, never in the container layer.

### 3.2 vast.ai control

- `deploy/vast/search.py` — query offers, filter by GPU model, VRAM, disk,
  bandwidth, price ceiling, and reliability. Print a ranked table. Never
  auto-select.
- `deploy/vast/up.sh` — create instance from a chosen offer, wait for SSH,
  write connection details to `deploy/vast/.current-instance` (gitignored).
- `deploy/vast/down.sh` — destroy, with confirmation and a final billing note.
- `deploy/vast/ssh.sh` — wrapper that reads `.current-instance` so no one is
  hand-typing IPs and ports.
- `deploy/vast/sync.sh` — push local repo state to the instance (rsync over
  SSH, excluding models and venv). This is how iteration happens: edit locally,
  sync, re-run the relevant provision stage.

### 3.3 Network posture

**Only port 22 is exposed on the instance. No exceptions.** ComfyUI binds to
`127.0.0.1` on the instance and is reached exclusively via SSH local port
forwarding.

- `deploy/vast/tunnel.sh` — establishes `-L 8188:127.0.0.1:8188` (plus any
  other needed forwards), runs in the foreground with clear status output, and
  handles reconnection. Getting this working is part of "done" for every
  deploy, every time — Sam will be using the ComfyUI web UI interactively
  through this tunnel.
- ComfyUI must be started under a process manager or `tmux`/`systemd` unit so
  it survives SSH disconnection. Provide `deploy/comfy_start.sh`,
  `comfy_stop.sh`, `comfy_logs.sh`.

### 3.4 Validation harness

This is the part that previously had no good answer. Unit tests are near-
useless here; what matters is *did an image come out, and does it resemble the
prompt*.

- `tests/smoke/` — cheap, fast, run first:
  - imports resolve, nodes register in ComfyUI without error
  - model loads and reports expected quantization type
  - a minimal generation at the smallest supported resolution returns a
    non-degenerate image (not blank, not uniform noise — check pixel variance
    and entropy before spending money on a vision model)
- `tests/prompts/cases.yaml` — a small, stable set of prompt→expectation cases.
  Expectations should be *checkable assertions* ("contains a red car",
  "contains exactly two people", "is a night scene"), not vibes.
- `tests/validate_image.py` — sends the generated image plus its assertion list
  to a vision model via OpenRouter, requests a strict JSON verdict per
  assertion (`{"assertion": ..., "pass": bool, "confidence": float,
  "reasoning": ...}`), and exits non-zero on failure.
  - Prompt the model for JSON only, no prose, no code fences; parse defensively.
  - Cache verdicts keyed by image hash + assertion so re-runs are free.
  - Treat this as *advisory but recorded*: the judge is noisy. Log every
    verdict with the image, so a regression can be reviewed by eye later.
  - Keep a golden set: images that previously passed, re-scored occasionally to
    detect judge drift.
- `tests/run_all.sh` — orchestrates the above and emits a single report into
  `deploy/manifests/`.

Model choice for the judge: pick a cheap, capable vision model and pin the
model string in config. Do not silently change judges between runs — a changed
judge invalidates comparison against previous results.

---

## 4. Definition of done for this phase

A clean run of:

```
deploy/vast/search.py  →  (human picks an offer)
deploy/vast/up.sh
deploy/vast/sync.sh
ssh → deploy/provision.sh
ssh → deploy/fetch_models.sh
deploy/vast/tunnel.sh
tests/run_all.sh
```

...on a **freshly created instance**, with no manual intervention beyond
choosing the offer, ending in a passing validation report and a working
browser-accessible ComfyUI over the tunnel.

Then, critically: **destroy the instance, create a new one, and do it again.**
If the second run needs any manual step the first run didn't, the scripts are
not done.

---

## 5. Interaction protocol with Sam

- Report at stage boundaries, not continuously.
- When something fails on the instance, state: what failed, the exact error,
  your hypothesis, and the *smallest* change that would test it. Then make that
  change locally, sync, retest. Do not fix five things at once.
- If you find yourself unable to explain why something started working, stop
  and say so. An unexplained fix is a future subtle bug.
- Flag anything that costs money before doing it.

---

## 6. Credentials to expose in the sandbox

Provide these as environment variables. None should ever be written into the
repo, into a manifest, or onto the instance except where noted.

| Variable | Purpose | Notes |
|---|---|---|
| `VAST_API_KEY` | create/destroy/query vast.ai instances | Full account control — treat as high-value. |
| `OPENROUTER_API_KEY` | vision-model image validation | Set a spend limit on the key. Used **host-side only**; never copy to the instance. |
| `HF_TOKEN` | Hugging Face model downloads | Read-only token. Needed on the instance for `huggingface-cli`; inject at download time via env, don't bake into an image. |
| `GITHUB_TOKEN` | push branches, open PRs on your fork | Fine-grained PAT scoped to `MorningDo/Comfy_HunyuanImage3` only, contents + PR write. |

Actual values live in `.env` at the repo root (git-ignored, never committed).

SSH keypair for connecting to the vast instance: private key at
`deploy/vast/keys/id_ed25519` (git-ignored, chmod 600), public key at
`deploy/vast/keys/id_ed25519.pub` (already registered with the vast.ai
account as key fingerprint for user `computer`).
Plus SSH: an SSH keypair for vast (public key registered with your vast
account, private key available in the sandbox). Use a **dedicated** key for
this, not your general-purpose one, so it can be revoked independently.

Optional, only if you decide you want them:

- `DISCORD_WEBHOOK` / similar, for long-running deploy notifications.
- A container registry token if you later layer a Docker image — but note the
  intended split is code/deps in the image, model weights on persistent
  storage, and that's a later phase.

Deliberately **not** exposed: anything giving write access to upstream, and any
key with billing/account-modification power beyond vast itself.

---

## 7. Explicit non-goals for this phase

- Fixing the six open upstream issues.
- Docker image layering.
- Multi-GPU support.
- Performance tuning, SageAttention, custom kernels.
- Any change to node behaviour beyond what is strictly required to make the
  pinned environment import and run.

If you believe one of these is blocking, say so and stop — don't proceed into it.
