# PyTorch / CUDA selection notes

This is the one deliberate, documented deviation point from "just use
Tencent's pins": PyTorch and its CUDA build are chosen to match the
*rented GPU and host driver*, not pinned blindly in
`deploy/pins/tencent-requirements.txt` (which itself only comments out a
suggested `torch==2.8.0` / cu128 install rather than hard-pinning it).

## Resolution order (see `deploy/provision.sh` stage 1 and stage 3)

1. Read the host's NVIDIA driver version with `nvidia-smi` first. The
   driver is the one thing that cannot be changed on a rented instance —
   everything else is negotiable.
2. Derive the CUDA major.minor family the driver supports (12.x vs 13.x
   are **not** interchangeable — a 12.x wheel needs a 12.x-driver host,
   generally driver >= 525 for CUDA 12).
3. Pick the torch wheel + `--index-url` accordingly. Tencent's suggested
   starting point is CUDA 12.8:
   `pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128`
   — use this unless the actual detected driver requires otherwise.
4. If the driver can't support the pinned torch/CUDA combination, the
   correct fix is picking a different vast.ai offer (a host with a newer
   driver), **not** downgrading the torch pin to fit a bad offer.
5. Record the exact wheel index URL and resolved torch/CUDA version used
   in the deploy manifest (see `deploy/verify_env.sh` / manifest schema).

## System vs. wheel-bundled CUDA

- Prefer torch's own bundled CUDA runtime for running torch itself.
- Only rely on the system CUDA toolkit (`nvcc`, headers) for
  *compilation* — this is why the base image must be a `devel` tag, not
  `runtime` (see `deploy/vast/config.sh`): `bitsandbytes` and any custom
  attention kernels need `nvcc` to build.
- If `LD_LIBRARY_PATH` ever needs to be set to make something resolve
  correctly, that is a finding for `deploy/pins/DEVIATIONS.md`, not
  something to bury silently in a script.

## Base image

Default: `vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py311`
(pinned in `deploy/vast/config.sh`) — confirmed to exist via a direct
Docker Hub API tag lookup (digest `sha256:cf12789c...`) after the
original date-suffixed tag turned out not to exist (a WebFetch
research artifact from phase-1 planning, caught by a real failed
`up.sh` run 2026-08-23 — the container runtime returned "manifest
unknown"). `devel`, not `runtime`; no pre-baked torch; no pre-built
ComfyUI template (see rationale in `CLAUDE_CODE_BRIEFING.md` §2a).
