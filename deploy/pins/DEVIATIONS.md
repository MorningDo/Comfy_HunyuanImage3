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
| _(none yet — no provisioning run has happened)_ | | | | |

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
