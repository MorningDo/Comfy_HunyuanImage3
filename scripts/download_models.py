#!/usr/bin/env python3
"""Download HunyuanImage-3.0 model weights from Hugging Face.

Standalone tool — deliberately has no dependency on torch/transformers/this
repo's own Python package, so it can run in its own small venv (see
../download_models.sh) instead of the much heavier ComfyUI/model venv. Just
needs `huggingface_hub` (and optionally `hf_transfer` for faster parallel
downloads — used automatically if installed).

Usage:
    download_models.py list
    download_models.py nf4 instruct-distil-int8
    download_models.py all [--skip-legacy] [--yes]

Run `download_models.py --help` for all options. See INSTALL.md and
extra_model_paths.yaml.example for how this fits into a ComfyUI-in-/opt,
models-in-/workspace deployment.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from typing import Optional


@dataclasses.dataclass(frozen=True)
class ModelEntry:
    key: str
    repo_id: str
    category: str  # "hunyuan" (base/T2I) or "hunyuan_instruct"
    local_dir_name: str
    approx_size_gb: float
    legacy: bool = False
    recommended: bool = False
    notes: str = ""


# Kept in sync with README.md's "Available Models on Hugging Face" table and
# download commands. approx_size_gb values are the same ballpark figures
# quoted there — actual on-disk size will vary a bit by repo revision.
CATALOG: list[ModelEntry] = [
    ModelEntry(
        "full-bf16", "tencent/HunyuanImage-3.0", "hunyuan", "HunyuanImage-3",
        80, notes="Tencent original, full precision. Needs ~80GB+ VRAM.",
    ),
    ModelEntry(
        "nf4", "EricRollei/HunyuanImage-3-NF4-v2", "hunyuan",
        "HunyuanImage-3-NF4-v2", 20, recommended=True,
        notes="v2, recommended for <96GB single-GPU setups.",
    ),
    ModelEntry(
        "nf4-v1", "EricRollei/HunyuanImage-3-NF4-ComfyUI", "hunyuan",
        "HunyuanImage-3-NF4-ComfyUI", 20, legacy=True,
    ),
    ModelEntry(
        "int8", "EricRollei/HunyuanImage-3-INT8-v2", "hunyuan",
        "HunyuanImage-3-INT8-v2", 85, recommended=True,
        notes="v2, higher fidelity than NF4.",
    ),
    ModelEntry(
        "int8-v1", "EricRollei/Hunyuan_Image_3_Int8", "hunyuan",
        "Hunyuan_Image_3_Int8", 85, legacy=True,
    ),
    ModelEntry(
        "instruct-distil-int8", "EricRollei/HunyuanImage-3.0-Instruct-Distil-INT8-v2",
        "hunyuan_instruct", "HunyuanImage-3.0-Instruct-Distil-INT8-v2", 81,
        recommended=True, notes="8-step fast inference. Recommended for 96GB GPUs.",
    ),
    ModelEntry(
        "instruct-distil-nf4", "EricRollei/HunyuanImage-3.0-Instruct-Distil-NF4-v2",
        "hunyuan_instruct", "HunyuanImage-3.0-Instruct-Distil-NF4-v2", 45,
        recommended=True, notes="8-step fast inference. Best for 48GB GPUs.",
    ),
    ModelEntry(
        "instruct-int8", "EricRollei/HunyuanImage-3.0-Instruct-INT8-v2",
        "hunyuan_instruct", "HunyuanImage-3.0-Instruct-INT8-v2", 81,
    ),
    ModelEntry(
        "instruct-nf4", "EricRollei/HunyuanImage-3.0-Instruct-NF4-v2",
        "hunyuan_instruct", "HunyuanImage-3.0-Instruct-NF4-v2", 45,
        notes="50-step, highest quality of the NF4 Instruct variants.",
    ),
    ModelEntry(
        "instruct-distil-int8-v1", "EricRollei/HunyuanImage-3.0-Instruct-Distil-INT8",
        "hunyuan_instruct", "HunyuanImage-3.0-Instruct-Distil-INT8", 81, legacy=True,
    ),
    ModelEntry(
        "instruct-distil-nf4-v1", "EricRollei/HunyuanImage-3.0-Instruct-Distil-NF4",
        "hunyuan_instruct", "HunyuanImage-3.0-Instruct-Distil-NF4", 45, legacy=True,
    ),
    ModelEntry(
        "instruct-int8-v1", "EricRollei/HunyuanImage-3.0-Instruct-INT8",
        "hunyuan_instruct", "HunyuanImage-3.0-Instruct-INT8", 81, legacy=True,
    ),
    ModelEntry(
        "instruct-nf4-v1", "EricRollei/HunyuanImage-3.0-Instruct-NF4",
        "hunyuan_instruct", "HunyuanImage-3.0-Instruct-NF4", 45, legacy=True,
    ),
    ModelEntry(
        "instruct-bf16", "tencent/HunyuanImage-3.0-Instruct",
        "hunyuan_instruct", "HunyuanImage-3.0-Instruct", 160,
        notes="Tencent original, full precision. 50-step.",
    ),
    ModelEntry(
        "instruct-distil-bf16", "tencent/HunyuanImage-3.0-Instruct-Distil",
        "hunyuan_instruct", "HunyuanImage-3.0-Instruct-Distil", 160,
        notes="Tencent original, full precision. 8-step distilled.",
    ),
]

CATALOG_BY_KEY = {entry.key: entry for entry in CATALOG}
assert len(CATALOG_BY_KEY) == len(CATALOG), "duplicate model key in CATALOG"

DEFAULT_MODELS_DIR = os.environ.get("HUNYUAN_MODELS_DIR", "/workspace/models")


def print_catalog() -> None:
    print(f"{'KEY':<26} {'CATEGORY':<17} {'~SIZE':>7}  NOTES")
    print("-" * 100)
    for entry in CATALOG:
        tags = []
        if entry.recommended:
            tags.append("recommended")
        if entry.legacy:
            tags.append("legacy")
        notes = entry.notes
        if tags:
            notes = f"[{', '.join(tags)}] {notes}".strip()
        print(f"{entry.key:<26} {entry.category:<17} {entry.approx_size_gb:>5.0f}GB  {notes}")
    print()
    print(f"Default destination base dir: {DEFAULT_MODELS_DIR}")
    print("  (override with --models-dir or $HUNYUAN_MODELS_DIR)")
    print()
    print("Usage: download_models.py <key> [<key> ...] | all   [options]")


def resolve_targets(requested: list[str], *, skip_legacy: bool) -> list[ModelEntry]:
    if [r.lower() for r in requested] == ["all"]:
        entries = list(CATALOG)
        if skip_legacy:
            entries = [e for e in entries if not e.legacy]
        return entries

    unknown = [r for r in requested if r not in CATALOG_BY_KEY]
    if unknown:
        valid = ", ".join(sorted(CATALOG_BY_KEY))
        raise SystemExit(
            f"Unknown model key(s): {', '.join(unknown)}\n\nValid keys:\n  {valid}\n\n"
            "(or 'all'; run with 'list' to see sizes/notes)"
        )

    seen: dict[str, ModelEntry] = {}
    for r in requested:
        seen[r] = CATALOG_BY_KEY[r]
    return list(seen.values())


def maybe_enable_hf_transfer() -> bool:
    try:
        import hf_transfer  # noqa: F401
    except ImportError:
        return False
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    return True


def download_one(
    entry: ModelEntry,
    *,
    models_dir: str,
    token: Optional[str],
    revision: Optional[str],
    max_workers: int,
    force: bool,
    dry_run: bool,
) -> bool:
    dest = os.path.join(models_dir, entry.category, entry.local_dir_name)
    print(f"\n--- {entry.key} ({entry.repo_id}) -> {dest}")

    if dry_run:
        print("  (dry run — not downloading)")
        return True

    os.makedirs(dest, exist_ok=True)

    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=entry.repo_id,
            local_dir=dest,
            revision=revision,
            token=token,
            max_workers=max_workers,
            force_download=force,
        )
    except Exception as exc:  # noqa: BLE001 - one bad repo shouldn't abort the batch
        print(f"  FAILED: {exc}", file=sys.stderr)
        return False

    print(f"  done: {dest}")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download HunyuanImage-3.0 model weights from Hugging Face.",
        epilog="Run 'download_models.py list' to see all available keys, sizes, and notes.",
    )
    parser.add_argument(
        "targets", nargs="*",
        help="Model key(s) to download, or 'all'. See 'list' for valid keys.",
    )
    parser.add_argument(
        "--models-dir", default=DEFAULT_MODELS_DIR,
        help=f"Base directory models are downloaded under, as <models-dir>/<category>/<name>. "
             f"Default: {DEFAULT_MODELS_DIR} (or $HUNYUAN_MODELS_DIR)",
    )
    parser.add_argument("--skip-legacy", action="store_true",
                         help="When downloading 'all', skip legacy v1 duplicates of v2 models.")
    parser.add_argument("-y", "--yes", action="store_true",
                         help="Don't prompt for confirmation before a large/multi-model download.")
    parser.add_argument("--force", action="store_true",
                         help="Re-verify/re-download files even if they already match locally "
                              "(passed through as huggingface_hub's force_download).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print what would be downloaded and where, without downloading.")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                         help="Hugging Face access token. Default: $HF_TOKEN, or the token cached "
                              "by `huggingface-cli login` if neither is set.")
    parser.add_argument("--revision", default=None,
                         help="Git revision/branch/tag to download (applies to every selected model).")
    parser.add_argument("--max-workers", type=int, default=8,
                         help="Parallel file downloads per model (default: 8).")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.targets or args.targets == ["list"]:
        print_catalog()
        return 0

    entries = resolve_targets(args.targets, skip_legacy=args.skip_legacy)

    total_gb = sum(e.approx_size_gb for e in entries)
    print(f"Selected {len(entries)} model(s), ~{total_gb:.0f}GB total, into: {args.models_dir}")
    for e in entries:
        flag = " [legacy]" if e.legacy else (" [recommended]" if e.recommended else "")
        print(f"  - {e.key}{flag}: {e.repo_id} (~{e.approx_size_gb:.0f}GB)")

    if not args.dry_run and not args.yes and len(entries) > 0:
        try:
            reply = input(f"\nProceed with ~{total_gb:.0f}GB download? [y/N] ").strip().lower()
        except EOFError:
            reply = ""
        if reply not in ("y", "yes"):
            print("Aborted.")
            return 1

    if not args.dry_run:
        if maybe_enable_hf_transfer():
            print("(hf_transfer available — using accelerated downloads)")

        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            print(
                "huggingface_hub is not installed in this Python environment.\n"
                "Run this script via ../download_models.sh, or:\n"
                "  pip install -r requirements-download.txt",
                file=sys.stderr,
            )
            return 1

    failures = []
    for entry in entries:
        ok = download_one(
            entry,
            models_dir=args.models_dir,
            token=args.token,
            revision=args.revision,
            max_workers=args.max_workers,
            force=args.force,
            dry_run=args.dry_run,
        )
        if not ok:
            failures.append(entry.key)

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)} (see errors above)", file=sys.stderr)
        return 1

    print(f"All {len(entries)} model(s) done.")
    if not args.dry_run:
        print(
            "\nPoint ComfyUI at these with extra_model_paths.yaml.example "
            "(see repo root) if it doesn't already scan this directory."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
