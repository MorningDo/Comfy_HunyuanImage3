#!/usr/bin/env python3
"""Send a generated image + a set of checkable assertions to a vision
model via OpenRouter, get a strict JSON verdict per assertion, and
record every verdict (pass or fail) for later review.

Host-side only, deliberately — OPENROUTER_API_KEY must never be copied
to the vast.ai instance (see CLAUDE_CODE_BRIEFING.md's credentials
table). Pull the generated image back to the host (e.g. via
deploy/vast/sync.sh in reverse, or scp) before running this.

Advisory but recorded: the judge is noisy, so a failed assertion does
not fail tests/run_all.sh's exit code unless --strict is passed —  but
every verdict is always logged with the image path/hash, so a
regression can be reviewed by eye later, and judge drift can be
spotted by re-running this against the same golden images over time.

Judge model is pinned (HY3_JUDGE_MODEL, default below) and never
silently changed between runs — a changed judge invalidates comparison
against previous results, per the definition of done.

Usage:
  tests/validate_image.py --image PATH --case-id red-apple-01
  tests/validate_image.py --image PATH --assertions "..." "..."
  tests/validate_image.py --image PATH --case-id red-apple-01 --strict
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_JUDGE_MODEL = "openai/gpt-5-nano"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_FILE = REPO_ROOT / "tests" / "prompts" / "cases.yaml"
DEFAULT_CACHE_FILE = REPO_ROOT / "tests" / ".validate_cache.json"
DEFAULT_LOG_FILE = REPO_ROOT / "tests" / ".validate_log.jsonl"
API_URL = "https://openrouter.ai/api/v1/chat/completions"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cases(cases_file: Path) -> list[dict]:
    import yaml  # host-side test dependency only, see tests/requirements.txt

    return yaml.safe_load(cases_file.read_text()) or []


def find_case(cases: list[dict], case_id: str) -> dict:
    for c in cases:
        if c.get("id") == case_id:
            return c
    raise KeyError(f"case id {case_id!r} not found")


def load_json_file(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def save_json_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def build_judge_prompt(assertions: list[str]) -> str:
    numbered = "\n".join(f"{i}. {a}" for i, a in enumerate(assertions, 1))
    return (
        "You are grading a generated image against a checklist of factual "
        "assertions. For EACH numbered assertion below, decide if it is true "
        "of the image.\n\n"
        f"{numbered}\n\n"
        "Respond with ONLY a JSON array, no prose, no markdown code fences. "
        "One object per assertion, in the same order, with exactly these keys: "
        '"assertion" (string, copy verbatim), "pass" (boolean), '
        '"confidence" (float 0-1), "reasoning" (short string, one sentence).'
    )


def parse_verdicts(raw_text: str, assertions: list[str]) -> list[dict]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge did not return valid JSON: {exc}\nraw: {raw_text[:500]}") from exc

    if not isinstance(parsed, list) or len(parsed) != len(assertions):
        raise ValueError(
            f"judge returned {len(parsed) if isinstance(parsed, list) else type(parsed)} "
            f"items, expected {len(assertions)}"
        )
    out = []
    for expected_assertion, item in zip(assertions, parsed):
        out.append({
            "assertion": item.get("assertion", expected_assertion),
            "pass": bool(item.get("pass")),
            "confidence": float(item.get("confidence", 0.0)),
            "reasoning": item.get("reasoning", ""),
        })
    return out


def call_openrouter(image_path: Path, assertions: list[str], model: str, api_key: str) -> list[dict]:
    mime, _ = mimetypes.guess_type(str(image_path))
    mime = mime or "image/png"
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": build_judge_prompt(assertions)},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }],
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            response = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter API error {exc.code}: {detail}") from exc

    raw_text = response["choices"][0]["message"]["content"]
    return parse_verdicts(raw_text, assertions)


def get_verdicts(
    image_path: Path,
    assertions: list[str],
    model: str,
    api_key: str,
    cache_file: Path,
) -> list[dict]:
    image_hash = sha256_file(image_path)
    cache = load_json_file(cache_file)

    pending = [a for a in assertions if f"{image_hash}:{model}:{a}" not in cache]
    if pending:
        fresh = call_openrouter(image_path, pending, model, api_key)
        for assertion, verdict in zip(pending, fresh):
            cache[f"{image_hash}:{model}:{assertion}"] = verdict
        save_json_file(cache_file, cache)

    return [cache[f"{image_hash}:{model}:{a}"] for a in assertions]


def log_verdicts(log_file: Path, image_path: Path, model: str, verdicts: list[dict]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "image": str(image_path),
        "image_sha256": sha256_file(image_path),
        "model": model,
        "verdicts": verdicts,
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--case-id", help="pull prompt/assertions from --cases-file by id")
    p.add_argument("--assertions", nargs="+", help="explicit assertions, instead of --case-id")
    p.add_argument("--cases-file", type=Path, default=DEFAULT_CASES_FILE)
    p.add_argument("--model", default=os.environ.get("HY3_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    p.add_argument("--cache-file", type=Path, default=DEFAULT_CACHE_FILE)
    p.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    p.add_argument("--strict", action="store_true", help="exit nonzero if any assertion fails")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if not args.image.is_file():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 2

    if args.assertions:
        assertions = args.assertions
    elif args.case_id:
        cases = load_cases(args.cases_file)
        try:
            assertions = find_case(cases, args.case_id)["assertions"]
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    else:
        print("error: pass either --case-id or --assertions", file=sys.stderr)
        return 2

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("error: OPENROUTER_API_KEY not set (source .env first)", file=sys.stderr)
        return 2

    try:
        verdicts = get_verdicts(args.image, assertions, args.model, api_key, args.cache_file)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    log_verdicts(args.log_file, args.image, args.model, verdicts)

    n_pass = sum(1 for v in verdicts if v["pass"])
    for v in verdicts:
        mark = "PASS" if v["pass"] else "FAIL"
        print(f"[{mark}] (conf={v['confidence']:.2f}) {v['assertion']}")
        print(f"       {v['reasoning']}")
    print(f"\n{n_pass}/{len(verdicts)} assertions passed (judge: {args.model}, advisory).")

    if args.strict and n_pass < len(verdicts):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
