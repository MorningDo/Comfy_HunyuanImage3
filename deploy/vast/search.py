#!/usr/bin/env python3
"""Query vast.ai for GPU offers matching a filter set and print a ranked
table. Read-only — never creates, rents, or modifies anything. Never
auto-selects: a human picks an offer id from the printed table and passes
it to deploy/vast/up.sh.

Stdlib-only (urllib/json), deliberately not depending on `pip install
vastai` or the vastai CLI — this sandbox has no pip, and the CLI is a
thin wrapper over the same REST API this script talks to directly.

API reference (undocumented publicly beyond the CLI source; confirmed
empirically 2026-08-23 against the live endpoint):
  GET https://console.vast.ai/api/v0/bundles/?q=<url-encoded JSON>
  Authorization: Bearer $VAST_API_KEY
  q is a dict of {field: {op: value}}, op in eq/neq/gte/gt/lte/lt/in/notin.
  An optional "order": [[field, "asc"|"desc"]] can be embedded inside q.
  Response: {"offers": [ {id, gpu_name, num_gpus, gpu_ram, disk_space,
    dph_total, reliability, inet_down, inet_up, driver_version,
    cuda_max_good, geolocation, verified, rentable, ...}, ... ]}

Usage:
  deploy/vast/search.py [--gpu NAME] [--min-vram-gb N] [--min-gpus N]
                         [--max-price DPH] [--min-disk-gb N]
                         [--min-inet-mbps N] [--min-reliability F]
                         [--limit N] [--include-unverified]
                         [--include-unrentable]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

API_BASE = os.environ.get("HY3_VAST_API_BASE", "https://console.vast.ai/api/v0")

# Per project convention (see CLAUDE_CODE_BRIEFING.md / project memory):
# the RTX 6000 Pro Blackwell (96GB) is the real test hardware, so that's
# the default floor — not guessed, confirmed with the user.
DEFAULT_MIN_VRAM_GB = 96


def build_query(args: argparse.Namespace) -> dict:
    q: dict = {}
    if not args.include_unverified:
        q["verified"] = {"eq": True}
    if not args.include_unrentable:
        q["rentable"] = {"eq": True}
    if args.gpu:
        q["gpu_name"] = {"eq": args.gpu}
    if args.min_gpus:
        q["num_gpus"] = {"gte": args.min_gpus}
    if args.min_vram_gb is not None:
        q["gpu_ram"] = {"gte": args.min_vram_gb * 1000}
    if args.max_price is not None:
        q["dph_total"] = {"lte": args.max_price}
    if args.min_disk_gb is not None:
        q["disk_space"] = {"gte": args.min_disk_gb}
    if args.min_inet_mbps is not None:
        q["inet_down"] = {"gte": args.min_inet_mbps}
    if args.min_reliability is not None:
        q["reliability"] = {"gte": args.min_reliability}
    q["order"] = [["dph_total", "asc"]]
    return q


def fetch_offers(query: dict, api_key: str, base_url: str = API_BASE) -> list[dict]:
    url = f"{base_url}/bundles/?" + urllib.parse.urlencode({"q": json.dumps(query)})
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vast.ai API error {exc.code}: {detail}") from exc
    if body.get("success") is False:
        raise RuntimeError(f"vast.ai API rejected query: {body.get('msg')}")
    return body.get("offers", [])


def rank_offers(offers: list[dict], limit: int) -> list[dict]:
    # API already orders by dph_total asc; re-sort defensively in case
    # that ever changes, then truncate.
    return sorted(offers, key=lambda o: o.get("dph_total", float("inf")))[:limit]


COLUMNS = [
    ("id", "ID", "{:>10}"),
    ("gpu_name", "GPU", "{:<16}"),
    ("num_gpus", "N", "{:>2}"),
    ("gpu_ram", "VRAM(GB)", "{:>8.1f}"),
    ("disk_space", "Disk(GB)", "{:>8.0f}"),
    ("dph_total", "$/hr", "{:>7.3f}"),
    ("reliability", "Reliab", "{:>6.3f}"),
    ("inet_down", "Down(Mbps)", "{:>10.0f}"),
    ("driver_version", "Driver", "{:<14}"),
    ("geolocation", "Location", "{:<20}"),
]


def format_table(offers: list[dict]) -> str:
    if not offers:
        return "No offers matched the given filters."
    header = "  ".join(label for _, label, _ in COLUMNS)
    lines = [header, "-" * len(header)]
    for o in offers:
        cells = []
        for field, _, fmt in COLUMNS:
            value = o.get(field)
            if field == "gpu_ram" and value is not None:
                value = value / 1000.0  # API reports MB
            try:
                cells.append(fmt.format(value if value is not None else ""))
            except (ValueError, TypeError):
                cells.append(str(value))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", help="exact gpu_name match, e.g. 'RTX 6000 Ada'")
    p.add_argument("--min-vram-gb", type=float, default=DEFAULT_MIN_VRAM_GB)
    p.add_argument("--min-gpus", type=int, default=None)
    p.add_argument("--max-price", type=float, default=None, help="max $/hr total")
    p.add_argument("--min-disk-gb", type=float, default=None)
    p.add_argument("--min-inet-mbps", type=float, default=None)
    p.add_argument("--min-reliability", type=float, default=None)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--include-unverified", action="store_true")
    p.add_argument("--include-unrentable", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        print("error: VAST_API_KEY not set (source .env first)", file=sys.stderr)
        return 2

    query = build_query(args)
    try:
        offers = fetch_offers(query, api_key)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    ranked = rank_offers(offers, args.limit)
    print(format_table(ranked))
    print(f"\n{len(ranked)} of {len(offers)} matching offers shown, cheapest first.")
    print("This script never creates an instance. Pick an id and run:")
    print("  deploy/vast/up.sh --offer-id <ID>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
