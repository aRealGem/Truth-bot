"""DC-2a seed worksheet: full domain->tier census from stored evidence ($0, read-only).

For every distinct host in metrics/journals + metrics/pca_runs evidence: how many URLs,
what tier(s) current classify_tier() assigns (revealing per-path splits and flapping),
what tier was STORED at adjudication time, and flags for the decision worksheet
(unmapped .gov/.mil/.int that would quarantine under fail-closed, whitehouse mirrors,
protected-statistical candidates). CSV to stdout.

Usage: .venv/bin/python scripts/emit_tier_census.py [--min-count N]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truthbot.verify.source_tiers import classify_tier  # noqa: E402

WHITEHOUSE_MIRRORS = (
    "whitehouse.gov", "obamawhitehouse.archives.gov", "bidenwhitehouse.archives.gov",
    "trumpwhitehouse.archives.gov", "georgewbush-whitehouse.archives.gov",
    "clintonwhitehouse3.archives.gov", "clintonwhitehouse4.archives.gov",
)
PROTECTED_HINTS = (
    "bls.gov", "bea.gov", "cbo.gov", "census.gov", "fbi.gov", "cdc.gov", "cms.gov",
    "irs.gov", "treasury.gov", "govinfo.gov", "clerk.house.gov", "senate.gov",
    "uscourts.gov", "supremecourt.gov",
)


def iter_items():
    for p in sorted((ROOT / "metrics" / "journals").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            items = rec.get("evidence") or rec.get("pool") or []
            if isinstance(items, dict):
                items = [x for v in items.values() for x in v]
            for it in items:
                if isinstance(it, dict) and it.get("source_url"):
                    yield it["source_url"], it.get("source_tier") or ""
    for p in sorted((ROOT / "metrics" / "pca_runs").glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        for items in (d.get("evidence") or {}).values():
            for it in items:
                if it.get("source_url"):
                    yield it["source_url"], it.get("source_tier") or ""


def flags_for(host: str, current_tiers: set[str]) -> str:
    out = []
    if any(host == m or host.endswith("." + m) for m in WHITEHOUSE_MIRRORS):
        out.append("WHITEHOUSE-MIRROR")
    if any(host == h or host.endswith("." + h) for h in PROTECTED_HINTS):
        out.append("PROTECTED-T1-CANDIDATE")
    if host.endswith((".gov", ".mil", ".int")) and not any(
        host == h or host.endswith("." + h) for h in PROTECTED_HINTS + WHITEHOUSE_MIRRORS
    ):
        out.append("GOV-NEEDS-MAPPING(else-quarantine)")
    if len(current_tiers) > 1:
        out.append("MULTI-TIER(path-split-or-flap)")
    return ";".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=1)
    args = ap.parse_args()

    per_host_urls: dict[str, set[str]] = defaultdict(set)
    per_host_current: dict[str, Counter] = defaultdict(Counter)
    per_host_stored: dict[str, Counter] = defaultdict(Counter)
    for url, stored in iter_items():
        host = (urlparse(url).hostname or "").lower().removeprefix("www.")
        if not host:
            continue
        per_host_urls[host].add(url)
        per_host_stored[host][stored] += 1

    for host, urls in per_host_urls.items():
        for u in urls:
            per_host_current[host][classify_tier(u).value] += 1

    w = csv.writer(sys.stdout)
    w.writerow(["host", "distinct_urls", "current_tiers", "stored_tiers", "flags", "example_url"])
    for host in sorted(per_host_urls, key=lambda h: -len(per_host_urls[h])):
        urls = per_host_urls[host]
        if len(urls) < args.min_count:
            continue
        cur = per_host_current[host]
        w.writerow([
            host,
            len(urls),
            " ".join(f"{t}:{n}" for t, n in cur.most_common()),
            " ".join(f"{t}:{n}" for t, n in per_host_stored[host].most_common()),
            flags_for(host, set(cur)),
            sorted(urls)[0][:120],
        ])


if __name__ == "__main__":
    main()
