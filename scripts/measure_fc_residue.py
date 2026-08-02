"""DC-1 data emitter: measure fact-check-exclusion residue on stored URLs ($0, read-only).

Applies the PROPOSED v2 exclusion rules (URL-path regex + named verticals + allowlist)
against every URL in metrics/journals/*.jsonl and metrics/pca_runs/*.json, and reports
what the current v1 rules (domain + path-prefix blocklist) miss. Output: JSON to stdout.

Usage: .venv/bin/python scripts/measure_fc_residue.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truthbot.verify.factcheck_exclusion import is_excluded_factchecker  # noqa: E402

FC_PATH_RE = re.compile(r"fact[-_ ]?check(s|ing|ed|er)?", re.I)

# Proposed v2 verticals: article-level fact-check sections on otherwise-allowed outlets.
VERTICALS = [
    ("apnews.com", re.compile(r"^/hub/ap-fact-check")),
    ("washingtonpost.com", re.compile(r"^/politics/fact-checker|fact-check", re.I)),
    ("cbsnews.com", re.compile(r"fact[-_]?check", re.I)),
    ("abcnews.go.com", re.compile(r"fact[-_ ]?check", re.I)),
    ("abcnews.com", re.compile(r"fact[-_ ]?check", re.I)),
]

ALLOWLIST = [("mn.gov", "/dhs/program-integrity/factcheck")]


def iter_urls():
    """Yield (source_label, url) over every stored evidence URL."""
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
                    yield p.name, it["source_url"]
    for p in sorted((ROOT / "metrics" / "pca_runs").glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        for items in (d.get("evidence") or {}).values():
            for it in items:
                if it.get("source_url"):
                    yield p.name, it["source_url"]


def allowlisted(host: str, path: str) -> bool:
    return any(host.endswith(d) and path.startswith(pfx) for d, pfx in ALLOWLIST)


def v2_reason(url: str) -> str:
    """'' when allowed; else the v2 rule class that excludes it."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path = parsed.path or "/"
    if allowlisted(host, path):
        return ""
    if is_excluded_factchecker(url):
        return "v1-blocklist"
    for dom, rx in VERTICALS:
        if host.endswith(dom) and rx.search(path):
            return f"vertical:{dom}"
    if FC_PATH_RE.search(path):
        return "path-regex"
    return ""


def main() -> None:
    by_reason: Counter[str] = Counter()
    residue: dict[str, list[str]] = {}
    seen: set[str] = set()
    gov_regex_hits: list[str] = []
    total = 0
    for src, url in iter_urls():
        total += 1
        if url in seen:
            continue
        seen.add(url)
        reason = v2_reason(url)
        if not reason:
            continue
        by_reason[reason] += 1
        if reason != "v1-blocklist":
            residue.setdefault(reason, []).append(url)
            host = (urlparse(url).hostname or "").lower()
            if host.endswith((".gov", ".mil", ".int")):
                gov_regex_hits.append(url)
    out = {
        "urls_total": total,
        "urls_distinct": len(seen),
        "excluded_by_reason": dict(by_reason),
        "v2_residue_count": sum(len(v) for v in residue.values()),
        "v2_residue": {k: sorted(v) for k, v in residue.items()},
        "gov_urls_hit_by_regex (false-positive candidates)": sorted(set(gov_regex_hits)),
    }
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
