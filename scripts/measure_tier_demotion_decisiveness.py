#!/usr/bin/env python3
"""Retrospective decisiveness register for the S5 political-communications tier.

Claim Eval v3 PR-A / D7. Zero API cost — reads stored ``metrics/pca_runs`` run
artifacts and re-classifies every evidence URL under the *new* tiering
(:func:`truthbot.verify.source_tiers.classify_tier`). A URL that now resolves to
``POLITICAL`` (S5) is *demoted* by PR-A.

The input-side question ("how many URLs get demoted?") was already answered when
the carve-out was measured against stored URLs. This answers the sharper,
outcome-side question the second-opinion review asked for: of the demoted URLs,
how many were actually **cited in the winning rationale of a DECIDED claim** —
i.e. were verdict-decisive under the old top-tiering the ruling corrects.

A claim is DECIDED when ``status == 'resolved'`` and its verdict is neither
``UNVERIFIABLE`` nor absent. Row ``citations`` are positional ``E<n>`` labels
into that claim's evidence list (``E1`` == evidence[0]).

Usage: ``python scripts/measure_tier_demotion_decisiveness.py [--json OUT]``
Defaults to the two P3-rerun artifacts (trump_2026, biden_2022); pass artifact
ids/paths as positional args to override.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

# Allow running from a checkout without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from truthbot.models import SourceTier
from truthbot.verify.source_tiers import classify_tier

RUNS_DIR = Path(__file__).resolve().parent.parent / "metrics" / "pca_runs"
DEFAULT_ARTIFACTS = (
    "23939712-59ea-449d-93f7-a0a0b449efd8",  # trump_2026, prod roster, v2 evidence
    "7208bbbb-c802-4155-932f-d0cc66803b24",  # biden_2022, prod roster, v2 evidence
)
UNDECIDED = {"UNVERIFIABLE", "ABSTAIN", None, ""}


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or url).lower()
    except ValueError:
        return url.lower()


def _is_decided(row: dict) -> bool:
    return row.get("status") == "resolved" and row.get("verdict") not in UNDECIDED


def measure_artifact(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = {r["sid"]: r for r in d.get("rows", [])}
    evidence = d.get("evidence", {})
    speaker = d.get("meta", {}).get("speech_id") or path.stem

    total_items = demoted_items = 0
    demoted_in_decided_pack = demoted_cited = decisive = 0
    demoted_hosts: Counter = Counter()
    decisive_records: list[dict] = []

    for sid, items in evidence.items():
        row = rows.get(sid)
        cited_labels = set(row.get("citations") or []) if row else set()
        decided = bool(row) and _is_decided(row)
        for i, item in enumerate(items):
            total_items += 1
            url = item.get("source_url", "")
            if classify_tier(url) is not SourceTier.POLITICAL:
                continue
            demoted_items += 1
            demoted_hosts[_host(url)] += 1
            label = f"E{i + 1}"
            cited = label in cited_labels
            if decided:
                demoted_in_decided_pack += 1
            if cited:
                demoted_cited += 1
            if cited and decided:
                decisive += 1
                decisive_records.append({
                    "sid": sid,
                    "verdict": row.get("verdict"),
                    "confidence": row.get("confidence"),
                    "label": label,
                    "host": _host(url),
                    "url": url,
                    "supports_claim": item.get("supports_claim"),
                    "old_tier": item.get("source_tier"),
                })

    return {
        "artifact": path.stem,
        "speaker": speaker,
        "claims": len(rows),
        "decided_claims": sum(1 for r in rows.values() if _is_decided(r)),
        "evidence_items": total_items,
        "demoted_items": demoted_items,
        "demoted_distinct_hosts": len(demoted_hosts),
        "demoted_in_decided_pack": demoted_in_decided_pack,
        "demoted_cited_anywhere": demoted_cited,
        "decisive_demoted": decisive,   # cited in a DECIDED claim = the damage register
        "top_demoted_hosts": demoted_hosts.most_common(15),
        "decisive_records": decisive_records,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifacts", nargs="*", default=list(DEFAULT_ARTIFACTS),
                    help="artifact ids or paths (default: the two P3-rerun runs)")
    ap.add_argument("--json", type=Path, help="write the full result object here")
    args = ap.parse_args()

    per = []
    for a in args.artifacts:
        p = Path(a) if Path(a).exists() else RUNS_DIR / f"{a}.json"
        if not p.exists():
            print(f"! missing artifact: {p}", file=sys.stderr)
            continue
        per.append(measure_artifact(p))

    agg = {
        "evidence_items": sum(r["evidence_items"] for r in per),
        "demoted_items": sum(r["demoted_items"] for r in per),
        "demoted_in_decided_pack": sum(r["demoted_in_decided_pack"] for r in per),
        "demoted_cited_anywhere": sum(r["demoted_cited_anywhere"] for r in per),
        "decisive_demoted": sum(r["decisive_demoted"] for r in per),
    }
    out = {"per_artifact": per, "aggregate": agg}

    for r in per:
        print(f"\n=== {r['speaker']} ({r['artifact'][:8]}) ===")
        print(f"  claims {r['claims']} ({r['decided_claims']} decided), "
              f"evidence items {r['evidence_items']}")
        print(f"  demoted→S5 items: {r['demoted_items']} "
              f"across {r['demoted_distinct_hosts']} hosts")
        print(f"  demoted & in a decided claim's pack: {r['demoted_in_decided_pack']}")
        print(f"  demoted & CITED (winning rationale, decided): "
              f"{r['decisive_demoted']}  <-- damage register")
        if r["top_demoted_hosts"]:
            print("  top demoted hosts: " +
                  ", ".join(f"{h}×{n}" for h, n in r["top_demoted_hosts"][:8]))
        for rec in r["decisive_records"]:
            print(f"    - {rec['sid']} [{rec['verdict']}] {rec['host']} "
                  f"(supports={rec['supports_claim']}, was {rec['old_tier']})")

    a = agg
    print("\n=== AGGREGATE ===")
    print(f"  evidence items: {a['evidence_items']}")
    print(f"  demoted→S5 items: {a['demoted_items']} "
          f"({100*a['demoted_items']/max(a['evidence_items'],1):.1f}% of evidence)")
    print(f"  of those, cited in a DECIDED verdict (verdict-decisive): "
          f"{a['decisive_demoted']}")

    if args.json:
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
