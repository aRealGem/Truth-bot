#!/usr/bin/env python3
"""Audit the evidence packs behind the 4 gold-FALSE claims (P67.2 Phase 1b).

F4 prior: CRM-114 (explicitly prompted not to soften) declined to flip the
unanimous-MISLEADING gold-FALSE rows in every config, on the same packs — so the
packs plausibly lack refuting evidence and RETRIEVAL, not judging, is the
bottleneck. This script tests that directly, with zero LLM spend (~20 Brave/
FactCheck queries).

Per gold-FALSE sid it:
  1. rebuilds the pack EXACTLY as the eval does — same provider
     (score_layerb_vs_gold._build_open_book_provider: Brave 5 + FactCheck 3),
     same window (evidence_pack.window_for), same dedup/tier-rank/cap-6
     (_dedup_rank_cap) — and also keeps the RAW pre-cap items so evidence dropped
     by the cap is visible (a refuting FactCheck item that exists but ranks past
     cap-6 is itself a finding: fix = tier-slot reservation, not queries);
  2. runs a COUNTER-EVIDENCE probe: extra Brave queries the pipeline could add
     systematically (claim + refutation keywords) plus hand-written gist probes
     (upper bound: does refuting evidence exist on the open web at all?). The two
     kinds are labelled — only "systematic" results argue for query augmentation;
     "gist" results only show the ceiling.

CAVEAT: live Brave = reconstruction, not replay — the index moved since the eval
runs; treat presence/absence directionally.

The refuting/supporting/neutral classification of each snippet is a human/agent
judgment made from the emitted artifact (this script fetches and organizes; it
does not judge).

Env: BRAVE_API_KEY (~/.config/truthbot/secrets.env). Run from repo root:
  PYTHONPATH=. .venv/bin/python eval/benchmarks/audit_evidence_packs.py
Writes eval/benchmarks/examples/evidence-pack-audit.json + prints a readable dump.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

from truthbot.models import Claim
from truthbot.verdict import evidence_pack
from truthbot.verify.sources.brave import BraveSearchConnector
from score_layerb_vs_gold import _build_open_book_provider

GOLD = HERE / "claim-set" / "verdict_gold.train.jsonl"
OUT = HERE / "examples" / "evidence-pack-audit.json"

# Counter-evidence probe queries. "systematic" = a scheme the pipeline could apply
# to ANY claim (claim text + refutation keywords — mirrors brave.py's existing
# "fact check " prefix). "gist" = hand-written, per-claim (existence ceiling only).
SYSTEMATIC_SUFFIXES = ["false", "debunked"]
GIST_PROBES: dict[str, list[str]] = {
    "trump_2026:0020": [
        "illegal border crossings gotaways 2025 numbers",
        "zero illegal aliens admitted claim fact check",
    ],
    "trump_2026:0056": [
        "DEI programs still exist companies 2025",
        "ended DEI in America fact check",
    ],
    "trump_2026:0556": [
        "Iran nuclear sites strike damage assessment intelligence set back",
        "obliterated Iran nuclear facilities fact check",
    ],
    "biden_2022:0342": [
        "PLCAA gun manufacturers liability immunity exceptions fact check",
        "industries with legal liability immunity vaccine manufacturers",
    ],
}


def _ev_dict(ev) -> dict:
    return {"source": ev.source_name, "tier": ev.source_tier.value,
            "url": ev.source_url, "snippet": (ev.snippet or "")[:400]}


def main() -> None:
    provider = _build_open_book_provider()
    if provider is None:
        sys.exit(1)
    # Probe queries go through a plain Brave connector (same key/params as the
    # provider's) with the query text substituted for the claim.
    brave = BraveSearchConnector(max_results=5)

    gold = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    false_rows = [g for g in gold if g["gold_verdict"] == "FALSE"]
    print(f"auditing packs for {len(false_rows)} gold-FALSE sids\n")

    audit = {}
    n_queries = 0
    for g in false_rows:
        sid, text = g["sid"], g["claim"]
        window = evidence_pack.window_for(sid)
        claim = Claim(transcript_id=sid.split(":", 1)[0], text=text)

        raw = provider.get_evidence(claim, window=window)       # Brave 5 + FactCheck 3
        n_queries += 2
        kept = evidence_pack._dedup_rank_cap(raw, evidence_pack.DEFAULT_MAX_ITEMS)
        kept_urls = {e.source_url for e in kept}
        dropped = [e for e in raw if e.source_url and e.source_url not in kept_urls]

        probes = {}
        for kind, queries in (
            ("systematic", [f"{text} {sfx}"[:200] for sfx in SYSTEMATIC_SUFFIXES]),
            ("gist", GIST_PROBES.get(sid, [])),
        ):
            for q in queries:
                probe_claim = Claim(transcript_id=sid.split(":", 1)[0], text=q)
                results = brave.search_windowed(probe_claim, window)
                n_queries += 1
                # only report items the base pack didn't already surface
                fresh = [e for e in results if e.source_url not in kept_urls]
                probes[f"{kind}: {q[:90]}"] = [_ev_dict(e) for e in fresh]

        audit[sid] = {
            "claim": text,
            "gold": "FALSE",
            "window": [str(w) for w in window] if window else None,
            "pack": [dict(_ev_dict(e), pack_id=f"E{i}") for i, e in enumerate(kept, 1)],
            "dropped_by_cap": [_ev_dict(e) for e in dropped],
            "probes": probes,
        }

        print(f"## {sid}  (gold FALSE)  window={audit[sid]['window']}")
        print(f"   {text[:120]}")
        print(f"  PACK ({len(kept)} kept, {len(dropped)} dropped by dedup/cap):")
        for it in audit[sid]["pack"]:
            print(f"   [{it['pack_id']}] ({it['tier']}) {it['source']} — {it['url']}")
            print(f"       {it['snippet'][:220]}")
        for it in audit[sid]["dropped_by_cap"]:
            print(f"   [DROPPED] ({it['tier']}) {it['source']} — {it['url']}")
            print(f"       {it['snippet'][:220]}")
        for label, items in probes.items():
            print(f"  PROBE {label}: {len(items)} new items")
            for it in items:
                print(f"   ({it['tier']}) {it['source']} — {it['url']}")
                print(f"       {it['snippet'][:220]}")
        print()

    OUT.write_text(json.dumps(audit, indent=2))
    print(f"# {n_queries} connector queries total; artifact → {OUT}")


if __name__ == "__main__":
    main()
