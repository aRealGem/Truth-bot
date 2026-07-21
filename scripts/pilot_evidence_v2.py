#!/usr/bin/env python3
"""Evidence-v2 pilot on the 15-claim gold fixture (P67.8 / T2.6).

Compares the OLD retrieval path (shared_pack_v1: Brave + FactCheck +
relevance middle step, 6-item packs) against the NEW path (shared_pack_v2:
R1 Claude-Opus/Lane-Worker native search + R2 GPT browsing → deterministic
consolidator, 10-item packs) on eval/benchmarks/claim-set/
sotu_gold_fixture_2026-07-10.json.

Metric — **decisive_source_recall**: for each gold evidence entry the
fixture says the verdict relies on, does the assembled pack contain that
source (registered-domain match on domains named in the entry, tier-class
keyword match otherwise)? Reported per claim and per path.

CONTAMINATION GUARD (T2.6, harness assertion): before any retriever call,
the prompt is asserted free of every gold verdict/rationale fragment
(truthbot.verify.retrievers.assert_no_contamination) — a leak raises, it
does not warn.

Cost: R1 is subscription-auth (zero marginal); R2 is metered OpenAI tokens;
the old path spends Brave + a haiku relevance call per claim. Use --limit
for spend metering and --skip-old/--skip-new to run one side.

Usage (repo root, needs repo .env + secrets for the old path):
  PYTHONPATH=. .venv/bin/python scripts/pilot_evidence_v2.py \
      --out metrics/evidence_v2_pilot.json [--limit 3] [--dry-run]
      [--skip-old] [--skip-new]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

FIXTURE = REPO / "eval" / "benchmarks" / "claim-set" / "sotu_gold_fixture_2026-07-10.json"

SPEECHES = {
    "biden2022": ("biden_2022", date(2022, 3, 1)),
    "trump2026": ("trump_2026", date(2026, 2, 24)),
}

_DOMAIN_RX = re.compile(r"\b([a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:gov|mil|org|com|int|edu|net))\b",
                        re.IGNORECASE)

_GOV_HINTS = ("bls", "bea", "cbo", "census", "treasury", "federal reserve",
              "white house", "omb", "cdc", "fbi", "dhs", "gao", "crs",
              "department of", "bureau of", ".gov", "official", "agency")
_WIRE_HINTS = ("associated press", "ap news", "reuters")


def gold_matchers(entry: dict) -> tuple[list[str], str]:
    """(explicit_domains, class_hint) for one gold evidence entry."""
    text = str(entry.get("source") or "")
    domains = [d.lower() for d in _DOMAIN_RX.findall(text)]
    low = text.lower()
    if any(h in low for h in _GOV_HINTS):
        cls = "gov"
    elif any(h in low for h in _WIRE_HINTS):
        cls = "wire"
    else:
        cls = ""
    return domains, cls


def recall_for_pack(gold_evidence: list[dict], pack_urls: list[str],
                    pack_tiers: list[str]) -> dict:
    from truthbot.domains import host_matches, url_host

    matched = unmatchable = 0
    details = []
    for entry in gold_evidence:
        domains, cls = gold_matchers(entry)
        hit = False
        if domains:
            hit = any(host_matches(url_host(u), d)
                      for u in pack_urls for d in domains)
        if not hit and cls == "gov":
            hit = "Government" in pack_tiers
        elif not hit and cls == "wire":
            hit = "Wire" in pack_tiers
        if not domains and not cls:
            unmatchable += 1
            details.append({"source": entry.get("source"), "result": "unmatchable"})
            continue
        matched += 1 if hit else 0
        details.append({"source": entry.get("source"),
                        "result": "hit" if hit else "miss",
                        "domains": domains, "class": cls})
    denom = len(gold_evidence) - unmatchable
    return {"matched": matched, "matchable": denom, "unmatchable": unmatchable,
            "recall": (matched / denom) if denom else None, "details": details}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="metrics/evidence_v2_pilot.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-old", action="store_true")
    ap.add_argument("--skip-new", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    claims = fixture["claims"]
    if args.limit:
        claims = claims[:args.limit]

    # T2.6 contamination guard inputs: every gold fragment that must never
    # reach a retriever prompt.
    gold_fragments = []
    for c in fixture["claims"]:
        gold_fragments.append(str(c.get("verdict_provisional") or ""))
        gold_fragments.append(str(c.get("rationale") or ""))
        for e in c.get("evidence") or []:
            gold_fragments.append(str(e.get("supports") or ""))

    from truthbot.verdict.consolidator import consolidate
    from truthbot.verdict.speech_context import expected_claim_window as _ecw
    from truthbot.verify.retrievers import (
        ClaudeWorkerRetriever, OpenAIBrowsingRetriever,
        assert_no_contamination, build_retrieval_prompt)

    r1, r2 = ClaudeWorkerRetriever(), OpenAIBrowsingRetriever()
    results = []
    for i, c in enumerate(claims):
        speech = c["speech"]
        sid_prefix, utt = SPEECHES[speech]
        window = _ecw(utt)
        text = c["paraphrase"]
        sid = f"{sid_prefix}:9{i:03d}"
        prompt = build_retrieval_prompt(text, utterance=utt, window=window)
        assert_no_contamination(prompt, gold_fragments)
        print(f"\n[{c['claim_id']}] {text[:90]}")
        if args.dry_run:
            print(prompt[:600])
            continue
        rec = {"claim_id": c["claim_id"], "speech": speech, "paraphrase": text}

        if not args.skip_new:
            sl1 = r1.shortlist(text, utterance=utt, window=window)
            sl2 = r2.shortlist(text, utterance=utt, window=window)
            new_pack = consolidate(sid, [("R1", sl1), ("R2", sl2)],
                                   utterance=utt, window=window)
            urls = [it.evidence.source_url for it in new_pack.items]
            tiers = [it.evidence.source_tier.value for it in new_pack.items]
            rec["new"] = {
                "r1_n": len(sl1), "r2_n": len(sl2),
                "pack": new_pack.to_payload(),
                "quota_met": new_pack.quota_met,
                "gate_code": new_pack.gate_code,
                "dropped": new_pack.dropped,
                "recall": recall_for_pack(c.get("evidence") or [], urls, tiers),
            }
            print(f"  NEW: R1 {len(sl1)} + R2 {len(sl2)} → pack {len(urls)}"
                  f" quota_met={new_pack.quota_met}"
                  f" recall={rec['new']['recall']['recall']}")

        if not args.skip_old:
            from truthbot.verdict.evidence_pack import build_evidence_pack
            from truthbot.pipeline import _build_open_book_provider
            provider = _build_open_book_provider()
            pack = build_evidence_pack(sid, text, provider, today=utt)
            urls = [it.source_url for it in pack.items]
            tiers = [it.tier.value for it in pack.items]
            rec["old"] = {
                "pack": [{"url": u, "tier": t} for u, t in zip(urls, tiers)],
                "recall": recall_for_pack(c.get("evidence") or [], urls, tiers),
            }
            print(f"  OLD: pack {len(urls)} recall={rec['old']['recall']['recall']}")
        results.append(rec)

    if args.dry_run:
        return
    summary = {}
    for path_key in ("old", "new"):
        vals = [r[path_key]["recall"]["recall"] for r in results
                if path_key in r and r[path_key]["recall"]["recall"] is not None]
        summary[path_key] = {
            "claims": len(vals),
            "mean_decisive_source_recall": (sum(vals) / len(vals)) if vals else None,
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"fixture": FIXTURE.name, "summary": summary, "claims": results},
        indent=2, ensure_ascii=False))
    print(f"\nSUMMARY: {json.dumps(summary, indent=2)}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
