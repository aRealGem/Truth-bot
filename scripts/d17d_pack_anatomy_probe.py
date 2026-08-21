#!/usr/bin/env python3
"""D17-d R7 — pack-anatomy probe. Analysis only, $0, no pipeline changes.

The structural probe (``d17d_structural_probe.py``) used only claim-level
fields: ``claim_type``, ``claim_shape``, ``series_rows`` presence. It never used
the PACK — how many items were retrieved, at what tiers, how many bore on the
claim. This asks whether that was a miss: does pack anatomy carry a decidability
signal the claim fields do not?

SAME FRAME AS R1-R6. Structured fields only, no claim text as prose, abstain
loudly, and NO THRESHOLD IS TUNED TO THE DESK PASS — a cut chosen to maximise
agreement would launder the fixture into the classifier and report its own
reflection as a finding. R7 therefore reports the anatomy DISTRIBUTION per desk
class and lets the separation (or absence of it) speak.

FIELD INVENTORY IS PART OF THE OUTPUT. Which per-item fields actually survive
into the stored pack is itself the answer to "can we read the gate's reasoning
back off the artifact?" — so the probe enumerates them rather than assuming.

$0. No model, network, or clock. Deterministic.

Usage (repo root):
  PYTHONPATH=src python3 scripts/d17d_pack_anatomy_probe.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "metrics" / "pca_runs"
DESK = REPO / "metrics" / "remediation_v2" / "d17d_triage.json"
OUT = REPO / "metrics" / "remediation_v2" / "d17d_pack_anatomy_probe.json"

HEADS = {
    "trump_2026": "799e71b6-2480-50ca-870e-1a95f0d0d5fe",
    "biden_2022": "c156d8f9-be85-5263-92a1-c08949afdedd",
    "obama_2014": "70748500-315a-5664-8474-c6632de57816",
    "clinton_1998": "d7ee7340-c07d-55da-b9db-9397d7141c35",
    "gwbush_2006": "6df77093-e328-596e-bfd5-afabd08a1679",
}

GATE = "insufficient-qualifying-evidence"

#: The consolidator's Tier-1..3 set (consolidator.py `_T13`), as stored strings.
T13 = {"Government", "Wire", "Established"}

#: Per-item fields that would let the gate's own reasoning be read back off the
#: artifact. Presence is REPORTED, never assumed.
WANTED_FIELDS = [
    "source_tier",        # tier — needed for the quota
    "supports_claim",     # the bearing flag
    "series_rows",        # D17-c series excerpt
    "one_line_why",       # the scorer's comparison
    "arithmetic_hinge",   # stance came from arithmetic
    "role",               # D11.2 evidential role — quota-relevant
    "era_note",           # era filtering — quota-relevant
    "utterance_rule",     # D15: credits 0
    "quota_credit",       # the decision itself
    "disqualification_code",
    "gate_code",
]


def _mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def _dist(xs):
    return {str(k): v for k, v in sorted(Counter(xs).items())}


def build() -> dict:
    desk_doc = json.loads(DESK.read_text(encoding="utf-8"))
    desk_cls = {c["sid"]: c["decidability_class"] for c in desk_doc["claims"]}
    desk_text = {c["sid"]: c["text"] for c in desk_doc["claims"]}

    field_presence = Counter()
    total_items = 0
    claims_out = []

    for speech, run in sorted(HEADS.items()):
        doc = json.loads((RUNS / f"{run}.json").read_text(encoding="utf-8"))
        evidence = doc.get("evidence", {})
        for row in doc["rows"]:
            if row.get("provenance_code") != GATE:
                continue
            sid = row["sid"]
            pack = evidence.get(sid, [])
            total_items += len(pack)
            for e in pack:
                for f in WANTED_FIELDS:
                    if f in e and e[f] is not None:
                        field_presence[f] += 1

            tiers = Counter(e.get("source_tier") for e in pack)
            bearing = sum(1 for e in pack if e.get("supports_claim") is not None)
            t13 = sum(1 for e in pack if e.get("source_tier") in T13)
            # Proxy for consolidator._quota_credit using ONLY stored fields.
            proxy_credits = sum(
                1 for e in pack
                if e.get("source_tier") in T13
                and e.get("supports_claim") is not None)
            rel = [e.get("relevance_score") for e in pack
                   if isinstance(e.get("relevance_score"), (int, float))]

            claims_out.append({
                "sid": sid,
                "speech": speech,
                "text": desk_text.get(sid, ""),
                "desk_class": desk_cls.get(sid, "MISSING-FROM-DESK"),
                "anatomy": {
                    "n_items": len(pack),
                    "n_tier13": t13,
                    "n_bearing": bearing,
                    "n_supports": sum(1 for e in pack
                                      if e.get("supports_claim") is True),
                    "n_refutes": sum(1 for e in pack
                                     if e.get("supports_claim") is False),
                    "n_neutral": sum(1 for e in pack
                                     if e.get("supports_claim") is None),
                    "proxy_quota_credits": proxy_credits,
                    "proxy_says_quota_met": proxy_credits >= 2,
                    "has_series_rows": any(bool(e.get("series_rows"))
                                           for e in pack),
                    "tier_counts": dict(sorted(tiers.items(),
                                               key=lambda kv: str(kv[0]))),
                    "mean_relevance": _mean(rel),
                },
            })

    # ── field inventory: what can actually be read back off the artifact ──
    inventory = {
        f: {"items_carrying": field_presence.get(f, 0),
            "present": field_presence.get(f, 0) > 0}
        for f in WANTED_FIELDS
    }
    absent = [f for f, v in inventory.items() if not v["present"]]

    # ── the gate-reproduction check ──
    # Every one of these 128 packs was REJECTED by the real gate. If the stored
    # fields sufficed to explain that, no pack would score >=2 proxy credits.
    proxy_pass = [c for c in claims_out if c["anatomy"]["proxy_says_quota_met"]]

    # ── anatomy distribution per desk class (no threshold fitted) ──
    by_class = defaultdict(list)
    for c in claims_out:
        by_class[c["desk_class"]].append(c["anatomy"])
    per_class = {}
    for cls, rows in sorted(by_class.items()):
        per_class[cls] = {
            "n_claims": len(rows),
            "n_items": {
                "mean": _mean([r["n_items"] for r in rows]),
                "distribution": _dist(r["n_items"] for r in rows),
            },
            "n_tier13_mean": _mean([r["n_tier13"] for r in rows]),
            "n_bearing_mean": _mean([r["n_bearing"] for r in rows]),
            "proxy_quota_credits_mean": _mean(
                [r["proxy_quota_credits"] for r in rows]),
            "proxy_quota_credits_distribution": _dist(
                r["proxy_quota_credits"] for r in rows),
            "mean_relevance": _mean([r["mean_relevance"] for r in rows
                                     if r["mean_relevance"] is not None]),
        }

    # Does any anatomy feature separate the two classes that matter?
    web = per_class.get("web-tier1", {})
    sub = per_class.get("substantive", {})
    separation = {
        "question": ("Does pack anatomy separate web-tier1 (a retrieval "
                     "backlog) from substantive (permanent abstention)? These "
                     "are the two classes a render must not confuse."),
        "web_tier1": {k: web.get(k) for k in
                      ("n_claims", "n_tier13_mean", "n_bearing_mean",
                       "proxy_quota_credits_mean", "mean_relevance")},
        "substantive": {k: sub.get(k) for k in
                        ("n_claims", "n_tier13_mean", "n_bearing_mean",
                         "proxy_quota_credits_mean", "mean_relevance")},
    }

    n = len(claims_out)
    return {
        "schema": "truthbot-d17d-pack-anatomy-probe v1",
        "generated_from": {sp: rid for sp, rid in sorted(HEADS.items())},
        "audit_fixture": str(DESK.relative_to(REPO)),
        "method": (
            "R7 reads ONLY stored pack fields (item count, per-tier counts, "
            "bearing flags, series_rows presence, relevance). No claim text as "
            "prose, no model, no network. NO THRESHOLD IS TUNED TO THE DESK "
            "PASS: a cut fitted to maximise agreement would launder the "
            "fixture into the classifier. The probe reports the anatomy "
            "distribution per desk class and lets the separation speak."),
        "totals": {"claims": n, "evidence_items": total_items},
        "field_inventory": inventory,
        "absent_fields": absent,
        "gate_reproduction_check": {
            "packs": n,
            "all_were_rejected_by_the_real_gate": True,
            "proxy_says_quota_met": len(proxy_pass),
            "proxy_disagreement_rate": round(len(proxy_pass) / n, 4) if n else None,
            "finding": (
                f"{len(proxy_pass)} of {n} packs score >=2 quota credits under "
                "a reconstruction built from the stored fields, yet the real "
                "gate rejected all of them. The stored artifact CANNOT "
                "reproduce the gate it is a record of."),
            "why": (
                "consolidator._quota_credit also consults role (D11.2 "
                "role-aware credit), utterance_rule (D15: credits 0), the "
                "post-speech band, and era mode. None of those survive into "
                "the stored evidence item, so the tier+bearing reconstruction "
                "is a PROXY, not the gate."),
        },
        "per_desk_class_anatomy": per_class,
        "separation": separation,
        "claims": claims_out,
    }


def main() -> int:
    doc = build()
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    t = doc["totals"]
    print(f"packs {t['claims']}  evidence items {t['evidence_items']}\n")
    print("field inventory (per-item fields surviving into the artifact):")
    for f, v in doc["field_inventory"].items():
        mark = "yes" if v["present"] else "NO "
        print(f"  [{mark}] {f:<24}{v['items_carrying']}")
    print(f"\nABSENT: {', '.join(doc['absent_fields'])}")
    g = doc["gate_reproduction_check"]
    print(f"\ngate reproduction: {g['proxy_says_quota_met']} of {g['packs']} "
          f"packs 'should have passed' ({g['proxy_disagreement_rate']:.0%}) "
          "-- yet all were rejected.")
    print("\nanatomy by desk class:")
    for cls, b in doc["per_desk_class_anatomy"].items():
        print(f"  {cls:<16} n={b['n_claims']:<4} items~{b['n_items']['mean']:<6} "
              f"t13~{b['n_tier13_mean']:<6} bearing~{b['n_bearing_mean']:<6} "
              f"credits~{b['proxy_quota_credits_mean']}")
    print(f"\n-> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
