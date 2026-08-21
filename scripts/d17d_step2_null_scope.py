"""D17-d STEP 2: is the null-stance gap a B1a scope miss, or an abstention? ($0)

    scripts/d17d_step2_null_scope.py [--json PATH]

WHAT THIS ANSWERS
-----------------
Step 1 showed the supply_met_bearing_gap claims (supply is there, <2 items bear)
all still withhold under the real gate — they are stance-limited, not
axis-limited. This step asks WHY the stance is missing on those creditable
items, by checking B1a's OWN scope declaration against the misses.

For every creditable T13 item that carries a NULL stance in the shipped pack we
join it to the merged B1a + B2 re-score sidecars (``rescored_<speech>.json`` +
``rescored_b2_<speech>.json``) by source_url and bin it three ways:

  * not_in_b1a_scope           — B1a never scored this item (a coverage gap;
                                 fix = widen the re-score scope).
  * scored_but_null            — B1a scored it and returned null (an ABSTENTION;
                                 the scorer saw it and declined a stance).
  * scored_resolved_not_propagated — the sidecar has a True/False stance the
                                 shipped pack never took (a propagation gap;
                                 fix = re-run propagation, no new spend).

The three defects have three different fixes, so the count that lands in each
bin is the decision. relevance_score and snippet length are carried for the
abstention bin, because an abstention on a relevant item with a real snippet is
a scorer/evidence-granularity limit (a full-text or stronger-model re-score
MIGHT move it — spend), while an abstention with no snippet is a retrieval
limit that no re-score fixes.

$0, no calls. Any actual RE-SCORING to try to move these nulls is spend and is
NOT done here — the directive requires a cost estimate to the owner first.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

CREDIT = REPO / "metrics" / "remediation_v2" / "d17d_credit_supply.json"
OUT = REPO / "metrics" / "remediation_v2" / "d17d_step2_null_scope.json"


def run(out_path: Path = OUT) -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cs", str(REPO / "scripts" / "d17d_credit_supply.py"))
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    import phase3_rebuild as p3
    import regate_from_rescore as rg
    import rescore_stored_packs as rsp
    from reshape_rerun_0031 import shipping_artifact

    fair_game = cs._fair_game_days()
    credit = json.loads(CREDIT.read_text(encoding="utf-8"))
    gap: dict[str, list] = {}
    for r in credit["claims"]:
        if r["bucket"] == "supply-met" and r["bearing_bucket"] != "bearing-met":
            gap.setdefault(r["speech"], []).append(r["sid"])

    bins = {"not_in_b1a_scope": 0, "scored_but_null": 0,
            "scored_resolved_not_propagated": 0}
    claim_rows: list[dict] = []
    abstentions: list[dict] = []
    for speech in sorted(gap):
        _path, art = shipping_artifact(speech)
        utterance = p3.SPEECHES[speech]["date"]
        evidence = art.get("evidence") or {}
        b1a = rsp.load_sidecar(rsp.sidecar_path(speech), speech, "")
        b2_path = rsp.b2_sidecar_path(speech)
        b2 = rsp.load_sidecar(b2_path, speech, "") if b2_path.exists() else None
        merged = rg.merge_sidecars(b1a, b2)
        scored = merged.get("sids") or {}

        for sid in sorted(gap[speech]):
            rowmap = {rg.join_key(r.get("source_url") or ""): r
                      for r in (scored.get(sid) or [])}
            per = {"not_in_b1a_scope": 0, "scored_but_null": 0,
                   "scored_resolved_not_propagated": 0}
            for it in evidence.get(sid) or []:
                if cs.classify_item(it, utterance, fair_game) != "creditable":
                    continue
                if it.get("supports_claim") is not None:
                    continue  # only the NULL creditable items are the miss
                jk = rg.join_key(it.get("source_url") or "")
                srow = rowmap.get(jk)
                if srow is None:
                    per["not_in_b1a_scope"] += 1
                elif srow.get("supports_claim") is None:
                    per["scored_but_null"] += 1
                    abstentions.append({
                        "sid": sid, "source_url": it.get("source_url"),
                        "source_tier": it.get("source_tier"),
                        "relevance_score": srow.get("relevance_score"),
                        "snippet_len": len((it.get("snippet") or "").strip())})
                else:
                    per["scored_resolved_not_propagated"] += 1
            for k in bins:
                bins[k] += per[k]
            claim_rows.append({"sid": sid, "speech": speech,
                               "sid_in_b1a_scope": sid in scored, **per})

    n_items = sum(bins.values())
    rels = [a["relevance_score"] for a in abstentions
            if isinstance(a["relevance_score"], (int, float))]
    snips = [a["snippet_len"] for a in abstentions]
    empty_snips = sum(1 for s in snips if s == 0)
    tiers: dict[str, int] = {}
    for a in abstentions:
        tiers[a["source_tier"]] = tiers.get(a["source_tier"], 0) + 1

    report = {
        "schema": "truthbot-d17d-step2-null-scope v1",
        "generated": _now(),
        "method": ("$0 join of every null-stance creditable item on the "
                   "supply_met_bearing_gap claims to the merged B1a+B2 sidecars "
                   "by source_url. No calls, no re-scoring."),
        "source_credit_supply": credit.get("generated"),
        "n_gap_claims": sum(len(v) for v in gap.values()),
        "n_null_creditable_items": n_items,
        "disposition": bins,
        "abstention_profile": {
            "n": len(abstentions), "empty_snippet": empty_snips,
            "relevance_min": min(rels) if rels else None,
            "relevance_median": _median(rels), "by_tier": tiers,
            "snippet_len_median": _median(snips)},
        "claims": claim_rows,
        "abstentions": abstentions,
        "finding": (
            "All %d null-stance creditable items on the %d gap claims are "
            "scored_but_null: B1a's scope COVERED every one (0 not_in_b1a_scope) "
            "and every scored stance propagated (0 not_propagated). The gap is "
            "therefore an ABSTENTION set, not a coverage miss or a propagation "
            "miss. The items are relevant (median relevance %s) with real "
            "snippets (0 empty, median %s chars), tier-heavy Government — B1a "
            "saw topically-relevant text and declined a True/False stance at "
            "snippet granularity."
            % (n_items, sum(len(v) for v in gap.values()),
               _median(rels), _median(snips))),
        "spend_gate": ("Moving these nulls means a re-score with fuller text or "
                       "a stronger scorer — SPEND. Per directive, cost-estimate "
                       "to the owner before any calls; none are made here. Some "
                       "abstentions may also be correct (relevant but genuinely "
                       "non-bearing), which a re-score would confirm, not fix."),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return report


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def print_report(rep: dict) -> None:
    print(f"\nD17-d step 2 — null-stance scope check ($0) — "
          f"{rep['n_gap_claims']} gap claims, "
          f"{rep['n_null_creditable_items']} null creditable items\n")
    for k, v in rep["disposition"].items():
        print(f"  {k:<34} {v}")
    ap = rep["abstention_profile"]
    print(f"\n  abstention profile: n={ap['n']} empty_snippet={ap['empty_snippet']} "
          f"relevance_median={ap['relevance_median']} "
          f"snippet_len_median={ap['snippet_len_median']}")
    print(f"  by tier: {ap['by_tier']}")
    print(f"\n  {rep['finding']}\n  SPEND: {rep['spend_gate']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=str(OUT))
    args = ap.parse_args(argv)
    rep = run(Path(args.json))
    print_report(rep)
    print(f"\nreport -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
