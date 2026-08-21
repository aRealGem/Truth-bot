"""D17-d STEP 2 cost estimate: what would re-scoring the null-stance set cost? ($0)

    scripts/d17d_step2_rescore_estimate.py [--json PATH]

WHY THIS EXISTS
---------------
Step 2 found that the 23 supply_met_bearing_gap claims fail the quota because
62 creditable items carry a NULL stance, and that every one of those was INSIDE
B1a's scope and scored — an ABSTENTION set, not a coverage gap. Moving them
means re-scoring, which is SPEND, and the standing rule is: cost estimate to the
owner BEFORE any calls. This is that estimate. It sends nothing.

WHAT IS MEASURED vs WHAT IS PROJECTED
-------------------------------------
The INPUT volume is MEASURED, not guessed: for each target claim we build the
exact payload ``score_evidence`` would send (``relevance.score_payload`` +
``_SCORE_SYSTEM``, via ``rescore_stored_packs.estimate_speech`` — the repo's own
sanctioned $0 estimator) and count its characters.

The OUTPUT volume cannot be measured (the reply does not exist yet) and is
priced from ``truthbot.costs``, whose per-item load is back-solved from what the
B1a and B2 runs actually cost. Carry ``costs.uncertainty_note()`` with any
number from here: +/-5% on a whole run, +/-20% on a single speech, and NOTHING
outside the calibrated prompt family / model.

SCOPE NOTE: a re-score rewrites a WHOLE pack (``score_evidence`` scores every
item), so the priced unit is all 195 items on the 23 claims — not just the 62
null ones. That is the real bill, so it is the one quoted.

THE THREE OPTIONS, AND WHY ONLY ONE IS REALLY A LEVER
------------------------------------------------------
1. SAME scorer (claude-haiku), same stored snippets — ON-PROXY, so a funded run
   is LEDGER-TRUE and the breaker (not this number) stops the spend. Cheapest by
   far. BUT the expected yield is near zero and this must not be buried: B1a
   already scored these exact items with this exact model on this exact text and
   abstained. Re-running an identical scorer over identical input mostly buys
   the same nulls back. Priced here for completeness, NOT recommended as a fix.

2. STRONGER scorer over the same snippets — the first option that could actually
   move an abstention. Two blockers before it is real: (a) the LiteLLM proxy must
   actually serve the model (``relevance.build_proxy_llm`` takes a ``model``
   argument, so the code path exists, but availability is a proxy-config fact
   this script does NOT assert); (b) the cost constants are calibrated on
   claude-haiku only, so the figure below is an ORDER-OF-MAGNITUDE
   EXTRAPOLATION, not an estimate of the same quality as option 1 — the exact
   error class ``costs.check_constant_applies`` refuses for pack constants.
   If the model is off-proxy it is also ESTIMATE-ONLY (no ledger truth).

3. FULLER TEXT (fetch the documents behind the URLs and score on those, not on a
   ~133-char snippet) — most likely to actually resolve a stance, since the
   abstentions look like snippet-granularity limits. NOT ESTIMABLE from stored
   data: it needs a fetch pass whose volume is unknown until the documents are
   pulled, and the prompt would grow by orders of magnitude. Quoting a number
   for it would be inventing one.

YIELD IS THE REAL UNKNOWN, AND IT IS NOT PRICED
------------------------------------------------
None of these buys a verdict. Even a resolved stance only clears the T13 quota;
and some abstentions are CORRECT (an item can be relevant and genuinely
non-bearing — an on-topic baseline report that does not affirm the specific
numeric claim). So the honest recommendation is a ONE-SPEECH PILOT to measure
the hit-rate before committing the full set; per-speech costs are broken out
below for exactly that purpose.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

STEP2 = REPO / "metrics" / "remediation_v2" / "d17d_step2_null_scope.json"
OUT = REPO / "metrics" / "remediation_v2" / "d17d_step2_rescore_estimate.json"

#: The scorer B1a used and the constants are calibrated on. ON-PROXY.
BASELINE_MODEL = "claude-haiku"
#: A stronger tier, priced only as an extrapolation (see module docstring).
STRONGER_MODEL = "gpt-5.5"


def run(out_path: Path = OUT) -> dict:
    import rescore_stored_packs as rsp
    from reshape_rerun_0031 import shipping_artifact
    from truthbot import costs

    step2 = json.loads(STEP2.read_text(encoding="utf-8"))
    gap: dict[str, list] = {}
    null_items: dict[str, int] = {}
    for r in step2["claims"]:
        gap.setdefault(r["speech"], []).append(r["sid"])
        null_items[r["speech"]] = (null_items.get(r["speech"], 0)
                                   + r["scored_but_null"])

    per_speech: dict[str, dict] = {}
    for speech in sorted(gap):
        _path, art = shipping_artifact(speech)
        sids = set(gap[speech])
        sub = {
            "claims": [c for c in (art.get("claims") or [])
                       if c.get("sid") in sids],
            "evidence": {k: v for k, v in (art.get("evidence") or {}).items()
                         if k in sids},
        }
        base = rsp.estimate_speech(sub, model=BASELINE_MODEL)
        strong = rsp.estimate_speech(sub, model=STRONGER_MODEL)
        per_speech[speech] = {
            "n_claims": len(sids), "calls": base["calls"],
            "items_rescored": base["items"],
            "null_items_targeted": null_items.get(speech, 0),
            "prompt_chars_measured": base["prompt_chars"],
            "tokens_in_est": base["tokens_in_est"],
            "tokens_out_est": base["tokens_out_est"],
            "cost_usd_est_baseline": base["cost_usd_est"],
            "cost_usd_est_stronger_EXTRAPOLATED": strong["cost_usd_est"],
        }

    tot_base = round(sum(v["cost_usd_est_baseline"]
                         for v in per_speech.values()), 4)
    tot_strong = round(sum(v["cost_usd_est_stronger_EXTRAPOLATED"]
                           for v in per_speech.values()), 4)
    cheapest = min(per_speech, key=lambda s: per_speech[s]["cost_usd_est_baseline"])
    biggest = max(per_speech, key=lambda s: per_speech[s]["null_items_targeted"])

    report = {
        "schema": "truthbot-d17d-step2-rescore-estimate v1",
        "generated": _now(),
        "method": ("Input volume MEASURED via rescore_stored_packs."
                   "estimate_speech (relevance.score_payload + _SCORE_SYSTEM) "
                   "over the step-2 target claims on the shipping head; output "
                   "volume priced from truthbot.costs. NO calls made."),
        "source_step2": step2.get("generated"),
        "n_claims": sum(len(v) for v in gap.values()),
        "n_null_items_targeted": sum(null_items.values()),
        "n_items_rescored": sum(v["items_rescored"] for v in per_speech.values()),
        "scope_note": ("score_evidence rewrites a WHOLE pack, so the priced unit "
                       "is every item on the target claims, not only the null "
                       "ones."),
        "per_speech": per_speech,
        "options": {
            "1_same_scorer_baseline": {
                "model": BASELINE_MODEL, "lane": "on-proxy (ledger-true)",
                "total_usd_est": tot_base,
                "recommended": False,
                "why": ("B1a already scored these exact items with this exact "
                        "model on this exact text and abstained; re-running it "
                        "mostly buys the same nulls back. Priced for "
                        "completeness, not offered as a fix."),
            },
            "2_stronger_scorer": {
                "model": STRONGER_MODEL,
                "lane": "UNCONFIRMED — verify proxy serves it; if off-proxy the "
                        "figure is ESTIMATE-ONLY with no ledger truth",
                "total_usd_est_EXTRAPOLATED": tot_strong,
                "recommended": "only via a one-speech pilot",
                "why": ("The first option that could actually move an "
                        "abstention. The number is an ORDER-OF-MAGNITUDE "
                        "extrapolation: costs.py's per-item output load is "
                        "calibrated on claude-haiku only."),
            },
            "3_fuller_text": {
                "lane": "NOT ESTIMABLE from stored data",
                "total_usd_est": None,
                "why": ("Most likely to resolve a stance (the abstentions look "
                        "like snippet-granularity limits at ~133 chars), but it "
                        "needs a document-fetch pass of unknown volume. A "
                        "number here would be invented."),
            },
        },
        "pilot_suggestion": {
            "cheapest_speech": cheapest,
            "cheapest_speech_usd_baseline": per_speech[cheapest]["cost_usd_est_baseline"],
            "most_null_items_speech": biggest,
            "most_null_items": per_speech[biggest]["null_items_targeted"],
            "note": ("Run ONE speech first and measure how many nulls actually "
                     "resolve, before committing the full 23. Yield is the "
                     "unpriced unknown and some abstentions are correct."),
        },
        "uncertainty": costs.uncertainty_note(model=BASELINE_MODEL),
        "spend_authorization": ("NONE. This is an estimate only; no calls were "
                                "made and none are authorized by this artifact. "
                                "A funded run needs an explicit owner budget "
                                "cap, which the breaker — not this number — "
                                "enforces."),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return report


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def print_report(rep: dict) -> None:
    print(f"\nD17-d step 2 re-score cost estimate ($0, nothing sent)")
    print(f"  {rep['n_claims']} claims / {rep['n_items_rescored']} items "
          f"re-scored ({rep['n_null_items_targeted']} null items targeted)\n")
    print(f"  {'speech':<13} {'claims':<7} {'items':<6} {'nulls':<6} "
          f"{'haiku':<10} {'stronger*':<10}")
    for sp, d in sorted(rep["per_speech"].items()):
        print(f"  {sp:<13} {d['n_claims']:<7} {d['items_rescored']:<6} "
              f"{d['null_items_targeted']:<6} "
              f"${d['cost_usd_est_baseline']:<9.4f} "
              f"${d['cost_usd_est_stronger_EXTRAPOLATED']:<9.4f}")
    o = rep["options"]
    print(f"\n  option 1 same scorer ({o['1_same_scorer_baseline']['model']}, "
          f"on-proxy): ${o['1_same_scorer_baseline']['total_usd_est']:.4f} "
          f"— NOT recommended")
    print(f"    {o['1_same_scorer_baseline']['why']}")
    print(f"  option 2 stronger scorer: "
          f"~${o['2_stronger_scorer']['total_usd_est_EXTRAPOLATED']:.4f} "
          f"*EXTRAPOLATED, lane unconfirmed")
    print(f"  option 3 fuller text: NOT ESTIMABLE — "
          f"{o['3_fuller_text']['why']}")
    p = rep["pilot_suggestion"]
    print(f"\n  PILOT: cheapest={p['cheapest_speech']} "
          f"(${p['cheapest_speech_usd_baseline']:.4f}), "
          f"most nulls={p['most_null_items_speech']} "
          f"({p['most_null_items']} items)")
    print(f"  {p['note']}")
    print(f"\n  {rep['uncertainty']}")
    print(f"\n  {rep['spend_authorization']}")


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
