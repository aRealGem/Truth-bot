"""D17-d: can the web-tier1 backlog EVER clear the T2.4 quota? ($0, no calls)

    scripts/d17d_credit_supply.py [--speech S] [--json PATH]

WHAT THIS ANSWERS
-----------------
The 2026-08-14 retrieval probe bought fresh evidence for three web-tier1 claims
and all three still gate-failed at exactly ONE credit against the required two
(``consolidator.MIN_BEARING_T13``). The binding constraint was not evidence
supply in general but WHEN the evidence was published: an item in the
post-speech band (utterance+1 .. utterance+7) is kept as context and can never
credit the quota, and the record for a recent speech is dominated by next-day
coverage of the speech itself.

n=3 is too small to close an 81-claim lane on. This script asks the same
question of every gate-withheld claim, for $0, from evidence already on disk.

WHY THIS IS NEWLY POSSIBLE
--------------------------
The R7 pack-anatomy probe concluded that stored packs "cannot reproduce the
gate", because role / utterance_rule / post-speech band / era mode are not
persisted. That is true of the GATE. It is not true of the question asked here:
``published_at`` is persisted on every stored item, and the speech date is
known, so the post-speech band can be RECOMPUTED even though it was never
stored. That reopens exactly the slice R7 closed.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not re-run the gate, and its counts are a CEILING, not a verdict: an
item counted here as creditable still has to bear on the claim (a resolved
True/False stance) to actually credit. So a claim reported with 0-1 creditable
items is a claim no amount of stance-scoring can rescue from stored evidence,
which is the decision-relevant direction. A claim with >=2 is NOT thereby
gateable — it merely fails to be excluded on supply grounds.

WHAT IT NOW ALSO SURFACES (still $0, still no gate re-run)
---------------------------------------------------------
``bearing_t13`` / ``null_stance_t13`` split each creditable item by its
persisted ``supports_claim``: True/False took a resolved stance (can count
toward the quota); None was retrieved but never stance-resolved (cannot bear).
This turns the ceiling into two decision-relevant sets among the supply-met
claims: ``supply_and_bearing_met`` (the stance to gate already exists on disk)
and the supply-met-but-bearing-short null-stance gap (supply is there, the
stance is missing). See the report's ``reconciliation`` block for why the
all-claims creditable total (179) and the supply-met-only total (157) differ.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

TRIAGE = REPO / "metrics" / "remediation_v2" / "d17d_triage.json"
OUT = REPO / "metrics" / "remediation_v2" / "d17d_credit_supply.json"

#: Tiers that can credit the decided-verdict quota (consolidator._T13).
#: POLITICAL is absent by design: a partisan release shows a claim was made,
#: never that it is true.
T13 = {"Government", "Wire", "Established"}

#: consolidator.MIN_BEARING_T13 — read, never restated, so a rule change here
#: cannot silently disagree with the pipeline.
def _min_bearing() -> int:
    from truthbot.verdict.consolidator import MIN_BEARING_T13
    return MIN_BEARING_T13


def _fair_game_days() -> int:
    from truthbot.verdict.era_lint import FAIR_GAME_DAYS
    return FAIR_GAME_DAYS


def _pub_date(ev: dict):
    raw = ev.get("published_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw)).date()
    except ValueError:
        try:
            return date.fromisoformat(str(raw)[:10])
        except ValueError:
            return None


def classify_item(ev: dict, utterance: date, fair_game: int) -> str:
    """Where an item sits relative to the speech — the axis the gate scores on.

    Mirrors consolidator's own test: ``post`` is
    ``utterance < d <= fair_game_end(utterance)``; later items are dropped
    outright, earlier/equal items are contemporaneous."""
    tier = str(ev.get("source_tier") or "")
    if tier not in T13:
        return "not-t13"
    d = _pub_date(ev)
    if d is None:
        return "undated"
    if d <= utterance:
        return "creditable"
    if d <= utterance + timedelta(days=fair_game):
        return "post-speech-band"
    return "dropped-after-fair-game"


def _stance(ev: dict) -> str:
    """Resolved bearing of a stored item on its claim.

    ``supports_claim`` is the only bearing signal persisted on a stored item
    (the gate's role / utterance_rule are not). It is tri-state: True/False =
    the item took a resolved stance and can COUNT toward MIN_BEARING_T13;
    None = retrieved but never stance-resolved, so it cannot bear however well
    it is tiered or dated. That None slice is the ``null_stance_t13`` gap."""
    sc = ev.get("supports_claim")
    if sc is True:
        return "supports"
    if sc is False:
        return "refutes"
    return "null"


def analyse_claim(sid: str, evidence: list, utterance: date,
                  fair_game: int) -> dict:
    buckets: dict[str, int] = {}
    bearing = null = 0
    provenance: list[dict] = []
    for ev in evidence or []:
        k = classify_item(ev, utterance, fair_game)
        buckets[k] = buckets.get(k, 0) + 1
        if k != "creditable":
            continue
        # Among the items that COULD credit on tier+date, split resolved
        # stance (bears) from null stance (cannot bear). retrieved_at is
        # carried so a later step can test whether a bearing stance POSTDATES
        # the original gate run (rescore-propagation gap vs non-persisted-axis
        # rejection are different defects).
        st = _stance(ev)
        if st == "null":
            null += 1
        else:
            bearing += 1
        provenance.append({"id": ev.get("id"),
                           "source_tier": ev.get("source_tier"),
                           "published_at": ev.get("published_at"),
                           "retrieved_at": ev.get("retrieved_at"),
                           "stance": st})
    creditable = buckets.get("creditable", 0)
    return {"sid": sid, "n_items": len(evidence or []),
            "creditable_t13": creditable,
            "bearing_t13": bearing,
            "null_stance_t13": null,
            "post_speech_band_t13": buckets.get("post-speech-band", 0),
            "undated_t13": buckets.get("undated", 0),
            "dropped_t13": buckets.get("dropped-after-fair-game", 0),
            "not_t13": buckets.get("not-t13", 0),
            "creditable_items": provenance}


def bucket_label(creditable: int, need: int) -> str:
    if creditable == 0:
        return "supply-0"
    if creditable < need:
        return "supply-short"
    return "supply-met"


def bearing_label(bearing: int, need: int) -> str:
    """Same thresholds as supply, applied to the RESOLVED-stance count.

    supply-met asks 'are there >=need creditable items?'; bearing-met asks the
    stricter 'are there >=need items that actually took a stance?'. The gap
    between the two is exactly the null-stance defect this artifact surfaces."""
    if bearing == 0:
        return "bearing-0"
    if bearing < need:
        return "bearing-short"
    return "bearing-met"


def run(speech_filter: str = "", out_path: Path = OUT) -> dict:
    from reshape_rerun_0031 import shipping_artifact

    import phase3_rebuild as p3

    need, fair_game = _min_bearing(), _fair_game_days()
    triage = json.loads(TRIAGE.read_text(encoding="utf-8"))
    wanted = {c["sid"]: c for c in triage["claims"]
              if c.get("decidability_class") == "web-tier1"}
    if not wanted:
        raise SystemExit(
            "no web-tier1 claims found in the triage — the schema's class "
            "field moved; refusing to report an empty analysis as a result")

    by_speech: dict[str, list] = {}
    for sid in wanted:
        by_speech.setdefault(sid.split(":", 1)[0], []).append(sid)

    rows, per_speech = [], {}
    for speech in sorted(by_speech):
        if speech_filter and speech != speech_filter:
            continue
        _path, art = shipping_artifact(speech)
        utterance = p3.SPEECHES[speech]["date"]
        evidence = art.get("evidence") or {}
        got = []
        for sid in sorted(by_speech[speech]):
            rec = analyse_claim(sid, evidence.get(sid) or [], utterance,
                                fair_game)
            rec["speech"] = speech
            rec["bucket"] = bucket_label(rec["creditable_t13"], need)
            rec["bearing_bucket"] = bearing_label(rec["bearing_t13"], need)
            rows.append(rec)
            got.append(rec)
        counts: dict[str, int] = {}
        for r in got:
            counts[r["bucket"]] = counts.get(r["bucket"], 0) + 1
        per_speech[speech] = {
            "run_id": art.get("run_id"), "utterance": utterance.isoformat(),
            "n_claims": len(got), "buckets": counts,
            "post_speech_band_items": sum(r["post_speech_band_t13"] for r in got),
            "creditable_items": sum(r["creditable_t13"] for r in got),
            "bearing_items": sum(r["bearing_t13"] for r in got),
            "null_stance_items": sum(r["null_stance_t13"] for r in got)}

    totals: dict[str, int] = {}
    for r in rows:
        totals[r["bucket"]] = totals.get(r["bucket"], 0) + 1

    # Bearing is the stance-resolved refinement of supply. supply_and_bearing_met
    # is the STEP-1 target set (supply says retrieval could help, bearing says
    # the stance to use it already exists); supply_met_bearing_gap is the
    # STEP-2 null-stance set (supply is there, the stance is missing).
    bearing_totals = {"supply_and_bearing_met": 0, "supply_met_bearing_gap": 0}
    for r in rows:
        if r["bucket"] != "supply-met":
            continue
        if r["bearing_bucket"] == "bearing-met":
            bearing_totals["supply_and_bearing_met"] += 1
        else:
            bearing_totals["supply_met_bearing_gap"] += 1

    cred_all = sum(r["creditable_t13"] for r in rows)
    null_all = sum(r["null_stance_t13"] for r in rows)
    cred_met = sum(r["creditable_t13"] for r in rows if r["bucket"] == "supply-met")
    null_met = sum(r["null_stance_t13"] for r in rows if r["bucket"] == "supply-met")

    report = {
        "schema": "truthbot-d17d-credit-supply v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "method": ("$0 recomputation of the post-speech band from stored "
                   "published_at + the registered speech date. No calls, no "
                   "gate re-run."),
        "min_bearing_t13": need, "fair_game_days": fair_game,
        "n_claims": len(rows), "bucket_totals": totals,
        "bearing_totals": bearing_totals,
        "per_speech": per_speech, "claims": rows,
        "ceiling_note": ("creditable_t13 is an UPPER BOUND: an item still has "
                         "to bear on the claim (resolved True/False stance) to "
                         "actually credit. supply-0 / supply-short are "
                         "therefore firm exclusions; supply-met is NOT a "
                         "prediction that the claim gates."),
        "bearing_note": ("bearing_t13 (+ null_stance_t13 == creditable_t13) "
                         "splits each creditable item by supports_claim: "
                         "True/False = a resolved stance that can count toward "
                         "min_bearing_t13; None = retrieved but never "
                         "stance-resolved, so it cannot bear. bearing-met is "
                         "the strict refinement of supply-met used to seed the "
                         "step-1/step-2 target sets."),
        "reconciliation": {
            "note": ("Two denominators are in play — do not conflate them. "
                     "creditable_t13 sums to %d over ALL %d web-tier1 claims; "
                     "restricted to the %d supply-met claims it sums to %d. "
                     "The %d-item difference is the %d supply-short claims, "
                     "which by definition hold exactly 1 creditable item each "
                     "(supply-0 holds none). null-stance is therefore %d/%d "
                     "(%.0f%%) over all claims and %d/%d (%.0f%%) over "
                     "supply-met only; the latter is the figure carried in "
                     "prior notes." % (
                         cred_all, len(rows),
                         totals.get("supply-met", 0), cred_met,
                         cred_all - cred_met, totals.get("supply-short", 0),
                         null_all, cred_all,
                         (100 * null_all / cred_all) if cred_all else 0,
                         null_met, cred_met,
                         (100 * null_met / cred_met) if cred_met else 0)),
            "creditable_t13_all": cred_all,
            "creditable_t13_supply_met": cred_met,
            "null_stance_t13_all": null_all,
            "null_stance_t13_supply_met": null_met,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return report


def print_report(rep: dict) -> None:
    need = rep["min_bearing_t13"]
    print(f"\nD17-d web-tier1 credit supply ($0) — {rep['n_claims']} claims")
    print(f"  quota needs {need} Tier-1..3 items published ON OR BEFORE the "
          f"utterance (post-speech band = +1..+{rep['fair_game_days']}d, "
          "context only)\n")
    t = rep["bucket_totals"]
    n = rep["n_claims"] or 1
    for key, label in (("supply-0", "0 creditable items  — retrieval cannot "
                                    "help from stored evidence"),
                       ("supply-short", f"1..{need - 1} creditable  — short of "
                                        "the quota"),
                       ("supply-met", f">={need} creditable  — not excluded on "
                                      "supply")):
        c = t.get(key, 0)
        print(f"  {c:>3} ({c / n:>4.0%})  {label}")
    bt = rep.get("bearing_totals", {})
    sabm = bt.get("supply_and_bearing_met", 0)
    gap = bt.get("supply_met_bearing_gap", 0)
    print(f"\n  of the {t.get('supply-met', 0)} supply-met claims: "
          f"{sabm} also bearing-met (step-1 set), "
          f"{gap} supply-met but bearing-short/0 (step-2 null-stance set)")
    rc = rep.get("reconciliation", {})
    if rc:
        print(f"  creditable_t13: {rc['creditable_t13_all']} over all claims, "
              f"{rc['creditable_t13_supply_met']} over supply-met only; "
              f"null-stance {rc['null_stance_t13_all']}/"
              f"{rc['creditable_t13_all']} all, "
              f"{rc['null_stance_t13_supply_met']}/"
              f"{rc['creditable_t13_supply_met']} supply-met")
    print("\n  per speech:")
    for sp, d in sorted(rep["per_speech"].items()):
        b = d["buckets"]
        print(f"    {sp:<13} n={d['n_claims']:>3}  "
              f"supply-0={b.get('supply-0', 0):>3} "
              f"short={b.get('supply-short', 0):>3} "
              f"met={b.get('supply-met', 0):>3}   "
              f"(creditable items {d['creditable_items']}, "
              f"bearing {d['bearing_items']}, "
              f"null-stance {d['null_stance_items']}, "
              f"post-speech-band {d['post_speech_band_items']})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--speech", default="")
    ap.add_argument("--json", default=str(OUT))
    args = ap.parse_args(argv)
    rep = run(args.speech, Path(args.json))
    print_report(rep)
    print(f"\nreport -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
