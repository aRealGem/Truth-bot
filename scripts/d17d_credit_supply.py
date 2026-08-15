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


def analyse_claim(sid: str, evidence: list, utterance: date,
                  fair_game: int) -> dict:
    buckets: dict[str, int] = {}
    for ev in evidence or []:
        k = classify_item(ev, utterance, fair_game)
        buckets[k] = buckets.get(k, 0) + 1
    creditable = buckets.get("creditable", 0)
    return {"sid": sid, "n_items": len(evidence or []),
            "creditable_t13": creditable,
            "post_speech_band_t13": buckets.get("post-speech-band", 0),
            "undated_t13": buckets.get("undated", 0),
            "dropped_t13": buckets.get("dropped-after-fair-game", 0),
            "not_t13": buckets.get("not-t13", 0)}


def bucket_label(creditable: int, need: int) -> str:
    if creditable == 0:
        return "supply-0"
    if creditable < need:
        return "supply-short"
    return "supply-met"


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
            rows.append(rec)
            got.append(rec)
        counts: dict[str, int] = {}
        for r in got:
            counts[r["bucket"]] = counts.get(r["bucket"], 0) + 1
        per_speech[speech] = {
            "run_id": art.get("run_id"), "utterance": utterance.isoformat(),
            "n_claims": len(got), "buckets": counts,
            "post_speech_band_items": sum(r["post_speech_band_t13"] for r in got),
            "creditable_items": sum(r["creditable_t13"] for r in got)}

    totals: dict[str, int] = {}
    for r in rows:
        totals[r["bucket"]] = totals.get(r["bucket"], 0) + 1

    report = {
        "schema": "truthbot-d17d-credit-supply v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "method": ("$0 recomputation of the post-speech band from stored "
                   "published_at + the registered speech date. No calls, no "
                   "gate re-run."),
        "min_bearing_t13": need, "fair_game_days": fair_game,
        "n_claims": len(rows), "bucket_totals": totals,
        "per_speech": per_speech, "claims": rows,
        "ceiling_note": ("creditable_t13 is an UPPER BOUND: an item still has "
                         "to bear on the claim (resolved True/False stance) to "
                         "actually credit. supply-0 / supply-short are "
                         "therefore firm exclusions; supply-met is NOT a "
                         "prediction that the claim gates."),
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
    print("\n  per speech:")
    for sp, d in sorted(rep["per_speech"].items()):
        b = d["buckets"]
        print(f"    {sp:<13} n={d['n_claims']:>3}  "
              f"supply-0={b.get('supply-0', 0):>3} "
              f"short={b.get('supply-short', 0):>3} "
              f"met={b.get('supply-met', 0):>3}   "
              f"(creditable items {d['creditable_items']}, "
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
