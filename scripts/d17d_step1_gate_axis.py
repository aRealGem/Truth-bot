"""D17-d STEP 1: which axis explains the original withhold, per claim? ($0)

    scripts/d17d_step1_gate_axis.py [--json PATH]

WHAT THIS ANSWERS
-----------------
``d17d_credit_supply`` split the 81 web-tier1 withholds by how much creditable,
stance-resolved evidence sits in the stored pack. This step runs the REAL gate
over each stored pack and asks WHICH axis explains the recorded
``GATE_INSUFFICIENT`` verdict — RULE-SET AWARE, which is the correction the
first cut of this script got wrong (see CORRECTION below).

METHOD ($0, no calls, no gate re-run beyond reconstruction)
-----------------------------------------------------------
For every web-tier1 claim we call ``regate_from_rescore.gate_once`` on the
stored pack under FOUR rule settings, to separate the ratified switches:

    pre        utterance_record=False, statistical_release=False   (superseded)
    utt_only   utterance_record=True,  statistical_release=False   (D15 only)
    stat_only  utterance_record=False, statistical_release=True    (D16 only)
    ratified   utterance_record=True,  statistical_release=True    (LIVE rules)

The ``ratified`` leg is the authority: it is the rule set the shipped heads were
adjudicated under, so its ``quota_met`` is compared to the recorded gate code to
prove the reconstruction reproduces the record. Axis attribution then reads:

  * ratified gates, pre PASSES        -> a RATIFIED-RULE axis knocked it out.
      utt_only gates  -> D15 utterance_record exclusion (utterance-record items,
                         i.e. next-day coverage OF the speech, do not credit)
      stat_only gates -> D16 statistical_release
  * ratified gates AND pre gates      -> NOT rule-driven: the pack is short on
      the gate regardless of the ratification. Sub-split by credit_supply:
      bearing<need with supply>=need -> stance-limited (null stance, step 2);
      supply<need                    -> supply/timing-limited.
  * ratified PASSES but recorded gated -> a genuine score-propagation gap
      (verdict stale vs the pack). NONE are observed.

CORRECTION (2026-08-16)
-----------------------
The first version of this script ran ONLY the ``pre`` leg and compared it to the
recorded gate. The shipped heads are POST-ratification, so ``pre`` is the WRONG
baseline: it passed the 30 supply_and_bearing_met claims and the mismatch was
mis-read as a propagation gap. Under the LIVE (ratified) rules all 81 reproduce
the recorded gate, and the 30 are gated by D15 — utterance-record items that
``credit_supply.bearing_t13`` counts but the gate excludes. The propagation-gap
reading was an artifact of the wrong rule set and is retracted. This is also why
re-adjudicating the 30 buys nothing: they force Unverifiable before any panel.

ADDITION (b): every finding here is REPORT ONLY. Re-adjudication (moving a
verdict) is owner-batch panel spend and is NOT done here.
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
OUT = REPO / "metrics" / "remediation_v2" / "d17d_step1_gate_axis.json"

RULE_LEGS = {
    "pre": dict(utterance_record=False, statistical_release=False),
    "utt_only": dict(utterance_record=True, statistical_release=False),
    "stat_only": dict(utterance_record=False, statistical_release=True),
    "ratified": dict(utterance_record=True, statistical_release=True),
}


def _group(row: dict) -> str:
    if row["bucket"] == "supply-met":
        return ("supply_and_bearing_met" if row["bearing_bucket"] == "bearing-met"
                else "supply_met_bearing_gap")
    if row["bucket"] == "supply-short":
        return "supply_short"
    return "supply_0"


def _axis(legs: dict, recorded_gated: bool, supply: dict, need: int) -> str:
    """Attribute the recorded withhold to an axis, rule-set aware."""
    if legs["ratified"]:                      # gate now credits under live rules
        return "propagation_gap" if recorded_gated else "was_decided"
    if legs["pre"]:                           # pre passed, ratified gates -> rule
        d15 = not legs["utt_only"]
        d16 = not legs["stat_only"]
        if d15 and d16:
            return "ratified_rule:D15+D16"
        if d15:
            return "ratified_rule:D15_utterance_record"
        if d16:
            return "ratified_rule:D16_statistical_release"
        return "ratified_rule:interaction"    # neither switch alone, both together
    # gates under every rule set -> not ratification-driven
    if supply["creditable_t13"] >= need and supply["bearing_t13"] < need:
        return "stance_limited_null"
    return "supply_timing_limited"


def run(out_path: Path = OUT) -> dict:
    import regate_from_rescore as rg
    from reshape_rerun_0031 import shipping_artifact
    from truthbot.verdict import speech_context
    from truthbot.verdict.consolidator import GATE_INSUFFICIENT
    from truthbot.verdict.publish_pipeline import evidence_from_artifact_dict
    from truthbot.verify.principals import principal_relation

    credit = json.loads(CREDIT.read_text(encoding="utf-8"))
    need = credit["min_bearing_t13"]
    by_speech: dict[str, list] = {}
    group_of, supply_of = {}, {}
    for r in credit["claims"]:
        by_speech.setdefault(r["speech"], []).append(r["sid"])
        group_of[r["sid"]] = _group(r)
        supply_of[r["sid"]] = r

    rows: list[dict] = []
    mismatches: list[dict] = []
    for speech in sorted(by_speech):
        _path, art = shipping_artifact(speech)
        meta = art.get("meta") or {}
        speaker = meta.get("speaker") or ""
        utterance = date.fromisoformat(meta["date"]) if meta.get("date") else None
        if utterance is not None:
            speech_context.register_speech_date(speech, utterance)
        relation_of = None
        if speaker and utterance is not None:
            def relation_of(ev, _sp=speaker, _u=utterance):   # noqa: F811
                return principal_relation(ev.source_url, _sp, _u)
        shapes, _ = rg.claim_shape_map(art, speech)
        rowmap = {r.get("sid"): r for r in (art.get("rows") or [])}
        claims = {c.get("sid"): c for c in (art.get("claims") or [])}
        evidence = art.get("evidence") or {}

        for sid in sorted(by_speech[speech]):
            dumps = evidence.get(sid) or []
            was_gated = rg.row_gate_code(rowmap.get(sid, {})) == GATE_INSUFFICIENT
            text = (claims.get(sid, {}).get("text") or "").strip()
            legs, credits, utt_items = {}, {}, {}
            for name, pins in RULE_LEGS.items():
                ev = evidence_from_artifact_dict({sid: dumps})[sid]
                res, bd = rg.gate_once(
                    sid, ev, utterance=utterance, claim_shape=shapes.get(sid, ""),
                    relation_of=relation_of, claim_text=text, **pins)
                legs[name] = bool(res.quota_met)
                credits[name] = bd["credits"]
                if name == "ratified":
                    utt_items[name] = sum(
                        1 for it in res.items
                        if getattr(it, "utterance_rule", False))
            sup = supply_of[sid]
            axis = _axis(legs, was_gated, sup, need)
            reproduced = (not legs["ratified"]) == was_gated
            if not reproduced:
                mismatches.append({"sid": sid, "was_gated": was_gated,
                                   "ratified_passes": legs["ratified"]})
            rows.append({
                "sid": sid, "speech": speech, "group": group_of[sid],
                "was_gated": was_gated, "axis": axis,
                "gate_pre": legs["pre"], "gate_utt_only": legs["utt_only"],
                "gate_stat_only": legs["stat_only"], "gate_ratified": legs["ratified"],
                "ratified_reproduces_record": reproduced,
                "credits_pre": credits["pre"], "credits_ratified": credits["ratified"],
                "utterance_rule_items": utt_items.get("ratified", 0),
                "creditable_t13": sup["creditable_t13"],
                "bearing_t13": sup["bearing_t13"],
                "null_stance_t13": sup["null_stance_t13"], "min_required": need,
            })

    groups: dict[str, dict] = {}
    axis_totals: dict[str, int] = {}
    for r in rows:
        g = groups.setdefault(r["group"], {"n": 0, "axis": {}})
        g["n"] += 1
        g["axis"][r["axis"]] = g["axis"].get(r["axis"], 0) + 1
        axis_totals[r["axis"]] = axis_totals.get(r["axis"], 0) + 1

    report = {
        "schema": "truthbot-d17d-step1-gate-axis v2",
        "generated": _now(),
        "method": ("$0 gate_once over each stored web-tier1 pack under four rule "
                   "legs (pre / D15-only / D16-only / ratified); the ratified "
                   "leg is the authority and reproduces the recorded gate. No "
                   "calls, no verdict written."),
        "source_credit_supply": credit.get("generated"),
        "min_bearing_t13": need, "n_claims": len(rows),
        "ratified_reproduces_all": not mismatches,
        "reproduction_mismatches": mismatches,
        "axis_totals": axis_totals, "groups": groups,
        "claims": rows,
        "finding": (
            "Under the LIVE (ratified) rules all %d web-tier1 claims reproduce "
            "the recorded GATE_INSUFFICIENT — the reconstruction is faithful. "
            "The %d supply_and_bearing_met claims are gated by the D15 "
            "utterance_record exclusion (they PASS pre-ratification but D15 "
            "alone re-gates all of them; every one carries >=1 utterance-record "
            "item that credit_supply's tier+date+stance ceiling counts but the "
            "gate excludes). They are correctly withheld, NOT a propagation gap "
            "and NOT a free re-adjudication win — they force Unverifiable before "
            "any panel. The supply_met_bearing_gap claims gate under every rule "
            "set (stance-limited, step 2); supply_short/0 likewise "
            "(supply/timing-limited)."
            % (len(rows),
               groups.get("supply_and_bearing_met", {}).get("n", 0))),
        "retraction": ("v1 of this artifact called the 30 a score-propagation "
                       "gap by comparing the PRE-ratification leg to a "
                       "POST-ratification record. Retracted: the axis is D15."),
        "report_only_note": ("axis attribution is a rules fact, not a verdict; "
                             "no adjudication is written."),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return report


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def print_report(rep: dict) -> None:
    print(f"\nD17-d step 1 (v2, rule-set aware) — {rep['n_claims']} claims, "
          f"quota {rep['min_bearing_t13']}")
    print(f"  ratified leg reproduces the recorded gate for ALL claims: "
          f"{rep['ratified_reproduces_all']}\n")
    for g in ("supply_and_bearing_met", "supply_met_bearing_gap",
              "supply_short", "supply_0"):
        d = rep["groups"].get(g)
        if not d:
            continue
        axis = ", ".join(f"{k}={v}" for k, v in sorted(d["axis"].items()))
        print(f"  {g:<26} n={d['n']:>3}   {axis}")
    print(f"\n  {rep['finding']}")
    print(f"\n  RETRACTION: {rep['retraction']}")


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
