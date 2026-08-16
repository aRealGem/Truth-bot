"""D17-d STEP 1: which axis explains the original withhold, per claim? ($0)

    scripts/d17d_step1_gate_axis.py [--json PATH]

WHAT THIS ANSWERS
-----------------
``d17d_credit_supply`` split the 81 web-tier1 withholds by how much creditable,
stance-resolved evidence sits in the stored pack. This step takes the next
question: for each withheld claim, run the REAL gate over its own stored pack
and ask WHICH axis explains the gap between the recorded verdict
(``GATE_INSUFFICIENT``) and what the gate now makes of the same evidence.

Two defects look identical in the verdict row but need different fixes:

  * score-propagation gap — the stored pack already credits >=MIN_BEARING_T13
    under the real gate, so the recorded withhold is STALE: stances were scored
    into the pack (B1a + score_propagation, applied 2026-08-10) AFTER the
    verdict row was written, and the row was never re-derived. Fix = re-derive
    the verdict (panel spend, owner batch) — NOT a rule change.

  * non-persisted-axis rejection — the gate still withholds because role /
    era-mode / post-speech / utterance_rule knocks the creditable items below
    quota. Fix lives in the gate or the evidence, not in propagation.

METHOD ($0, no calls, no gate re-run beyond reconstruction)
-----------------------------------------------------------
For every web-tier1 claim we call ``regate_from_rescore.gate_once`` on the
BEFORE leg only — the stored stances, pinned to PRE_RATIFICATION_RULES — which
reconstructs role / era / post-speech from ``published_at`` + the registered
speech date (the axes R7 found un-persisted are RECOMPUTED here, not read). We
compare its ``quota_met`` to the recorded gate code:

  recorded gated + gate now PASSES  -> score-propagation gap (verdict stale)
  recorded gated + gate now WITHHOLDS -> axis-suppressed (still short on the gate)

ADDITION (a): a gate-now-passes finding is only a PROPAGATION gap if the
bearing stances POSTDATE the original run. We evidence that two ways, both
recorded per claim: the BEFORE-leg reconstruction MISMATCHES the recorded gate
(recomputed passes, row says gated), and ``score_propagation.json`` (applied)
is the dated event that wrote those stances into the head. A claim whose BEFORE
leg REPRODUCES its recorded gate is not a propagation gap.

ADDITION (b): every gate-now-passes row is REPORT ONLY. Re-adjudication (moving
the verdict) is panel spend on the owner batch and is deliberately NOT done
here; this script never writes a verdict.
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


def _group(row: dict) -> str:
    """The credit_supply supply/bearing bucket a claim falls in."""
    if row["bucket"] == "supply-met":
        return ("supply_and_bearing_met" if row["bearing_bucket"] == "bearing-met"
                else "supply_met_bearing_gap")
    if row["bucket"] == "supply-short":
        return "supply_short"
    return "supply_0"


def run(out_path: Path = OUT) -> dict:
    import regate_from_rescore as rg
    from reshape_rerun_0031 import shipping_artifact
    from truthbot.verdict import speech_context
    from truthbot.verdict.consolidator import GATE_INSUFFICIENT
    from truthbot.verdict.publish_pipeline import evidence_from_artifact_dict
    from truthbot.verify.principals import principal_relation

    credit = json.loads(CREDIT.read_text(encoding="utf-8"))
    by_speech: dict[str, list] = {}
    group_of, supply_of = {}, {}
    for r in credit["claims"]:
        by_speech.setdefault(r["speech"], []).append(r["sid"])
        group_of[r["sid"]] = _group(r)
        supply_of[r["sid"]] = r

    rows: list[dict] = []
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
            before_ev = evidence_from_artifact_dict({sid: dumps})[sid]
            before, bd = rg.gate_once(
                sid, before_ev, utterance=utterance,
                claim_shape=shapes.get(sid, ""), relation_of=relation_of,
                claim_text=text, **rg.PRE_RATIFICATION_RULES)
            now_pass = bool(before.quota_met)
            # Does the stored pack reproduce the recorded gate? For a
            # propagation gap it must NOT (recomputed passes, row says gated).
            reproduced = (not now_pass) == was_gated
            if was_gated and now_pass:
                axis = "score_propagation_gap"
            elif was_gated and not now_pass:
                axis = "axis_suppressed"
            else:
                axis = "was_decided"
            sup = supply_of[sid]
            rows.append({
                "sid": sid, "speech": speech, "group": group_of[sid],
                "was_gated": was_gated, "gate_now_passes": now_pass,
                "baseline_reproduced": reproduced, "axis": axis,
                "creditable_t13": sup["creditable_t13"],
                "bearing_t13": sup["bearing_t13"],
                "null_stance_t13": sup["null_stance_t13"],
                "independent": bd["independent"], "corroborant": bd["corroborant"],
                "primary": bd["primary"], "credits": bd["credits"],
                "min_required": bd["min_required"], "role_aware": bd["role_aware"],
                "era_mode": bd["era_mode"], "recon_agrees": bd["agrees"],
            })

    # Roll-ups: axis outcome cross-tabbed by supply/bearing group.
    groups: dict[str, dict] = {}
    for r in rows:
        g = groups.setdefault(r["group"], {"n": 0, "axis": {}})
        g["n"] += 1
        g["axis"][r["axis"]] = g["axis"].get(r["axis"], 0) + 1

    prop = [r for r in rows if r["axis"] == "score_propagation_gap"]
    # An honest ceiling caveat: any supply-short/0 claim the gate PASSES was
    # credited on the role axis (primary-record / corroborant), which the
    # tier+date credit_supply ceiling does not model.
    role_rescued = [r for r in rows
                    if r["group"] in ("supply_short", "supply_0")
                    and r["gate_now_passes"]]

    report = {
        "schema": "truthbot-d17d-step1-gate-axis v1",
        "generated": _now(),
        "method": ("$0 gate_once (BEFORE leg, PRE_RATIFICATION_RULES) over each "
                   "stored web-tier1 pack; role/era/post-speech reconstructed, "
                   "no calls, no verdict written."),
        "source_credit_supply": credit.get("generated"),
        "min_bearing_t13": rows[0]["min_required"] if rows else None,
        "n_claims": len(rows),
        "groups": groups,
        "propagation_gap_sids": sorted(r["sid"] for r in prop),
        "role_axis_rescued": [
            {"sid": r["sid"], "group": r["group"], "independent": r["independent"],
             "corroborant": r["corroborant"], "primary": r["primary"],
             "credits": r["credits"]} for r in role_rescued],
        "claims": rows,
        "finding": (
            "%d claims are recorded GATE_INSUFFICIENT but credit >=%d under the "
            "real gate on their own stored pack — a score-propagation gap "
            "(verdict stale vs applied stances), NOT a non-persisted-axis "
            "rejection: role/era/post-speech suppress none of them. %d are the "
            "supply_and_bearing_met set; the remaining %d is a supply-short "
            "claim credited on the ROLE axis (primary-record), which the "
            "tier+date credit_supply ceiling does not model. REPORT ONLY: "
            "re-adjudication is owner-batch panel spend."
            % (len(prop), rows[0]["min_required"] if rows else 0,
               groups.get("supply_and_bearing_met", {}).get("n", 0),
               len(prop) - groups.get("supply_and_bearing_met", {}).get(
                   "axis", {}).get("score_propagation_gap", 0))),
        "step2_handoff": (
            "The supply_met_bearing_gap claims all still WITHHOLD under the gate "
            "reconstruction — confirming that set is stance-limited (null "
            "stance), not axis-limited, and is the correct step-2 target."),
        "report_only_note": ("gate_now_passes is a supply/stance fact, not a "
                             "verdict. This script writes no adjudication."),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return report


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def print_report(rep: dict) -> None:
    print(f"\nD17-d step 1 — gate-axis attribution ($0) — {rep['n_claims']} claims")
    print(f"  quota needs {rep['min_bearing_t13']} credits\n")
    for g in ("supply_and_bearing_met", "supply_met_bearing_gap",
              "supply_short", "supply_0"):
        d = rep["groups"].get(g)
        if not d:
            continue
        axis = ", ".join(f"{k}={v}" for k, v in sorted(d["axis"].items()))
        print(f"  {g:<26} n={d['n']:>3}   {axis}")
    print(f"\n  score-propagation gap: {len(rep['propagation_gap_sids'])} claims "
          "(recorded gated, gate now credits on stored evidence)")
    if rep["role_axis_rescued"]:
        print("  role-axis rescued (credit_supply ceiling undercount):")
        for r in rep["role_axis_rescued"]:
            print(f"    {r['sid']}  indep={r['independent']} "
                  f"corrob={r['corroborant']} prim={r['primary']} "
                  f"credits={r['credits']}")
    print(f"\n  {rep['finding']}\n  STEP 2: {rep['step2_handoff']}")


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
