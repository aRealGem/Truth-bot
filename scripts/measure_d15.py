#!/usr/bin/env python3
"""D15 blast radius — what the utterance-record exclusion would cost. $0.

NO model calls, no keys, no network: pure arithmetic over the five rebuilt run
artifacts (metrics/pca_runs) and the B1a re-score sidecars. It answers the two
questions a ratification decision needs:

  1. HOW MANY items would be reclassified ``utterance-record`` — split by rule,
     by tier, and (the number that matters) by whether the item is currently
     BEARING, since only a bearing Tier-1..3 item can credit the quota today;
  2. HOW MANY claims would change gate outcome — each stored pack is run
     through the REAL gate (``consolidator.consolidate``) twice, once with the
     D15 switch off and once with it on, and the two answers are compared.

Both are reported against TWO stance vintages, because they disagree and the
disagreement is the point:

  * ``stored``   — the stances as the rebuilt artifacts recorded them;
  * ``rescored`` — those stances with the B1a sidecars overlaid. This is the
    live state of the corpus, and it is the larger blast radius, because B1a is
    exactly what gave the transcripts a bearing stance in the first place.

The switch is passed EXPLICITLY (``consolidate(utterance_record=...)``), never
by setting the environment, so this measurement can never leave a flag on
behind it.

Usage (repo root, always $0):
  PYTHONPATH=.:src .venv/bin/python scripts/measure_d15.py
  PYTHONPATH=.:src .venv/bin/python scripts/measure_d15.py --json PATH
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from regate_from_rescore import (claim_shape_map, gate_once,  # noqa: E402
                                 load_rescore_sidecar, overlay_rescores,
                                 row_gate_code)
from rescore_stored_packs import (REBUILT_RUNS, artifact_path,  # noqa: E402
                                  load_artifact, sidecar_path)

OUT_DIR = REPO / "metrics" / "remediation_v2"
OUT_STEM = "d15_blast_radius"

#: The two stance vintages measured, in report order.
VINTAGES = ("stored", "rescored")


def _bearing(ev) -> bool:
    return ev.supports_claim is True or ev.supports_claim is False


def measure_speech(speech: str, artifact: dict,
                   sidecar: Optional[dict]) -> dict:
    """Both vintages for one speech. Pure: no I/O, no mutation of ``artifact``."""
    from truthbot.verdict import speech_context
    from truthbot.verdict.consolidator import _T13
    from truthbot.verdict.publish_pipeline import evidence_from_artifact_dict
    from truthbot.verify.principals import principal_relation

    meta = artifact.get("meta") or {}
    speaker = meta.get("speaker") or ""
    utterance = date.fromisoformat(meta["date"]) if meta.get("date") else None
    if utterance is not None:
        speech_context.register_speech_date(speech, utterance)

    relation_of = None
    if speaker and utterance is not None:
        def relation_of(ev):                      # noqa: F811 — mirrors pipeline
            return principal_relation(ev.source_url, speaker, utterance)

    claims = {c.get("sid"): c for c in (artifact.get("claims") or [])}
    shapes, _ = claim_shape_map(artifact, speech)
    rows = {r.get("sid"): r for r in (artifact.get("rows") or [])}
    scored = (sidecar or {}).get("sids") or {}

    out = {v: {"claims": 0, "flagged_items": 0, "flagged_bearing": 0,
               "flagged_bearing_t13": 0, "by_rule": Counter(),
               "by_tier": Counter(), "gate_changed": 0,
               "newly_gated_sids": [], "released_sids": [],
               "claims_touched": 0}
           for v in VINTAGES}

    for sid, dumps in (artifact.get("evidence") or {}).items():
        claim = claims.get(sid) or {}
        text = (claim.get("text") or "").strip()
        shape = shapes.get(sid, "")
        old_gated = bool(row_gate_code(rows.get(sid, {})))

        for vintage in VINTAGES:
            if vintage == "rescored" and sid not in scored:
                # Nothing to overlay; the rescored vintage has nothing to say
                # about this sid, so it is left out of that column entirely.
                continue
            out[vintage]["claims"] += 1

            def _pack():
                ev = evidence_from_artifact_dict({sid: dumps})[sid]
                if vintage == "rescored":
                    overlay_rescores(ev, scored[sid])
                return ev

            # A SECOND reconstruction per side, so the on-run cannot leak into
            # the off-run's arithmetic.
            off, _ = gate_once(sid, _pack(), utterance=utterance,
                               claim_shape=shape, relation_of=relation_of,
                               claim_text=text, utterance_record=False)
            on, _ = gate_once(sid, _pack(), utterance=utterance,
                              claim_shape=shape, relation_of=relation_of,
                              claim_text=text, utterance_record=True)

            flagged = [it for it in on.pre_cap_items if it.utterance_rule]
            if flagged:
                out[vintage]["claims_touched"] += 1
            for it in flagged:
                out[vintage]["flagged_items"] += 1
                out[vintage]["by_rule"][it.utterance_rule] += 1
                out[vintage]["by_tier"][it.evidence.source_tier.value] += 1
                if _bearing(it.evidence):
                    out[vintage]["flagged_bearing"] += 1
                    if it.evidence.source_tier in _T13:
                        out[vintage]["flagged_bearing_t13"] += 1

            if bool(off.quota_met) != bool(on.quota_met):
                out[vintage]["gate_changed"] += 1
                key = ("newly_gated_sids" if off.quota_met
                       else "released_sids")
                out[vintage][key].append(
                    {"sid": sid, "old_verdict": (rows.get(sid) or {}).get("verdict"),
                     "was_gated_in_artifact": old_gated,
                     "claim": text[:120],
                     "rules": sorted({it.utterance_rule for it in flagged})})

    for v in VINTAGES:
        out[v]["by_rule"] = dict(out[v]["by_rule"])
        out[v]["by_tier"] = dict(out[v]["by_tier"])
    return {"speech": speech, "source_run": artifact.get("run_id"),
            "vintages": out}


def build_report(speeches: list[str]) -> dict:
    per_speech = []
    missing: list[str] = []
    for sp in speeches:
        art = load_artifact(artifact_path(sp))
        side = None
        p = sidecar_path(sp)
        if p.exists():
            side = load_rescore_sidecar(p, sp, art.get("run_id", ""))
        else:
            missing.append(sp)
        per_speech.append(measure_speech(sp, art, side))

    corpus = {v: {"claims": 0, "flagged_items": 0, "flagged_bearing": 0,
                  "flagged_bearing_t13": 0, "gate_changed": 0,
                  "claims_touched": 0, "by_rule": Counter(),
                  "by_tier": Counter()}
              for v in VINTAGES}
    for s in per_speech:
        for v in VINTAGES:
            row = s["vintages"][v]
            for k in ("claims", "flagged_items", "flagged_bearing",
                      "flagged_bearing_t13", "gate_changed", "claims_touched"):
                corpus[v][k] += row[k]
            corpus[v]["by_rule"].update(row["by_rule"])
            corpus[v]["by_tier"].update(row["by_tier"])
    for v in VINTAGES:
        corpus[v]["by_rule"] = dict(corpus[v]["by_rule"])
        corpus[v]["by_tier"] = dict(corpus[v]["by_tier"])

    return {
        "schema": "truthbot-d15-blast-radius v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "flag": "TRUTHBOT_D15_UTTERANCE_RECORD (default OFF — NOT enabled)",
        "speeches": speeches,
        "speeches_missing_sidecar": missing,
        "corpus": corpus,
        "per_speech": per_speech,
    }


def render_text(report: dict) -> str:
    L: list[str] = []
    A = L.append
    A("D15 utterance-record — blast radius ($0, no model calls)")
    A(f"flag: {report['flag']}")
    A("")
    for v in VINTAGES:
        c = report["corpus"][v]
        A(f"[{v}] {c['claims']} claims · {c['flagged_items']} items flagged "
          f"across {c['claims_touched']} claims")
        A(f"      bearing: {c['flagged_bearing']}  "
          f"(bearing AND Tier-1..3, i.e. quota-crediting today: "
          f"{c['flagged_bearing_t13']})")
        A(f"      by rule: {c['by_rule']}")
        A(f"      by tier: {c['by_tier']}")
        A(f"      GATE OUTCOMES CHANGED: {c['gate_changed']}")
        A("")
    A(f"  {'speech':<14}{'items':>7}{'bearing':>9}{'T13':>6}{'gate Δ':>8}"
      f"{'items':>8}{'bearing':>9}{'T13':>6}{'gate Δ':>8}")
    A(f"  {'':<14}{'stored':>30}{'rescored':>31}")
    for s in report["per_speech"]:
        a, b = s["vintages"]["stored"], s["vintages"]["rescored"]
        A(f"  {s['speech']:<14}{a['flagged_items']:>7}{a['flagged_bearing']:>9}"
          f"{a['flagged_bearing_t13']:>6}{a['gate_changed']:>8}"
          f"{b['flagged_items']:>8}{b['flagged_bearing']:>9}"
          f"{b['flagged_bearing_t13']:>6}{b['gate_changed']:>8}")
    A("")
    for v in VINTAGES:
        flips = [(f, s["speech"]) for s in report["per_speech"]
                 for f in s["vintages"][v]["newly_gated_sids"]]
        A(f"[{v}] newly gated by D15 — {len(flips)} claim(s):")
        for f, _sp in flips:
            A(f"    {f['sid']:<20} {f['old_verdict'] or '-':<13} "
              f"{','.join(f['rules'])}")
            A(f"      {f['claim']}")
        rel = [f for s in report["per_speech"]
               for f in s["vintages"][v]["released_sids"]]
        if rel:
            A(f"[{v}] RELEASED by D15 (unexpected — investigate): {rel}")
        A("")
    if report["speeches_missing_sidecar"]:
        A(f"(no B1a sidecar, 'rescored' column empty: "
          f"{report['speeches_missing_sidecar']})")
    return "\n".join(L)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--speech", choices=sorted(REBUILT_RUNS), default=None)
    ap.add_argument("--json", default=None, metavar="PATH",
                    help=f"write the machine-readable report (default "
                         f"{OUT_DIR / (OUT_STEM + '.json')})")
    args = ap.parse_args(argv)

    speeches = [args.speech] if args.speech else list(REBUILT_RUNS)
    report = build_report(speeches)
    print(render_text(report))
    out = Path(args.json) if args.json else OUT_DIR / f"{OUT_STEM}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
