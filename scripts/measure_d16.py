#!/usr/bin/env python3
"""D16(α) blast radius — what the statistical-release allowlist would RELEASE. $0.

NO model calls, no keys, no network: pure arithmetic over the five rebuilt run
artifacts (metrics/pca_runs) and the B1a re-score sidecars. The mirror image of
``scripts/measure_d15.py``, deliberately built to the same shape so the two
proposals can be priced on one page:

  1. HOW MANY post-speech items the allowlist would release — split by AGENCY,
     by period rule, by tier, and by whether the item is currently BEARING,
     since only a bearing Tier-1..3 item can credit the quota;
  2. HOW MANY claims would change gate outcome — each stored pack is run
     through the REAL gate (``consolidator.consolidate``) twice, once with the
     D16 switch off and once with it on, and the two answers are compared.

D16 can only ADD credits, so the expected direction is one-way: claims move
from gated to released, never the reverse. A ``newly_gated`` entry in this
report would be a defect, and it is reported rather than assumed away.

Both are measured against TWO stance vintages, for the same reason D15 is:

  * ``stored``   — the stances as the rebuilt artifacts recorded them;
  * ``rescored`` — those stances with the B1a sidecars overlaid, i.e. the live
    state of the corpus.

The switch is passed EXPLICITLY (``consolidate(statistical_release=...)``),
never by setting the environment, so this measurement can never leave a flag on
behind it.

Usage (repo root, always $0):
  PYTHONPATH=.:src .venv/bin/python scripts/measure_d16.py
  PYTHONPATH=.:src .venv/bin/python scripts/measure_d16.py --json PATH
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
OUT_STEM = "d16_blast_radius"

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
    from truthbot.verify.statistical_agency import agency_for

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

    out = {v: {"claims": 0, "released_items": 0, "released_bearing": 0,
               "released_bearing_t13": 0, "by_rule": Counter(),
               "by_agency": Counter(), "by_tier": Counter(),
               "gate_changed": 0, "released_sids": [], "newly_gated_sids": [],
               "claims_touched": 0}
           for v in VINTAGES}

    for sid, dumps in (artifact.get("evidence") or {}).items():
        claim = claims.get(sid) or {}
        text = (claim.get("text") or "").strip()
        shape = shapes.get(sid, "")
        old_gated = bool(row_gate_code(rows.get(sid, {})))

        for vintage in VINTAGES:
            if vintage == "rescored" and sid not in scored:
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
                               claim_text=text, statistical_release=False)
            on, _ = gate_once(sid, _pack(), utterance=utterance,
                              claim_shape=shape, relation_of=relation_of,
                              claim_text=text, statistical_release=True)

            freed = [it for it in on.pre_cap_items if it.stat_release_rule]
            if freed:
                out[vintage]["claims_touched"] += 1
            for it in freed:
                out[vintage]["released_items"] += 1
                out[vintage]["by_rule"][it.stat_release_rule] += 1
                out[vintage]["by_agency"][
                    agency_for(it.evidence.source_url) or "?"] += 1
                out[vintage]["by_tier"][it.evidence.source_tier.value] += 1
                if _bearing(it.evidence):
                    out[vintage]["released_bearing"] += 1
                    if it.evidence.source_tier in _T13:
                        out[vintage]["released_bearing_t13"] += 1

            if bool(off.quota_met) != bool(on.quota_met):
                out[vintage]["gate_changed"] += 1
                key = ("newly_gated_sids" if off.quota_met
                       else "released_sids")
                out[vintage][key].append(
                    {"sid": sid,
                     "old_verdict": (rows.get(sid) or {}).get("verdict"),
                     "was_gated_in_artifact": old_gated,
                     "claim": text[:120],
                     "rules": sorted({it.stat_release_rule for it in freed}),
                     "agencies": sorted({agency_for(it.evidence.source_url)
                                         for it in freed}),
                     "urls": [it.evidence.source_url for it in freed]})

    for v in VINTAGES:
        for k in ("by_rule", "by_agency", "by_tier"):
            out[v][k] = dict(out[v][k])
    return {"speech": speech, "source_run": artifact.get("run_id"),
            "vintages": out}


def build_report(speeches: list[str]) -> dict:
    from truthbot.verify.statistical_agency import load_registry

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

    corpus = {v: {"claims": 0, "released_items": 0, "released_bearing": 0,
                  "released_bearing_t13": 0, "gate_changed": 0,
                  "claims_touched": 0, "by_rule": Counter(),
                  "by_agency": Counter(), "by_tier": Counter()}
              for v in VINTAGES}
    for s in per_speech:
        for v in VINTAGES:
            row = s["vintages"][v]
            for k in ("claims", "released_items", "released_bearing",
                      "released_bearing_t13", "gate_changed", "claims_touched"):
                corpus[v][k] += row[k]
            for k in ("by_rule", "by_agency", "by_tier"):
                corpus[v][k].update(row[k])
    for v in VINTAGES:
        for k in ("by_rule", "by_agency", "by_tier"):
            corpus[v][k] = dict(corpus[v][k])

    reg = load_registry()
    return {
        "schema": "truthbot-d16-blast-radius v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "flag": "TRUTHBOT_D16_STATISTICAL_RELEASE (default OFF — NOT enabled)",
        "registry_version": reg.version,
        "registry_hosts": sorted(reg.entries_by_domain),
        "speeches": speeches,
        "speeches_missing_sidecar": missing,
        "corpus": corpus,
        "per_speech": per_speech,
    }


def render_text(report: dict) -> str:
    L: list[str] = []
    A = L.append
    A("D16(α) statistical-release — blast radius ($0, no model calls)")
    A(f"flag: {report['flag']}")
    A(f"allowlist: statistical_agency_registry {report['registry_version']} "
      f"({len(report['registry_hosts'])} hosts)")
    A("")
    for v in VINTAGES:
        c = report["corpus"][v]
        A(f"[{v}] {c['claims']} claims · {c['released_items']} post-speech "
          f"items released across {c['claims_touched']} claims")
        A(f"      bearing: {c['released_bearing']}  "
          f"(bearing AND Tier-1..3, i.e. able to credit: "
          f"{c['released_bearing_t13']})")
        A(f"      by agency: {c['by_agency']}")
        A(f"      by rule:   {c['by_rule']}")
        A(f"      by tier:   {c['by_tier']}")
        A(f"      GATE OUTCOMES CHANGED: {c['gate_changed']}")
        A("")
    A(f"  {'speech':<14}{'items':>7}{'bearing':>9}{'T13':>6}{'gate Δ':>8}"
      f"{'items':>8}{'bearing':>9}{'T13':>6}{'gate Δ':>8}")
    A(f"  {'':<14}{'stored':>30}{'rescored':>31}")
    for s in report["per_speech"]:
        a, b = s["vintages"]["stored"], s["vintages"]["rescored"]
        A(f"  {s['speech']:<14}{a['released_items']:>7}{a['released_bearing']:>9}"
          f"{a['released_bearing_t13']:>6}{a['gate_changed']:>8}"
          f"{b['released_items']:>8}{b['released_bearing']:>9}"
          f"{b['released_bearing_t13']:>6}{b['gate_changed']:>8}")
    A("")
    for v in VINTAGES:
        rel = [f for s in report["per_speech"]
               for f in s["vintages"][v]["released_sids"]]
        A(f"[{v}] RELEASED by D16(α) — {len(rel)} claim(s):")
        for f in rel:
            A(f"    {f['sid']:<20} {f['old_verdict'] or '-':<13} "
              f"{','.join(f['agencies'])} · {','.join(f['rules'])}")
            A(f"      {f['claim']}")
            for u in f["urls"]:
                A(f"      · {u}")
        gated = [f for s in report["per_speech"]
                 for f in s["vintages"][v]["newly_gated_sids"]]
        if gated:
            A(f"[{v}] NEWLY GATED by D16 (a defect — D16 only adds credits): "
              f"{gated}")
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
