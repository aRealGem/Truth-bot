"""D17-d STEP 6: map reason codes onto the 35 substantive rows ($0, PROPOSAL only).

    scripts/d17d_step6_map_codes.py [--json PATH]

WHAT THIS IS
------------
An assignment of the owner-approved reason codes (``data/reason_codes.json``) to
every row recorded as ``undecidable-from-public-record``, made from the claim
text plus the desk rationale already in ``data/decidability.json``.

IT IS A PROPOSAL. Nothing here is written into ``data/decidability.json`` and
nothing is flipped to ``owner-ratified``. Fable maps the same 35 independently
and the owner sees only the disagreements plus the UNCODED rows; the codes
become real on that one owner reply, not here.

THE RULE THAT MATTERS: NEVER FORCE-FIT
--------------------------------------
A code is assigned only on a CLEAR fit. A stretch, or a tie between two codes,
resolves to UNCODED. That is not a failure mode -- it is the mechanism that
surfaces a missing code instead of hiding it under an approximate one, and an
UNCODED row simply keeps its current unpublished state.

That rule earns its keep here: the 7 UNCODED rows are not scattered noise, they
cluster into three coherent gaps (see ``uncoded_gap_clusters``), each of which
is a candidate new code with its spawning row already identified.

The assignments are a literal table below, not a heuristic. Mapping a claim to
a reason it cannot be checked is a judgement about meaning; pretending a
keyword rule made it would misrepresent how it was actually done. The AUDIT is
computed, so the counts cannot drift from the table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

DECIDABILITY = REPO / "data" / "decidability.json"
CODES = REPO / "data" / "reason_codes.json"
OUT = REPO / "metrics" / "remediation_v2" / "d17d_step6_code_map.json"

SPEAKERS = {"biden_2022": "Joe Biden", "clinton_1998": "Bill Clinton",
            "gwbush_2006": "George W. Bush", "obama_2014": "Barack Obama",
            "trump_2026": "Donald Trump"}

#: sid -> (reason_code, reason_code_2 or None, mapper_note or None).
#: mapper_note is carried ONLY for UNCODED rows, per the step-6 spec.
ASSIGNMENTS: dict[str, tuple] = {
    # -- evaluative comparison with no stated measure ------------------------
    "biden_2022:0045": ("NO-METRIC", None, None),
    "biden_2022:0124": ("NO-METRIC", None, None),
    # -- a view attributed to an uncountable group ---------------------------
    "biden_2022:0194": ("MASS-VOICE", None, None),
    "biden_2022:0373": ("MASS-VOICE", None, None),
    "trump_2026:0514": ("MASS-VOICE", "COUNTERFACTUAL", None),
    # -- interior state: intends / believes / aims --------------------------
    "gwbush_2006:0033": ("INTENT", None, None),
    "clinton_1998:0240": ("INTENT", None, None),
    "trump_2026:0110": ("INTENT", None, None),
    # -- a number for a population nobody measures --------------------------
    "clinton_1998:0350": ("NO-INSTRUMENT", None, None),
    "trump_2026:0334": ("NO-INSTRUMENT", None, None),
    # -- private life: no public witness or record ---------------------------
    "biden_2022:0100": ("PRIVATE-EVENT", None, None),
    "biden_2022:0431": ("PRIVATE-EVENT", None, None),
    "clinton_1998:0134": ("PRIVATE-EVENT", None, None),
    "clinton_1998:0135": ("PRIVATE-EVENT", None, None),
    "obama_2014:0004": ("PRIVATE-EVENT", None, None),
    "obama_2014:0123": ("PRIVATE-EVENT", None, None),
    "obama_2014:0125": ("PRIVATE-EVENT", None, None),
    "obama_2014:0126": ("PRIVATE-EVENT", None, None),
    "trump_2026:0137": ("PRIVATE-EVENT", None, None),
    "trump_2026:0153": ("PRIVATE-EVENT", None, None),
    "trump_2026:0255": ("PRIVATE-EVENT", None, None),
    "trump_2026:0279": ("PRIVATE-EVENT", None, None),
    "trump_2026:0327": ("PRIVATE-EVENT", None, None),
    "trump_2026:0328": ("PRIVATE-EVENT", None, None),
    "trump_2026:0329": ("PRIVATE-EVENT", None, None),
    "trump_2026:0482": ("PRIVATE-EVENT", "INTENT", None),
    "trump_2026:0487": ("PRIVATE-EVENT", None, None),
    "trump_2026:0638": ("PRIVATE-EVENT", None, None),
    # -- no clear fit: UNCODED, each with the reason it did not fit ----------
    "clinton_1998:0210": (
        "UNCODED", None,
        "Causal attribution: that NATO CONTAINED communism and KEPT two "
        "continents secure over 50 years. No code covers 'X caused Y'; "
        "NO-METRIC is the nearest but this is not a ranking or comparison. "
        "Pairs with trump_2026:0666."),
    "clinton_1998:0243": (
        "UNCODED", None,
        "Attributes agreement to a BOUNDED, SPECIFIED group (the members in "
        "the chamber). MASS-VOICE's copy says 'a large, unspecified group ... "
        "no source can count', which would misdescribe this row to a reader -- "
        "this group is enumerable, it is their inner agreement that is not. "
        "Pairs with trump_2026:0667."),
    "trump_2026:0106": (
        "UNCODED", None,
        "'I was there.' carries no proposition standing alone; its referent "
        "sits in an adjacent sentence. That is a segmentation problem, closer "
        "to needs-decomposition than to any reason code."),
    "trump_2026:0161": (
        "UNCODED", None,
        "A forward-looking conditional PROJECTION about future account "
        "balances. COUNTERFACTUAL covers what WOULD HAVE happened in a past "
        "that did not occur; a projection is unrecorded for the different "
        "reason that it has not happened yet. Coding it COUNTERFACTUAL would "
        "state the wrong reason to the reader."),
    "trump_2026:0450": (
        "UNCODED", None,
        "The bearing content is an unrecorded facial expression that had "
        "public witnesses (people on the train), so PRIVATE-EVENT's 'no public "
        "witnesses or records' is false of it. The 'no one will ever forget' "
        "wrapper is rhetorical framing rather than a countable attribution."),
    "trump_2026:0666": (
        "UNCODED", None,
        "Causal attribution: that the mission and the crew's lives HINGED on "
        "one man. A citation can evidence what he did, not that everything "
        "turned on it. Desk records it in the same conflation family as ruling "
        "(d). Pairs with clinton_1998:0210."),
    "trump_2026:0667": (
        "UNCODED", None,
        "Inner awareness of a bounded, specified group (those in the "
        "helicopter). Same gap as clinton_1998:0243: MASS-VOICE would "
        "misdescribe an enumerable group as an uncountable one."),
}

#: Clusters the UNCODED rows fall into -- each a candidate code for the owner,
#: with its spawning row already named as the precedent a new code would need.
UNCODED_GAP_CLUSTERS = [
    {"gap": "causal-attribution",
     "sids": ["clinton_1998:0210", "trump_2026:0666"],
     "description": ("A claim that one thing CAUSED or DEPENDED ON another. "
                     "Evidence of the events does not establish the link, and "
                     "none of the six codes names causation."),
     "candidate_precedent": "trump_2026:0666"},
    {"gap": "bounded-group-agreement",
     "sids": ["clinton_1998:0243", "trump_2026:0667"],
     "description": ("A view or awareness attributed to a SPECIFIED, "
                     "enumerable group. MASS-VOICE is the right family but its "
                     "copy says 'large, unspecified', which is false of these "
                     "rows. Resolvable either by a new code or by widening "
                     "MASS-VOICE's wording at ratification."),
     "candidate_precedent": "clinton_1998:0243"},
    {"gap": "forward-projection",
     "sids": ["trump_2026:0161"],
     "description": ("A conditional claim about the future. Distinct from "
                     "COUNTERFACTUAL, which is about a past that did not "
                     "happen."),
     "candidate_precedent": "trump_2026:0161"},
    {"gap": "no-standalone-proposition",
     "sids": ["trump_2026:0106"],
     "description": ("The utterance has no checkable content without its "
                     "antecedent. Arguably belongs on the decomposition track "
                     "rather than carrying a reason code at all."),
     "candidate_precedent": "trump_2026:0106"},
    {"gap": "unrecorded-but-witnessed",
     "sids": ["trump_2026:0450"],
     "description": ("An observation that had witnesses but left no record. "
                     "PRIVATE-EVENT explicitly requires 'no public witnesses', "
                     "so it does not cover this."),
     "candidate_precedent": "trump_2026:0450"},
]

#: Observations about the APPROVED COPY, raised rather than acted on. The
#: wording is owner-approved and verbatim; this pass does not edit it.
COPY_OBSERVATIONS = [
    {"code": "PRIVATE-EVENT",
     "observation": (
         "The copy says 'a private MOMENT', but most rows it correctly covers "
         "are private CIRCUMSTANCES, HISTORIES or ONGOING STATES rather than "
         "moments -- 'she's a dispatcher', 'she'd never collected unemployment "
         "benefits', '13 years on and off welfare', 'she is now in the first "
         "grade'. The operative clause ('no public witnesses or records') fits "
         "all of them, so they are coded rather than force-fitted elsewhere, "
         "but a reader may find 'moment' odd on a multi-year history."),
     "affected_rows": 18,
     "recommendation": (
         "Consider widening to 'a private moment or circumstance' at "
         "ratification. NOT changed here: copy is owner-approved and verbatim.")},
]


def run(out_path: Path = OUT) -> dict:
    from truthbot.publish.decidability import load_decidability
    from truthbot.publish.reason_codes import known, load_reason_codes

    registry = load_reason_codes(CODES)
    entries = load_decidability(DECIDABILITY, reason_codes=registry)
    targets = [e for e in entries
               if e["decidability"] == "undecidable-from-public-record"]

    # Refuse to emit a partial or invented map: every target row must be
    # assigned, and every assignment must name a target row and a real code.
    by_sid = {e["sid"]: e for e in targets}
    missing = sorted(set(by_sid) - set(ASSIGNMENTS))
    extra = sorted(set(ASSIGNMENTS) - set(by_sid))
    if missing or extra:
        raise SystemExit(f"assignment table out of sync: missing={missing} "
                         f"not-a-substantive-row={extra}")
    vocabulary = known(registry)
    for sid, (primary, secondary, note) in ASSIGNMENTS.items():
        for code in (primary, secondary):
            if code is not None and code not in vocabulary:
                raise SystemExit(f"{sid}: {code!r} is not a defined reason code")
        if primary == "UNCODED" and not note:
            raise SystemExit(f"{sid}: UNCODED rows require a mapper_note")
        if primary != "UNCODED" and note:
            raise SystemExit(f"{sid}: mapper_note is for UNCODED rows only")

    rows = []
    for sid in sorted(ASSIGNMENTS):
        primary, secondary, note = ASSIGNMENTS[sid]
        speech = by_sid[sid]["speech_id"]
        row = {"sid": sid, "speech": speech,
               "speaker": SPEAKERS.get(speech, "(unmapped)"),
               "reason_code": primary}
        if secondary:
            row["reason_code_2"] = secondary
        if note:
            row["mapper_note"] = note
        rows.append(row)

    # M-6 audit: per-speaker distribution and per-speaker UNCODED rate.
    by_speaker: dict[str, dict] = {}
    by_code: dict[str, int] = {}
    for r in rows:
        s = by_speaker.setdefault(r["speaker"], {"rows": 0, "codes": {},
                                                 "uncoded": 0})
        s["rows"] += 1
        s["codes"][r["reason_code"]] = s["codes"].get(r["reason_code"], 0) + 1
        if r["reason_code"] == "UNCODED":
            s["uncoded"] += 1
        by_code[r["reason_code"]] = by_code.get(r["reason_code"], 0) + 1
    for s in by_speaker.values():
        s["uncoded_rate_pct"] = round(100 * s["uncoded"] / s["rows"], 1)

    n_uncoded = by_code.get("UNCODED", 0)
    report = {
        "schema": "truthbot-d17d-step6-code-map v1",
        "generated": _now(),
        "status": "PROPOSAL",
        "status_note": ("Not written into data/decidability.json and nothing "
                        "flipped to owner-ratified. These assignments become "
                        "real only on the owner reply that ratifies them "
                        "against Fable's independent mapping."),
        "reason_codes_source": str(CODES.relative_to(REPO)),
        "n_rows": len(rows),
        "rule": ("Never force-fit: a code is assigned only on a clear fit; a "
                 "stretch or a tie resolves to UNCODED."),
        "assignments": rows,
        "m6_audit": {
            "by_speaker": by_speaker,
            "by_code": by_code,
            "uncoded_total": n_uncoded,
            "uncoded_rate_pct": round(100 * n_uncoded / (len(rows) or 1), 1),
            "dual_coded": [r["sid"] for r in rows if r.get("reason_code_2")],
        },
        "uncoded_gap_clusters": UNCODED_GAP_CLUSTERS,
        "copy_observations": COPY_OBSERVATIONS,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return report


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def print_report(rep: dict) -> None:
    a = rep["m6_audit"]
    print(f"\nD17-d step 6 code map ({rep['status']}) -- {rep['n_rows']} rows\n")
    print("  by code:")
    for code, n in sorted(a["by_code"].items(), key=lambda kv: -kv[1]):
        print(f"    {code:<16} {n}")
    print(f"\n  UNCODED {a['uncoded_total']}/{rep['n_rows']} "
          f"({a['uncoded_rate_pct']}%), dual-coded: {a['dual_coded']}")
    print("\n  by speaker (M-6):")
    for who, d in sorted(a["by_speaker"].items(), key=lambda kv: -kv[1]["rows"]):
        codes = ", ".join(f"{k}={v}" for k, v in sorted(d["codes"].items()))
        print(f"    {who:<16} rows={d['rows']:<3} uncoded={d['uncoded']} "
              f"({d['uncoded_rate_pct']}%)  {codes}")
    print("\n  UNCODED clusters (candidate new codes):")
    for c in rep["uncoded_gap_clusters"]:
        print(f"    {c['gap']:<28} {c['sids']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=str(OUT))
    args = ap.parse_args(argv)
    rep = run(Path(args.json))
    print_report(rep)
    print(f"\nmap -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
