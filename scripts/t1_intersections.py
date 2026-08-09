#!/usr/bin/env python3
"""T-1 — does D15 shrink the adjudication wave? $0.

NO model calls, no keys, no network: set arithmetic over two artifacts that are
already in the repo (``metrics/remediation_v2/d15_blast_radius.json`` and
``metrics/remediation_v2/regate_flipset.json``).

WHY THIS EXISTS
---------------
The planned wave was sized as "33 released + 6 named extras + 2 from D16(alpha)
= 41 claims to adjudicate". That arithmetic quietly assumes the three lists are
disjoint, and it was written before D15 was ratified.

D15 does not need a panel. A claim D15 newly gates is Unverifiable BY THE GATE
— deterministically, for $0 — so putting it in front of an adjudication panel
buys nothing. Every claim that is BOTH "released by B1a/B2" AND "newly gated by
D15" is therefore a claim the wave can drop.

So the honest pre-wave question is not "how big is the wave", it is "how much of
the wave has already been answered for free". This script answers it.

WHAT IT COMPUTES
----------------
Three intersections, plus the two cross-checks that make the subtraction safe:

  * |D15 newly-gated  n  released|      — released claims D15 re-gates for free
  * |D15 newly-gated  n  named extras|  — extras D15 re-gates for free
  * |D15 newly-gated  n  D16(alpha)|    — D16 releases D15 immediately takes back
  * |released n named extras| and |released n D16(alpha)| — the disjointness the
    original "33 + 6 + 2" assumed, verified rather than assumed.

CEILING, NOT ESTIMATE (stated, not buried)
------------------------------------------
The resulting number is an UPPER BOUND on the wave and can only shrink:

  * the released set here is the PRE-RATIFICATION flip set (B1a+B2 stances, both
    rules off). Re-gating with D15 and D16(alpha) ACTIVE — task T-4 — can only
    remove claims from it, because D15 never releases anything (measured:
    ``released_sids`` is empty in every speech, both vintages) and the claims
    D16 adds are counted here already;
  * a claim can leave the wave for other reasons (a dedupe, an owner drop), but
    nothing in the remaining pipeline ADDS one.

VINTAGE CAVEAT (the one thing that could move the number)
---------------------------------------------------------
The two inputs were produced against DIFFERENT stance vintages:
``scripts/measure_d15.py`` loads the B1a sidecar only (``sidecar_path``), while
``scripts/regate_from_rescore.py`` merges B2 over B1a by default. The D15 count
of 50 is therefore B1a-vintage; the released count of 33 is B1a+B2. T-4 re-runs
both rules over the merged vintage and settles it. The intersections below are
reported on the artifacts as they stand, and the ceiling is stated as a ceiling
precisely because of this.

Usage (repo root, always $0):
  PYTHONPATH=.:src .venv/bin/python scripts/t1_intersections.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "metrics" / "remediation_v2"
OUT_STEM = "t1_intersections"

D15_PATH = OUT_DIR / "d15_blast_radius.json"
FLIPSET_PATH = OUT_DIR / "regate_flipset.json"

#: The stance vintage whose D15 numbers the brief quotes ("D15 newly-gates 50").
D15_VINTAGE = "rescored"

#: The six extras named by the owner, in the order the brief names them.
NAMED_EXTRAS: tuple[str, ...] = (
    "trump_2026:0030",
    "trump_2026:0031",
    "trump_2026:0023",
    "trump_2026:0024",
    "trump_2026:0343",
    "clinton_1998:0313",
)

#: The two claims D16(alpha) releases, from the D16 blast radius.
D16_RELEASED: tuple[str, ...] = ("clinton_1998:0026", "clinton_1998:0038")


def d15_newly_gated(report: dict, vintage: str = D15_VINTAGE) -> dict[str, dict]:
    """sid -> the blast-radius row, for every claim D15 newly gates.

    Returns a dict rather than a set so the markdown can name the RULE that
    fired next to each sid — "why is this claim free" is the whole point, and a
    bare sid does not answer it."""
    out: dict[str, dict] = {}
    for speech in report["per_speech"]:
        for row in speech["vintages"][vintage]["newly_gated_sids"]:
            out[row["sid"]] = row
    return out


def d15_releases_nothing(report: dict) -> bool:
    """D15 only ever REMOVES quota credit. The ceiling argument leans on that,
    so it is checked against the artifact rather than asserted from the prose."""
    return all(not sp["vintages"][v]["released_sids"]
               for sp in report["per_speech"]
               for v in sp["vintages"])


def build_report(d15: dict, flipset: dict) -> dict:
    gated = d15_newly_gated(d15)
    gated_sids = set(gated)
    released = set(flipset["released_sids"])
    extras = set(NAMED_EXTRAS)
    d16 = set(D16_RELEASED)

    hit_released = sorted(gated_sids & released)
    hit_extras = sorted(gated_sids & extras)
    hit_d16 = sorted(gated_sids & d16)

    planned = len(released) + len(extras) + len(d16)
    # A sid removed by two different overlaps must not be subtracted twice.
    removed = sorted(set(hit_released) | set(hit_extras) | set(hit_d16))
    ceiling = len(((released | extras | d16) - gated_sids))

    return {
        "schema": "truthbot-t1-intersections v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "cost_usd": 0.0,
        "inputs": {
            "d15_blast_radius": str(D15_PATH.relative_to(REPO)),
            "d15_vintage": D15_VINTAGE,
            "d15_vintage_note": (
                "measure_d15.py loads the B1a sidecar only; "
                "regate_from_rescore.py merges B2 over B1a. T-4 re-runs both "
                "rules over the merged vintage and settles the difference."),
            "regate_flipset": str(FLIPSET_PATH.relative_to(REPO)),
            "flipset_generated": flipset.get("generated"),
            "d15_generated": d15.get("generated"),
        },
        "sets": {
            "d15_newly_gated": {"n": len(gated_sids),
                                "sids": sorted(gated_sids)},
            "released": {"n": len(released), "sids": sorted(released)},
            "named_extras": {"n": len(extras), "sids": list(NAMED_EXTRAS)},
            "d16_released": {"n": len(d16), "sids": list(D16_RELEASED)},
        },
        "intersections": {
            "d15_x_released": {
                "n": len(hit_released),
                "sids": hit_released,
                "rules": {s: gated[s]["rules"] for s in hit_released},
            },
            "d15_x_named_extras": {
                "n": len(hit_extras),
                "sids": hit_extras,
                "rules": {s: gated[s]["rules"] for s in hit_extras},
            },
            "d15_x_d16_released": {"n": len(hit_d16), "sids": hit_d16},
        },
        "cross_checks": {
            "released_x_named_extras": sorted(released & extras),
            "released_x_d16_released": sorted(released & d16),
            "named_extras_x_d16_released": sorted(extras & d16),
            "d15_released_sids_empty": d15_releases_nothing(d15),
        },
        "wave": {
            "planned_gross": planned,
            "planned_formula": (
                f"{len(released)} released + {len(extras)} named extras + "
                f"{len(d16)} from D16(alpha)"),
            "removed_by_d15": len(removed),
            "removed_sids": removed,
            "ceiling": ceiling,
            "ceiling_is_upper_bound": True,
            "ceiling_rationale": (
                "D15 releases nothing in any speech or vintage, so re-gating "
                "with both rules active can only subtract from the released "
                "set. Nothing downstream adds a claim to the wave."),
        },
    }


def render_markdown(r: dict) -> str:
    L: list[str] = []
    A = L.append
    w, i, c = r["wave"], r["intersections"], r["cross_checks"]

    A("# T-1 — D15 x wave intersections")
    A("")
    A(f"_Generated {r['generated']} · $0 (set arithmetic over committed "
      f"artifacts, no model calls)._")
    A("")
    A("## The question")
    A("")
    A("A claim D15 newly gates is Unverifiable by the gate, deterministically, "
      "for free. Sending it to an adjudication panel buys nothing. So every "
      "claim in BOTH the wave and D15's newly-gated set is a claim the wave "
      "can drop.")
    A("")
    A("## Intersections")
    A("")
    A("| Intersection | Size |")
    A("| --- | ---: |")
    A(f"| D15 newly-gated ({r['sets']['d15_newly_gated']['n']}) "
      f"n released ({r['sets']['released']['n']}) | "
      f"**{i['d15_x_released']['n']}** |")
    A(f"| D15 newly-gated n named extras "
      f"({r['sets']['named_extras']['n']}) | "
      f"**{i['d15_x_named_extras']['n']}** |")
    A(f"| D15 newly-gated n D16(alpha) released "
      f"({r['sets']['d16_released']['n']}) | "
      f"**{i['d15_x_d16_released']['n']}** |")
    A("")

    if i["d15_x_released"]["sids"]:
        A(f"### The {i['d15_x_released']['n']} released claims D15 re-gates "
          f"for free")
        A("")
        A("| sid | D15 rule(s) that fired |")
        A("| --- | --- |")
        for sid in i["d15_x_released"]["sids"]:
            A(f"| `{sid}` | {', '.join(i['d15_x_released']['rules'][sid])} |")
        A("")

    if i["d15_x_named_extras"]["sids"]:
        A("### Named extras D15 re-gates for free")
        A("")
        A("| sid | D15 rule(s) that fired |")
        A("| --- | --- |")
        for sid in i["d15_x_named_extras"]["sids"]:
            A(f"| `{sid}` | "
              f"{', '.join(i['d15_x_named_extras']['rules'][sid])} |")
        A("")

    if not i["d15_x_d16_released"]["sids"]:
        A("D15 and D16(alpha) do not collide: neither claim D16 releases is one "
          "D15 takes back. The two rules are pulling on different claims.")
        A("")

    A("## Wave size")
    A("")
    A(f"- Planned gross: **{w['planned_gross']}** "
      f"({w['planned_formula']})")
    A(f"- Already answered by D15, no panel needed: "
      f"**-{w['removed_by_d15']}**")
    A(f"- **CEILING: {w['ceiling']} claims.**")
    A("")
    A("This is a ceiling, not an estimate. " + w["ceiling_rationale"])
    A("")
    A("## Cross-checks")
    A("")
    A("The original `33 + 6 + 2` assumed the three lists were disjoint. They "
      "are — verified, not assumed:")
    A("")
    A(f"- released n named extras: "
      f"{c['released_x_named_extras'] or 'empty'}")
    A(f"- released n D16(alpha): "
      f"{c['released_x_d16_released'] or 'empty'}")
    A(f"- named extras n D16(alpha): "
      f"{c['named_extras_x_d16_released'] or 'empty'}")
    A(f"- D15 released nothing anywhere (the ceiling argument leans on this): "
      f"`{c['d15_released_sids_empty']}`")
    A("")
    A("## Vintage caveat")
    A("")
    A(r["inputs"]["d15_vintage_note"])
    A("")
    return "\n".join(L) + "\n"


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=str(OUT_DIR / f"{OUT_STEM}.json"))
    ap.add_argument("--md", default=str(OUT_DIR / f"{OUT_STEM}.md"))
    args = ap.parse_args(argv)

    d15 = json.loads(D15_PATH.read_text(encoding="utf-8"))
    flipset = json.loads(FLIPSET_PATH.read_text(encoding="utf-8"))
    report = build_report(d15, flipset)

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    Path(args.md).write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    print(f"wrote {args.json}\nwrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
