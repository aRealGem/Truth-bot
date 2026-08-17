"""D17-d STEP 6: decomposition spec for the 5 compound rows ($0, RECORD ONLY).

    scripts/d17d_step6_decomposition_spec.py [--json PATH]

RECORD ONLY. Segmentation itself is deferred: all five rows stay fail-closed
excluded until they are actually segmented, and nothing here changes a lane, a
verdict or a published state.

Three rows carry OWNER GUIDANCE (recorded verbatim in substance from the
2026-08-17 batch). The remaining two are DESK DRAFTS by ccagent, queued for the
same owner ratification reply -- they are marked as such and carry no more
authority than a proposal.

The five rows are READ FROM ``data/decidability.json`` rather than hard-coded,
so if the registry's compound set ever changes this script fails loudly instead
of silently speccing a stale list.

A NOTE ON WHAT DECOMPOSITION IS FOR. A compound utterance hides a checkable core
inside uncheckable wrapping. Splitting is not a way to manufacture a verdict --
it is how the checkable part stops being suppressed by the part that never could
be. Each split below therefore labels every fragment with where it goes: a lane
(web / series), a reason code family, or non-bearing.
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
OUT = REPO / "metrics" / "remediation_v2" / "d17d_step6_decomposition_spec.json"

OWNER = "owner-guidance (2026-08-17 batch)"
DESK = "desk-draft (ccagent) -- queued for the same owner ratification reply"

GUIDANCE: dict[str, dict] = {
    "trump_2026:0057": {
        "source": OWNER,
        "fragments": [
            {"fragment": "record number of job-killing regulations cut",
             "route": "web lane",
             "detail": "Federal Register / Unified Agenda regulation counts."},
            {"fragment": "2.4 million Americans lifted off food stamps",
             "route": "series lane",
             "detail": "SNAP participation series."},
            {"fragment": "'job-killing'",
             "route": "non-bearing",
             "detail": "Characterization; carries no checkable proposition."},
        ],
    },
    "trump_2026:0130": {
        "source": OWNER,
        "fragments": [
            {"fragment": "I urged this Congress to begin the mission",
             "route": "non-bearing",
             "detail": ("Trivially true and near-empty; the owner noted the "
                        "'begin' hedge drains it further.")},
            {"fragment": "the largest tax cuts in American history",
             "route": "series lane / web lane once operationalized",
             "detail": ("This is the core. Operationalize as revenue effect, "
                        "share of GDP, on a standard CBO/Treasury comparison "
                        "basis. If it is NOT operationalized, the claim is "
                        "NO-METRIC.")},
        ],
    },
    "trump_2026:0343": {
        "source": OWNER,
        "fragments": [
            {"fragment": "her heartbroken mother is in the gallery",
             "route": "trivially checkable",
             "detail": "Presence in the chamber."},
            {"fragment": "deporting ... at record numbers",
             "route": "series lane",
             "detail": ("ICE ERO removals series. VINTAGE CAVEAT: the "
                        "comparison must use figures published on or before "
                        "the utterance.")},
            {"fragment": "'why we are deporting'",
             "route": "non-bearing (INTENT family)",
             "detail": "A motive wrapper; intent is not established by records."},
        ],
    },
    "biden_2022:0154": {
        "source": DESK,
        "fragments": [
            {"fragment": "Intel's CEO ... told me they are ready to ...",
             "route": "non-bearing (PRIVATE-EVENT family)",
             "detail": ("A private remark reported by the speaker. Nothing "
                        "public settles what was said to him.")},
            {"fragment": ("Intel is ready to increase its investment from "
                          "$20 billion to $100 billion"),
             "route": "web lane",
             "detail": ("The checkable core the desk said to keep: company "
                        "announcements and SEC filings carry the figures. "
                        "VINTAGE CAVEAT: sources must be published on or "
                        "before the utterance date.")},
            {"fragment": "who is here tonight",
             "route": "trivially checkable",
             "detail": "Gallery presence."},
        ],
        "draft_note": ("Split keeps the figure and drops only the private "
                       "remark, which is exactly the desk's instruction."),
    },
    "clinton_1998:0132": {
        "source": DESK,
        "fragments": [
            {"fragment": "we have also met that goal, two full years ahead of schedule",
             "route": "blocked on referent binding, then lane by substance",
             "detail": ("'That goal' is named in a PRIOR sentence, so this "
                        "utterance carries no standalone proposition. This is "
                        "referent binding, not a multi-claim compound: "
                        "re-segment to include the antecedent, producing a "
                        "self-contained claim of the form '<goal> was met two "
                        "years ahead of schedule', and only then route -- a "
                        "numeric target goes to the series lane, a program "
                        "milestone to the web lane.")},
        ],
        "draft_note": ("SAME UNDERLYING PROBLEM as trump_2026:0106 ('I was "
                       "there'), which the code map returns as UNCODED under "
                       "the no-standalone-proposition gap. One sits on the "
                       "decomposition track and the other on the reason-code "
                       "track, but both are unbound referents. Worth handling "
                       "uniformly rather than fixing twice."),
    },
}


def run(out_path: Path = OUT) -> dict:
    from truthbot.publish.decidability import load_decidability

    entries = load_decidability(DECIDABILITY)
    compound = [e for e in entries if e["decidability"] == "needs-decomposition"]
    sids = sorted(e["sid"] for e in compound)

    # Read, never assume: if the registry's compound set has moved, say so
    # rather than speccing a list that no longer exists.
    if sorted(GUIDANCE) != sids:
        raise SystemExit(
            "decomposition set out of sync with data/decidability.json:\n"
            f"  registry: {sids}\n  spec:     {sorted(GUIDANCE)}")

    by_sid = {e["sid"]: e for e in compound}
    rows = []
    for sid in sids:
        g = GUIDANCE[sid]
        rows.append({
            "sid": sid,
            "speech": by_sid[sid]["speech_id"],
            "desk_why": by_sid[sid]["why"],
            "guidance_source": g["source"],
            "fragments": g["fragments"],
            **({"draft_note": g["draft_note"]} if g.get("draft_note") else {}),
        })

    report = {
        "schema": "truthbot-d17d-step6-decomposition-spec v1",
        "generated": _now(),
        "status": "RECORD ONLY -- segmentation deferred",
        "status_note": ("All five rows remain fail-closed excluded until they "
                        "are segmented. Nothing here changes a lane, a verdict "
                        "or a published state, and no row is ratified."),
        "n_rows": len(rows),
        "owner_guided": [r["sid"] for r in rows if r["guidance_source"] == OWNER],
        "desk_drafted": [r["sid"] for r in rows if r["guidance_source"] == DESK],
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return report


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def print_report(rep: dict) -> None:
    print(f"\nD17-d step 6 decomposition spec -- {rep['n_rows']} rows "
          f"({rep['status']})\n")
    for r in rep["rows"]:
        tag = "OWNER" if r["guidance_source"].startswith("owner") else "DESK DRAFT"
        print(f"  {r['sid']:<20} [{tag}]")
        for f in r["fragments"]:
            print(f"      - {f['fragment'][:64]:<64} -> {f['route']}")
    print(f"\n  owner-guided: {rep['owner_guided']}")
    print(f"  desk-drafted: {rep['desk_drafted']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=str(OUT))
    args = ap.parse_args(argv)
    rep = run(Path(args.json))
    print_report(rep)
    print(f"\nspec -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
