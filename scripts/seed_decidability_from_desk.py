#!/usr/bin/env python3
"""Seed data/decidability.json from the D17-d desk pass — $0, deterministic.

Records what the desk pass already concluded into the decidability registry,
every row stamped ``provenance: "desk"``.

THIS RATIFIES NOTHING. ``desk`` is not a publishable provenance: the registry's
``publishable_entries`` returns only ``owner-ratified`` rows, so nothing seeded
here can reach a page. Step 6 (the owner ratifying the substantive
classifications) is what flips provenance to ``owner-ratified``, one considered
row at a time. Seeding exists so that step is a review of recorded rows rather
than a re-typing of 128 of them.

Mapping from the desk's four classes to the axis:

  web-tier1      -> retrievable-pending-lane   (lane: web-tier1)
  series-core    -> retrievable-pending-lane   (lane: series)
  compound-split -> needs-decomposition
  substantive    -> undecidable-from-public-record  + a review_trigger

``retrieved-insufficient`` is deliberately NOT emitted here. It describes a pack
that was retrieved and fell short; the desk's whole point is that these lanes
were never run. Using it now would restate the gate's existing message and lose
the distinction the axis exists to draw.

Every ``undecidable-from-public-record`` row must name what would reopen it —
the registry validator rejects the file otherwise, so the triggers below are
load-bearing, not decoration.

Usage (repo root):
  PYTHONPATH=src python3 scripts/seed_decidability_from_desk.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DESK = REPO / "metrics" / "remediation_v2" / "d17d_triage.json"
OUT = REPO / "data" / "decidability.json"

sys.path.insert(0, str(REPO / "src"))
from truthbot.publish.decidability import (  # noqa: E402
    SCHEMA, load_decidability, summary,
)

#: Seed date. Passed as a constant, never read from the clock, so re-running is
#: byte-identical (the artifact is an audit fixture).
SEED_DATE = "2026-08-14"

CLASS_MAP = {
    "web-tier1": ("retrievable-pending-lane", "web-tier1"),
    "series-core": ("retrievable-pending-lane", "series"),
    "compound-split": ("needs-decomposition", None),
    "substantive": ("undecidable-from-public-record", None),
}

#: Why a substantive claim is beyond the public record decides what would
#: reopen it. Keyed on the shape of the undecidability, not on the claim.
_TRIGGERS = {
    "private-exchange": (
        "A published first-hand account by a participant, or a "
        "contemporaneous record of the exchange."),
    "interior-state": (
        "A first-hand published statement by the person about their own "
        "state at the time."),
    "unmeasured-population": (
        "A published measurement of the population with a stated method "
        "and period."),
    "attributed-intent": (
        "A published statement of intent by the actor, or a record "
        "establishing it (court finding, declassified assessment)."),
    "evaluative": (
        "A stipulated measure and period that makes the comparison "
        "checkable, agreed before retrieval."),
}

#: Which trigger applies, by the desk's own stated reason. Assigned by reading
#: the desk's ``why``, which is a human judgement being transcribed -- NOT a
#: derivation from claim text. Anything unmatched falls back to evaluative,
#: the weakest claim to permanence.
_TRIGGER_KEYS = (
    ("private-exchange", ("private conversation", "in private", "hospital",
                          "a room", "said in a room", "private moment",
                          "private fertility", "private remark")),
    ("interior-state", ("thought", "felt", "inner", "knew", "faces",
                        "closeness", "loved")),
    ("unmeasured-population", ("unmeasured", "quantifier", "many, if not most",
                               "mass attribution", "population")),
    ("attributed-intent", ("intent", "motive", "aim", "causal")),
)


def _trigger_for(why: str) -> str:
    low = (why or "").lower()
    for key, needles in _TRIGGER_KEYS:
        if any(n in low for n in needles):
            return _TRIGGERS[key]
    return _TRIGGERS["evaluative"]


def build() -> dict:
    desk = json.loads(DESK.read_text(encoding="utf-8"))
    entries = []
    for c in desk["claims"]:
        value, lane = CLASS_MAP[c["decidability_class"]]
        entry = {
            "sid": c["sid"],
            "speech_id": c["speech"],
            "decidability": value,
            "provenance": "desk",
            "date": SEED_DATE,
            "why": c["why"],
            "desk_class": c["decidability_class"],
        }
        if lane:
            entry["lane"] = lane
        if c.get("candidate_series"):
            entry["candidate_series"] = c["candidate_series"]
        if value == "undecidable-from-public-record":
            entry["review_trigger"] = _trigger_for(c["why"])
        entries.append(entry)
    entries.sort(key=lambda e: e["sid"])
    return {
        "schema": SCHEMA,
        "comment": (
            "Seeded from the D17-d desk pass. Every row is provenance 'desk' "
            "and therefore NOT publishable -- publishable_entries() returns "
            "only 'owner-ratified'. Step 6 ratification flips rows one at a "
            "time. Regenerate with scripts/seed_decidability_from_desk.py."),
        "seeded_from": str(DESK.relative_to(REPO)),
        "entries": entries,
    }


def main() -> int:
    doc = build()
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    # Round-trip through the real validator: a file this script cannot load is
    # a file that must not ship.
    loaded = load_decidability(OUT)
    s = summary(loaded)
    print(f"seeded {s['total']} entries -> {OUT.relative_to(REPO)}")
    print(f"  publishable (owner-ratified): {s['publishable']}")
    print("  by value:")
    for k, v in sorted(s["by_value"].items()):
        print(f"    {k:<34}{v:>4}")
    print("  by provenance:")
    for k, v in sorted(s["by_provenance"].items()):
        print(f"    {k:<34}{v:>4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
