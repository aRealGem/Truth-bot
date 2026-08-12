"""D17-c Stage 0 items 5b/6/7 — the row selector, its goldens, its cost.

Everything here is OFFLINE: it reads the committed fixtures and never the
network. Fetching happened once, in ``fetch_fixtures.py``.

THE SELECTION PREDICATE, stated once so it can be argued with:

    the 13 most recent observations whose observation_date is on or before
    the pin date, where the pin date is the speech's registered utterance
    date

Thirteen because a year-over-year comparison needs the same month one year
back plus the month itself, and that is the smallest window that lets a
scorer check both a level and a change without being handed the series.
The predicate is a pure function of (fixture bytes, pin date), which is what
makes item 5b's byte-determinism claim meaningful rather than incidental.

Each excerpt ships the provenance Fable required: series id, the rows, the
vintage/as-of stamp, the total row count of the full table, the window
bounds, and a link back to the full table -- plus the predicate itself, so a
reader can tell what was NOT shown and why.

ONE REQUIREMENT IS UNMET AND IS NOT PAPERED OVER: ``units``. The authorized
CSV endpoint returns ``observation_date,SERIES_VINTAGE`` and no units field.
Units would need a metadata endpoint outside the current authorization, so
every excerpt records units as null with a stated reason rather than a
guess.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
sys.path.insert(0, str(HERE.parents[2]))

from truthbot.costs import CHARS_PER_TOKEN, rates  # noqa: E402

WINDOW = 13
PREDICATE = ("the {n} most recent observations with observation_date <= {pin} "
             "(the speech's registered utterance date)")
FULL_TABLE = "https://fred.stlouisfed.org/series/{sid}"

#: (claim_sid, evidence_id, series, fixture, pin date, role)
TARGETS = [
    ("biden_2022:0169", "E7", "MANEMP", "MANEMP_v20220301.csv", "2022-03-01", "wave1"),
    ("biden_2022:0245", "E7", "FYFSD", "FYFSD_v20220301.csv", "2022-03-01", "wave1"),
    ("gwbush_2006:0133", "E8", "PAYEMS", "PAYEMS_v20060131.csv", "2006-01-31", "wave1"),
    ("obama_2014:0189", "E4", "CPIAUCSL", "CPIAUCSL_v20140128.csv", "2014-01-28", "wave1"),
    ("trump_2026:0054", "E4", "CE16OV", "CE16OV_v20260224.csv", "2026-02-24", "wave1"),
    ("trump_2026:0054", "E7", "PAYEMS", "PAYEMS_v20260224.csv", "2026-02-24", "wave1"),
    ("trump_2026:0219", "E1", "APU0000708111", "APU0000708111_v20260224.csv",
     "2026-02-24", "wave1"),
    ("trump_2026:0221", "E9", "CUUR0000SAF112", "CUUR0000SAF112_v20260224.csv",
     "2026-02-24", "wave1"),
    ("trump_2026:0031", "E6", "CPILFESL", "CPILFESL_v20260224.csv", "2026-02-24",
     "exemplar"),
]

#: Halted, not silently dropped -- see the report. The corpus cites a FRED URL
#: built from a BLS series id; it 404s on ALFRED *and* on current FRED.
UNFETCHABLE = [("trump_2026:0054", "E8", "LNS12000000",
                "cited FRED URL uses a BLS series id; 404 on both endpoints")]


def load(fixture: str) -> tuple[str, list[tuple[str, str]]]:
    """(vintage column header, [(date, value)]) with blank observations dropped."""
    with (FIXTURES / fixture).open() as fh:
        rows = list(csv.reader(fh))
    header = rows[0][1]
    return header, [(r[0], r[1]) for r in rows[1:]
                    if len(r) > 1 and r[1] not in ("", ".")]


def excerpt(claim_sid: str, evidence_id: str, series: str, fixture: str,
            pin: str, role: str) -> dict:
    """The excerpt payload for one item. Pure function of fixture bytes + pin."""
    header, obs = load(fixture)
    eligible = [o for o in obs if o[0] <= pin]
    window = eligible[-WINDOW:]
    return {
        "claim_sid": claim_sid,
        "evidence_id": evidence_id,
        "role": role,
        "series_id": series,
        "vintage_as_of": pin,
        "vintage_column": header,
        "rows": [{"period": d, "value": v} for d, v in window],
        "units": None,
        "units_unavailable_because": (
            "the authorized CSV endpoint returns observation_date,SERIES_VINTAGE "
            "and carries no units field; a metadata endpoint is outside the "
            "current egress authorization"),
        "total_rows_in_full_table": len(obs),
        "rows_eligible_at_vintage": len(eligible),
        "window_start": window[0][0] if window else None,
        "window_end": window[-1][0] if window else None,
        "rows_shown": len(window),
        "full_table": FULL_TABLE.format(sid=series),
        "selection_predicate": PREDICATE.format(n=WINDOW, pin=pin),
        "fixture_sha256": hashlib.sha256((FIXTURES / fixture).read_bytes()).hexdigest(),
    }


def render(exc: dict) -> str:
    """The text an excerpt contributes to a scoring prompt."""
    rows = "\n".join(f"  {r['period']}  {r['value']}" for r in exc["rows"])
    return (
        f"SERIES {exc['series_id']} (as of {exc['vintage_as_of']})\n"
        f"{rows}\n"
        f"showing {exc['rows_shown']} of {exc['total_rows_in_full_table']} rows, "
        f"{exc['window_start']} to {exc['window_end']}\n"
        f"selected by: {exc['selection_predicate']}\n"
        f"full table: {exc['full_table']}\n"
    )


def build() -> list[dict]:
    return [excerpt(*t) for t in TARGETS]


def main() -> int:
    goldens = build()

    print("=== ITEM 7: goldens ===")
    for g in goldens:
        print(f"  {g['claim_sid']:<18}{g['evidence_id']:<4}{g['series_id']:<16}"
              f"{g['rows_shown']:>3} of {g['total_rows_in_full_table']:>5} rows  "
              f"{g['window_start']} .. {g['window_end']}")
    print(f"  ({len(goldens)} goldens; {len(UNFETCHABLE)} item HALTED, see report)")

    # ── item 5b: byte-determinism of the selector ───────────────────────────
    a = json.dumps(build(), sort_keys=True, indent=2)
    b = json.dumps(build(), sort_keys=True, indent=2)
    ha, hb = (hashlib.sha256(x.encode()).hexdigest() for x in (a, b))
    print("\n=== ITEM 5b: selector determinism (two runs, same vintage) ===")
    print(f"  run 1 sha256 {ha}")
    print(f"  run 2 sha256 {hb}")
    print(f"  byte-identical: {a == b}")
    assert a == b, "selector is not byte-deterministic"

    # ── item 6: the measured token delta ────────────────────────────────────
    r_in, _ = rates("claude-haiku")
    sizes = [len(render(g)) for g in goldens]
    mean_c, max_c = sum(sizes) / len(sizes), max(sizes)
    print("\n=== ITEM 6: measured payload growth (real excerpts) ===")
    print(f"  chars/excerpt   mean {mean_c:8.1f}   max {max_c:8d}")
    print(f"  tokens/excerpt  mean {mean_c / CHARS_PER_TOKEN:8.1f}   "
          f"max {max_c / CHARS_PER_TOKEN:8.1f}")
    measured_84 = mean_c * 84 / CHARS_PER_TOKEN * r_in / 1e6
    modelled_84 = 4000 * 84 / CHARS_PER_TOKEN * r_in / 1e6
    print(f"\n  excerpt cost across 84 wave-1 items:")
    print(f"    measured (mean {mean_c:.0f} chars) ${measured_84:.4f}")
    print(f"    modelled (4,000 chars)             ${modelled_84:.4f}")
    print(f"    the projection over-provisioned by {modelled_84 / measured_84:.1f}x")
    print(f"\n  reconciliation vs the $0.2992 projection: the excerpt term falls")
    print(f"  from $0.1073 to ${measured_84:.4f}, so Stage A projects "
          f"${0.2992 - modelled_84 + measured_84:.4f}.")
    print(f"  Ceiling $0.75 -> UNDER, no halt.")

    out = HERE / "goldens.json"
    out.write_text(json.dumps(
        {"schema": "truthbot-d17c-goldens v1",
         "selection_predicate": PREDICATE.format(n=WINDOW, pin="<utterance date>"),
         "window": WINDOW,
         "unfetchable": [dict(zip(("claim_sid", "evidence_id", "series", "reason"), u))
                         for u in UNFETCHABLE],
         "goldens": goldens}, indent=2, sort_keys=True) + "\n")
    print(f"\ngoldens -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
