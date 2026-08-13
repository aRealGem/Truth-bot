"""D17-c Stage 0 items 5b/6/7 — the row selector, its goldens, its cost.

Everything here is OFFLINE: it reads the committed fixtures and never the
network. Fetching happened once, in ``fetch_fixtures.py``.

THE SELECTION PREDICATE, stated once so it can be argued with:

    DEFAULT -- the trailing K observations at the series' native frequency
    whose observation_date is on or before the pin date, where the pin date
    is the speech's registered utterance date and K is 25 monthly, 9
    quarterly, 13 annual. Frequency is the median spacing of the last four
    eligible observations, and is recorded in the predicate rather than
    assumed.

    EXTENSIONS -- four fixed regex rules. Each rule that fires proposes a
    start date; the WIDEST proposal wins, ties broken on (date, name), and
    every fired rule is named in the predicate so a reader can see which one
    set the window:
        explicit years              -> start Jan 1 of (min_year - 1)
        last/past N years           -> N+1 years back from the pin
        record|ever|history|never   -> the full eligible history
        took office|administration  -> trailing 5 years

    RULE SCOPE IS NOT UNIFORM (R1 = (b)). Three of the four read claim text
    + " || " + context. ``record|ever|history|all-time|never`` reads the
    CLAIM TEXT ONLY. Under text+context it fired on biden_2022:0169 -- "we
    created 369,000 new manufacturing jobs just last year", a claim with no
    superlative of its own -- because a NEIGHBOURING sentence carried one,
    pulling the full 997-row MANEMP history for a claim about a single year.
    A rule that keys on a claim's own assertion should read the claim's own
    words. The other three describe a period rather than assert a
    superlative, so context legitimately informs them.

An excerpt asserts at most MAX_ROWS rows and halts rather than truncating,
so a runaway window is a stop, not a silent trim. The predicate is a pure
function of (fixture bytes, pin date, claim text, context), which is what
makes item 5b's byte-determinism claim meaningful rather than incidental.

WHY NOT A FLAT OBSERVATION COUNT (the Stage 0 limitation, now closed): the
old window counted OBSERVATIONS, not time, so thirteen meant thirteen months
on a monthly series and thirteen YEARS on FYFSD. Ruled on before Stage A;
the frequency-aware form above replaces it. Determinism is unaffected either
way.

KNOWN LIMITATION, carried: a claim whose comparison anchor is a proper noun
rather than a date escapes all four rules. ``obama_2014:0189`` compares the
minimum wage to "when Ronald Reagan first stood here" (circa 1982); no rule
fires, so it takes the default 25-month window, which does not reach the
period the claim is actually about. Recorded, not papered over -- closing it
needs a new rule and a ruling.

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
import datetime as dt
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
RUNS = HERE.parents[1] / "pca_runs"
sys.path.insert(0, str(HERE.parents[2]))

from truthbot.costs import CHARS_PER_TOKEN, rates  # noqa: E402

#: Trailing observation count per native frequency (Fable's D17-c ruling).
WINDOW_BY_FREQ = {"monthly": 25, "quarterly": 9, "annual": 13}

#: Median spacing in days -> frequency label. Ordered, first match wins.
FREQ_BANDS = ((45, "monthly"), (180, "quarterly"), (10**6, "annual"))

#: Halt rather than truncate. A runaway window is a stop, not a silent trim.
MAX_ROWS = 1500

#: Fable's Stage A-FRED spend ceiling. Halt-and-report, never trim to fit.
CEILING = 0.15

#: The five publishing heads, for claim text + context. Committed, offline.
HEADS = {
    "trump_2026": "91dd7a34-7a3c-4f40-bcdc-276b2cb15d26",
    "biden_2022": "ddb05ee3-7d9c-4b2c-beaf-e197b9354379",
    "obama_2014": "2cbda3e4-c578-442a-aee7-c5c28a388048",
    "clinton_1998": "49b2e3e8-1667-4460-8989-b265914d4450",
    "gwbush_2006": "5c923c25-b065-4a9f-80bf-d23db4f9bcd1",
}

#: The committed window extensions. Fixed regexes, each named when it fires.
WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
RE_EXPLICIT_YEARS = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
RE_LAST_N_YEARS = re.compile(
    r"\b(?:last|past)\s+(\d{1,2}|" + "|".join(WORD_NUMBERS) + r")\s+years?\b",
    re.I)
RE_RECORD_EVER = re.compile(r"\b(?:record|ever|history|all[-\s]?time|never)\b",
                            re.I)
RE_TOOK_OFFICE = re.compile(r"\b(?:took office|administration|inherited)\b",
                            re.I)

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


_CLAIM_CACHE: dict[str, dict[str, dict]] = {}


def claim_record(claim_sid: str) -> dict:
    """(text, context) for a claim sid, read from its committed publishing head."""
    speech = claim_sid.split(":")[0]
    if speech not in _CLAIM_CACHE:
        head = json.loads((RUNS / f"{HEADS[speech]}.json").read_text())
        _CLAIM_CACHE[speech] = {c["sid"]: c for c in head["claims"]}
    return _CLAIM_CACHE[speech][claim_sid]


def claim_pack(claim_sid: str) -> list[dict]:
    """The evidence pack for a claim, from its committed publishing head."""
    speech = claim_sid.split(":")[0]
    head = json.loads((RUNS / f"{HEADS[speech]}.json").read_text())
    return head["evidence"][claim_sid]


def frequency(eligible: list[tuple[str, str]]) -> tuple[str, float]:
    """(label, median spacing in days) from the last four eligible observations."""
    tail = [dt.date.fromisoformat(d) for d, _ in eligible[-4:]]
    gaps = [(b - a).days for a, b in zip(tail, tail[1:])]
    spacing = statistics.median(gaps)
    for ceiling, label in FREQ_BANDS:
        if spacing < ceiling:
            return label, spacing
    raise AssertionError("unreachable: FREQ_BANDS has an open top band")


def resolve_window(claim_sid: str, series: str, pin: str,
                   eligible: list[tuple[str, str]]) -> tuple[list, str, dict]:
    """Apply the default and any fired extensions. Widest proposal wins.

    Returns (window rows, frequency label, provenance dict naming every rule
    that fired and the start date each one proposed).
    """
    rec = claim_record(claim_sid)
    haystack = f"{rec['text']} || {rec.get('context', '')}"
    # R1 = (b): the superlative rule reads the CLAIM'S OWN WORDS. The other
    # three rules stay on text + context. See the docstring for why.
    text_only = rec["text"]
    pin_date = dt.date.fromisoformat(pin)
    label, spacing = frequency(eligible)

    k = WINDOW_BY_FREQ[label]
    default_start = dt.date.fromisoformat(eligible[-k:][0][0])
    proposals: dict[str, dt.date] = {f"default trailing {k} ({label})": default_start}

    years = [int(y) for y in RE_EXPLICIT_YEARS.findall(haystack)]
    if years:
        proposals["explicit_years"] = dt.date(min(years) - 1, 1, 1)

    m = RE_LAST_N_YEARS.search(haystack)
    if m:
        raw = m.group(1).lower()
        n = int(raw) if raw.isdigit() else WORD_NUMBERS[raw]
        proposals["last_past_n_years"] = pin_date.replace(year=pin_date.year - (n + 1))

    if RE_RECORD_EVER.search(text_only):
        proposals["record_ever_history"] = dt.date.fromisoformat(eligible[0][0])

    if RE_TOOK_OFFICE.search(haystack):
        proposals["took_office"] = pin_date.replace(year=pin_date.year - 5)

    winner = min(proposals, key=lambda name: (proposals[name], name))
    start = proposals[winner]
    window = [o for o in eligible if dt.date.fromisoformat(o[0]) >= start]

    assert len(window) <= MAX_ROWS, (
        f"{claim_sid} {series}: window of {len(window)} rows exceeds "
        f"MAX_ROWS={MAX_ROWS} (rule '{winner}') -- halt and ask, do not truncate")

    return window, label, {
        "frequency": label,
        "median_spacing_days": spacing,
        "rules_fired": sorted(k2 for k2 in proposals if k2 != winner),
        "rule_applied": winner,
        "window_start_proposed": {k2: v.isoformat() for k2, v in sorted(proposals.items())},
    }


def predicate_text(pin: str, prov: dict, shown: int) -> str:
    """The predicate as it ships inside the excerpt, naming the rule that won."""
    fired = ", ".join(prov["rules_fired"]) or "none"
    return (
        f"{shown} observations with observation_date <= {pin} (the speech's "
        f"registered utterance date), selected by rule '{prov['rule_applied']}' "
        f"from start {prov['window_start_proposed'][prov['rule_applied']]}; "
        f"series frequency {prov['frequency']} (median spacing "
        f"{prov['median_spacing_days']:.0f} days); other rules fired: {fired}")


def excerpt(claim_sid: str, evidence_id: str, series: str, fixture: str,
            pin: str, role: str) -> dict:
    """The excerpt payload for one item.

    Pure function of (fixture bytes, pin date, claim text, context).
    """
    header, obs = load(fixture)
    eligible = [o for o in obs if o[0] <= pin]
    window, _freq, prov = resolve_window(claim_sid, series, pin, eligible)
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
        "window_selection": prov,
        "selection_predicate": predicate_text(pin, prov, len(window)),
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

    print("=== ITEM 7: goldens (frequency-aware windows) ===")
    for g in goldens:
        w = g["window_selection"]
        print(f"  {g['claim_sid']:<18}{g['evidence_id']:<4}{g['series_id']:<16}"
              f"{g['rows_shown']:>5} of {g['total_rows_in_full_table']:>5} rows  "
              f"{g['window_start']} .. {g['window_end']}")
        print(f"      {w['frequency']:<10} applied '{w['rule_applied']}'"
              f"   also fired: {', '.join(w['rules_fired']) or 'none'}")
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

    # ── item 6: the measured token delta, at Stage A scope ──────────────────
    #
    # SCOPE CORRECTION. The Stage 0 form of this block projected across all 84
    # wave-1 items against the $0.75 whole-programme bound. Stage A-FRED is a
    # whole-pack rescore of the 7 claims carrying the 8 excerptable FRED items
    # -- 67 pack items, not 84 excerpts -- against a $0.15 ceiling. Projecting
    # the old number against the new ceiling would compare two different runs.
    r_in, _ = rates("claude-haiku")
    wave1 = [g for g in goldens if g["role"] == "wave1"]
    sizes = [len(render(g)) for g in goldens]
    mean_c, max_c = sum(sizes) / len(sizes), max(sizes)
    print("\n=== ITEM 6: measured payload growth (real excerpts) ===")
    print(f"  chars/excerpt   mean {mean_c:8.1f}   max {max_c:8d}")
    print(f"  tokens/excerpt  mean {mean_c / CHARS_PER_TOKEN:8.1f}   "
          f"max {max_c / CHARS_PER_TOKEN:8.1f}")

    settlement = json.loads((HERE / "b2_settlement.json").read_text())
    per_item = settlement["measured_cost_usd"] / settlement["items"]
    pack_items = sum(len(claim_pack(sid))
                     for sid in sorted({g["claim_sid"] for g in wave1}))
    excerpt_chars = sum(len(render(g)) for g in wave1)
    base_term = per_item * pack_items
    excerpt_term = excerpt_chars / CHARS_PER_TOKEN * r_in / 1e6
    projection = base_term + excerpt_term

    print(f"\n  STAGE A-FRED projection ({len(wave1)} excerpts / "
          f"{len(set(g['claim_sid'] for g in wave1))} claims / "
          f"{pack_items} pack items):")
    print(f"    base term   {pack_items} items x ${per_item:.6f} "
          f"(B2 measured)   ${base_term:.4f}")
    print(f"    excerpt term {excerpt_chars:,} chars in                    "
          f"  ${excerpt_term:.4f}")
    print(f"    projection                                       ${projection:.4f}")
    print(f"    ceiling $0.15 -> {'UNDER' if projection <= CEILING else 'OVER'}, "
          f"headroom ${CEILING - projection:.4f}")
    assert projection <= CEILING, (
        f"projected ${projection:.4f} exceeds the ${CEILING:.2f} Stage A "
        f"ceiling -- halt and report, do not trim windows to fit")

    out = HERE / "goldens.json"
    out.write_text(json.dumps(
        {"schema": "truthbot-d17c-goldens v2 (frequency-aware)",
         "selection_predicate": (
             "trailing K observations at the series' native frequency with "
             "observation_date <= the utterance date, K = 25 monthly / 9 "
             "quarterly / 13 annual, frequency = median spacing of the last "
             "four eligible observations; widened by the first of four fixed "
             "regex rules to propose an earlier start, widest proposal wins, "
             "every fired rule named per item"),
         "window_by_frequency": WINDOW_BY_FREQ,
         "max_rows": MAX_ROWS,
         "extension_rules": {
             "explicit_years": {
                 "start": "Jan 1 of (min_year - 1)",
                 "scope": "claim text + ' || ' + context"},
             "last_past_n_years": {
                 "start": "N+1 years back from the pin",
                 "scope": "claim text + ' || ' + context"},
             "record_ever_history": {
                 "start": "the full eligible history",
                 "scope": "claim TEXT ONLY (R1=(b)); a rule keying on a "
                          "claim's own superlative reads the claim's own words"},
             "took_office": {
                 "start": "trailing 5 years",
                 "scope": "claim text + ' || ' + context"}},
         "precedence": "widest proposal wins, ties broken on (date, name)",
         "unfetchable": [dict(zip(("claim_sid", "evidence_id", "series", "reason"), u))
                         for u in UNFETCHABLE],
         "goldens": goldens}, indent=2, sort_keys=True) + "\n")
    print(f"\ngoldens -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
