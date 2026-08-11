#!/usr/bin/env python3
"""Computed exhibit builder — pinned-vintage core-CPI arithmetic (A8 / R-2).

trump_2026:0031 says core inflation "was down to 1.7 percent" in the last three
months of 2025. The pre-remediation run called that FALSE by checking it
against the 2.7% year-over-year figure — which is a different number about a
different window. The claim is about the three-month ANNUALIZED rate, and on
the right series it is right.

Making that legible needs three things on the page, and R-2 requires all three
to be VISIBLE, not merely stored:

* the formula, ``(Dec/Sep)^4 - 1`` — a three-month change raised to the fourth
  power is what "annualized" means here;
* BOTH input levels, so a reader can redo the division by hand;
* the VINTAGE DATE, because it is load-bearing. Pinned to the speech-day
  vintage (2026-02-24) the answer is 1.701%. On the pre-revision 2026-02-09
  vintage the same formula over the same months gives 1.605% — about 10 basis
  points apart. An exhibit that showed the arithmetic but hid which vintage it
  ran on would be reproducible only by luck.

ALFRED (the archival arm of FRED) serves exactly this: the series AS IT STOOD
on a given day. No API key, no spend.

Usage (repo root)::

    PYTHONPATH=. .venv/bin/python scripts/build_computed_exhibit.py           # fetch + write
    PYTHONPATH=. .venv/bin/python scripts/build_computed_exhibit.py --check   # verify, offline

``--check`` recomputes the arithmetic from the COMMITTED inputs and never
touches the network — the same check CI runs on every commit. The network
re-fetch is a separate, ``network``-marked test so a silent ALFRED revision
gets caught without making the default suite depend on the internet.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

REPO = Path(__file__).resolve().parents[1]

#: Core CPI (CPI for All Urban Consumers: All Items Less Food and Energy).
SERIES = "CPILFESL"
SOURCE = "ALFRED"

#: The pinned vintage: the series as it stood on the day of the speech. Moving
#: this date changes the answer (see the module docstring) — it is a decision,
#: not a default.
VINTAGE_DATE = "2026-02-24"

#: Q4 2025: the September base and the December endpoint.
START_DATE = "2025-09-01"
END_DATE = "2025-12-01"

#: Months between the two observations. 3 months → the ^(12/3) = ^4 exponent.
SPAN_MONTHS = 3

FORMULA = "(Dec/Sep)^4 - 1"
CLAIM_REF = "trump_2026:0031"

ALFRED_BASE = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"

#: Committed output.
EXHIBIT_PATH = (REPO / "metrics" / "computed_exhibits"
                / "cpilfesl_q4_2025_annualized.json")

SCHEMA = "truthbot-computed-exhibit v1"


def alfred_url(series: str = SERIES, vintage_date: str = VINTAGE_DATE) -> str:
    """The ALFRED VINTAGE endpoint. ``vintage_date`` is the whole point: drop
    it and you get today's revised series, which is not what the speaker or
    the fact-check saw."""
    return f"{ALFRED_BASE}?id={series}&vintage_date={vintage_date}"


def parse_levels(csv_text: str, dates: Iterable[str]) -> dict[str, float]:
    """``observation_date`` → level, for the requested dates only.

    ALFRED names the value column after the series AND the vintage
    (``CPILFESL_20260224``), so the column is read positionally: whatever
    follows ``observation_date``. Blank cells (the series has gaps — October
    2025 is empty in this vintage) are skipped rather than coerced to zero."""
    wanted = set(dates)
    out: dict[str, float] = {}
    reader = csv.reader(io.StringIO(csv_text))
    header = next(reader, None)
    if not header or header[0].strip().lower() not in {"observation_date", "date"}:
        raise ValueError(f"unexpected ALFRED header: {header!r}")
    for row_ in reader:
        if len(row_) < 2:
            continue
        day, raw = row_[0].strip(), row_[1].strip()
        if day in wanted and raw:
            out[day] = float(raw)
    missing = wanted - set(out)
    if missing:
        raise ValueError(f"ALFRED vintage is missing {sorted(missing)}")
    return out


def annualized(start_level: float, end_level: float,
               span_months: int = SPAN_MONTHS) -> float:
    """Compound the observed change out to a year: ``(end/start)^(12/n) - 1``.

    Pure arithmetic on two numbers — this is the function CI re-runs against
    the committed inputs, with no network anywhere near it."""
    if start_level <= 0:
        raise ValueError("start level must be positive")
    return (end_level / start_level) ** (12 / span_months) - 1


def build_exhibit(levels: dict[str, float], *, vintage_date: str = VINTAGE_DATE,
                  claim_ref: str = CLAIM_REF) -> dict:
    """The exhibit record. ``result`` is rounded to 5 decimal places — one
    more digit than the 1.701% the exhibit renders, so the published figure is
    never a rounding artifact of the stored one."""
    result = annualized(levels[START_DATE], levels[END_DATE])
    return {
        "schema": SCHEMA,
        "series": SERIES,
        "source": SOURCE,
        "vintage_date": vintage_date,
        "inputs": {START_DATE: levels[START_DATE], END_DATE: levels[END_DATE]},
        "formula": FORMULA,
        "result": round(result, 5),
        "claim_ref": claim_ref,
        "source_url": alfred_url(vintage_date=vintage_date),
        "span_months": SPAN_MONTHS,
        "note": ("Vintage-pinned: the same formula on the 2026-02-09 "
                 "pre-revision vintage returns 1.605%, ~10bp lower. The "
                 "vintage is part of the result."),
    }


def fetch_csv(url: str, timeout: float = 30.0) -> str:
    """GET the pinned vintage CSV. Free, keyless, read-only."""
    import httpx

    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def fetch_exhibit(vintage_date: str = VINTAGE_DATE) -> dict:
    """Network path: fetch the pinned vintage and build the exhibit from it."""
    levels = parse_levels(fetch_csv(alfred_url(vintage_date=vintage_date)),
                          (START_DATE, END_DATE))
    return build_exhibit(levels, vintage_date=vintage_date)


def load_exhibit(path: Path = EXHIBIT_PATH) -> dict:
    return json.loads(path.read_text("utf-8"))


def recompute(exhibit: dict) -> float:
    """Redo the exhibit's own arithmetic from its own stored inputs. Offline,
    total, and the thing CI asserts against ``exhibit['result']``."""
    inputs = exhibit["inputs"]
    dates = sorted(inputs)
    return annualized(inputs[dates[0]], inputs[dates[-1]],
                      int(exhibit.get("span_months", SPAN_MONTHS)))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify the committed exhibit's arithmetic; no network")
    ap.add_argument("--out", type=Path, default=EXHIBIT_PATH)
    args = ap.parse_args(argv)

    if args.check:
        ex = load_exhibit(args.out)
        got = recompute(ex)
        ok = round(got, 5) == ex["result"]
        print(f"{args.out}: stored {ex['result']} vs recomputed {got:.6f} "
              f"— {'OK' if ok else 'MISMATCH'}")
        return 0 if ok else 1

    exhibit = fetch_exhibit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(exhibit, indent=2) + "\n", "utf-8")
    print(f"wrote {args.out}")
    print(f"  {START_DATE}={exhibit['inputs'][START_DATE]} "
          f"{END_DATE}={exhibit['inputs'][END_DATE]} "
          f"vintage={exhibit['vintage_date']} → {exhibit['result']:.5f} "
          f"({exhibit['result'] * 100:.3f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
