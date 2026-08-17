"""D17-c Stage 0 item 4 — fetch the FRED/ALFRED parser fixtures, once.

Run under Fable's 2026-08-12 egress authorization and nothing wider:
read-only GET, only ``fred.stlouisfed.org`` and ``alfred.stlouisfed.org``,
25 requests total, >=5s spacing, plain curl User-Agent (a browser-like UA
draws bot-protection 503s -- measured, do not "improve" it), <=3 retries,
30s timeout.

The point of saving raw bytes is that every later step is offline: the
parser tests, the determinism check and the token-delta measurement all
read these files and never the network. Each payload is recorded with its
URL, the UTC timestamp of the fetch, the HTTP status, the byte count and a
sha256, so a reader can tell whether a fixture is the one a result was
derived from.

Vintage pinning is the default fetch mode (R2): each wave-1 item is pinned
to its speech's registered utterance date, so the rows are what the speaker
could have been talking about rather than what the series says today.

This script is idempotent by refusal, not by re-fetch: a fixture that
already exists on disk is skipped and costs no request.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
MANIFEST = FIXTURES / "manifest.json"

ALLOWED_HOSTS = ("fred.stlouisfed.org", "alfred.stlouisfed.org")
CURRENT = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
PINNED = ("https://alfred.stlouisfed.org/graph/alfredgraph.csv"
          "?id={sid}&vintage_date={vintage}")

MAX_REQUESTS = 24          # 25 authorized, 1 spent on the endpoint probe
SPACING_S = 5.0
TIMEOUT_S = 30
MAX_RETRIES = 3

#: Registered utterance dates (truthbot.verdict.speech_context) -- the ONE map.
UTTERANCE = {"gwbush_2006": "2006-01-31", "clinton_1998": "1998-01-27",
             "obama_2014": "2014-01-28", "biden_2022": "2022-03-01",
             "trump_2026": "2026-02-24"}

#: The 9 wave-1 FRED items, from wave1_items.tsv. ``url_series`` preserves the
#: series id exactly as the corpus URL spells it (obama's is lowercased there).
WAVE1 = [
    ("biden_2022:0169", "E7", "MANEMP", "biden_2022", "MANEMP"),
    ("biden_2022:0245", "E7", "FYFSD", "biden_2022", "FYFSD"),
    ("gwbush_2006:0133", "E8", "PAYEMS", "gwbush_2006", "PAYEMS"),
    ("obama_2014:0189", "E4", "CPIAUCSL", "obama_2014", "cpiaucsl"),
    ("trump_2026:0054", "E4", "CE16OV", "trump_2026", "CE16OV"),
    ("trump_2026:0054", "E7", "PAYEMS", "trump_2026", "PAYEMS"),
    ("trump_2026:0054", "E8", "LNS12000000", "trump_2026", "LNS12000000"),
    ("trump_2026:0219", "E1", "APU0000708111", "trump_2026", "APU0000708111"),
    ("trump_2026:0221", "E9", "CUUR0000SAF112", "trump_2026", "CUUR0000SAF112"),
]

#: Determinism fixtures only -- stance-BEARING, never rescored in wave 1 (R2).
EXEMPLARS = [
    ("trump_2026:0031", "CPILFESL", "2026-02-24",
     "the exhibit's series; rows are diffed against the published values"),
    ("trump_2026:0054", "LNU02000000", "2026-02-24",
     "the stance-bearing ALFRED item's series"),
]

#: Current-vintage payloads for parser breadth (monthly index, monthly level,
#: and an average-price series whose row shape differs).
BREADTH = ["PAYEMS", "CPIAUCSL", "APU0000708111"]


class Budget:
    def __init__(self, cap: int) -> None:
        self.cap, self.used, self.last = cap, 0, 0.0

    def spend(self) -> None:
        if self.used >= self.cap:
            raise SystemExit(f"HALT: request cap {self.cap} reached")
        gap = time.monotonic() - self.last
        if self.last and gap < SPACING_S:
            time.sleep(SPACING_S - gap)
        self.used += 1
        self.last = time.monotonic()


def fetch(url: str, dest: Path, budget: Budget) -> dict:
    """One rate-limited GET, saved raw. Returns its manifest row."""
    assert any(f"//{h}/" in url for h in ALLOWED_HOSTS), f"host not authorized: {url}"
    if dest.exists():
        raw = dest.read_bytes()
        return {"url": url, "file": dest.name, "status": "cached",
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                "fetched_utc": None}

    last_err = ""
    for attempt in range(1, MAX_RETRIES + 1):
        budget.spend()
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        proc = subprocess.run(
            ["curl", "-sS", "--max-time", str(TIMEOUT_S),
             "-w", "\n%{http_code}", url],
            capture_output=True, timeout=TIMEOUT_S + 15)
        body, _, code = proc.stdout.rpartition(b"\n")
        status = code.decode(errors="replace").strip()
        if status == "200" and body.startswith(b"observation_date"):
            dest.write_bytes(body)
            row = {"url": url, "file": dest.name, "status": 200,
                   "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                   "fetched_utc": stamp, "attempts": attempt}
            print(f"  OK   {dest.name:<34} {len(body):>8,}B  {row['sha256'][:12]}")
            return row
        last_err = f"status={status} head={body[:80]!r}"
        print(f"  retry {attempt}/{MAX_RETRIES} {dest.name}: {last_err}")

    print(f"  FAIL {dest.name}: {last_err}")
    return {"url": url, "file": dest.name, "status": "FAILED",
            "error": last_err, "fetched_utc": None}


def main() -> int:
    FIXTURES.mkdir(exist_ok=True)
    budget = Budget(MAX_REQUESTS)
    rows: list[dict] = []

    print("=== wave-1 items, pinned to their speech's utterance date ===")
    for sid, eid, series, speech, url_series in WAVE1:
        vintage = UTTERANCE[speech]
        row = fetch(PINNED.format(sid=series, vintage=vintage),
                    FIXTURES / f"{series}_v{vintage.replace('-', '')}.csv", budget)
        rows.append({**row, "role": "wave1", "claim_sid": sid, "evidence_id": eid,
                     "series": series, "series_in_corpus_url": url_series,
                     "speech": speech, "vintage": vintage,
                     "vintage_rule": "utterance-date"})

    print("\n=== determinism exemplars (stance-bearing; never rescored) ===")
    for sid, series, vintage, why in EXEMPLARS:
        row = fetch(PINNED.format(sid=series, vintage=vintage),
                    FIXTURES / f"{series}_v{vintage.replace('-', '')}.csv", budget)
        rows.append({**row, "role": "exemplar", "claim_sid": sid, "series": series,
                     "vintage": vintage, "note": why})

    print("\n=== current-vintage payloads (parser breadth) ===")
    for series in BREADTH:
        row = fetch(CURRENT.format(sid=series), FIXTURES / f"{series}_current.csv", budget)
        rows.append({**row, "role": "breadth", "series": series, "vintage": "current"})

    print("\n=== determinism (a): same pinned URL twice, in-session ===")
    series, vintage = "CE16OV", "2026-02-24"
    url = PINNED.format(sid=series, vintage=vintage)
    row = fetch(url, FIXTURES / f"{series}_v{vintage.replace('-', '')}__repeat.csv", budget)
    rows.append({**row, "role": "determinism-repeat", "series": series,
                 "vintage": vintage})

    manifest = {
        "schema": "truthbot-d17c-fixtures v1",
        "authorization": ("Fable 2026-08-12: read-only GET, fred/alfred only, "
                          "25 requests, >=5s spacing, plain curl UA, <=3 retries"),
        "requests_used_here": budget.used,
        "requests_reserved_for_probe": 1,
        "endpoints": {"current": CURRENT, "pinned": PINNED},
        "fixtures": rows,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    failed = [r for r in rows if r.get("status") == "FAILED"]
    print(f"\nrequests used: {budget.used} of {MAX_REQUESTS} (cap 25 incl. probe)")
    print(f"fixtures: {len(rows)}  failed: {len(failed)}")
    print(f"manifest -> {MANIFEST.relative_to(HERE.parents[2])}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
