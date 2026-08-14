#!/usr/bin/env python3
"""Wave 2, lane 2c — mint the five successor artifacts, deterministically, at $0.

WHAT A SUCCESSOR CARRIES. The D17-c series excerpts ride on the evidence itself
(``series_rows``), so a claim's observations reach the rendered page instead of
living only in a sidecar nobody opens. Four speeches carry excerpts; clinton_1998
carries none and mints an id-only successor so the stable-id deep-link rotation
completes site-wide in ONE event rather than trickling across five publishes.

WHY THIS IS $0. The two adjudicated speeches were already paid for: the
d17c-wave2 escape run cost $0.3266 and produced the verdicts for
``trump_2026:0054`` and ``trump_2026:0219``. Those rows are REUSED here, not
recomputed. Every other speech is a pure carry-forward: same rows, same
verdicts, evidence gaining only the excerpts. No model call, no adjudication.

WHY THE IDS ARE DERIVED. The first mint used ``uuid4``, so re-minting produced
different ids and the committed verdict diffs ended up citing artifacts that no
longer existed. ``wave_adjudicate.successor_run_id`` derives the id from
(parent, tag, claim set), so an identical re-mint reproduces an identical id and
a record can cite one safely.

WHAT MAY MOVE. Only the sids the run DECLARES. For the two adjudicated speeches
that is the escape set; for the other three it is empty, so nothing may move —
and the 1B guard proves that rather than this script asserting it.

Usage (repo root):
  PYTHONPATH=src .venv/bin/python scripts/mint_d17c_successors.py --dry-run
  PYTHONPATH=src .venv/bin/python scripts/mint_d17c_successors.py --write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "metrics" / "pca_runs"
QUARANTINE = RUNS / "_quarantine"
GOLDENS = REPO / "metrics" / "remediation_v2" / "d17c_stage0" / "goldens.json"

sys.path.insert(0, str(REPO / "scripts"))
from wave_adjudicate import successor_run_id  # noqa: E402

TAG = "d17c-wave2"
REMEDIATION = f"D17-c wave 2 successor ({TAG})"

#: Publishing heads the successors derive from.
HEADS = {
    "trump_2026": "91dd7a34-7a3c-4f40-bcdc-276b2cb15d26",
    "biden_2022": "ddb05ee3-7d9c-4b2c-beaf-e197b9354379",
    "obama_2014": "2cbda3e4-c578-442a-aee7-c5c28a388048",
    "clinton_1998": "49b2e3e8-1667-4460-8989-b265914d4450",
    "gwbush_2006": "5c923c25-b065-4a9f-80bf-d23db4f9bcd1",
}

#: The paid-for adjudication whose rows are reused rather than recomputed.
PRIOR_MINTS = {
    "trump_2026": "5d3d54f0-e3fd-4fd2-85d2-0e9b2215d61a",
    "biden_2022": "44009622-3916-47cd-8a6c-79448c300861",
}

#: Sids the escape run declared and paid to adjudicate.
DECLARED = {
    "trump_2026": ["trump_2026:0054", "trump_2026:0219"],
    "biden_2022": ["biden_2022:0169"],
}

PERIOD_MISMATCH = {("obama_2014:0189", "E4")}

MISMATCH_NOTE = (
    "This window does not reach the period the claim compares against. The "
    "claim's reference point is named rather than dated, so the selection "
    "rules did not extend the window to cover it. Shown for transparency; it "
    "cannot settle the claim.")


def series_index() -> dict:
    """``(sid, stored_source_url) -> series_rows`` for the committed goldens.

    Keyed on the URL the SHIPPED PACK actually carries, resolved through the
    golden's ``evidence_id``, NOT on the golden's ``full_table``. Those two are
    not the same string: ``full_table`` is constructed as
    ``fred.stlouisfed.org/series/{SERIES}``, while ``obama_2014:0189`` is stored
    at ``fred.stlouisfed.org/data/cpiaucsl``. Keying on ``full_table`` silently
    dropped that excerpt — and it is the one carrying the period-mismatch
    warning, so the miss would have removed a caveat rather than a nicety.

    The E-number is used HERE, against the shipped head it was recorded from,
    and the resulting URL is what later matching uses — so re-gating that
    reorders or drops items still cannot mis-attach a table.
    """
    doc = json.loads(GOLDENS.read_text(encoding="utf-8"))
    heads = {sp: json.loads((RUNS / f"{rid}.json").read_text(encoding="utf-8"))
             for sp, rid in HEADS.items()}
    out = {}
    for g in doc.get("goldens") or []:
        if g.get("role") != "wave1":
            continue
        sid = g["claim_sid"]
        pack = heads[sid.split(":")[0]]["evidence"][sid]
        url = pack[int(g["evidence_id"][1:]) - 1]["source_url"]
        rows = dict(g)
        if (sid, g["evidence_id"]) in PERIOD_MISMATCH:
            rows["window_period_mismatch"] = True
            rows["window_period_mismatch_note"] = MISMATCH_NOTE
        out[(sid, url)] = rows
    return out


def build(speech: str) -> dict:
    head = json.loads((RUNS / f"{HEADS[speech]}.json").read_text(encoding="utf-8"))
    art = json.loads(json.dumps(head))          # deep copy; head is never touched
    index = series_index()

    # Rows: reuse the paid-for adjudication where there was one.
    reused = None
    prior = PRIOR_MINTS.get(speech)
    if prior:
        p = QUARANTINE / f"{prior}.json"
        if not p.exists():
            raise SystemExit(
                f"{speech}: prior mint {prior[:8]} not found in _quarantine — "
                "its rows were paid for and must not be recomputed")
        reused = json.loads(p.read_text(encoding="utf-8"))
        art["rows"] = reused["rows"]

    # Evidence: attach the excerpts, keyed on URL rather than E-number.
    attached = 0
    for sid, items in art.get("evidence", {}).items():
        for it in items:
            rows = index.get((sid, it.get("source_url")))
            if rows is not None:
                it["series_rows"] = rows
                attached += 1

    declared = DECLARED.get(speech, [])
    meta = dict(art.get("meta") or {})
    meta["rebuild_of"] = HEADS[speech]
    meta["remediation"] = REMEDIATION
    meta["d17c_wave2"] = {
        "tag": TAG,
        "series_rows_attached": attached,
        "rows_reused_from": prior or None,
        "rows_recomputed": False,
        "adjudication": ("reused from the paid d17c-wave2 escape run"
                         if prior else "none — carry-forward only"),
        "note": ("Successor carrying the D17-c series excerpts onto the "
                 "evidence so the observations render for a reader. Verdicts "
                 "move only for the declared sids; every other row is the "
                 "parent's."),
    }
    # The declared set the 1B guard reads. Empty for a pure carry-forward, so
    # the scoped guard collapses to the absolute one and nothing may move.
    esc = dict(meta.get("escape_run") or {})
    esc["sids_adjudicated"] = declared
    esc["tag"] = TAG
    meta["escape_run"] = esc
    meta.pop("wave", None)   # the parent's wave set is not this run's declaration
    art["meta"] = meta
    art["run_id"] = successor_run_id(HEADS[speech], REMEDIATION, declared)
    return art


def verdict_map(art: dict) -> dict:
    return {r["sid"]: r.get("verdict") for r in art.get("rows") or []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"{'speech':<14}{'run_id':<38}{'excerpts':>9}{'moved':>7}  declared")
    total_moved = 0
    for speech in sorted(HEADS):
        art = build(speech)
        parent = json.loads(
            (RUNS / f"{HEADS[speech]}.json").read_text(encoding="utf-8"))
        cm, pm = verdict_map(art), verdict_map(parent)
        moved = sorted(s for s in set(cm) | set(pm) if cm.get(s) != pm.get(s))
        declared = set(DECLARED.get(speech, []))
        undeclared = set(moved) - declared
        if undeclared:
            raise SystemExit(
                f"{speech}: {len(undeclared)} undeclared verdict move(s): "
                f"{sorted(undeclared)[:5]}")
        total_moved += len(moved)
        n = art["meta"]["d17c_wave2"]["series_rows_attached"]
        print(f"{speech:<14}{art['run_id']:<38}{n:>9}{len(moved):>7}  "
              f"{sorted(declared) or '-'}")
        if args.write:
            (RUNS / f"{art['run_id']}.json").write_text(
                json.dumps(art, indent=2, sort_keys=True) + "\n")

    print(f"\nverdict deltas across all five: {total_moved} "
          f"(expected: only the declared escape sids)")
    if not args.write:
        print("dry run — nothing written. Re-run with --write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
