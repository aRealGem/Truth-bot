#!/usr/bin/env python3
"""D17-c wave 2 — publish the Stage A stances as a re-score sidecar pass.

The Stage A census is an ANALYSIS record: it says what the series rows did to
the scorer. The gate cannot read it. ``consolidator._bearing()`` needs a
True/False stance on the pack the gate is running over, and until those stances
reach a sidecar, ``trump_2026:0054`` keeps failing T2.4 with
``insufficient-qualifying-evidence`` and is forced Unverifiable WITHOUT a panel
call — the flip that Stage A paid to measure never reaches the verdict.

So this emits the census as a third re-score pass, ``d17c``, merged after B1a
and B2. Precedence is deliberate and is the same argument the B1a→B2 order
rests on: the later pass saw strictly more evidence (the series rows), so where
it disagrees it is better informed, not merely newer.

WHOLE PACKS, NOT JUST THE FLIPS. ``merge_sidecars`` is per-SID — "a sid B2
touched takes B2's rows entirely" — so a sidecar carrying only the 8 excerpted
items would silently DELETE the B1a/B2 stances for the other 59 items in those
packs. Every row of every covered claim is emitted.

SPEND IS NOT RESTATED HERE. The Stage A treatment cost $0.053984 as one run
across four speeches, and splitting that per speech would be an invented
number. Each sidecar records 0.0 with a pointer to the census, so the merged
total neither double-counts nor fabricates.

$0 and offline: reads the committed census and the shipped heads, nothing else.

Usage (repo root):
  PYTHONPATH=src .venv/bin/python \\
      metrics/remediation_v2/d17c_stage0/emit_d17c_sidecars.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = REPO / "metrics" / "pca_runs"
OUT_DIR = REPO / "metrics" / "remediation_v2"
CENSUS = HERE / "stage_a_census.json"

sys.path.insert(0, str(HERE))
import select_rows as S  # noqa: E402

SCHEMA = "truthbot-rescore-sidecar v1"
PASS_LABEL = "d17c"

#: The artifact revision a sidecar JOINS against. Stage A measured its stances
#: on the PUBLISHING HEADS; the re-score sidecars are keyed to the REBUILT runs,
#: and ``load_rescore_sidecar`` refuses a mismatch — "joining them would attach
#: stance to the wrong evidence". That guard is right, and it is satisfied here
#: with evidence rather than bypassed: the two revisions carry byte-identical
#: evidence (verified per run by :func:`assert_same_evidence` over source_url,
#: snippet, source_name, source_tier and published_at), so the head and the
#: rebuilt run differ in verdicts, not in what the stances attach to.
#
# IMPORTED, never restated. These are full UUIDs and there is exactly one
# correct value for each; transcribing them by hand from truncated output is
# how a sidecar ends up pointing at an artifact that does not exist, or worse,
# at one that does and is wrong.
sys.path.insert(0, str(REPO / "scripts"))
from wave_adjudicate import REBUILT_RUNS  # noqa: E402

#: Fields the join depends on. A difference in any of them means the stance
#: would land on different evidence.
JOIN_FIELDS = ("source_url", "snippet", "source_name", "source_tier",
               "published_at")


def assert_same_evidence(speech: str, sids) -> None:
    """The head and the rebuilt run must agree on the evidence being scored.

    Checked EVERY run, not once by hand: if a future revision diverges, this
    stops the sidecar being written rather than letting a stale stance attach
    to a changed pack."""
    head = json.loads((RUNS / f"{S.HEADS[speech]}.json").read_text())["evidence"]
    reb = json.loads(
        (RUNS / f"{REBUILT_RUNS[speech]}.json").read_text())["evidence"]
    for sid in sids:
        a = [{k: i.get(k) for k in JOIN_FIELDS} for i in head.get(sid) or []]
        b = [{k: i.get(k) for k in JOIN_FIELDS} for i in reb.get(sid) or []]
        if a != b:
            raise AssertionError(
                f"{sid}: evidence differs between publishing head "
                f"{S.HEADS[speech][:8]} and rebuilt run "
                f"{REBUILT_RUNS[speech][:8]} — the D17-c stances were measured "
                "on the head and must not be joined to a different pack")


def emit() -> dict:
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    rows = census["rows"]

    by_speech: dict[str, dict] = {}
    for r in rows:
        by_speech.setdefault(r["claim_sid"].split(":")[0], {}).setdefault(
            r["claim_sid"], []).append(r)

    written = {}
    for speech, claims in sorted(by_speech.items()):
        assert_same_evidence(speech, sorted(claims))
        head = json.loads((RUNS / f"{S.HEADS[speech]}.json").read_text())
        stored = head["evidence"]
        sids: dict[str, list] = {}
        for sid, items in sorted(claims.items()):
            pack = stored[sid]
            out = []
            for r in sorted(items, key=lambda x: int(x["evidence_id"][1:])):
                idx = int(r["evidence_id"][1:]) - 1
                out.append({
                    "source_url": pack[idx]["source_url"],
                    "relevance_score": r["relevance_after"],
                    "supports_claim": r["stance_after"],
                    "one_line_why": r.get("one_line_why"),
                })
            assert len(out) == len(pack), (
                f"{sid}: emitting {len(out)} of {len(pack)} items — a partial "
                "sid would delete the B1a/B2 stances the merge does not restore")
            sids[sid] = out

        doc = {
            "schema": SCHEMA,
            "speech_id": speech,
            # The revision this JOINS against — see REBUILT_RUNS. The head it
            # was MEASURED on is recorded in provenance, so both are legible.
            "source_run": REBUILT_RUNS[speech],
            "model": "claude-haiku (D17-c Stage A, series-row augmented)",
            "generated": census.get("generated", ""),
            "spend_usd": 0.0,
            "spend_note": (
                "Stage A cost $0.053984 as ONE run across four speeches; see "
                "stage_a_census.json. Not split per speech here — that would "
                "be an invented number — and not restated, to avoid "
                "double-counting in the merged total."),
            "provenance": {
                "census": str(CENSUS.relative_to(REPO)),
                "measured_on_publishing_head": S.HEADS[speech],
                "joins_against_rebuilt_run": REBUILT_RUNS[speech],
                "evidence_identical_verified": True,
                "selector_run_sha256": census.get("selector_run_sha256"),
                "note": ("stances measured with D17-c series excerpts in the "
                         "scoring payload; control arm produced 0 flips, so "
                         "these are excerpt-attributable"),
            },
            "soft_failures": [],
            "sids": sids,
        }
        path = OUT_DIR / f"rescored_d17c_{speech}.json"
        path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        written[speech] = (path, sum(len(v) for v in sids.values()), len(sids))
    return written


def main() -> int:
    written = self_check(emit())
    for speech, (path, items, claims) in sorted(written.items()):
        print(f"  {speech:<14}{claims} claims, {items:>3} items -> {path.name}")
    print(f"\n{len(written)} sidecars written. $0, no model call.")
    return 0


def self_check(written: dict) -> dict:
    """The stances that reach the gate must equal the stances in the census."""
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    want = {(r["claim_sid"], r["evidence_id"]): r["stance_after"]
            for r in census["rows"]}
    seen = 0
    for speech, (path, _items, _claims) in written.items():
        doc = json.loads(path.read_text())
        head = json.loads((RUNS / f"{S.HEADS[speech]}.json").read_text())
        for sid, rows in doc["sids"].items():
            for i, row in enumerate(rows, start=1):
                assert row["source_url"] == head["evidence"][sid][i - 1]["source_url"], (
                    f"{sid} E{i}: sidecar row is not the pack's item {i}")
                assert row["supports_claim"] == want[(sid, f"E{i}")], (
                    f"{sid} E{i}: sidecar stance disagrees with the census")
                seen += 1
    assert seen == len(want), f"checked {seen} of {len(want)} census rows"
    print(f"self-check: {seen} stances match the census, row order verified")
    return written


if __name__ == "__main__":
    sys.exit(main())
