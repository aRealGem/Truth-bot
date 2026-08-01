"""Journal-layer verdict-count reconciliation (PR-A2.0 / T0.1).

The Obama-2014 measurement reported a verdict distribution summing to 95
against 96 check-worthy claims. The published site was never wrong — the
renderer buckets split claims explicitly — but the JOURNAL layer invites the
error: a PCA disagreement row carries ``verdict=null`` + ``split=true``, so a
naive ``Counter(row["verdict"])`` silently sheds one named bucket per split.

These tests pin the canonical tally (``verdict_bucket_tally``) against a
frozen slice of the real Obama-2014 pilot journal, document the naive-tally
failure mode so it stays recognizable, and (when the untracked local journals
are present) reconcile the live artifacts against ``meta.n_check_worthy``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from truthbot.verdict.publish_pipeline import verdict_bucket_tally

_REPO = Path(__file__).resolve().parent.parent
_FIXTURE = Path(__file__).parent / "fixtures" / "obama_2014_verdict_rows.json"
_PILOT_JOURNAL = _REPO / "metrics" / "journals" / "obama_2014_pilot.jsonl"


def _fixture() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


# ── Unit contract ─────────────────────────────────────────────────────────────


def test_tally_buckets_split_and_malformed_rows_explicitly() -> None:
    rows = [
        {"sid": "s:1", "verdict": "TRUE"},
        {"sid": "s:2", "verdict": "UNVERIFIABLE"},
        {"sid": "s:3", "verdict": None, "split": True},   # PCA disagreement
        {"sid": "s:4", "verdict": None, "split": False},  # malformed — never silent
    ]
    tally = verdict_bucket_tally(rows)
    assert tally == {"TRUE": 1, "UNVERIFIABLE": 1, "Models split": 1, "No verdict": 1}
    assert sum(tally.values()) == len(rows)


def test_tally_of_empty_rows_is_empty() -> None:
    assert verdict_bucket_tally([]) == {}


# ── Frozen Obama-2014 regression anchor ───────────────────────────────────────


def test_obama_2014_buckets_sum_to_checkworthy_count() -> None:
    fx = _fixture()
    tally = verdict_bucket_tally(fx["rows"])
    assert sum(tally.values()) == fx["n_check_worthy"] == 96
    assert tally == {"TRUE": 67, "FALSE": 4, "MISLEADING": 3,
                     "UNVERIFIABLE": 21, "Models split": 1}


def test_naive_verdict_counter_reproduces_the_95_bug() -> None:
    """Documents the failure mode this module exists to kill: tallying on
    ``row["verdict"]`` alone drops the ``obama_2014:0086`` disagreement row
    from every named bucket and reads 95 of 96."""
    fx = _fixture()
    named = [r["verdict"] for r in fx["rows"] if r["verdict"] is not None]
    assert len(named) == 95  # the bug
    split_rows = [r for r in fx["rows"] if r["verdict"] is None]
    assert [r["sid"] for r in split_rows] == ["obama_2014:0086"]
    assert split_rows[0]["split"] is True
    assert split_rows[0]["status"] == "disagreement"


# ── Live-artifact reconciliation (local journals are untracked) ───────────────


@pytest.mark.skipif(not _PILOT_JOURNAL.exists(),
                    reason="local Obama-2014 pilot journal not present")
def test_live_pilot_journal_reconciles_against_run_meta() -> None:
    rows: list[dict] = []
    for line in _PILOT_JOURNAL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.extend(json.loads(line).get("rows") or [])
    run_files = sorted((_REPO / "metrics" / "pca_runs").glob("*.json"))
    metas = [json.loads(p.read_text(encoding="utf-8")).get("meta", {})
             for p in run_files]
    meta = next((m for m in metas if m.get("speech_id") == "obama_2014"), None)
    assert meta is not None, "no obama_2014 run artifact next to the journal"
    tally = verdict_bucket_tally(rows)
    assert sum(tally.values()) == meta["n_check_worthy"] == len(rows)
