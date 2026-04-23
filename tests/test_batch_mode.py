"""Batch job descriptor persistence."""

from __future__ import annotations

from pathlib import Path

from truthbot.verify.batch import BatchDispatcher, read_batch_job


def test_record_and_read_batch_job(tmp_path: Path) -> None:
    md = tmp_path / "metrics"
    d = BatchDispatcher(md)
    path = d.record_job(
        "run-abc",
        transcript_meta={"speaker": "X"},
        work_units=[{"claim_id": "1", "claim_text": "t"}],
    )
    assert path.exists()
    data = read_batch_job(md, "run-abc")
    assert data is not None
    assert data["run_id"] == "run-abc"
    assert data["status"] == "pending"
    assert len(data["work_units"]) == 1


def test_poll_missing(tmp_path: Path) -> None:
    assert BatchDispatcher(tmp_path / "m").poll("nope") == "missing"


def test_poll_pending(tmp_path: Path) -> None:
    md = tmp_path / "metrics"
    BatchDispatcher(md).record_job("r1", transcript_meta={}, work_units=[])
    assert BatchDispatcher(md).poll("r1") == "pending"
