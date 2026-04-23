"""finalize_run aggregates JSONL by run_id."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from truthbot.metrics.telemetry import finalize_run


def test_finalize_run_writes_summary_and_roi_row(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TRUTHBOT_METRICS_DIR", str(tmp_path))
    jsonl = tmp_path / "adapter_calls.jsonl"
    rid = "run-test-1"
    row = {
        "adapter_name": "anthropic",
        "model_id": "claude",
        "claim_id": "c1",
        "estimated_cost_usd": 0.01,
        "run_id": rid,
        "tier": "triage",
        "mode": "live",
    }
    jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")

    summary = finalize_run(rid, jsonl_path=jsonl)
    assert summary["total_calls"] == 1
    assert summary["total_cost_usd"] == pytest.approx(0.01)
    assert "triage" in summary["by_tier"]

    out = tmp_path / "run_summaries" / f"{rid}.json"
    assert out.exists()
    roi = tmp_path / "triage_roi.csv"
    assert roi.exists()
    assert rid in roi.read_text(encoding="utf-8")
