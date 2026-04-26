"""Tests for Layer 5 — fabrication-rate telemetry.

Validates that ``CallRecord``, ``TelemetryLogger.measure``, and
``finalize_run`` correctly capture and aggregate the new fabrication
fields (``model_reported_source_count`` and ``stripped_source_count``).
"""
from __future__ import annotations

import json
from pathlib import Path

from truthbot.metrics.telemetry import (
    CallRecord,
    TelemetryLogger,
    finalize_run,
)


def test_call_record_has_fabrication_fields_with_defaults():
    rec = CallRecord(
        timestamp="2026-04-23T00:00:00",
        adapter_name="openai",
        model_id="gpt-x",
        claim_id="c1",
        wall_clock_ms=1234,
        input_tokens=100,
        output_tokens=50,
        tool_call_count=2,
        retrieved_url_count=3,
        estimated_cost_usd=0.001,
        status="ok",
    )
    assert rec.model_reported_source_count == 0
    assert rec.stripped_source_count == 0


def test_measure_records_fabrication_fields(tmp_path: Path):
    log_path = tmp_path / "adapter_calls.jsonl"
    log = TelemetryLogger(log_path)

    with log.measure(
        adapter_name="openai", model_id="gpt-x", claim_id="c1"
    ) as td:
        td["input_tokens"] = 100
        td["output_tokens"] = 20
        td["model_reported_source_count"] = 4
        td["stripped_source_count"] = 1

    assert log_path.exists()
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["model_reported_source_count"] == 4
    assert rows[0]["stripped_source_count"] == 1


def test_finalize_run_aggregates_fabrication(tmp_path: Path, monkeypatch):
    """Across mixed-adapter rows, finalize_run computes overall and
    per-adapter fabrication rates correctly."""
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    jsonl = metrics_dir / "adapter_calls.jsonl"

    monkeypatch.setenv("TRUTHBOT_METRICS_DIR", str(metrics_dir))

    rows = [
        # OpenAI reported 4, stripped 1 → 25% fabrication
        dict(
            run_id="r1",
            adapter_name="openai",
            estimated_cost_usd=0.001,
            tier="frontier",
            mode="live",
            model_reported_source_count=4,
            stripped_source_count=1,
        ),
        # Anthropic reported 5, stripped 0 → 0% fabrication
        dict(
            run_id="r1",
            adapter_name="anthropic",
            estimated_cost_usd=0.002,
            tier="frontier",
            mode="batch",
            model_reported_source_count=5,
            stripped_source_count=0,
        ),
        # OpenAI second call: reported 6, stripped 3 → cumulative 4/10 = 40%
        dict(
            run_id="r1",
            adapter_name="openai",
            estimated_cost_usd=0.001,
            tier="frontier",
            mode="live",
            model_reported_source_count=6,
            stripped_source_count=3,
        ),
        # Different run — must be ignored.
        dict(
            run_id="r2",
            adapter_name="openai",
            estimated_cost_usd=0.005,
            tier="frontier",
            mode="live",
            model_reported_source_count=10,
            stripped_source_count=10,
        ),
    ]
    with jsonl.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary = finalize_run("r1", jsonl_path=jsonl)

    fab = summary["fabrication"]
    # Overall: openai 4+6 + anthropic 5 = 15 reported, 1+3+0 = 4 stripped
    assert fab["model_reported_sources_total"] == 15
    assert fab["stripped_sources_total"] == 4
    assert abs(fab["fabrication_rate"] - 4 / 15) < 1e-9

    # Per-adapter
    assert fab["by_adapter"]["openai"]["reported"] == 10
    assert fab["by_adapter"]["openai"]["stripped"] == 4
    assert abs(fab["by_adapter"]["openai"]["rate"] - 0.4) < 1e-9
    assert fab["by_adapter"]["anthropic"]["reported"] == 5
    assert fab["by_adapter"]["anthropic"]["stripped"] == 0
    assert fab["by_adapter"]["anthropic"]["rate"] == 0.0


def test_finalize_run_no_calls_keeps_zero_rate(tmp_path: Path, monkeypatch):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    jsonl = metrics_dir / "adapter_calls.jsonl"
    jsonl.write_text("")

    monkeypatch.setenv("TRUTHBOT_METRICS_DIR", str(metrics_dir))

    summary = finalize_run("r-empty", jsonl_path=jsonl)

    assert summary["fabrication"]["model_reported_sources_total"] == 0
    assert summary["fabrication"]["stripped_sources_total"] == 0
    assert summary["fabrication"]["fabrication_rate"] == 0.0
    assert summary["fabrication"]["by_adapter"] == {}
