"""Synthetic cross-module checks for run telemetry + batch descriptor."""

from __future__ import annotations

from pathlib import Path

import truthbot.metrics.telemetry as tel
from truthbot.metrics.telemetry import finalize_run, get_telemetry, telemetry_run_context
from truthbot.verify.batch import BatchDispatcher


def test_run_telemetry_finalize_and_batch_job(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TRUTHBOT_METRICS_DIR", str(tmp_path))
    tel._telemetry_instance = None

    rid = "integration-run-001"
    BatchDispatcher(tmp_path).record_job(
        rid,
        transcript_meta={"speaker": "S"},
        work_units=[{"claim_id": "1", "claim_text": "claim"}],
    )
    job_path = tmp_path / "batch_jobs" / f"{rid}.json"
    assert job_path.exists()

    with telemetry_run_context(
        run_id=rid,
        evidence_injected=False,
        synthesis_mode="batch",
    ):
        log = get_telemetry()
        with log.measure("openai", "gpt-test", "claim-1", tier="triage") as d:
            d["input_tokens"] = 50
            d["output_tokens"] = 5
            d["tool_call_count"] = 0
            d["retrieved_url_count"] = 0
            d["status"] = "ok"

    summ = finalize_run(rid)
    assert summ["run_id"] == rid
    assert summ["total_calls"] >= 1
    assert (tmp_path / "run_summaries" / f"{rid}.json").is_file()

    lines = (tmp_path / "adapter_calls.jsonl").read_text(encoding="utf-8").strip().splitlines()
    import json

    last = json.loads(lines[-1])
    assert last["run_id"] == rid
    assert last["evidence_injected"] is False
    assert last["mode"] == "batch"
