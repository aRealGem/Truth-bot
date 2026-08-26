"""Spend telemetry for the metered retriever lanes (R2 OpenAI / R3 xAI).

These lanes billed ~$313 across Jul/Aug 2026 while writing zero rows to
``adapter_calls.jsonl``: they POST via raw urllib, read the response's ``usage``
block, and logged it to nowhere. Every assertion here exists to keep one of the
ways that happened from happening again — above all
``test_r2_logs_one_record_per_model_attempt``, since falling down the model
chain bills each rung and the old code could not tell you that.
"""
from __future__ import annotations

import json
import urllib.error

import pytest

from truthbot.metrics.telemetry import (claim_context, claim_spend_context,
                                        telemetry_run_context)
from truthbot.verify import retrievers as R


def _records(tmp_path) -> list[dict]:
    path = tmp_path / "metrics" / "adapter_calls.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _doc(*, items=1, input_tokens=1000, output_tokens=200, cached=0) -> dict:
    payload = {"items": [{"url": f"https://bls.gov/{i}", "date": "2014-01-02",
                          "stance": "supports", "one_line_why": "why"}
                         for i in range(items)]}
    return {
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                  "prompt_tokens_details": {"cached_tokens": cached}},
        "output": [{"content": [{"type": "output_text",
                                 "text": json.dumps(payload)}]}],
    }


@pytest.fixture(autouse=True)
def _keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")


def test_r2_logs_one_record_per_model_attempt(tmp_path, monkeypatch):
    """Falling down the chain bills every rung — so it must log every rung.

    gpt-5-mini returns an empty shortlist twice (a soft failure that retries
    the same model), then gpt-5.4 answers. That is three billable calls.
    """
    seen: list[str] = []

    def fake_post(self, model, prompt):
        seen.append(model)
        return _doc(items=0) if model == "gpt-5-mini" else _doc(items=1)

    monkeypatch.setattr(R.OpenAIBrowsingRetriever, "_post", fake_post)
    R.OpenAIBrowsingRetriever(model="gpt-5-mini").shortlist("claim")

    recs = _records(tmp_path)
    assert seen == ["gpt-5-mini", "gpt-5-mini", "gpt-5.4"]
    assert [r["model_id"] for r in recs] == ["gpt-5-mini", "gpt-5-mini", "gpt-5.4"]
    assert [r["status"] for r in recs] == [
        "empty_shortlist", "empty_shortlist", "ok"]
    assert {r["adapter_name"] for r in recs} == {"openai"}
    assert {r["tier"] for r in recs} == {"retrieval"}


def test_r2_logs_record_on_post_failure(tmp_path, monkeypatch):
    """A failed call still costs latency and may have cost money — log it."""
    def fake_post(self, model, prompt):
        raise RuntimeError("boom")

    monkeypatch.setattr(R.OpenAIBrowsingRetriever, "_post", fake_post)
    assert R.OpenAIBrowsingRetriever(model="gpt-5-mini").shortlist("claim") == []

    recs = _records(tmp_path)
    # One per model in the chain, and NO same-model retry after a hard failure.
    assert [r["model_id"] for r in recs] == ["gpt-5-mini", "gpt-5.4", "gpt-4o"]
    assert {r["status"] for r in recs} == {"api_error"}
    assert {r["input_tokens"] for r in recs} == {0}
    assert {r["estimated_cost_usd"] for r in recs} == {0.0}


def test_r2_logs_rate_limit_and_still_signals_the_governor(tmp_path, monkeypatch):
    """Telemetry and the pool-governor 429 signal must not drift apart."""
    fired: list[int] = []

    def fake_post(self, model, prompt):
        raise urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(R.OpenAIBrowsingRetriever, "_post", fake_post)
    R.OpenAIBrowsingRetriever(
        model="gpt-5-mini", on_rate_limit=lambda: fired.append(1)).shortlist("c")

    assert fired, "the governor must still learn about the 429"
    assert {r["status"] for r in _records(tmp_path)} == {"rate_limited"}


def test_r2_maps_usage_and_prices_gpt55_with_cached_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(
        R.OpenAIBrowsingRetriever, "_post",
        lambda self, model, prompt: _doc(input_tokens=1000, output_tokens=200,
                                         cached=400))
    R.OpenAIBrowsingRetriever().shortlist("claim")

    rec = _records(tmp_path)[0]
    assert rec["model_id"] == "gpt-5.5"          # the R2 default
    assert rec["input_tokens"] == 1000
    assert rec["output_tokens"] == 200
    assert rec["openai_cached_prompt_tokens"] == 400
    assert rec["retrieved_url_count"] == 1
    # 600 uncached @ $5/MTok + 400 cached @ $0.50/MTok + 200 out @ $30/MTok.
    # Before this change gpt-5.5 had no rate row and silently priced its output
    # at the $15/MTok fallback — half its real rate.
    assert rec["estimated_cost_usd"] == pytest.approx(
        600 * 5e-6 + 400 * 0.5e-6 + 200 * 30e-6)
    assert rec["cost_basis"] == "table"


def test_records_carry_claim_and_run_id(tmp_path, monkeypatch):
    monkeypatch.setattr(R.OpenAIBrowsingRetriever, "_post",
                        lambda self, model, prompt: _doc())
    with telemetry_run_context(run_id="run-42"), claim_context("sp:0007"):
        R.OpenAIBrowsingRetriever().shortlist("claim")

    rec = _records(tmp_path)[0]
    assert rec["claim_id"] == "sp:0007"
    assert rec["run_id"] == "run-42"


def test_missing_key_is_recorded_as_zero_cost(tmp_path, monkeypatch):
    """No key means no call was billed — say so, don't invent spend."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert R.OpenAIBrowsingRetriever().shortlist("claim") == []

    recs = _records(tmp_path)
    assert recs, "a keyless lane must still be visible in telemetry"
    assert {r["status"] for r in recs} == {"no_key"}
    assert {r["estimated_cost_usd"] for r in recs} == {0.0}


def test_claim_spend_accumulates_across_retrieval(tmp_path, monkeypatch):
    monkeypatch.setattr(R.OpenAIBrowsingRetriever, "_post",
                        lambda self, model, prompt: _doc())
    with claim_spend_context() as spend:
        R.OpenAIBrowsingRetriever().shortlist("claim")
        R.OpenAIBrowsingRetriever().shortlist("claim")

    snap = spend.snapshot()
    assert snap["calls"] == 2
    assert snap["unpriced_calls"] == 0
    assert snap["cost_usd"] == pytest.approx(2 * (1000 * 5e-6 + 200 * 30e-6))
    assert set(snap["by_adapter"]) == {"openai"}


def test_r3_logs_xai_retrieval_records(tmp_path, monkeypatch):
    """R3 posts to xAI over the same raw-urllib path and was equally invisible."""
    monkeypatch.setenv("TRUTHBOT_R3_MODEL", "grok-4.3")
    monkeypatch.setattr(R.GrokSearchRetriever, "_post",
                        lambda self, model, prompt, tool: _doc(
                            input_tokens=800, output_tokens=100))
    R.GrokSearchRetriever().shortlist("claim")

    rec = _records(tmp_path)[0]
    assert rec["adapter_name"] == "xai"
    assert rec["model_id"] == "grok-4.3"
    assert rec["tier"] == "retrieval"
    assert rec["status"] == "ok"
    assert rec["estimated_cost_usd"] == pytest.approx(
        800 * 1.25e-6 + 100 * 2.50e-6)
    assert rec["cost_basis"] == "table"
