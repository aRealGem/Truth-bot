"""TRUTHBOT_OPENAI_RESPONSES_PROBE structured logging on live Responses path."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from truthbot.models import Claim
from truthbot.verify.adapters.openai import OpenAIAdapter


@pytest.fixture(autouse=True)
def _key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _fake_response_min(text: str, *, searches: int = 2) -> SimpleNamespace:
    output: list[Any] = []
    for i in range(searches):
        output.append(
            SimpleNamespace(
                type="web_search_call",
                id=f"ws_{i}",
                action=SimpleNamespace(type="search", queries=["q"], query="q"),
            )
        )
    output.append(
        SimpleNamespace(
            type="message",
            content=[
                SimpleNamespace(
                    type="output_text",
                    text=text,
                    annotations=[
                        SimpleNamespace(url="https://example.com/article", type="url_citation")
                    ],
                )
            ],
        )
    )
    return SimpleNamespace(
        status="completed",
        output=output,
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
    )


class _FakeResponses:
    def __init__(self, response: Any) -> None:
        self._response = response

    def create(self, **kwargs: Any) -> Any:
        return self._response


class _FakeClient:
    def __init__(self, response: Any) -> None:
        self.responses = _FakeResponses(response)


def _patch_openai(monkeypatch, response: Any) -> None:
    import openai

    client = _FakeClient(response)
    monkeypatch.setattr(openai, "OpenAI", lambda **_kw: client)


def test_probe_log_live_single_when_env_truthy(monkeypatch, caplog) -> None:
    monkeypatch.setenv("TRUTHBOT_OPENAI_RESPONSES_PROBE", "1")
    text = json.dumps(
        {
            "label": "True",
            "confidence": "High",
            "explanation": "ok",
            "sources": ["https://example.com/article"],
        }
    )
    _patch_openai(monkeypatch, _fake_response_min(text, searches=3))
    adapter = OpenAIAdapter()
    claim = Claim(transcript_id="t1", text="ping", speaker="Test")

    with caplog.at_level(logging.WARNING):
        adapter.call(claim, [], inject_evidence=False, telemetry_tier="frontier")

    assert any("OpenAIAdapter RESPONSES_PROBE" in r.message for r in caplog.records)
    assert any("web_search_calls=3" in r.message for r in caplog.records)


def test_probe_log_live_multi_includes_batch_call_id(monkeypatch, caplog) -> None:
    monkeypatch.setenv("TRUTHBOT_OPENAI_RESPONSES_PROBE", "1")
    claims = [
        Claim(transcript_id="t1", text="A", speaker="Test"),
        Claim(transcript_id="t1", text="B", speaker="Test"),
    ]
    text = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High"},
        ]
    )
    _patch_openai(monkeypatch, _fake_response_min(text, searches=1))
    adapter = OpenAIAdapter()

    with caplog.at_level(logging.WARNING):
        adapter.call_multi(claims, {c.id: [] for c in claims}, inject_evidence=False)

    assert any("context=live_multi" in r.message for r in caplog.records)
    assert any("batch_call_id=openai-live-multi-" in r.message for r in caplog.records)


def test_probe_log_suppressed_without_env(monkeypatch, caplog) -> None:
    monkeypatch.delenv("TRUTHBOT_OPENAI_RESPONSES_PROBE", raising=False)
    text = json.dumps(
        {"label": "True", "confidence": "High", "explanation": "ok", "sources": []}
    )
    _patch_openai(monkeypatch, _fake_response_min(text))
    adapter = OpenAIAdapter()
    claim = Claim(transcript_id="t1", text="ping", speaker="Test")

    with caplog.at_level(logging.WARNING):
        adapter.call(claim, [], inject_evidence=False)

    assert not any(
        "OpenAIAdapter RESPONSES_PROBE" in r.message for r in caplog.records
    )
