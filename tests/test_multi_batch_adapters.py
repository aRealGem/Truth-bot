"""Multi-claim batch-response parsing for the AnthropicAdapter and OpenAIAdapter.

Exercises only the ``build_multi_batch_payload`` / ``parse_multi_batch_response``
pair on each adapter with hand-built response envelopes that mimic what the
real SDKs return. No live calls, no mocked SDKs.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from truthbot.models import Claim, VerdictLabel
from truthbot.verify.adapters.anthropic import AnthropicAdapter
from truthbot.verify.adapters.openai import OpenAIAdapter


@pytest.fixture(autouse=True)
def _set_api_keys(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _claim(text: str) -> Claim:
    return Claim(transcript_id="t1", text=text, speaker="Test")


# ── Anthropic multi-claim envelope (mimics messages.batches.results message) ──


def _anthropic_message(
    verdicts_json: str, *, urls: list[str] | None = None, cached: int = 0
) -> dict:
    content = []
    if urls:
        content.append(
            {
                "type": "web_search_tool_result",
                "content": [{"url": u} for u in urls],
            }
        )
    content.append({"type": "text", "text": verdicts_json})
    return {
        "model": "claude-opus-4-7",
        "content": content,
        "usage": {"input_tokens": 500, "output_tokens": 200, "cache_read_input_tokens": cached},
    }


def test_anthropic_build_multi_payload_scales_max_tokens_with_n() -> None:
    adapter = AnthropicAdapter()
    claims = [_claim(f"Claim {i}") for i in range(5)]
    payload = adapter.build_multi_batch_payload(
        claims, {c.id: [] for c in claims}, inject_evidence=False
    )
    assert payload["model"] == "claude-opus-4-7"
    assert payload["max_tokens"] >= 1024 * 5
    assert payload["messages"][0]["role"] == "user"
    assert claims[0].id in payload["messages"][0]["content"]
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_parse_multi_all_succeed() -> None:
    adapter = AnthropicAdapter()
    claims = [_claim("A"), _claim("B"), _claim("C")]
    verdicts_json = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High", "explanation": "a"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High", "explanation": "b"},
            {"claim_id": claims[2].id, "label": "Misleading", "confidence": "Medium", "explanation": "c"},
        ]
    )
    raw = _anthropic_message(
        verdicts_json, urls=["https://example.gov/a"], cached=42
    )
    verdicts = adapter.parse_multi_batch_response(raw, claims, batch_call_id="anthropic::multi::xyz")

    assert [v.label for v in verdicts] == [
        VerdictLabel.TRUE,
        VerdictLabel.FALSE,
        VerdictLabel.MISLEADING,
    ]
    assert verdicts[0].batch_call_index == 0
    assert verdicts[1].batch_call_index == 1
    assert verdicts[2].batch_call_index == 2
    assert all(v.batch_call_id == "anthropic::multi::xyz" for v in verdicts)
    assert verdicts[0].cached_input_tokens == 42
    assert verdicts[1].cached_input_tokens == 0
    assert verdicts[2].cached_input_tokens == 0


def test_anthropic_parse_multi_partial_fills_missing_with_no_response() -> None:
    adapter = AnthropicAdapter()
    claims = [_claim("A"), _claim("B"), _claim("C")]
    verdicts_json = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High", "explanation": "a"},
            {"claim_id": claims[2].id, "label": "False", "confidence": "High", "explanation": "c"},
        ]
    )
    raw = _anthropic_message(verdicts_json)
    verdicts = adapter.parse_multi_batch_response(raw, claims)

    assert verdicts[0].label == VerdictLabel.TRUE
    assert verdicts[1].label == VerdictLabel.UNVERIFIABLE
    assert verdicts[1].no_response is True
    assert "partial" in verdicts[1].explanation.lower()
    assert verdicts[2].label == VerdictLabel.FALSE


def test_anthropic_parse_multi_malformed_json_marks_all_as_no_response() -> None:
    adapter = AnthropicAdapter()
    claims = [_claim("A"), _claim("B")]
    raw = _anthropic_message("this is not json")
    verdicts = adapter.parse_multi_batch_response(raw, claims)
    assert all(v.no_response for v in verdicts)
    assert all(v.label == VerdictLabel.UNVERIFIABLE for v in verdicts)


def test_anthropic_parse_multi_reordered_ids_still_key_correctly() -> None:
    adapter = AnthropicAdapter()
    claims = [_claim("A"), _claim("B"), _claim("C")]
    # Model returned them in reverse; IDs still line up.
    verdicts_json = json.dumps(
        [
            {"claim_id": claims[2].id, "label": "Misleading", "confidence": "Medium"},
            {"claim_id": claims[0].id, "label": "True", "confidence": "High"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High"},
        ]
    )
    raw = _anthropic_message(verdicts_json)
    verdicts = adapter.parse_multi_batch_response(raw, claims)
    assert verdicts[0].label == VerdictLabel.TRUE
    assert verdicts[1].label == VerdictLabel.FALSE
    assert verdicts[2].label == VerdictLabel.MISLEADING


def test_anthropic_parse_multi_backfills_web_sources_on_first_verdict() -> None:
    adapter = AnthropicAdapter()
    claims = [_claim("A"), _claim("B")]
    # Model omitted per-verdict web_sources; envelope has harvested URLs.
    verdicts_json = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High"},
        ]
    )
    raw = _anthropic_message(
        verdicts_json, urls=["https://example.gov/x", "https://example.gov/y"]
    )
    verdicts = adapter.parse_multi_batch_response(raw, claims)
    assert verdicts[0].web_sources == ["https://example.gov/x", "https://example.gov/y"]
    assert verdicts[1].web_sources == []


# ── OpenAI multi-claim envelope (mimics Responses API body) ───────────────────


def _openai_body(text: str, *, cached: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        model="gpt-4.1",
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text=text, annotations=[])
                ],
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached)
        ),
    )


def test_openai_build_multi_payload_scales_tool_budget_with_n() -> None:
    adapter = OpenAIAdapter()
    claims = [_claim(f"Claim {i}") for i in range(4)]
    payload = adapter.build_multi_batch_payload(
        claims, {c.id: [] for c in claims}, inject_evidence=False
    )
    assert payload["model"] == "gpt-4.1"
    assert payload["max_tool_calls"] == 2 * 4
    assert payload["max_output_tokens"] >= 1024 * 4
    # Prompt-cache parity: the OpenAI system text must match OPENAI_SYNTHESIS_SYSTEM exactly.
    from truthbot.verify.adapters.base import OPENAI_SYNTHESIS_SYSTEM

    assert payload["input"][0]["content"][0]["text"] == OPENAI_SYNTHESIS_SYSTEM


def test_openai_parse_multi_all_succeed() -> None:
    adapter = OpenAIAdapter()
    claims = [_claim("A"), _claim("B")]
    text = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High", "explanation": "a"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High", "explanation": "b"},
        ]
    )
    raw = _openai_body(text, cached=321)
    verdicts = adapter.parse_multi_batch_response(
        raw, claims, batch_call_id="openai::multi::9"
    )
    assert verdicts[0].label == VerdictLabel.TRUE
    assert verdicts[1].label == VerdictLabel.FALSE
    assert verdicts[0].cached_input_tokens == 321
    assert verdicts[1].cached_input_tokens == 0
    assert all(v.batch_call_id == "openai::multi::9" for v in verdicts)


def test_openai_parse_multi_malformed_marks_all_no_response() -> None:
    adapter = OpenAIAdapter()
    claims = [_claim("A"), _claim("B")]
    raw = _openai_body("garbage not json")
    verdicts = adapter.parse_multi_batch_response(raw, claims)
    assert all(v.no_response for v in verdicts)


def test_openai_parse_multi_duplicated_claim_id_uses_first() -> None:
    adapter = OpenAIAdapter()
    claims = [_claim("A"), _claim("B")]
    text = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High"},
            {"claim_id": claims[0].id, "label": "False", "confidence": "High"},
            {"claim_id": claims[1].id, "label": "Mostly True", "confidence": "Medium"},
        ]
    )
    raw = _openai_body(text)
    verdicts = adapter.parse_multi_batch_response(raw, claims)
    # First occurrence wins for claims[0]; claims[1] still gets its own verdict.
    assert verdicts[0].label == VerdictLabel.TRUE
    assert verdicts[1].label == VerdictLabel.MOSTLY_TRUE


def test_adapters_expose_multi_claim_caps() -> None:
    assert AnthropicAdapter.max_claims_per_request >= 2
    assert OpenAIAdapter.max_claims_per_request >= 2
    assert AnthropicAdapter.supports_batch is True
    assert OpenAIAdapter.supports_batch is True
