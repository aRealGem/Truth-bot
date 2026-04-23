"""Unit tests for the provider-agnostic multi-claim batching helpers.

Covers the pure-Python surface that the Anthropic/OpenAI adapters delegate
to: ``build_multi_user_message``, ``parse_multi_claim_json``,
``build_multi_verdicts``, and ``chunk_claims_with_evidence``. No SDK mocks
here — those live in ``test_multi_batch_adapters.py``.
"""

from __future__ import annotations

import json

import pytest

from truthbot.models import Claim, Confidence, Evidence, SourceTier, VerdictLabel
from truthbot.verify.adapters.base import (
    build_multi_user_message,
    build_multi_verdicts,
    parse_multi_claim_json,
)
from truthbot.verify.batch import chunk_claims_with_evidence


def _claim(text: str, *, speaker: str = "Test") -> Claim:
    return Claim(transcript_id="t1", text=text, speaker=speaker)


def _evidence(claim_id: str, name: str = "BLS") -> Evidence:
    return Evidence(
        claim_id=claim_id,
        source_name=name,
        source_url="https://example.gov/foo",
        source_tier=SourceTier.GOVERNMENT,
        snippet=f"Evidence for {claim_id[:8]}",
    )


# ── chunk_claims_with_evidence ────────────────────────────────────────────────


def test_chunk_claims_empty_list_returns_empty() -> None:
    assert chunk_claims_with_evidence([], 3) == []


def test_chunk_claims_single_item_smaller_than_chunk() -> None:
    c = _claim("X")
    assert chunk_claims_with_evidence([(c, [])], 5) == [[(c, [])]]


def test_chunk_claims_exact_multiple() -> None:
    items = [(_claim(f"c{i}"), []) for i in range(6)]
    chunks = chunk_claims_with_evidence(items, 3)
    assert [len(ch) for ch in chunks] == [3, 3]


def test_chunk_claims_non_multiple_last_chunk_is_shorter() -> None:
    items = [(_claim(f"c{i}"), []) for i in range(7)]
    chunks = chunk_claims_with_evidence(items, 3)
    assert [len(ch) for ch in chunks] == [3, 3, 1]


def test_chunk_claims_size_of_one_is_per_claim() -> None:
    items = [(_claim(f"c{i}"), []) for i in range(4)]
    chunks = chunk_claims_with_evidence(items, 1)
    assert [len(ch) for ch in chunks] == [1, 1, 1, 1]


def test_chunk_claims_size_zero_falls_back_to_one() -> None:
    items = [(_claim(f"c{i}"), []) for i in range(3)]
    chunks = chunk_claims_with_evidence(items, 0)
    assert [len(ch) for ch in chunks] == [1, 1, 1]


# ── build_multi_user_message ──────────────────────────────────────────────────


def test_build_multi_user_message_embeds_every_claim_id() -> None:
    claims = [_claim("Claim one."), _claim("Claim two."), _claim("Claim three.")]
    msg = build_multi_user_message(claims, {}, inject_evidence=False)
    for c in claims:
        assert c.id in msg
    assert "3 claims" in msg
    assert "Claim one." in msg
    assert "Claim three." in msg


def test_build_multi_user_message_includes_evidence_when_injected() -> None:
    c1 = _claim("First")
    c2 = _claim("Second")
    ev_map = {c1.id: [_evidence(c1.id)]}
    msg = build_multi_user_message(
        [c1, c2], ev_map, inject_evidence=True, max_evidence_per_claim=5
    )
    assert "BLS" in msg
    assert "Evidence for" in msg


def test_build_multi_user_message_caps_evidence_per_claim() -> None:
    c = _claim("Single")
    ev_map = {c.id: [_evidence(c.id, name=f"src{i}") for i in range(12)]}
    msg = build_multi_user_message(
        [c], ev_map, inject_evidence=True, max_evidence_per_claim=3
    )
    assert msg.count("src") == 3


def test_build_multi_user_message_rejects_empty_claim_list() -> None:
    with pytest.raises(ValueError):
        build_multi_user_message([], {})


# ── parse_multi_claim_json ────────────────────────────────────────────────────


def test_parse_multi_claim_json_keys_by_claim_id() -> None:
    claims = [_claim("A"), _claim("B")]
    payload = json.dumps(
        [
            {"claim_id": claims[0].id, "label": "True", "confidence": "High"},
            {"claim_id": claims[1].id, "label": "False", "confidence": "High"},
        ]
    )
    out = parse_multi_claim_json(payload, claims)
    assert out[claims[0].id]["label"] == "True"
    assert out[claims[1].id]["label"] == "False"


def test_parse_multi_claim_json_falls_back_to_position_when_ids_missing() -> None:
    claims = [_claim("A"), _claim("B"), _claim("C")]
    payload = json.dumps(
        [
            {"label": "True", "confidence": "High"},
            {"label": "False", "confidence": "High"},
            {"label": "Mostly True", "confidence": "Medium"},
        ]
    )
    out = parse_multi_claim_json(payload, claims)
    assert out[claims[0].id]["label"] == "True"
    assert out[claims[1].id]["label"] == "False"
    assert out[claims[2].id]["label"] == "Mostly True"


def test_parse_multi_claim_json_partial_response_returns_subset() -> None:
    claims = [_claim("A"), _claim("B"), _claim("C")]
    payload = json.dumps(
        [{"claim_id": claims[0].id, "label": "True", "confidence": "High"}]
    )
    out = parse_multi_claim_json(payload, claims)
    assert claims[0].id in out
    assert claims[1].id not in out
    assert claims[2].id not in out


def test_parse_multi_claim_json_tolerates_markdown_fences() -> None:
    claims = [_claim("A")]
    payload = f"```json\n[{{\"claim_id\": \"{claims[0].id}\", \"label\": \"True\", \"confidence\": \"High\"}}]\n```"
    out = parse_multi_claim_json(payload, claims)
    assert out[claims[0].id]["label"] == "True"


def test_parse_multi_claim_json_tolerates_verdicts_dict_wrapper() -> None:
    claims = [_claim("A")]
    payload = json.dumps(
        {"verdicts": [{"claim_id": claims[0].id, "label": "True", "confidence": "High"}]}
    )
    out = parse_multi_claim_json(payload, claims)
    assert out[claims[0].id]["label"] == "True"


def test_parse_multi_claim_json_raises_on_garbage() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_multi_claim_json("not json at all", [_claim("A")])


# ── build_multi_verdicts ──────────────────────────────────────────────────────


def test_build_multi_verdicts_fills_missing_as_unverifiable_no_response() -> None:
    claims = [_claim("A"), _claim("B"), _claim("C")]
    raw_by_claim = {
        claims[0].id: {
            "label": "True",
            "confidence": "High",
            "explanation": "ok",
            "web_sources": ["https://example.gov"],
        },
        # claims[1] omitted — partial response
        claims[2].id: {
            "label": "False",
            "confidence": "Medium",
            "explanation": "nope",
        },
    }
    out = build_multi_verdicts(
        claims,
        raw_by_claim,
        adapter_name="anthropic",
        model_id="claude-opus-4-7",
    )
    assert len(out) == 3
    assert out[0].label == VerdictLabel.TRUE
    assert out[0].batch_call_index == 0
    assert out[1].label == VerdictLabel.UNVERIFIABLE
    assert out[1].no_response is True
    assert out[1].batch_call_index == 1
    assert out[2].label == VerdictLabel.FALSE
    assert out[2].batch_call_index == 2


def test_build_multi_verdicts_attributes_usage_only_to_first_verdict() -> None:
    claims = [_claim("A"), _claim("B")]
    raw_by_claim = {
        c.id: {"label": "True", "confidence": "High", "explanation": "x"} for c in claims
    }
    out = build_multi_verdicts(
        claims,
        raw_by_claim,
        adapter_name="openai",
        model_id="gpt-4.1",
        call_usage={"cached_input_tokens": 999},
    )
    assert out[0].cached_input_tokens == 999
    assert out[1].cached_input_tokens == 0


def test_build_multi_verdicts_handles_invalid_label_gracefully() -> None:
    claims = [_claim("A")]
    raw_by_claim = {claims[0].id: {"label": "NotALabel", "confidence": "High"}}
    out = build_multi_verdicts(
        claims,
        raw_by_claim,
        adapter_name="anthropic",
        model_id="claude-opus-4-7",
    )
    assert out[0].label == VerdictLabel.UNVERIFIABLE
    assert out[0].no_response is True


def test_build_multi_verdicts_propagates_batch_call_id() -> None:
    claims = [_claim("A"), _claim("B")]
    raw_by_claim = {
        c.id: {"label": "True", "confidence": "High", "explanation": "x"} for c in claims
    }
    out = build_multi_verdicts(
        claims,
        raw_by_claim,
        adapter_name="anthropic",
        model_id="claude-opus-4-7",
        batch_call_id="anthropic::multi::abc123",
    )
    assert all(v.batch_call_id == "anthropic::multi::abc123" for v in out)
