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


def test_build_multi_user_message_demands_per_claim_web_sources() -> None:
    """Regression for the Phase 3a calibration finding (Bug A): OpenAI/Grok
    multi-claim verdicts returned ``web_sources: []`` for every claim despite
    invoking the search tool 7-17 times, because the original multi-claim
    preamble only mentioned ``web_sources`` once and relied on inheritance
    from the single-claim CITATION DISCIPLINE block in the system prompt.

    The strengthened preamble must explicitly require per-claim attribution
    (each claim's ``web_sources`` reflects URLs retrieved for THAT claim),
    forbid the field being omitted, and reiterate that empty-list is the
    only acceptable empty signal."""
    claims = [_claim("A"), _claim("B")]
    msg = build_multi_user_message(claims, {}, inject_evidence=False)

    lower = msg.lower()
    assert "per-claim" in lower or "per claim" in lower
    assert "that specific claim" in lower or "that claim" in lower
    assert "do not omit" in lower or "not omit" in lower or "must" in lower
    assert "web_sources" in msg
    assert "MUST include the `web_sources`" in msg


def test_build_multi_user_message_schema_block_is_valid_json_shape() -> None:
    """Regression for the post-Step-1 calibration finding: an earlier draft
    of the strengthened schema put a JavaScript-style ``// REQUIRED ...``
    inline comment inside the JSON-shaped example block. JSON does not allow
    comments, and Anthropic + Gemini both responded with prose explaining
    the schema instead of valid JSON arrays — every multi-claim batch row
    was logged as ``parse_error`` / ``api_error`` (zero verdicts).

    The schema example block (between ``[`` and ``]``) must remain
    JSON-shaped: a ``//`` comment may appear in the surrounding free-text
    instructions, but never inside the bracketed schema example.
    """
    from truthbot.verify.adapters.base import _MULTI_CLAIM_OUTPUT_SCHEMA

    bracket_block_start = _MULTI_CLAIM_OUTPUT_SCHEMA.index("[")
    bracket_block_end = _MULTI_CLAIM_OUTPUT_SCHEMA.rindex("]")
    json_block = _MULTI_CLAIM_OUTPUT_SCHEMA[bracket_block_start : bracket_block_end + 1]
    assert "//" not in json_block, (
        f"Schema example block must be JSON-shaped (no // comments). "
        f"Got: {json_block!r}"
    )


def test_parse_multi_claim_json_preserves_explicit_empty_web_sources() -> None:
    """Regression: when the model explicitly returns ``web_sources: []`` for
    a claim, ``parse_multi_claim_json`` must preserve that signal so
    downstream ``apply_url_grounding`` records it as model-asserted "no
    sources" (mrs=[], stripped=0) rather than treating the field as absent.

    This complements the strengthened multi-claim prompt: the prompt makes
    the model's intent explicit, and this test guarantees the parser
    propagates that intent without coercion."""
    claims = [_claim("A"), _claim("B")]
    payload = json.dumps(
        [
            {
                "claim_id": claims[0].id,
                "label": "True",
                "confidence": "High",
                "web_sources": [],
            },
            {
                "claim_id": claims[1].id,
                "label": "False",
                "confidence": "High",
                "web_sources": ["https://example.gov/foo"],
            },
        ]
    )
    out = parse_multi_claim_json(payload, claims)
    assert out[claims[0].id]["web_sources"] == []
    assert out[claims[1].id]["web_sources"] == ["https://example.gov/foo"]
    assert "web_sources" in out[claims[0].id]


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


def test_build_multi_verdicts_attributes_tool_call_count_to_first_only() -> None:
    """Regression guard for finding C6 — tool_call_count must not N-count
    a single API call's tool usage across the per-claim verdict rows."""
    claims = [_claim("A"), _claim("B"), _claim("C")]
    raw_by_claim = {
        c.id: {"label": "True", "confidence": "High", "explanation": "x"} for c in claims
    }
    out = build_multi_verdicts(
        claims,
        raw_by_claim,
        adapter_name="openai",
        model_id="gpt-5.4",
        call_usage={
            "input_tokens": 1200,
            "output_tokens": 450,
            "tool_call_count": 7,
        },
    )
    assert out[0].tool_call_count == 7
    assert out[1].tool_call_count == 0
    assert out[2].tool_call_count == 0
    assert sum(v.tool_call_count for v in out) == 7


def test_build_multi_verdicts_tool_call_count_defaults_to_zero() -> None:
    """call_usage may omit tool_call_count entirely (Grok / unset path)."""
    claims = [_claim("A"), _claim("B")]
    raw_by_claim = {
        c.id: {"label": "True", "confidence": "High", "explanation": "x"} for c in claims
    }
    out = build_multi_verdicts(
        claims,
        raw_by_claim,
        adapter_name="grok",
        model_id="grok-4",
        call_usage={"input_tokens": 100},
    )
    assert out[0].tool_call_count == 0
    assert out[1].tool_call_count == 0


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


# ── Defensive ``model_reported_sources`` backfill (2026-04-26) ───────────────
# When OpenAI / Gemini / xAI multi-claim drop per-claim attribution despite
# the search tool firing, we backfill ``model_reported_sources`` for every
# claim so audit trails / cross-claim consensus see the grounding signal.
# Index-0 also gets ``web_sources`` populated for visible publish-layer
# grounding, but siblings keep ``web_sources`` empty to preserve attribution
# fidelity (see STATUS.md L75-80 trade-off).


def test_build_multi_verdicts_backfills_mrs_when_web_sources_omitted() -> None:
    """Model omits ``web_sources`` entirely → tool URLs land on every mrs."""
    claims = [_claim("A"), _claim("B"), _claim("C")]
    raw_by_claim = {
        c.id: {"label": "True", "confidence": "High", "explanation": "x"}
        for c in claims
    }
    tool_urls = ["https://bls.gov/x", "https://bea.gov/y"]

    out = build_multi_verdicts(
        claims,
        raw_by_claim,
        adapter_name="xai",
        model_id="grok-4",
        tool_retrieved_urls=tool_urls,
    )

    assert all(v.model_reported_sources == tool_urls for v in out), (
        "all chunk indices must have model_reported_sources populated from "
        "tool URLs when the model dropped per-claim attribution"
    )


def test_build_multi_verdicts_backfills_mrs_when_web_sources_explicit_empty() -> None:
    """Model emits ``web_sources: []`` explicitly → same backfill applies."""
    claims = [_claim("A"), _claim("B")]
    raw_by_claim = {
        c.id: {
            "label": "True",
            "confidence": "High",
            "explanation": "x",
            "web_sources": [],
        }
        for c in claims
    }
    tool_urls = ["https://example.gov/a", "https://example.gov/b"]

    out = build_multi_verdicts(
        claims,
        raw_by_claim,
        adapter_name="gemini",
        model_id="gemini-2.5-pro",
        tool_retrieved_urls=tool_urls,
    )

    assert out[0].model_reported_sources == tool_urls
    assert out[1].model_reported_sources == tool_urls


def test_build_multi_verdicts_backfill_visible_grounding_index_zero_only() -> None:
    """``web_sources`` (publish-layer field) is backfilled on index-0 only.

    Siblings stay empty so the published report doesn't claim each claim
    was independently grounded by the same URL set.
    """
    claims = [_claim("A"), _claim("B"), _claim("C")]
    raw_by_claim = {
        c.id: {"label": "True", "confidence": "High", "explanation": "x"}
        for c in claims
    }
    tool_urls = ["https://bls.gov/x", "https://bea.gov/y"]

    out = build_multi_verdicts(
        claims,
        raw_by_claim,
        adapter_name="openai",
        model_id="gpt-5.4",
        tool_retrieved_urls=tool_urls,
    )

    assert out[0].web_sources == tool_urls, "index-0 gets visible grounding"
    assert out[1].web_sources == [], "siblings keep web_sources empty"
    assert out[2].web_sources == [], "siblings keep web_sources empty"


def test_build_multi_verdicts_no_backfill_when_tool_urls_empty() -> None:
    """If the search tool didn't fire, neither field is backfilled."""
    claims = [_claim("A"), _claim("B")]
    raw_by_claim = {
        c.id: {"label": "True", "confidence": "High", "explanation": "x"}
        for c in claims
    }

    out = build_multi_verdicts(
        claims,
        raw_by_claim,
        adapter_name="xai",
        model_id="grok-4",
        tool_retrieved_urls=[],
    )

    assert all(v.web_sources == [] for v in out)
    assert all(v.model_reported_sources == [] for v in out)


def test_build_multi_verdicts_no_backfill_when_tool_retrieved_urls_is_none() -> None:
    """Legacy callers (no Layer 1d wiring) keep the pre-grounding behavior."""
    claims = [_claim("A"), _claim("B")]
    raw_by_claim = {
        c.id: {"label": "True", "confidence": "High", "explanation": "x"}
        for c in claims
    }

    out = build_multi_verdicts(
        claims,
        raw_by_claim,
        adapter_name="anthropic",
        model_id="claude-opus-4-7",
        # tool_retrieved_urls omitted → defaults to None
    )

    assert all(v.model_reported_sources == [] for v in out)


def test_build_multi_verdicts_preserves_partial_attribution() -> None:
    """When the model attributes some URLs validly, no backfill clobbers them.

    Anthropic-style gold-standard path: model emits real per-claim
    web_sources that intersect with tool URLs; the new backfill must not
    overwrite this.
    """
    claims = [_claim("A"), _claim("B")]
    raw_by_claim = {
        claims[0].id: {
            "label": "True",
            "confidence": "High",
            "explanation": "x",
            "web_sources": ["https://bls.gov/x"],
        },
        claims[1].id: {
            "label": "False",
            "confidence": "High",
            "explanation": "y",
            "web_sources": ["https://bea.gov/y"],
        },
    }
    tool_urls = ["https://bls.gov/x", "https://bea.gov/y", "https://other.gov/z"]

    out = build_multi_verdicts(
        claims,
        raw_by_claim,
        adapter_name="anthropic",
        model_id="claude-opus-4-7",
        tool_retrieved_urls=tool_urls,
    )

    assert out[0].web_sources == ["https://bls.gov/x"]
    assert out[0].model_reported_sources == ["https://bls.gov/x"]
    assert out[1].web_sources == ["https://bea.gov/y"]
    assert out[1].model_reported_sources == ["https://bea.gov/y"]


def test_build_multi_verdicts_backfill_caps_tool_urls_at_ten() -> None:
    """Tool URL backfill is capped at 10 to avoid exploding the audit list."""
    claims = [_claim("A")]
    raw_by_claim = {
        claims[0].id: {"label": "True", "confidence": "High", "explanation": "x"}
    }
    tool_urls = [f"https://example.gov/{i}" for i in range(20)]

    out = build_multi_verdicts(
        claims,
        raw_by_claim,
        adapter_name="xai",
        model_id="grok-4",
        tool_retrieved_urls=tool_urls,
    )

    assert len(out[0].model_reported_sources) == 10
    assert len(out[0].web_sources) == 10
    assert out[0].model_reported_sources == tool_urls[:10]
