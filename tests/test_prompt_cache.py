"""Prompt prefix stability, user-message builder, and cached-token cost math."""

from __future__ import annotations

import pytest

from truthbot.metrics.costs import estimate_cost
from truthbot.models import Claim, Evidence, SourceTier
from truthbot.verify.adapters.base import (
    OPENAI_SYNTHESIS_SYSTEM,
    SYNTHESIS_SYSTEM,
    build_user_message,
)


def test_openai_system_prefix_stable_across_claims() -> None:
    """OpenAI automatic caching requires identical system prefix bytes per request."""
    assert OPENAI_SYNTHESIS_SYSTEM.startswith(SYNTHESIS_SYSTEM)
    assert "Operational constraints (OpenAI)" in OPENAI_SYNTHESIS_SYSTEM
    # Heuristic: ~1024+ tokens for English prose (~3–4 chars/token).
    assert len(OPENAI_SYNTHESIS_SYSTEM) >= 4000


def test_build_user_message_stable_with_same_inputs() -> None:
    claim = Claim(transcript_id="t1", text="Unemployment fell.", speaker="Test")
    ev = [
        Evidence(
            claim_id=claim.id,
            source_name="BLS",
            source_url="https://bls.gov",
            source_tier=SourceTier.GOVERNMENT,
            snippet="Data shows a decline.",
        )
    ]
    a = build_user_message(claim, ev, inject_evidence=True)
    b = build_user_message(claim, ev, inject_evidence=True)
    assert a == b
    assert "Pre-gathered evidence" in a
    assert "Evidence:" in a


def test_build_user_message_no_inject_skips_evidence_block() -> None:
    claim = Claim(transcript_id="t1", text="GDP grew.", speaker="Test")
    ev = [
        Evidence(
            claim_id=claim.id,
            source_name="BEA",
            source_url="https://bea.gov",
            source_tier=SourceTier.GOVERNMENT,
            snippet="2% growth.",
        )
    ]
    msg = build_user_message(claim, ev, inject_evidence=False)
    assert "Evidence:" not in msg
    assert "No pre-gathered evidence was supplied" in msg


def test_estimate_cost_anthropic_cache_reads() -> None:
    """Anthropic: cache reads use discounted rate; writes use 1.25× input."""
    in_rate = 5.0 / 1_000_000
    cached_read_rate = 0.5 / 1_000_000  # 0.1 × in_rate in COST_TABLE
    out_rate = 25.0 / 1_000_000
    total_in = 1000
    cread = 800
    cwrite = 50
    fresh = total_in - cread - cwrite
    expected = (
        fresh * in_rate
        + cwrite * in_rate * 1.25
        + cread * cached_read_rate
        + 100 * out_rate
    )
    got = estimate_cost(
        "anthropic",
        "claude-opus-4-7",
        total_in,
        100,
        cache_read_input_tokens=cread,
        cache_creation_input_tokens=cwrite,
    )
    assert abs(got - expected) < 1e-9


def test_estimate_cost_openai_cached_split() -> None:
    got = estimate_cost(
        "openai",
        "gpt-4.1",
        2000,
        500,
        openai_cached_prompt_tokens=1500,
    )
    in_rate, cached_in_rate, out_rate = (2.0 / 1_000_000, 0.5 / 1_000_000, 8.0 / 1_000_000)
    expected = 500 * in_rate + 1500 * cached_in_rate + 500 * out_rate
    assert abs(got - expected) < 1e-9


def test_estimate_cost_batch_multiplier() -> None:
    base = estimate_cost("openai", "gpt-4.1", 1000, 200, mode="live")
    batched = estimate_cost("openai", "gpt-4.1", 1000, 200, mode="batch")
    assert abs(batched - base * 0.5) < 1e-9
