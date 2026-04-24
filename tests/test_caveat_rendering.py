"""Tests for per-model caveat attribution + normalized dedup.

Covers the fix for findings C8 (no attribution) and C9 (exact-string dedup)
in the SOTU 2026 review. Lives separately from `test_site_social.py` to
keep the caveat-render test surface close to the helpers that produced
the bugs: ``_normalize_caveat_signature`` and ``_render_caveat_block``.
"""
from __future__ import annotations

import pytest

from truthbot.models import Confidence, ModelVerdict, VerdictLabel
from truthbot.publish.site import (
    _normalize_caveat_signature,
    _render_caveat_block,
)


def _mv(
    adapter: str,
    caveat: str,
    *,
    model_id: str = "",
    no_response: bool = False,
) -> ModelVerdict:
    return ModelVerdict(
        adapter_name=adapter,
        model_id=model_id or f"{adapter}-test",
        claim_id="c1",
        label=VerdictLabel.TRUE,
        confidence=Confidence.HIGH,
        explanation="x",
        caveats=caveat,
        no_response=no_response,
    )


# ── _normalize_caveat_signature ───────────────────────────────────────────────


def test_normalize_signature_strips_whitespace_and_lowercases() -> None:
    s1 = _normalize_caveat_signature("Source reliability may vary.")
    s2 = _normalize_caveat_signature("  SOURCE   reliability may vary.  ")
    s3 = _normalize_caveat_signature("source reliability may vary,\n  as noted")
    assert s1 == s2
    assert s3.startswith(s1)


def test_normalize_signature_truncates_to_prefix_length() -> None:
    long = "a" * 500
    assert len(_normalize_caveat_signature(long)) == 80


def test_normalize_signature_empty_returns_empty() -> None:
    assert _normalize_caveat_signature("") == ""
    assert _normalize_caveat_signature("   \n\t  ") == ""


def test_normalize_signature_strips_trailing_punctuation() -> None:
    assert _normalize_caveat_signature("A short note.") == _normalize_caveat_signature(
        "A short note"
    )


# ── _render_caveat_block ──────────────────────────────────────────────────────


def test_render_caveat_block_empty_when_no_caveats() -> None:
    assert _render_caveat_block([]) == ""
    assert _render_caveat_block([_mv("anthropic", "")]) == ""


def test_render_caveat_block_single_model_shows_attribution() -> None:
    """Fix for C8 — every surviving caveat is attributed to the contributing
    adapter brand, never rendered as an anonymous block."""
    html = _render_caveat_block([_mv("anthropic", "Projections assume 2025 data.")])
    assert 'class="caveat"' in html
    assert "Model notes" in html
    assert 'class="caveat-attribution"' in html
    assert "Anthropic" in html
    assert "Projections assume 2025 data." in html


def test_render_caveat_block_distinct_caveats_render_separately() -> None:
    html = _render_caveat_block(
        [
            _mv("anthropic", "Projections assume 2025 data."),
            _mv("openai", "Source tier not yet verified."),
        ]
    )
    assert html.count('class="caveat-item"') == 2
    assert "Anthropic" in html
    assert "OpenAI" in html


def test_render_caveat_block_groups_near_duplicates_by_normalized_prefix() -> None:
    """Fix for C9 — exact-string dedup let semantically identical caveats
    double up. Normalized-prefix dedup must fold them into one item with
    multi-model attribution."""
    html = _render_caveat_block(
        [
            _mv("anthropic", "Source reliability may vary."),
            _mv("openai", "  SOURCE reliability may vary.  "),
            _mv("gemini", "Tier 1 citation not independently verified."),
        ]
    )
    assert html.count('class="caveat-item"') == 2
    assert "Anthropic, OpenAI" in html
    assert "Google" in html


def test_render_caveat_block_preserves_first_seen_model_text() -> None:
    """When deduped, the first-seen caveat text wins so the rendered
    phrasing traces back to a specific model's output. Both caveats share
    the first 80 normalized characters so the dedup key collides."""
    shared_prefix = (
        "Underlying dataset covers the period through mid-2025 and may lag on "
        "the latest reporting"
    )
    html = _render_caveat_block(
        [
            _mv("anthropic", f"{shared_prefix}; more recent data still pending."),
            _mv("openai", f"{shared_prefix}; updates trickle in later."),
        ]
    )
    assert "Anthropic, OpenAI" in html
    assert "more recent data still pending" in html
    assert "updates trickle in later" not in html
    assert html.count('class="caveat-item"') == 1


def test_render_caveat_block_skips_no_response_verdicts() -> None:
    """A model that failed to respond has no verdict-level caveat worth
    displaying; its empty/placeholder caveat must not leak into the block."""
    html = _render_caveat_block(
        [
            _mv("anthropic", "Valid caveat here."),
            _mv("openai", "ignored", no_response=True),
        ]
    )
    assert "Anthropic" in html
    assert "OpenAI" not in html


def test_render_caveat_block_grok_attribution_uses_xai_brand() -> None:
    html = _render_caveat_block([_mv("grok", "xAI-only caveat text.")])
    assert "xAI" in html


def test_render_caveat_block_unknown_adapter_falls_back_to_model_id() -> None:
    html = _render_caveat_block(
        [_mv("nova", "Caveat from unrecognized adapter.", model_id="nova-1")]
    )
    assert "Nova" in html or "nova" in html.lower()


def test_render_caveat_block_does_not_concat_into_single_paragraph() -> None:
    """Regression guard for C9 — the old code joined every caveat with ' '
    which stitched opposed claims into one paragraph. The new block renders
    each distinct caveat in its own <li> so contradictory notes can't bleed
    into each other."""
    html = _render_caveat_block(
        [
            _mv("anthropic", "Operation confirmed by Tier 1 sources."),
            _mv("openai", "Appears to be speculative fiction with no real source."),
        ]
    )
    assert "</li>" in html
    assert html.count("<li") == 2
