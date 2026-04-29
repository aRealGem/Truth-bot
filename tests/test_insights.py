"""Unit tests for `truthbot.publish.insights.compute_model_panel_insights`.

These tests pin the per-model panel insights data layer. The *rendering*
of the same data is covered separately in
[`tests/test_site_render_insights.py`](test_site_render_insights.py).
"""
from __future__ import annotations

import pytest

from truthbot.publish.insights import (
    EXTREME_DIFF_THRESHOLD,
    LABEL_SCORE,
    ModelPanelInsights,
    _adapter_brand,
    compute_model_panel_insights,
)


def _claim(
    text: str,
    consensus: str,
    panel: dict[str, str],
    *,
    claim_id: str | None = None,
    report_id: str = "r-1",
    speaker: str = "Speaker",
) -> dict:
    """Build a claim dict mirroring the SitePublisher._claim_meta shape."""
    return {
        "id": claim_id or text[:8],
        "report_id": report_id,
        "claim_text": text,
        "speaker": speaker,
        "consensus_verdict": consensus,
        "model_verdicts_summary": [
            {"adapter": adapter, "label": label, "confidence": "High"}
            for adapter, label in panel.items()
        ],
        "url": f"claims/{claim_id or text[:8]}.html",
    }


# ── Constants in lockstep with eval/opus_vs_rest_scan.py ─────────────────────


def test_label_score_matches_opus_scan() -> None:
    """The Truthy-axis score table must agree with the eval-side scan."""
    from eval.opus_vs_rest_scan import LABEL_SCORE as EVAL_SCORE
    from eval.opus_vs_rest_scan import EXTREME_DIFF_THRESHOLD as EVAL_THRESHOLD

    assert LABEL_SCORE == EVAL_SCORE
    assert EXTREME_DIFF_THRESHOLD == EVAL_THRESHOLD


def test_adapter_brand_handles_known_adapters_and_falls_back() -> None:
    assert _adapter_brand("anthropic") == "Anthropic"
    assert _adapter_brand("xai") == "xAI"
    assert _adapter_brand("grok") == "xAI"
    assert _adapter_brand("UnknownVendor").lower() == "unknownvendor"


# ── Empty / degenerate inputs ────────────────────────────────────────────────


def test_empty_claims_returns_empty_insights() -> None:
    out = compute_model_panel_insights([])
    assert out.per_model == []
    assert out.pairwise == []
    assert out.top_extreme_splits == []
    assert out.total_claims == 0
    assert out.most_divergent is None
    assert out.most_lenient is None
    assert out.top_pair is None


def test_skips_claims_with_unknown_labels() -> None:
    """A model verdict whose ``label`` isn't in LABEL_SCORE is silently
    dropped — we don't want junk labels skewing the bias math."""
    claims = [
        _claim("a", "True", {"anthropic": "True", "openai": "totally bogus"}),
    ]
    ins = compute_model_panel_insights(claims)
    # openai has no scoreable verdicts → not in adapter universe
    adapters = {m.adapter for m in ins.per_model}
    assert "anthropic" in adapters
    assert "openai" not in adapters


# ── Dedup ────────────────────────────────────────────────────────────────────


def test_dedups_by_normalized_claim_text() -> None:
    """Same claim text appearing in multiple report runs counts once."""
    claims = [
        _claim("Trump claims X.", "True",
               {"anthropic": "True", "openai": "True"}, claim_id="c1", report_id="r1"),
        _claim("trump claims x.", "True",       # case + whitespace variant
               {"anthropic": "True", "openai": "True"}, claim_id="c2", report_id="r2"),
        _claim("  Trump claims X.  ", "True",
               {"anthropic": "True", "openai": "True"}, claim_id="c3", report_id="r3"),
    ]
    ins = compute_model_panel_insights(claims)
    assert ins.total_claims == 1


# ── Dissent rate ─────────────────────────────────────────────────────────────


def test_dissent_rate_counts_label_disagreement_with_consensus() -> None:
    claims = [
        _claim("a", "True",  {"anthropic": "True",        "openai": "False"}),
        _claim("b", "True",  {"anthropic": "Mostly True", "openai": "True"}),
        _claim("c", "False", {"anthropic": "False",       "openai": "False"}),
    ]
    ins = compute_model_panel_insights(claims)
    by = {m.adapter: m for m in ins.per_model}

    # OpenAI dissents on claim a (False vs True). Mostly True dissents on b.
    assert by["openai"].dissent_count == 1
    assert by["anthropic"].dissent_count == 1
    assert by["openai"].claims_seen == 3
    assert by["anthropic"].claims_seen == 3
    assert by["openai"].dissent_rate == pytest.approx(1 / 3)


# ── Truthy bias ──────────────────────────────────────────────────────────────


def test_truthy_bias_signed_average_against_panel_mean() -> None:
    """Anthropic always votes Truthy in a panel that doesn't — bias ≫ 0."""
    claims = [
        _claim("a", "False",
               {"anthropic": "True", "openai": "False", "gemini": "False"}),
        _claim("b", "False",
               {"anthropic": "True", "openai": "False", "gemini": "False"}),
    ]
    ins = compute_model_panel_insights(claims)
    by = {m.adapter: m for m in ins.per_model}
    # Per claim: anthropic=+2, others mean=-2 → bias=+4 each → avg=+4
    assert by["anthropic"].truthy_bias == pytest.approx(4.0)
    # Each non-Anthropic peer sees others_mean = (anthropic + other_peer)/2 = (2+-2)/2 = 0
    # so its bias is -2 - 0 = -2 per claim → avg = -2
    assert by["openai"].truthy_bias == pytest.approx(-2.0)
    assert by["gemini"].truthy_bias == pytest.approx(-2.0)


# ── Extreme splits ───────────────────────────────────────────────────────────


def test_lone_optimist_split_recorded_for_anthropic_against_three_falses() -> None:
    panel = {
        "anthropic": "True",
        "openai":    "False",
        "gemini":    "False",
        "xai":       "False",
    }
    claims = [_claim("opus stands alone", "False", panel)]
    ins = compute_model_panel_insights(claims)
    assert len(ins.top_extreme_splits) == 1
    e = ins.top_extreme_splits[0]
    assert e.odd_one_out == "anthropic"
    assert e.direction == "optimist"
    assert e.diff == 4  # +2 - (-2) = 4
    assert e.odd_label == "True"
    assert "openai" in e.other_labels


def test_lone_pessimist_split_recorded() -> None:
    panel = {
        "anthropic": "True",
        "openai":    "True",
        "gemini":    "True",
        "xai":       "False",
    }
    claims = [_claim("xai is the lone pessimist", "True", panel)]
    ins = compute_model_panel_insights(claims)
    assert len(ins.top_extreme_splits) == 1
    e = ins.top_extreme_splits[0]
    assert e.odd_one_out == "xai"
    assert e.direction == "pessimist"


def test_no_extreme_split_when_diff_below_threshold() -> None:
    """A 2-point gap (Mostly True +1 vs Exaggerated -1) shouldn't qualify."""
    panel = {
        "anthropic": "Mostly True",
        "openai":    "Exaggerated",
        "gemini":    "Exaggerated",
        "xai":       "Exaggerated",
    }
    claims = [_claim("close call", "Exaggerated", panel)]
    ins = compute_model_panel_insights(claims)
    assert ins.top_extreme_splits == []


def test_extreme_split_ranks_by_diff_desc() -> None:
    claims = [
        # diff=3 (Mostly True vs False trio)
        _claim("medium", "False", {
            "anthropic": "Mostly True", "openai": "False",
            "gemini": "False", "xai": "False",
        }),
        # diff=4 (True vs False trio)
        _claim("max", "False", {
            "anthropic": "True", "openai": "False",
            "gemini": "False", "xai": "False",
        }),
    ]
    ins = compute_model_panel_insights(claims)
    diffs = [e.diff for e in ins.top_extreme_splits]
    assert diffs == sorted(diffs, reverse=True)
    assert diffs[0] == 4


# ── Pairwise agreement ───────────────────────────────────────────────────────


def test_pairwise_agreement_uses_fine_label_identity() -> None:
    """Pairwise agreement is the share of co-checked claims where two
    models cast IDENTICAL fine labels — not Truthy-axis projection."""
    claims = [
        _claim("a", "True",
               {"anthropic": "True", "openai": "True"}),         # agree
        _claim("b", "Mostly True",
               {"anthropic": "Mostly True", "openai": "True"}),  # disagree
        _claim("c", "Mostly True",
               {"anthropic": "Mostly True", "openai": "Mostly True"}),  # agree
    ]
    ins = compute_model_panel_insights(claims)
    pair = next(p for p in ins.pairwise if p.a == "anthropic" and p.b == "openai")
    assert pair.agreement_rate == pytest.approx(2 / 3)
    assert pair.claims_both_present == 3


def test_pairwise_sorted_descending_by_agreement_rate() -> None:
    claims = [
        _claim("a", "True", {"anthropic": "True", "openai": "True", "gemini": "False"}),
        _claim("b", "True", {"anthropic": "True", "openai": "True", "gemini": "True"}),
    ]
    ins = compute_model_panel_insights(claims)
    rates = [p.agreement_rate for p in ins.pairwise]
    assert rates == sorted(rates, reverse=True)


# ── Convenience accessors ────────────────────────────────────────────────────


def test_most_lenient_and_strict_pick_correct_models() -> None:
    claims = [
        _claim("a", "False", {
            "anthropic": "True",
            "openai": "False",
            "gemini": "False",
        }),
        _claim("b", "False", {
            "anthropic": "Mostly True",
            "openai": "False",
            "gemini": "Exaggerated",
        }),
    ]
    ins = compute_model_panel_insights(claims)
    assert ins.most_lenient is not None
    assert ins.most_lenient.adapter == "anthropic"
    assert ins.most_strict is not None
    assert ins.most_strict.adapter in ("openai", "gemini")
    assert ins.most_strict.adapter != "anthropic"
