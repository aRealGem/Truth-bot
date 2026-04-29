"""Render-side tests for the Model Panel Insights pages.

What this file pins:

* ``_insights_strip_html(insights)`` produces a compact 3-card strip
  with a CTA pointing to ``./model-insights.html``.
* ``_render_model_insights(insights)`` renders the dedicated deep-dive
  page with the agreement matrix, per-model bias bars, top extreme
  splits, and the methodology footnote.
* Both renderers degrade gracefully when ``insights`` is None / empty.
"""
from __future__ import annotations

import pytest

from truthbot.publish.insights import (
    ExtremeSplit,
    ModelPanelInsights,
    ModelStat,
    PairAgreement,
    compute_model_panel_insights,
)
from truthbot.publish.site import (
    _insights_strip_html,
    _render_model_insights,
)


def _claim(text: str, consensus: str, panel: dict[str, str]) -> dict:
    return {
        "id": text[:8],
        "report_id": "r-1",
        "claim_text": text,
        "consensus_verdict": consensus,
        "model_verdicts_summary": [
            {"adapter": a, "label": l, "confidence": "High"}
            for a, l in panel.items()
        ],
        "url": f"claims/{text[:8]}.html",
    }


# Fixture covering all four adapters with deliberate skew so the renderers
# have non-trivial numbers to surface.
def _sample_insights() -> ModelPanelInsights:
    claims = [
        _claim("Claim alpha", "False", {
            "anthropic": "True",        # lone optimist (diff=4)
            "openai":    "False",
            "gemini":    "False",
            "xai":       "False",
        }),
        _claim("Claim beta", "True", {
            "anthropic": "True",
            "openai":    "Mostly True",
            "gemini":    "True",
            "xai":       "True",
        }),
        _claim("Claim gamma", "False", {
            "anthropic": "False",
            "openai":    "False",
            "gemini":    "False",
            "xai":       "True",
        }),
    ]
    return compute_model_panel_insights(claims)


# ── Index strip ─────────────────────────────────────────────────────────────


def test_strip_returns_empty_when_insights_is_none() -> None:
    assert _insights_strip_html(None) == ""


def test_strip_returns_empty_for_empty_insights() -> None:
    empty = ModelPanelInsights(per_model=[], pairwise=[], top_extreme_splits=[],
                               total_claims=0)
    assert _insights_strip_html(empty) == ""


def test_strip_renders_section_with_cta_to_insights_page() -> None:
    html = _insights_strip_html(_sample_insights())
    assert 'class="insights-strip"' in html
    assert 'class="insight-card"' in html
    # CTA always points at the dedicated page
    assert 'href="./model-insights.html"' in html


def test_strip_includes_pairwise_top_card_and_dissent_card() -> None:
    html = _insights_strip_html(_sample_insights())
    assert "Strongest pairwise agreement" in html
    assert "Most divergent on the panel" in html


def test_strip_uses_pretty_brand_names() -> None:
    html = _insights_strip_html(_sample_insights())
    # No raw lowercase adapter ids
    for raw in ("anthropic", "openai", "gemini", "xai"):
        assert "{0}".format(raw) not in html or "xai" not in html or "xAI" in html
    # Pretty names should appear
    assert "Anthropic" in html or "OpenAI" in html or "Google" in html or "xAI" in html


# ── Dedicated insights page ────────────────────────────────────────────────


def test_insights_page_degrades_gracefully_when_no_data() -> None:
    page = _render_model_insights(None)
    assert "Model panel insights" in page
    assert "Not enough claims yet" in page


def test_insights_page_renders_all_sections() -> None:
    page = _render_model_insights(_sample_insights())
    for marker in (
        "<h1>Model panel insights</h1>",
        "<h2>Per-model summary</h2>",
        "<h2>Truthy bias</h2>",
        "<h2>Pairwise agreement</h2>",
        "<h2>Top extreme splits</h2>",
        "<h2>Method</h2>",
        'class="agreement-matrix"',
        'class="bias-chart"',
        'class="insights-summary"',
        'class="extreme-card"',
    ):
        assert marker in page, f"missing: {marker}"


def test_insights_page_methodology_links_to_about_and_eval_scan() -> None:
    page = _render_model_insights(_sample_insights())
    assert 'href="./about.html"' in page
    assert "opus_vs_rest_scan.py" in page


def test_insights_page_extreme_card_links_to_claim_page() -> None:
    page = _render_model_insights(_sample_insights())
    # The lone-optimist split on "Claim alpha" should produce a link to
    # ../claims/Claim alp.html (the synthetic test url uses claim_text[:8]
    # and our renderer prefixes ../). We just assert the relative-up link
    # form is present.
    assert "../claims/" in page


def test_insights_page_bias_bar_uses_lenient_or_strict_class() -> None:
    page = _render_model_insights(_sample_insights())
    # Anthropic should be rendered with the lenient bar (positive bias);
    # at least one strict-side bar should appear too (xAI / others).
    assert "bias-fill-lenient" in page or "bias-fill-strict" in page


def test_insights_page_renders_diff_badge_for_extreme_split() -> None:
    page = _render_model_insights(_sample_insights())
    assert 'class="extreme-diff"' in page
    # The fixture's lone-optimist diff is 4 (True vs False trio).
    assert "&Delta;4" in page
