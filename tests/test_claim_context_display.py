"""Claim-in-context rendering (2026-08-01, jackie escalation).

Half the Obama-2014 claims open with deictic words ("Tonight, I'm announcing
we'll launch six more this year.") that are unreadable as a bare quote. The
panel always judged with ``claim.context`` (the segmenter's ``prev || claim ||
next``); these tests pin that the pages now show the reader the same thing:
neighbors greyed, the checked claim emphasized, legacy bundles unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone

from truthbot.models import (
    Claim,
    Confidence,
    ConsensusVerdict,
    ModelVerdict,
    VerdictBundle,
    VerdictLabel,
)
from truthbot.publish.site import (SitePublisher, _claim_card,
                                   _claim_quote_html, _esc)

_PREV = ("My administration has launched two hubs for high-tech manufacturing "
         "in Raleigh and Youngstown.")
_TEXT = "Tonight, I'm announcing we'll launch six more this year."
_NEXT = "Bipartisan bills in both houses could double the number of these hubs."


def _bundle(context: str | None) -> VerdictBundle:
    claim = Claim(transcript_id="t", text=_TEXT, speaker="Barack Obama",
                  context=context, category="economy", is_checkable=True)
    mv = ModelVerdict(adapter_name="panel", model_id="m", claim_id=claim.id,
                      label=VerdictLabel.TRUE, confidence=Confidence.HIGH,
                      explanation="r")
    consensus = ConsensusVerdict(
        claim_id=claim.id, model_verdicts=[mv],
        consensus_label=VerdictLabel.TRUE, consensus_verdict="True",
        confidence=Confidence.HIGH, agreement=True,
        consensus_strength="strong", explanation="x")
    return VerdictBundle(claim=claim, speaker="Barack Obama",
                         date_str="2014-01-28", model_verdicts=[mv],
                         consensus=consensus)


def test_quote_renders_inside_greyed_neighbors() -> None:
    html = _claim_quote_html(_bundle(f"{_PREV} || {_TEXT} || {_NEXT}").claim)
    assert "claim-quote-ctx" in html
    assert html.count('class="ccq-side"') == 2
    assert "Raleigh and Youngstown" in html          # prev sentence visible
    assert "Bipartisan bills" in html                # next sentence visible
    # The checked claim itself is the emphasized element, in quote marks.
    assert f'<span class="ccq-claim">"' in html
    # Reading order: prev … claim … next.
    assert html.index("Raleigh") < html.index("six more") < html.index("Bipartisan")


def test_legacy_bundle_without_context_renders_bare_quote_unchanged() -> None:
    html = _claim_quote_html(_bundle(None).claim)
    assert html == f'<blockquote class="claim-quote">"{_esc(_TEXT)}"</blockquote>'
    assert "ccq" not in html
    # Context equal to the claim text alone is also the bare quote.
    assert _claim_quote_html(_bundle(_TEXT).claim) == html


def test_unsplittable_context_falls_back_below_the_quote() -> None:
    # Context that doesn't contain the claim as a clean `||` element must
    # still reach the reader — below the bare quote, never dropped.
    html = _claim_quote_html(_bundle("Earlier remarks about manufacturing hubs.").claim)
    assert 'class="claim-quote"' in html
    assert "claim-context-fallback" in html
    assert "Earlier remarks about manufacturing hubs." in html


def test_claim_card_and_meta_carry_the_context() -> None:
    b = _bundle(f"{_PREV} || {_TEXT} || {_NEXT}")
    card = _claim_card(b, 23, 96)
    assert "claim-quote-ctx" in card and "ccq-side" in card
    publisher = SitePublisher.__new__(SitePublisher)
    meta = publisher._claim_meta(b, type("SR", (), {"report_id": "r"})())
    assert meta["claim_context"].startswith(_PREV)
