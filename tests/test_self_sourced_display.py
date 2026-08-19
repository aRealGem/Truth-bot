"""Render-side tests for the honest-abstention sub-state (PR-A2.1 T1.1/T1.2).

Pins the display-only contract: a gate-forced Unverifiable whose only bearing
sources are the speaker's own organization renders "Unverified — self-sourced
only", its self-records are badged on the source strip, the
verdict panel gains the decomposition chip, and claims.json exports enough
provenance (``evidence_gate``, ``self_sourced_only``) for the consistency
checker to re-derive every chip number. NO verdict, gate, or weight changes.
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
    VerdictProvenance,
)
from truthbot.publish.site import (
    GATE_INSUFFICIENT,
    SELF_SOURCED_PILL,
    SitePublisher,
    SiteReport,
    _claim_card,
    _is_self_sourced_unverified,
    _self_source_ids,
    _verdict_panel,
)

_WH = "https://www.whitehouse.gov/the-press-office/2014/01/28/fact-sheet"
_AP = "https://apnews.com/article/education-summit"


def _bundle(
    label: VerdictLabel = VerdictLabel.UNVERIFIABLE,
    *,
    gate: str = GATE_INSUFFICIENT,
    sources: list[dict] | None = None,
    claim_type: str = "statistical",
    consensus_verdict: str | None = None,
) -> VerdictBundle:
    claim = Claim(transcript_id="t", text="We convened 150 university presidents.",
                  speaker="Barack Obama", context="ctx", category="education",
                  is_checkable=True)
    mv = ModelVerdict(adapter_name="panel", model_id="m", claim_id=claim.id,
                      label=label, confidence=Confidence.HIGH, explanation="r")
    verdict_text = label.value if consensus_verdict is None else consensus_verdict
    # Production split bundles carry "Models split" in the STORED coarse labels
    # (set by the projection layer); mirror that so the strict/lenient
    # distributions bucket them the way real reports do.
    coarse = "Models split" if verdict_text == "Models split" else ""
    consensus = ConsensusVerdict(
        claim_id=claim.id, model_verdicts=[mv], consensus_label=label,
        consensus_verdict=verdict_text,
        coarse_lenient_label=coarse, coarse_strict_label=coarse,
        confidence=Confidence.HIGH, agreement=True,
        consensus_strength="strong", explanation="x",
        provenance=VerdictProvenance(layer_a_label="check-worthy",
                                     layer_a_source="A2",
                                     layer_a_claim_type=claim_type,
                                     evidence_gate=gate))
    return VerdictBundle(
        claim=claim, speaker="Barack Obama", date_str="2014-01-28",
        model_verdicts=[mv], consensus=consensus,
        sources_consulted=sources if sources is not None else [
            {"id": "E1", "source": "R1", "url": _WH, "tier": "Political",
             "snippet": "Fact sheet", "supports_claim": True},
            {"id": "E2", "source": "R1", "url": "https://example.com/blog",
             "tier": "Other", "snippet": "context", "supports_claim": None},
        ])


def _site_report(bundles: list[VerdictBundle]) -> SiteReport:
    return SiteReport(
        report_id="00000000-1111-2222-3333-444444444444",
        speaker="Barack Obama", role="President",
        date=datetime(2014, 1, 28, tzinfo=timezone.utc), venue="U.S. Capitol",
        transcript_source_url="", bundles=bundles,
        source_of_claims="Barack Obama",
        source_of_claims_professional_public_title="President",
        event="State of the Union Address", channel="")


# ── Predicate ─────────────────────────────────────────────────────────────────


def test_gate_failed_self_bearing_only_is_self_sourced() -> None:
    assert _is_self_sourced_unverified(_bundle()) is True


def test_independent_bearing_s13_item_defeats_the_substate() -> None:
    b = _bundle(sources=[
        {"id": "E1", "url": _WH, "tier": "Political", "supports_claim": True},
        {"id": "E2", "url": _AP, "tier": "Wire", "supports_claim": True},
    ])
    assert _is_self_sourced_unverified(b) is False


def test_nonbearing_self_item_alone_does_not_qualify() -> None:
    b = _bundle(sources=[
        {"id": "E1", "url": _WH, "tier": "Political", "supports_claim": None},
    ])
    assert _is_self_sourced_unverified(b) is False


def test_passing_gate_or_decided_verdicts_are_untouched() -> None:
    assert _is_self_sourced_unverified(_bundle(gate="")) is False
    assert _is_self_sourced_unverified(
        _bundle(VerdictLabel.TRUE, gate="")) is False


def test_anecdote_pill_takes_precedence() -> None:
    assert _is_self_sourced_unverified(
        _bundle(claim_type="personal-anecdote")) is False


def test_split_claims_are_excluded() -> None:
    assert _is_self_sourced_unverified(
        _bundle(consensus_verdict="Models split")) is False


def test_gate_string_pins_the_consolidator_constant() -> None:
    from truthbot.verdict import consolidator
    assert GATE_INSUFFICIENT == consolidator.GATE_INSUFFICIENT


# ── Claim card + source strip ─────────────────────────────────────────────────


def test_claim_card_renders_self_sourced_pill() -> None:
    html = _claim_card(_bundle(), 1, 1)
    assert SELF_SOURCED_PILL in html
    assert "pill-self-sourced" in html
    # The visible pill text is the sub-state — never a bare "Unverifiable"
    # headline pill (same mechanism as the anecdote pill).
    assert f'>{SELF_SOURCED_PILL}</span>' in html


def test_claim_card_badges_the_self_records() -> None:
    html = _claim_card(_bundle(), 1, 1)
    assert "ev-self" in html
    assert _self_source_ids(_bundle()) == {"E1"}


def test_decided_claim_renders_no_substate_markup() -> None:
    html = _claim_card(_bundle(VerdictLabel.TRUE, gate=""), 1, 1)
    assert SELF_SOURCED_PILL not in html
    assert "pill-self-sourced" not in html


# ── Verdict panel chip ────────────────────────────────────────────────────────


def test_panel_chip_decomposes_abstentions_and_sums_to_claim_count() -> None:
    bundles = [
        _bundle(VerdictLabel.TRUE, gate=""),
        _bundle(VerdictLabel.TRUE, gate=""),
        _bundle(),                                  # self-sourced-only UV
        _bundle(sources=[]),                        # gate-failed, no self items
        _bundle(consensus_verdict="Models split", gate=""),
    ]
    html = _verdict_panel(_site_report(bundles))
    # D17-d: what used to fall into "unverifiable — other" is now named. The
    # gate-failed bundle is a WITHHELD verdict, not an undecidable claim, and
    # the chip says which. Terms still sum to claim_count.
    assert ("2 decided · 1 unverified — self-sourced only · "
            "1 insufficient qualifying evidence retrieved · "
            "1 models split") in html


def test_panel_chip_appears_for_gate_withheld_alone() -> None:
    """D17-d: the chip is no longer self-sourced-only.

    It used to be absent unless a self-sourced claim existed — so a report
    whose abstentions were ALL gate-withheld decomposed nothing and published a
    bare Unverifiable count, which is the exact reading this split exists to
    fix."""
    bundles = [_bundle(VerdictLabel.TRUE, gate=""), _bundle(sources=[])]
    html = _verdict_panel(_site_report(bundles))
    assert "vp-abstention-chip" in html  # A3: class renamed (was vp-selfsource-chip)
    assert "1 insufficient qualifying evidence retrieved" in html


def test_the_chip_tooltip_describes_the_substates_it_shows() -> None:
    """A chip listing only gate-withheld counts must not explain itself with
    "every source bearing on this claim is the speaker's own organization"."""
    from truthbot.publish.site import GATE_WITHHELD_TITLE, SELF_SOURCED_TITLE
    html = _verdict_panel(_site_report(
        [_bundle(VerdictLabel.TRUE, gate=""), _bundle(sources=[])]))
    assert GATE_WITHHELD_TITLE.split(".")[0] in html
    assert SELF_SOURCED_TITLE.split(".")[0] not in html


# ── claims.json export ────────────────────────────────────────────────────────


def test_claim_meta_exports_gate_and_substate() -> None:
    publisher = SitePublisher.__new__(SitePublisher)  # bypass __init__
    sr = _site_report([_bundle()])
    meta = publisher._claim_meta(sr.bundles[0], sr)
    assert meta["provenance"]["evidence_gate"] == GATE_INSUFFICIENT
    assert meta["provenance"]["self_sourced_only"] is True
    decided = publisher._claim_meta(_bundle(VerdictLabel.TRUE, gate=""), sr)
    assert decided["provenance"]["evidence_gate"] == ""
    assert decided["provenance"]["self_sourced_only"] is False
