"""Render-side tests for the lens-aware aggregate displays + frontier cleanup.

What this file pins (everything that round 2 of the projection work shipped):

* ``SiteReport.verdict_distribution_lenient`` / ``verdict_distribution_strict``
  produce the projected 5-bucket histograms and round-trip the engine's
  ``LENIENT_PROJECTION`` / ``STRICT_PROJECTION``.
* ``_verdict_panel(site_report)`` renders BOTH lens variants of the
  headline + ratio + verdict bar, with ``data-lens-axis`` markup so the
  lens toggle JS can flip them in lockstep with per-claim pills.
* ``_toc(bundles)`` mini-pills carry ``data-coarse-lenient`` /
  ``data-coarse-strict`` attrs and the shared ``.lens-pill`` class
  (so the toggle finds them alongside headline pills).
* ``_report_card(report_meta)`` renders Lenient + Strict aggregate
  segment bars and headlines in paired ``[data-lens-axis]`` blocks.
* The per-model strip on a claim card no longer renders any
  ``model-tier-wrap`` element — that "frontier"/"batch" chip was
  retired (2026-04-29) per the editorial decision that the panel is
  always frontier modulo the bundle-level Triage pill.
* The methodology line on each report still says
  "frontier language model(s)" (intentionally kept).
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
from truthbot.publish.site import (
    COARSE_LENIENT_PROJECTION,
    COARSE_STRICT_PROJECTION,
    SiteReport,
    _claim_card,
    _headline_verdict_coarse,
    _project_dist,
    _report_card,
    _render_report,
    _toc,
    _verdict_panel,
)
from truthbot.verify.engine import (
    LENIENT_PROJECTION,
    STRICT_PROJECTION,
)


# ── Fixture builders ──────────────────────────────────────────────────────────


def _make_bundle(
    fine_label: VerdictLabel,
    *,
    coarse_lenient: str = "",
    coarse_strict: str = "",
    model_labels: list[VerdictLabel] | None = None,
) -> VerdictBundle:
    """Build a minimal VerdictBundle suitable for renderer assertions."""
    if model_labels is None:
        # A 4-model panel that's unanimous on the fine label keeps the
        # renderer focused on aggregates rather than dissent paths.
        model_labels = [fine_label] * 4
    claim = Claim(
        transcript_id="t",
        text="Test claim.",
        speaker="Speaker",
        context="ctx",
        category="economy",
        is_checkable=True,
    )
    mvs = [
        ModelVerdict(
            adapter_name=f"adapter-{i}",
            model_id=f"model-{i}",
            claim_id=claim.id,
            label=lbl,
            confidence=Confidence.HIGH,
            explanation="r",
        )
        for i, lbl in enumerate(model_labels)
    ]
    consensus = ConsensusVerdict(
        claim_id=claim.id,
        model_verdicts=mvs,
        consensus_label=fine_label,
        consensus_verdict=fine_label.value,
        confidence=Confidence.HIGH,
        agreement=True,
        consensus_strength="strong",
        explanation="x",
        coarse_lenient_label=coarse_lenient,
        coarse_lenient_strength="strong" if coarse_lenient else "none",
        coarse_strict_label=coarse_strict,
        coarse_strict_strength="strong" if coarse_strict else "none",
    )
    return VerdictBundle(
        claim=claim,
        speaker="Speaker",
        date_str="2026-03-04",
        model_verdicts=mvs,
        consensus=consensus,
    )


def _make_site_report(bundles: list[VerdictBundle]) -> SiteReport:
    return SiteReport(
        report_id="00000000-1111-2222-3333-444444444444",
        speaker="Test Speaker",
        role="President",
        date=datetime(2026, 3, 4, tzinfo=timezone.utc),
        venue="Test Venue",
        transcript_source_url="",
        bundles=bundles,
        source_of_claims="Test Speaker",
        source_of_claims_professional_public_title="President",
        event="Test Event",
        channel="",
    )


# ── Projection mapping invariants ─────────────────────────────────────────────


def test_site_projection_constants_match_engine() -> None:
    """site.py keys these mappings on the fine label *string* (not the enum),
    but the projection itself MUST agree with engine.py for any two callers
    to produce the same 5-bucket distribution. This pin catches drift."""
    for fine_label in VerdictLabel:
        engine_lenient = LENIENT_PROJECTION[fine_label]
        engine_strict = STRICT_PROJECTION[fine_label]
        assert COARSE_LENIENT_PROJECTION[fine_label.value] == engine_lenient, fine_label
        assert COARSE_STRICT_PROJECTION[fine_label.value] == engine_strict, fine_label


def test_project_dist_collapses_mostly_true_plus_exaggerated_under_lenient() -> None:
    """The whole point of the Lenient axis: directionally aligned
    Mostly True + Exaggerated counts merge into Truthy."""
    fine = {"True": 1, "Mostly True": 3, "Exaggerated": 2, "Misleading": 0,
            "False": 0, "Unverifiable": 1}
    out = _project_dist(fine, COARSE_LENIENT_PROJECTION)
    assert out["True"] == 1
    assert out["Truthy"] == 5            # 3 Mostly True + 2 Exaggerated
    assert out["Falsey"] == 0
    assert out["False"] == 0
    assert out["Unverifiable"] == 1


def test_project_dist_separates_exaggerated_from_truthy_under_strict() -> None:
    fine = {"True": 1, "Mostly True": 3, "Exaggerated": 2, "Misleading": 1,
            "False": 0, "Unverifiable": 0}
    out = _project_dist(fine, COARSE_STRICT_PROJECTION)
    assert out["Truthy"] == 3            # only Mostly True under Strict
    assert out["Falsey"] == 3            # 2 Exaggerated + 1 Misleading


# ── SiteReport coarse distributions ───────────────────────────────────────────


def test_site_report_lenient_distribution_uses_stored_coarse_label_when_present() -> None:
    bundles = [
        _make_bundle(VerdictLabel.MOSTLY_TRUE, coarse_lenient="Truthy", coarse_strict="Truthy"),
        _make_bundle(VerdictLabel.EXAGGERATED, coarse_lenient="Truthy", coarse_strict="Falsey"),
        _make_bundle(VerdictLabel.FALSE, coarse_lenient="False", coarse_strict="False"),
    ]
    sr = _make_site_report(bundles)
    dist_l = sr.verdict_distribution_lenient
    dist_s = sr.verdict_distribution_strict
    assert dist_l["Truthy"] == 2
    assert dist_l["False"] == 1
    assert dist_s["Truthy"] == 1
    assert dist_s["Falsey"] == 1
    assert dist_s["False"] == 1


def test_site_report_falls_back_to_projection_for_legacy_bundles() -> None:
    """A bundle whose coarse_* fields are still empty (cached before the
    projection layer landed) must still contribute to the aggregates by
    projecting the fine label on the fly."""
    bundles = [
        _make_bundle(VerdictLabel.EXAGGERATED, coarse_lenient="", coarse_strict=""),
        _make_bundle(VerdictLabel.MISLEADING, coarse_lenient="", coarse_strict=""),
    ]
    sr = _make_site_report(bundles)
    assert sr.verdict_distribution_lenient["Truthy"] == 1   # Exaggerated -> Truthy
    assert sr.verdict_distribution_lenient["Falsey"] == 1   # Misleading  -> Falsey
    assert sr.verdict_distribution_strict["Falsey"] == 2    # Excg+Misl   -> Falsey


# ── _headline_verdict_coarse ──────────────────────────────────────────────────


def test_headline_verdict_coarse_speaks_truthy_scale_vocabulary() -> None:
    # 4 of 5 Truthy → "Truthy" (already-qualified; no "Mostly" prefix).
    label, cls = _headline_verdict_coarse(
        {"True": 1, "Truthy": 4, "Unverifiable": 0, "Falsey": 0, "False": 0,
         "Models split": 0}
    )
    assert label == "Truthy"
    assert cls == "vt-truthy"

    # 4 True out of 5 → "Largely True"
    label, cls = _headline_verdict_coarse(
        {"True": 4, "Truthy": 0, "Unverifiable": 0, "Falsey": 1, "False": 0,
         "Models split": 0}
    )
    assert label == "Largely True"
    assert cls == "vt-true"

    # Tie → Mixed verdict
    label, cls = _headline_verdict_coarse(
        {"True": 0, "Truthy": 2, "Unverifiable": 0, "Falsey": 2, "False": 0,
         "Models split": 0}
    )
    assert label == "Mixed verdict"
    assert cls == "neutral"


def test_headline_verdict_coarse_dominant_models_split_reads_as_mixed() -> None:
    """A panel that's mostly Models-split shouldn't read as "Mostly Models
    split" — that's nonsense English. Surface it as Mixed verdict."""
    label, cls = _headline_verdict_coarse(
        {"True": 1, "Truthy": 0, "Unverifiable": 0, "Falsey": 0, "False": 0,
         "Models split": 4}
    )
    assert label == "Mixed verdict"
    assert cls == "neutral"


# ── _verdict_panel renders both lens axes ─────────────────────────────────────


def test_verdict_panel_renders_both_lens_aggregates() -> None:
    bundles = [
        _make_bundle(VerdictLabel.MOSTLY_TRUE, coarse_lenient="Truthy", coarse_strict="Truthy"),
        _make_bundle(VerdictLabel.EXAGGERATED, coarse_lenient="Truthy", coarse_strict="Falsey"),
        _make_bundle(VerdictLabel.FALSE, coarse_lenient="False", coarse_strict="False"),
    ]
    sr = _make_site_report(bundles)
    html = _verdict_panel(sr)
    # Both lens-axis blocks present, only Strict starts hidden.
    assert 'data-lens-axis="lenient"' in html
    assert 'data-lens-axis="strict"' in html
    # Strict block comes after Lenient in source order; assert hidden attr
    # appears between the two markers so the page renders Lenient by default
    # for non-JS clients.
    lenient_idx = html.index('data-lens-axis="lenient"')
    strict_idx  = html.index('data-lens-axis="strict"')
    assert lenient_idx < strict_idx
    assert 'data-lens-axis="strict" hidden' in html
    assert 'data-lens-axis="lenient" hidden' not in html


def test_verdict_panel_uses_coarse_labels_in_headline() -> None:
    """The 6-bucket label (e.g. "Exaggerated") should NOT appear as the
    headline anymore. We only ever speak the Truthy scale on aggregates."""
    bundles = [_make_bundle(VerdictLabel.EXAGGERATED,
                             coarse_lenient="Truthy", coarse_strict="Falsey")] * 3
    sr = _make_site_report(bundles)
    html = _verdict_panel(sr)
    # Truthy reads as headline under Lenient; Falsey under Strict. Neither
    # axis should leak the fine "Exaggerated" string into the headline pill.
    # (Other places — e.g. an aria-label inside the verdict bar — may still
    # contain it, so we scope the assertion to the .vp-verdict block.)
    import re
    headlines = re.findall(r'<div class="vp-verdict[^"]*">([^<]+)</div>', html)
    assert any("Truthy" in h for h in headlines)
    assert any("Falsey" in h for h in headlines)
    assert not any("Exaggerated" in h for h in headlines)


# ── TOC mini-pill carries lens-pill class + both data attrs ──────────────────


def test_toc_pill_carries_both_coarse_attrs_and_lens_pill_class() -> None:
    bundles = [
        _make_bundle(VerdictLabel.MOSTLY_TRUE, coarse_lenient="Truthy", coarse_strict="Truthy"),
        _make_bundle(VerdictLabel.EXAGGERATED, coarse_lenient="Truthy", coarse_strict="Falsey"),
    ]
    html = _toc(bundles)
    # Shared marker class so the JS finds TOC pills alongside headline pills.
    assert "lens-pill" in html
    # Both data attrs present per pill.
    assert html.count('data-coarse-lenient="Truthy"') == 2
    # Strict diverges on the second pill (Exaggerated → Falsey under Strict).
    assert 'data-coarse-strict="Truthy"' in html
    assert 'data-coarse-strict="Falsey"' in html


def test_toc_pill_falls_back_for_legacy_bundles() -> None:
    bundles = [_make_bundle(VerdictLabel.MOSTLY_TRUE)]   # empty coarse_*
    html = _toc(bundles)
    assert 'data-coarse-lenient="Truthy"' in html        # projected on the fly
    assert 'data-coarse-strict="Truthy"' in html


# ── _report_card (index per-report card) renders both axes ───────────────────


def test_report_card_renders_paired_lens_axis_blocks() -> None:
    r = {
        "id": "rid", "url": "reports/r.html",
        "speaker": "X", "date": "2026-03-04", "venue": "v",
        "claim_count": 5,
        "verdict_distribution": {"Mostly True": 3, "Exaggerated": 2,
                                 "False": 0, "Misleading": 0, "True": 0,
                                 "Unverifiable": 0},
        "verdict_distribution_lenient": {"True": 0, "Truthy": 5,
                                         "Unverifiable": 0,
                                         "Falsey": 0, "False": 0,
                                         "Models split": 0},
        "verdict_distribution_strict":  {"True": 0, "Truthy": 3,
                                         "Unverifiable": 0,
                                         "Falsey": 2, "False": 0,
                                         "Models split": 0},
        "tier_counts": {},
    }
    html = _report_card(r)
    assert 'data-lens-axis="lenient"' in html
    assert 'data-lens-axis="strict"' in html
    assert 'data-lens-axis="strict" hidden' in html
    # Lenient says all 5 are Truthy (Mostly True + Exaggerated collapse).
    # Strict splits 3 Truthy / 2 Falsey.
    # Headlines should be self-descriptive (Truthy / Mixed verdict respectively).
    assert "Truthy" in html
    assert "Falsey" in html


def test_report_card_legacy_entry_falls_back_to_on_the_fly_projection() -> None:
    """An older reports.json entry that only has the 6-bucket
    ``verdict_distribution`` must still render lens-aware aggregates."""
    r = {
        "id": "rid", "url": "reports/r.html",
        "speaker": "X", "date": "2026-03-04", "venue": "v",
        "claim_count": 5,
        "verdict_distribution": {"Mostly True": 3, "Exaggerated": 2},
        # NO verdict_distribution_lenient / _strict — pre-projection era.
        "tier_counts": {},
    }
    html = _report_card(r)
    assert 'data-lens-axis="lenient"' in html
    assert 'data-lens-axis="strict"' in html
    # Lenient should still surface Truthy.
    assert "Truthy" in html


# ── Frontier badge cleanup + methodology pin ──────────────────────────────────


def test_per_model_card_has_no_frontier_tier_chip() -> None:
    """The ``model-tier-wrap`` chip used to render under each model card with
    text like "frontier" or "triage". Editorial decision (2026-04-29) was
    to retire it — the panel is always frontier modulo the bundle-level
    Triage pill. Pin: this element must not render in any state."""
    # Bundle with non-default tier/mode would have triggered the chip.
    bundle = _make_bundle(VerdictLabel.TRUE,
                          coarse_lenient="True", coarse_strict="True")
    # Manually force tier/mode that previously triggered the chip.
    for mv in bundle.model_verdicts:
        mv.tier = "frontier"
        mv.synthesis_mode = "batch"   # would have rendered the chip pre-cleanup
    html = _claim_card(bundle, idx=1, total=1, rel="../", standalone=True)
    assert "model-tier-wrap" not in html
    assert "model-tier-sum" not in html
    assert "Review tier:" not in html


def test_methodology_line_still_says_frontier_language_models() -> None:
    """The methodology line at the top of every report intentionally still
    says "verified by N frontier language models" (per 2026-04-29 editorial
    decision). Pin that wording so a future drive-by edit can't quietly
    change it."""
    bundles = [_make_bundle(VerdictLabel.TRUE,
                             coarse_lenient="True", coarse_strict="True")]
    sr = _make_site_report(bundles)
    html = _render_report(sr)
    # 4-adapter panel from _make_bundle → "4 frontier language models".
    assert "frontier language model" in html
