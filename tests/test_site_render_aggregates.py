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
    CSS,
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


def test_verdict_panel_renders_both_lens_aggregates_strict_first() -> None:
    """2026-04-30: Strict became the published default. Both lens
    blocks still render server-side, but Strict comes first in DOM
    order and stays visible on initial paint while Lenient ships
    ``hidden``. Non-JS clients therefore see Strict."""
    bundles = [
        _make_bundle(VerdictLabel.MOSTLY_TRUE, coarse_lenient="Truthy", coarse_strict="Truthy"),
        _make_bundle(VerdictLabel.EXAGGERATED, coarse_lenient="Truthy", coarse_strict="Falsey"),
        _make_bundle(VerdictLabel.FALSE, coarse_lenient="False", coarse_strict="False"),
    ]
    sr = _make_site_report(bundles)
    html = _verdict_panel(sr)
    assert 'data-lens-axis="lenient"' in html
    assert 'data-lens-axis="strict"' in html
    # Strict comes first now.
    strict_idx  = html.index('data-lens-axis="strict"')
    lenient_idx = html.index('data-lens-axis="lenient"')
    assert strict_idx < lenient_idx
    assert 'data-lens-axis="lenient" hidden' in html
    assert 'data-lens-axis="strict" hidden' not in html


def test_verdict_panel_bar_blocks_carry_lens_caption() -> None:
    """Each bar block is now self-labeled so the reader knows which
    lens they're seeing without consulting the chip."""
    bundles = [
        _make_bundle(VerdictLabel.MOSTLY_TRUE, coarse_lenient="Truthy", coarse_strict="Truthy"),
    ]
    sr = _make_site_report(bundles)
    html = _verdict_panel(sr)
    assert "Strict lens" in html
    assert "Lenient lens" in html
    # Captions live inside their own data-lens-axis block — assert that
    # the Strict caption appears before the Lenient caption (strict-first).
    assert html.index("Strict lens") < html.index("Lenient lens")


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


def test_css_hidden_attribute_hides_lens_axis_under_flex_display() -> None:
    """Without this rule, `.report-bar` / `.report-counts` ``display:flex``
    overrides the HTML ``hidden`` attribute and *both* lens bars show on
    index cards — only one bar should be visible at a time."""
    assert "[data-lens-axis][hidden]" in CSS
    assert "display: none !important" in CSS


def test_report_card_renders_paired_lens_axis_blocks_strict_first() -> None:
    """Same Strict-first DOM order as the verdict panel — landing-page
    cards reflect the published default."""
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
    assert 'data-lens-axis="lenient" hidden' in html
    assert 'data-lens-axis="strict" hidden' not in html
    # Captions name the active lens.
    assert "Strict lens" in html
    assert "Lenient lens" in html
    # Lenient says all 5 are Truthy; Strict splits 3 Truthy / 2 Falsey.
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


# ── Round 3: "% truthy or better" lens-aware stat ─────────────────────────────


def test_verdict_panel_promotes_truthy_and_false_into_headline_frames() -> None:
    """2026-04-30: '% Truthy or better' moved out of the stats grid
    into a dedicated 2-frame block above the grid; '% False or worse'
    joined it. Stats grid reverts to 4 columns (no more stats-5)."""
    bundles = [
        _make_bundle(VerdictLabel.MOSTLY_TRUE, coarse_lenient="Truthy", coarse_strict="Truthy"),
        _make_bundle(VerdictLabel.EXAGGERATED, coarse_lenient="Truthy", coarse_strict="Falsey"),
        _make_bundle(VerdictLabel.FALSE,       coarse_lenient="False",  coarse_strict="False"),
    ]
    sr = _make_site_report(bundles)
    html = _verdict_panel(sr)
    # Frame markup present, both labels visible.
    assert 'vp-headline-stats' in html
    assert 'vp-stat-truthy' in html
    assert 'vp-stat-false' in html
    assert 'Truthy or better' in html
    assert 'False or worse' in html
    # Stats grid reverted to 4 columns; no more stats-5.
    assert 'class="stats stats-4"' in html
    assert 'class="stats stats-5"' not in html
    # Truthy-or-better is no longer a tile inside the .stats grid.
    truthy_idx = html.index("Truthy or better")
    grid_idx   = html.index('class="stats stats-4"')
    assert truthy_idx < grid_idx, "Truthy frame must precede the stats grid"


def test_headline_frames_are_lens_paired_strict_first() -> None:
    """Each frame holds two ``data-lens-axis`` spans (strict + lenient);
    Strict ships visible, Lenient ships ``hidden``."""
    bundles = [
        _make_bundle(VerdictLabel.MOSTLY_TRUE, coarse_lenient="Truthy", coarse_strict="Truthy"),
        _make_bundle(VerdictLabel.EXAGGERATED, coarse_lenient="Truthy", coarse_strict="Falsey"),
    ]
    sr = _make_site_report(bundles)
    html = _verdict_panel(sr)
    # Both axes present in headline-frame markup
    assert '<span class="lens-target" data-lens-axis="strict">' in html
    assert '<span class="lens-target" data-lens-axis="lenient" hidden>' in html


def test_truthy_or_better_includes_unverifiable_in_denominator() -> None:
    """Editorial choice: a leader citing an unverifiable claim is itself
    a fact-check failure, so Unverifiable counts in the denominator
    (NOT just the numerator-eligible Truthy/True buckets)."""
    bundles = [
        _make_bundle(VerdictLabel.TRUE,         coarse_lenient="True",        coarse_strict="True"),
        _make_bundle(VerdictLabel.MOSTLY_TRUE,  coarse_lenient="Truthy",      coarse_strict="Truthy"),
        _make_bundle(VerdictLabel.UNVERIFIABLE, coarse_lenient="Unverifiable", coarse_strict="Unverifiable"),
        _make_bundle(VerdictLabel.FALSE,        coarse_lenient="False",       coarse_strict="False"),
    ]
    sr = _make_site_report(bundles)
    html = _verdict_panel(sr)
    # 2 of 4 are Truthy-or-better → 50%. If Unverifiable were excluded
    # from the denominator we'd see 2/3 = 67%.
    assert '<span class="lens-target" data-lens-axis="strict">50%</span>' in html
    assert '<span class="lens-target" data-lens-axis="lenient" hidden>50%</span>' in html


def test_false_or_worse_uses_falsey_plus_false_numerator() -> None:
    """% False or worse mirrors % Truthy or better on the negative
    end of the scale: numerator = (False + Falsey), denominator =
    full claim count (Unverifiable counts against)."""
    bundles = [
        _make_bundle(VerdictLabel.MOSTLY_TRUE,  coarse_lenient="Truthy",      coarse_strict="Truthy"),
        _make_bundle(VerdictLabel.MISLEADING,   coarse_lenient="Falsey",      coarse_strict="Falsey"),
        _make_bundle(VerdictLabel.FALSE,        coarse_lenient="False",       coarse_strict="False"),
        _make_bundle(VerdictLabel.UNVERIFIABLE, coarse_lenient="Unverifiable", coarse_strict="Unverifiable"),
    ]
    sr = _make_site_report(bundles)
    html = _verdict_panel(sr)
    # 2 of 4 (Misleading + False) → 50% on both axes for this fixture.
    # We pin the strict-side default since that's what non-JS sees.
    assert "False or worse" in html
    # Find the False-or-worse frame and check its numbers.
    false_frame_idx = html.index("vp-stat-false")
    next_frame_close = html.index('</div>', html.index('</div>', false_frame_idx) + 1)
    false_frame_html = html[false_frame_idx : next_frame_close + 200]
    assert 'data-lens-axis="strict">50%' in false_frame_html
    assert 'data-lens-axis="lenient" hidden>50%' in false_frame_html


# ── Round 3: site-wide Truthy mute persistence contract ─────────────────────


def test_embedded_js_contains_truthy_mute_storage_key() -> None:
    """The mute-toggle IIFE persists state under localStorage["truthy-mute"].
    We pin the key here so a rename can't silently break stored prefs."""
    from truthbot.publish.site import JS
    assert "'truthy-mute'" in JS
    assert "isTruthyFunPage" in JS  # fun-page exclusion path stays in place


def test_embedded_js_default_lens_is_strict() -> None:
    """2026-04-30 editorial flip: the published default editorial-lens
    flipped from Lenient to Strict. Pin the JS constant so a rename or
    accidental flip-back surfaces immediately. Stored user preference
    still wins on revisit — only the unset default is asserted here."""
    from truthbot.publish.site import JS
    assert "var DEFAULT_LENS = 'strict';" in JS
    # Storage key itself unchanged (existing localStorage values still work).
    assert "var STORAGE_KEY = 'editorial-lens';" in JS


def test_status_bar_lens_chip_defaults_to_strict() -> None:
    """The chip ships with ``data-lens="strict"`` and the visible value
    is "Strict" — matches the JS default so non-JS clients stay in
    sync with what the JS would write on first paint."""
    from truthbot.publish.site import _status_bar
    html = _status_bar(model_count=4, stamp="x")
    assert 'data-lens="strict"' in html
    assert "<span class=\"lens-value\">Strict</span>" in html


def test_truthy_tap_hint_includes_label_span_for_state_aware_text() -> None:
    """The tap-hint label is now JS-controlled (Tap / Tap to mute / Muted)
    so the JS needs a stable hook to find. Pin the marker class."""
    from truthbot.publish.site import _TRUTHY_TAP_HINT
    assert 'class="tap-hint-label"' in _TRUTHY_TAP_HINT


# ── Round 3: index strip wiring ─────────────────────────────────────────────


def test_compute_stats_populates_insights_when_claims_present() -> None:
    """`SitePublisher._compute_stats` should produce an `insights` entry
    when the claims index is populated, so the index renderer can pick
    it up. Drives the data flow end-to-end without a live publisher."""
    from truthbot.publish.site import SitePublisher
    publisher = SitePublisher.__new__(SitePublisher)   # bypass __init__
    publisher._root = None  # type: ignore[assignment]  # not used by _compute_stats
    reports = [{"id": "r1", "claim_count": 1, "model_agreement_rate": 1.0,
                "verdict_distribution": {"True": 1}}]
    claims = [{
        "id": "c1", "report_id": "r1",
        "claim_text": "alpha",
        "consensus_verdict": "True",
        "model_verdicts_summary": [
            {"adapter": "anthropic", "label": "True", "confidence": "High"},
            {"adapter": "openai",    "label": "True", "confidence": "High"},
        ],
        "url": "claims/c1.html",
    }]
    stats = publisher._compute_stats(reports, claims)
    assert "insights" in stats
    assert stats["insights"] is not None
    assert stats["insights"].total_claims == 1


# ── Model-cited (unverified) tier — strip-audit 2026-05 follow-up ─────────────


def _bundle_with_sources(
    *,
    web_sources_per_model: list[list[str]],
    mrs_per_model: list[list[str]],
) -> VerdictBundle:
    """Build a bundle with explicit per-model web_sources / model_reported_sources.

    Both lists must have the same length. Each entry produces one
    ModelVerdict on the bundle.
    """
    assert len(web_sources_per_model) == len(mrs_per_model)
    claim = Claim(
        transcript_id="t",
        text="Test claim with sources.",
        speaker="Speaker",
        context="ctx",
        category="economy",
        is_checkable=True,
    )
    mvs: list[ModelVerdict] = []
    for i, (ws, mrs) in enumerate(
        zip(web_sources_per_model, mrs_per_model)
    ):
        mvs.append(
            ModelVerdict(
                adapter_name=f"adapter-{i}",
                model_id=f"model-{i}",
                claim_id=claim.id,
                label=VerdictLabel.TRUE,
                confidence=Confidence.HIGH,
                explanation="r",
                web_sources=list(ws),
                model_reported_sources=list(mrs),
                stripped_source_count=max(0, len(mrs) - len(ws)),
            )
        )
    consensus = ConsensusVerdict(
        claim_id=claim.id,
        model_verdicts=mvs,
        consensus_label=VerdictLabel.TRUE,
        consensus_verdict=VerdictLabel.TRUE.value,
        confidence=Confidence.HIGH,
        agreement=True,
        consensus_strength="strong",
        explanation="x",
        coarse_lenient_label="True",
        coarse_lenient_strength="strong",
        coarse_strict_label="True",
        coarse_strict_strength="strong",
    )
    return VerdictBundle(
        claim=claim,
        speaker="Speaker",
        date_str="2026-03-04",
        model_verdicts=mvs,
        consensus=consensus,
    )


def test_evidence_block_surfaces_model_cited_unverified_tier() -> None:
    """When a model emits citations the tool didn't return (model_reported_sources
    minus web_sources is non-empty), the rendered claim card MUST surface those
    URLs as a separate "Model-cited URLs that didn't validate" sub-list under
    the Combined evidence/sources block. Domain-only, non-clickable, with a
    "didn't validate" badge so readers see the audit trail without us implying
    we vouched for them. Mirrors the 2026-04-30 arm-B finding that OpenAI batch
    stripped 25/26 URLs (all real *.gov domains)."""
    kept = "https://www.bls.gov/news.release/archives/cpi_12182025.pdf"
    stripped_a = "https://www.bls.gov/news.release/archives/cpi_12182025.htm"
    stripped_b = "https://www.bls.gov/news.release/archives/cpi_01132026.htm"
    bundle = _bundle_with_sources(
        web_sources_per_model=[[kept], []],
        mrs_per_model=[[kept, stripped_a, stripped_b], []],
    )
    html = _claim_card(bundle, idx=1, total=1, rel="../", standalone=True)
    # Kept URL renders in the existing verified evidence-list as a
    # clickable link.
    assert f'<a href="{kept}"' in html
    # Stripped URLs render as host+path text (so readers can verify
    # them) inside the new model-only sub-list — but MUST NOT be
    # clickable, since we did not vouch for them.
    assert "cpi_12182025.htm" in html
    assert "cpi_01132026.htm" in html
    assert f'<a href="{stripped_a}"' not in html
    assert f'<a href="{stripped_b}"' not in html
    # The unverified sub-list and header are present.
    assert "evidence-list-model-only" in html
    assert "Model-cited URLs that didn’t validate" in html
    # Exactly 2 model-only items (the kept URL must NOT appear in the
    # unverified list, even though its host shares a domain with the
    # stripped URLs).
    assert html.count('class="source-model-only"') == 2
    # "didn't validate" caveat badge present.
    assert "didn’t validate" in html


def test_evidence_block_unverified_only_when_no_web_sources() -> None:
    """Edge case: model emitted citations but tool returned NONE for any model
    (web_sources empty across the panel). The card MUST suppress the legacy
    "No sources retrieved." note and render only the unverified block — the
    audit trail is the right answer in that case. Without this branch readers
    would see "No sources retrieved." even though the model cited 4 URLs.
    """
    stripped = [
        "https://www.cbp.gov/newsroom/national-media-release/cbp-releases-march-2024-monthly-update",
        "https://www.cbp.gov/newsroom/stats/cbp-enforcement-statistics",
        "https://www.fbi.gov/news/press-releases/fbi-releases-2024-reported-crimes-in-the-nation-statistics",
        "https://www.fbi.gov/services/cjis/ucr/",
    ]
    bundle = _bundle_with_sources(
        web_sources_per_model=[[], []],
        mrs_per_model=[stripped[:2], stripped[2:]],
    )
    html = _claim_card(bundle, idx=1, total=1, rel="../", standalone=True)
    # Legacy "No sources retrieved." note SUPPRESSED since the model
    # did cite URLs — the unverified block is the audit trail.
    assert "No sources retrieved." not in html
    assert "evidence-list-model-only" in html
    assert html.count('class="source-model-only"') == 4
    # No clickable <a href> for any of the stripped URLs.
    for u in stripped:
        assert f'<a href="{u}"' not in html
    # Hosts visible (host+path form may be truncated for long URLs but
    # the domain is always intact).
    assert "cbp.gov" in html
    assert "fbi.gov" in html


def test_evidence_block_no_unverified_when_model_reported_empty() -> None:
    """Backward compat: legacy bundles without model_reported_sources (or with
    everything already in web_sources) render unchanged — the new sub-list
    must NOT appear. Avoids visual regression on cached pre-2026-04-26 reports
    that don't carry the MRS field."""
    bundle = _bundle_with_sources(
        web_sources_per_model=[
            ["https://apnews.com/article/foo"],
            ["https://www.reuters.com/world/bar"],
        ],
        mrs_per_model=[[], []],
    )
    html = _claim_card(bundle, idx=1, total=1, rel="../", standalone=True)
    assert "evidence-list-model-only" not in html
    assert "Model-cited URLs that didn’t validate" not in html
    assert "didn’t validate" not in html


def test_evidence_block_model_only_url_validated_by_other_model_excluded() -> None:
    """A URL stripped on model A but validated (kept in web_sources) on model B
    is treated as VALIDATED at the bundle level — the unverified block only
    contains URLs no model successfully grounded. Ensures we don't surface
    spurious doubt when at least one search tool returned the citation."""
    shared = "https://www.bls.gov/news.release/cpi.htm"
    only_stripped = "https://www.bls.gov/news.release/archives/cpi_01132026.htm"
    bundle = _bundle_with_sources(
        web_sources_per_model=[[shared], []],
        mrs_per_model=[[shared, only_stripped], [shared]],
    )
    html = _claim_card(bundle, idx=1, total=1, rel="../", standalone=True)
    # Shared URL kept once as a clickable link.
    assert f'<a href="{shared}"' in html
    # Only the truly-stripped URL appears in the unverified sub-list,
    # rendered with its path so the reader can verify it themselves.
    assert "evidence-list-model-only" in html
    assert html.count('class="source-model-only"') == 1
    assert "cpi_01132026.htm" in html
    assert f'<a href="{only_stripped}"' not in html

