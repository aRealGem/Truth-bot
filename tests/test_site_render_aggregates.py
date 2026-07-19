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
    _adapter_run_stats,
    _claim_card,
    _headline_verdict_coarse,
    _project_dist,
    _report_card,
    _render_report,
    _run_manifest_html,
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
    consensus_verdict: str | None = None,
) -> VerdictBundle:
    """Build a minimal VerdictBundle suitable for renderer assertions.

    ``consensus_verdict`` overrides the display verdict text (defaults to the fine
    label's value); pass "Models split" to model a PCA split claim, which carries
    ``consensus_label=UNVERIFIABLE`` but its own headline bucket.
    """
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
        consensus_verdict=fine_label.value if consensus_verdict is None else consensus_verdict,
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


# ── Fine distribution: split claims get their own bucket ──────────────────────


def test_verdict_distribution_gives_models_split_its_own_bucket() -> None:
    # A genuine Unverifiable and a PCA "Models split" both carry
    # consensus_label=UNVERIFIABLE. They must NOT be merged: the split gets its
    # own headline bucket, matching the coarse distributions.
    bundles = [
        _make_bundle(VerdictLabel.UNVERIFIABLE),
        _make_bundle(VerdictLabel.UNVERIFIABLE, consensus_verdict="Models split"),
        _make_bundle(VerdictLabel.TRUE),
    ]
    dist = _make_site_report(bundles).verdict_distribution
    assert dist["Unverifiable"] == 1
    assert dist["Models split"] == 1
    assert dist["True"] == 1


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


def test_headline_verdict_coarse_family_aggregation() -> None:
    # 2026-07-19 editorial (jackie): headlines aggregate FAMILIES over decided
    # claims — a spread of adverse buckets must read as false-leaning, not
    # "Mixed" because no single bucket dominates.
    # 5/5 true-family (True + Truthy) → Largely True.
    label, cls = _headline_verdict_coarse(
        {"True": 1, "Truthy": 4, "Unverifiable": 0, "Falsey": 0, "False": 0,
         "Models split": 0}
    )
    assert label == "Largely True"
    assert cls == "vt-true"

    # The Trump-2026 shape: adverse spread (Falsey 51 + False 44) vs True 37
    # → 72% false-leaning of decided → Largely False (was "Mixed verdict").
    label, cls = _headline_verdict_coarse(
        {"True": 37, "Truthy": 0, "Unverifiable": 28, "Falsey": 51, "False": 44,
         "Models split": 7}
    )
    assert label == "Largely False"
    assert cls == "vt-false"

    # 60% adverse → Mostly False ("<40% true is damning", not Mixed).
    label, cls = _headline_verdict_coarse(
        {"True": 4, "Truthy": 0, "Unverifiable": 0, "Falsey": 6, "False": 0,
         "Models split": 0}
    )
    assert label == "Mostly False"

    # Genuine coin-flip → Mixed verdict.
    label, cls = _headline_verdict_coarse(
        {"True": 0, "Truthy": 2, "Unverifiable": 0, "Falsey": 2, "False": 0,
         "Models split": 0}
    )
    assert label == "Mixed verdict"
    assert cls == "neutral"


def test_headline_verdict_coarse_dominant_models_split_is_abstention() -> None:
    """Models-split rows are abstentions: excluded from the decided denominator
    (like Unverifiable), never headlined. The decided remainder speaks."""
    label, cls = _headline_verdict_coarse(
        {"True": 1, "Truthy": 0, "Unverifiable": 0, "Falsey": 0, "False": 0,
         "Models split": 4}
    )
    assert label == "Largely True"          # 1/1 decided
    # nothing decided at all → Unverifiable headline, not nonsense English
    label, cls = _headline_verdict_coarse(
        {"True": 0, "Truthy": 0, "Unverifiable": 1, "Falsey": 0, "False": 0,
         "Models split": 4}
    )
    assert label == "Unverifiable"
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


def test_verdict_panel_lens_semantics_binary_vs_graded() -> None:
    """2026-07-19 lens semantics: Lenient = simple Truthy/Falsey lean,
    Strict = graded family bands. The 6-bucket label (e.g. "Exaggerated")
    never appears as a headline. Here: all-Truthy under Lenient → 'Truthy';
    all-Falsey under Strict → 'Largely False'."""
    bundles = [_make_bundle(VerdictLabel.EXAGGERATED,
                             coarse_lenient="Truthy", coarse_strict="Falsey")] * 3
    sr = _make_site_report(bundles)
    html = _verdict_panel(sr)
    import re
    headlines = re.findall(r'<div class="vp-verdict[^"]*">([^<]+)</div>', html)
    assert any(h.strip() == "Truthy" for h in headlines)          # lenient/simple
    assert any("Largely False" in h for h in headlines)           # strict/graded
    assert not any("Exaggerated" in h for h in headlines)


def test_models_engaged_counts_panel_seats_not_reconciled_cards() -> None:
    """2026-07-19 review find: 'Models Engaged' was stuck at 1 on PCA reports
    because the bridge emits one reconciled ModelVerdict — the count must come
    from the roster seats (+ the Severity Classifier when it fired)."""
    from truthbot.publish.site import _models_engaged
    bundles = [_make_bundle(VerdictLabel.TRUE, coarse_lenient="True", coarse_strict="True")]
    sr = _make_site_report(bundles)
    sr.panel_roster = {"name": "dev", "seats": {"proposer": ["mistral"],
                                                "critic": ["dsv4-flash"],
                                                "arbiter": ["claude-haiku"]}}
    n, hint = _models_engaged(sr)
    assert n == 3 and "mistral" in hint
    # a stage-2 override anywhere in the report adds the Severity Classifier
    bundles[0].consensus.provenance.crm114_final = "FALSE"
    n, hint = _models_engaged(sr)
    assert n == 4 and "Severity Classifier" in hint
    # legacy (no roster): distinct adapter names as before
    sr.panel_roster = {}
    n, _hint = _models_engaged(sr)
    assert n == len({mv.adapter_name for b in bundles for mv in b.model_verdicts})


def test_binary_verdict_simple_lean_with_mixed_band() -> None:
    """The Lenient-lens rule: Truthy/Falsey by dominant family, with the SAME
    Mixed band as the graded lens (dominant share < 0.55 → Mixed) so the two
    lenses always agree on when a report is a toss-up. Abstentions stay out of
    the denominator."""
    from truthbot.publish.site import _binary_verdict
    label, cls, ratio = _binary_verdict({"True": 3, "Falsey": 1, "Unverifiable": 4})
    assert label == "Truthy" and ratio == "3 of 4 decided claims true-leaning"
    label, _cls, _r = _binary_verdict({"True": 1, "Misleading": 3})
    assert label == "Falsey"
    label, _cls, _r = _binary_verdict({"True": 2, "Misleading": 2})
    assert label == "Mixed verdict"                    # exact tie → Mixed
    label, _cls, _r = _binary_verdict({"True": 10, "Falsey": 9})   # 52.6%
    assert label == "Mixed verdict"                    # coin-flip band
    label, _cls, _r = _binary_verdict({"True": 11, "Falsey": 9})   # 55%
    assert label == "Truthy"                           # band edge is exclusive
    label, _cls, _r = _binary_verdict({"Unverifiable": 3, "Models split": 1})
    assert label == "Unverifiable"
    label, _cls, _r = _binary_verdict({})
    assert label == "No claims evaluated"


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


# ── Run manifest panel (roadmap [4]) ──────────────────────────────────────────


def _bundle_with_panel(
    *,
    claim_id_suffix: str,
    panel: list[tuple[str, str, str, bool]],
    web_sources_per_model: list[list[str]] | None = None,
    mrs_per_model: list[list[str]] | None = None,
    consensus_strength: str = "strong",
) -> VerdictBundle:
    """Build a bundle from an explicit per-model panel spec.

    Each panel entry is ``(adapter_name, model_id, synthesis_mode, no_response)``.
    Optional ``web_sources_per_model`` / ``mrs_per_model`` align by index;
    ignored when ``no_response`` is True for that model.
    """
    if web_sources_per_model is None:
        web_sources_per_model = [[] for _ in panel]
    if mrs_per_model is None:
        mrs_per_model = [[] for _ in panel]
    claim = Claim(
        transcript_id="t",
        text=f"claim {claim_id_suffix}",
        speaker="Speaker",
        context="ctx",
        category="economy",
        is_checkable=True,
    )
    mvs: list[ModelVerdict] = []
    for (adapter, model_id, mode, nr), ws, mrs in zip(
        panel, web_sources_per_model, mrs_per_model
    ):
        mvs.append(
            ModelVerdict(
                adapter_name=adapter,
                model_id=model_id,
                claim_id=claim.id,
                label=VerdictLabel.UNVERIFIABLE if nr else VerdictLabel.TRUE,
                confidence=Confidence.HIGH,
                explanation="r",
                no_response=nr,
                web_sources=list(ws) if not nr else [],
                model_reported_sources=list(mrs) if not nr else [],
                synthesis_mode=mode,
                tier="frontier",
            )
        )
    consensus = ConsensusVerdict(
        claim_id=claim.id,
        model_verdicts=mvs,
        consensus_label=VerdictLabel.TRUE,
        consensus_verdict=VerdictLabel.TRUE.value,
        confidence=Confidence.HIGH,
        agreement=True,
        consensus_strength=consensus_strength,
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


_FOUR_ADAPTER_PANEL = [
    ("anthropic", "claude-opus-4-7", "batch", False),
    ("openai", "gpt-5.4", "batch", False),
    ("gemini", "gemini-2.5-pro", "live", False),
    ("xai", "grok-4", "live", False),
]


def test_run_manifest_renders_per_adapter_table_when_no_degradation() -> None:
    """Happy path: 4 adapters, all 100% coverage. Manifest renders with 4
    rows, NO degraded-consensus banner, details collapsed by default. The
    aside is always present (audit trail) — but the banner only appears
    when something actually degraded."""
    bundles = [
        _bundle_with_panel(
            claim_id_suffix=str(i),
            panel=_FOUR_ADAPTER_PANEL,
        )
        for i in range(3)
    ]
    sr = _make_site_report(bundles)
    html = _run_manifest_html(sr)
    # Aside always present.
    assert '<aside class="run-manifest">' in html
    # No degraded banner when all adapters fully covered.
    assert "run-manifest-banner" not in html
    assert "Degraded consensus" not in html
    # Details collapsed by default (no ` open` attr) when no degradation.
    assert "<details class=\"run-manifest-details\">" in html
    # 4 adapter rows in the table.
    assert html.count('<tr class="degraded">') == 0
    # Each adapter renders.
    for adapter in ("anthropic", "openai", "gemini", "xai"):
        # _adapter_pretty title-cases for some (Anthropic / Google / OpenAI / xAI).
        # Just check the model_id shows up since that's adapter-specific.
        pass
    assert "claude-opus-4-7" in html.lower() or "Claude Opus" in html
    assert "gpt" in html.lower()
    assert "gemini" in html.lower()
    assert "grok" in html.lower()


def test_run_manifest_degraded_banner_when_adapter_misses_a_claim() -> None:
    """When any adapter has at least one no_response across the report,
    the manifest renders the degraded-consensus banner with the adapter
    name + "X of Y claims (Z unavailable)" copy. Details opens by
    default so the reader sees the row immediately."""
    panel_with_gemini_miss = [
        ("anthropic", "claude-opus-4-7", "batch", False),
        ("openai", "gpt-5.4", "batch", False),
        ("gemini", "gemini-2.5-pro", "live", True),  # no_response
        ("xai", "grok-4", "live", False),
    ]
    bundles = [
        _bundle_with_panel(claim_id_suffix="0", panel=panel_with_gemini_miss),
        _bundle_with_panel(claim_id_suffix="1", panel=_FOUR_ADAPTER_PANEL),
        _bundle_with_panel(claim_id_suffix="2", panel=_FOUR_ADAPTER_PANEL),
    ]
    sr = _make_site_report(bundles)
    html = _run_manifest_html(sr)
    assert '<div class="run-manifest-banner"' in html
    assert "Degraded consensus" in html
    assert "gemini contributed 2 of 3 claims (1 unavailable)" in html
    # Details opens by default when degraded.
    assert 'details class="run-manifest-details" open' in html
    # The gemini row carries the .degraded class so CSS highlights it.
    assert '<tr class="degraded">' in html
    # Coverage cell for gemini bolded.
    assert "<strong>2/3" in html


def test_run_manifest_consensus_strength_distribution_visible() -> None:
    """Manifest shows the consensus_strength distribution across claims —
    "strong"/"weak"/"single"/"none" tally — so readers can see how
    confident each claim's panel was on average."""
    bundles = [
        _bundle_with_panel(claim_id_suffix="0", panel=_FOUR_ADAPTER_PANEL,
                           consensus_strength="strong"),
        _bundle_with_panel(claim_id_suffix="1", panel=_FOUR_ADAPTER_PANEL,
                           consensus_strength="strong"),
        _bundle_with_panel(claim_id_suffix="2", panel=_FOUR_ADAPTER_PANEL,
                           consensus_strength="weak"),
    ]
    sr = _make_site_report(bundles)
    html = _run_manifest_html(sr)
    assert "Consensus strength" in html
    # Sorted by descending count, so strong (2) renders before weak (1).
    assert "2 strong" in html
    assert "1 weak" in html


def test_run_manifest_tool_url_grounding_caveat_de_emphasized() -> None:
    """The audit-revised framing demands the tool-URL-grounding metric be
    surfaced WITH a caveat explaining it isn't pure fabrication rate, since
    the multi-claim batch path produces apparent strips for harness reasons.
    Pin the caveat copy + the per-adapter grounding column."""
    real_url = "https://www.bls.gov/cpi.htm"
    bundles = [
        _bundle_with_panel(
            claim_id_suffix="0",
            panel=_FOUR_ADAPTER_PANEL,
            # Anthropic: 100% grounding (web == mrs); OpenAI: 0% (mrs > web=0);
            # Gemini: nothing cited (—); xAI: 100%.
            web_sources_per_model=[[real_url], [], [], [real_url]],
            mrs_per_model=[[real_url], [real_url], [], [real_url]],
        )
    ]
    sr = _make_site_report(bundles)
    html = _run_manifest_html(sr)
    # Caveat copy (the "doesn't mean fabrication" framing).
    assert "Tool-URL grounding" in html
    assert "harness-capture asymmetry" in html
    assert "didn’t validate" in html
    # Em-dash placeholder for adapters that cited zero URLs.
    assert ">—<" in html


def test_run_manifest_extra_models_shown_when_adapter_used_multiple_ids() -> None:
    """Anthropic primary→fallback (e.g., opus-4-7 → haiku-4-5) routes some
    claims through a different model_id within the same adapter. Manifest
    surfaces the most-common model_id and adds "+N more" so readers know
    a fallback occurred."""
    primary_panel = [
        ("anthropic", "claude-opus-4-7", "batch", False),
        ("openai", "gpt-5.4", "batch", False),
        ("gemini", "gemini-2.5-pro", "live", False),
        ("xai", "grok-4", "live", False),
    ]
    fallback_panel = [
        ("anthropic", "claude-haiku-4-5-20251001", "live", False),  # fallback
        ("openai", "gpt-5.4", "batch", False),
        ("gemini", "gemini-2.5-pro", "live", False),
        ("xai", "grok-4", "live", False),
    ]
    bundles = [
        _bundle_with_panel(claim_id_suffix="0", panel=primary_panel),
        _bundle_with_panel(claim_id_suffix="1", panel=primary_panel),
        _bundle_with_panel(claim_id_suffix="2", panel=fallback_panel),
    ]
    sr = _make_site_report(bundles)
    html = _run_manifest_html(sr)
    # Primary model_id wins (2 of 3 verdicts).
    assert "claude-opus-4-7" in html.lower() or "Claude Opus" in html
    # +1 more indicator for the haiku fallback.
    assert "+1 more" in html


def test_run_manifest_handles_legacy_bundle_with_no_mrs() -> None:
    """Pre-2026-04-26 bundles don't carry model_reported_sources at all
    (or carry empty). Manifest must render — grounding column shows '—'
    rather than crashing — to keep older reports re-publishable."""
    bundles = [_bundle_with_panel(claim_id_suffix="0", panel=_FOUR_ADAPTER_PANEL)]
    sr = _make_site_report(bundles)
    rows = _adapter_run_stats(sr)
    assert all(r["mrs_total"] == 0 for r in rows)
    html = _run_manifest_html(sr)
    # All four adapters render the em-dash for grounding.
    assert html.count(">—<") >= 4


def test_render_report_inserts_run_manifest_after_methodology() -> None:
    """End-to-end: _render_report places the run-manifest aside AFTER the
    methodology aside so the editorial 'how this works' copy reads first
    and the per-run audit trail reads second."""
    bundles = [_bundle_with_panel(claim_id_suffix="0", panel=_FOUR_ADAPTER_PANEL)]
    sr = _make_site_report(bundles)
    html = _render_report(sr)
    methodology_idx = html.find('<aside class="methodology">')
    manifest_idx = html.find('<aside class="run-manifest">')
    assert methodology_idx >= 0, "methodology aside should render"
    assert manifest_idx >= 0, "run-manifest aside should render"
    assert methodology_idx < manifest_idx, (
        "methodology aside must precede run-manifest aside"
    )


# ── Family-aware dissent flagging (roadmap [6]) ──────────────────────────────
#
# Findings-review C4: dissent is computed by exact-fine-label match against
# the consensus, so [Mostly True, Mostly True, True, True] flags both True
# voters as dissenting despite directional agreement. This degrades the
# dissent panel that the lens toggle exposes. Family-aware flagging fixes
# the canonical case + a few neighbors. See ``_verdict_family`` in
# publish/site.py for the family map + rationale.


from truthbot.publish.site import _verdict_family  # noqa: E402


def _bundle_with_panel_labels(
    labels: list[VerdictLabel],
    *,
    fine_consensus: VerdictLabel | None = None,
) -> VerdictBundle:
    """Build a bundle with per-model fine-axis labels — consensus optional.

    When ``fine_consensus`` is None, falls back to ``labels[0]`` so the
    test author specifies the consensus explicitly only when it diverges
    from the panel majority (e.g., synthetic edge cases).
    """
    consensus_label = fine_consensus if fine_consensus is not None else labels[0]
    claim = Claim(
        transcript_id="t",
        text="dissent test claim",
        speaker="X",
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
        for i, lbl in enumerate(labels)
    ]
    consensus = ConsensusVerdict(
        claim_id=claim.id,
        model_verdicts=mvs,
        consensus_label=consensus_label,
        consensus_verdict=consensus_label.value,
        confidence=Confidence.HIGH,
        agreement=True,
        consensus_strength="strong",
        explanation="x",
        coarse_lenient_label=consensus_label.value,
        coarse_lenient_strength="strong",
        coarse_strict_label=consensus_label.value,
        coarse_strict_strength="strong",
    )
    return VerdictBundle(
        claim=claim,
        speaker="X",
        date_str="2026-03-04",
        model_verdicts=mvs,
        consensus=consensus,
    )


# ── Family-mapping pins ──────────────────────────────────────────────────────


def test_verdict_family_maps_truthy_pair_to_same_family() -> None:
    """The findings-review C4 canonical case: True and Mostly True share
    a family so a True voter against a Mostly True consensus does not
    flag as dissent."""
    assert _verdict_family("True") == _verdict_family("Mostly True")


def test_verdict_family_maps_falsey_pair_to_same_family() -> None:
    """Misleading and False share a falsey family — a Misleading voter
    against a False consensus is directional agreement, not dissent."""
    assert _verdict_family("Misleading") == _verdict_family("False")


def test_verdict_family_keeps_exaggerated_in_its_own_family() -> None:
    """Exaggerated is editorially the most ambiguous label (Lenient
    projects it to Truthy, Strict projects to Falsey). Putting it in
    its own family means a Mostly True consensus + Exaggerated voter
    still flags as dissent — that's a genuine framing disagreement."""
    assert _verdict_family("Exaggerated") != _verdict_family("Mostly True")
    assert _verdict_family("Exaggerated") != _verdict_family("Misleading")


def test_verdict_family_unverifiable_separate_from_truthy_falsey() -> None:
    """Unverifiable stays its own family — defensive votes against a
    confident consensus must still surface as dissent."""
    assert _verdict_family("Unverifiable") not in {
        _verdict_family("True"),
        _verdict_family("Mostly True"),
        _verdict_family("Misleading"),
        _verdict_family("False"),
        _verdict_family("Exaggerated"),
    }


# ── Render-level dissent pin: the canonical findings-review C4 case ──────────


def test_panel_with_true_and_mostly_true_voters_shows_zero_dissents() -> None:
    """The smoking-gun case from findings-review C4:
    [Mostly True, Mostly True, True, True] → consensus Mostly True. The
    two True voters are directionally aligned with the consensus, so the
    rendered card MUST show "4 of 4 agree" with NO dissent flags. Pre-
    fix: 2 dissents flagged (the True voters). Pin so this regression
    can't return."""
    bundle = _bundle_with_panel_labels(
        [VerdictLabel.MOSTLY_TRUE, VerdictLabel.MOSTLY_TRUE,
         VerdictLabel.TRUE, VerdictLabel.TRUE],
        fine_consensus=VerdictLabel.MOSTLY_TRUE,
    )
    html = _claim_card(bundle, idx=1, total=1, rel="../", standalone=True)
    assert "4 of 4</span> agree" in html, (
        "All four voters in the truthy family — must read as full agreement"
    )
    # No `class="model dissent"` (or with-leading-space variant) should appear.
    assert 'class="model dissent"' not in html
    # And no "N dissent..." copy in the agreement note.
    assert " dissent" not in html.split("Model consensus")[1].split("</span>")[0]


def test_panel_mixed_truthy_and_falsey_still_flags_dissent() -> None:
    """Anti-regression on the family-aware fix: dissent must still flag
    when a voter is in a different family than the consensus."""
    bundle = _bundle_with_panel_labels(
        [VerdictLabel.MOSTLY_TRUE, VerdictLabel.MOSTLY_TRUE,
         VerdictLabel.MOSTLY_TRUE, VerdictLabel.FALSE],
        fine_consensus=VerdictLabel.MOSTLY_TRUE,
    )
    html = _claim_card(bundle, idx=1, total=1, rel="../", standalone=True)
    # The False voter is in a different family → dissent flagged.
    assert 'class="model dissent"' in html
    assert "3 of 4</span> agree" in html


def test_panel_with_exaggerated_voter_against_mostly_true_consensus_flags_dissent() -> None:
    """Exaggerated is intentionally NOT collapsed with truthy (despite
    Lenient projection grouping them) — it represents a genuine framing
    disagreement that's worth surfacing on the dissent panel."""
    bundle = _bundle_with_panel_labels(
        [VerdictLabel.MOSTLY_TRUE, VerdictLabel.MOSTLY_TRUE,
         VerdictLabel.MOSTLY_TRUE, VerdictLabel.EXAGGERATED],
        fine_consensus=VerdictLabel.MOSTLY_TRUE,
    )
    html = _claim_card(bundle, idx=1, total=1, rel="../", standalone=True)
    assert 'class="model dissent"' in html
    assert "3 of 4</span> agree" in html


def test_panel_with_misleading_voter_against_false_consensus_shows_zero_dissents() -> None:
    """Symmetric to the truthy case: Misleading and False share the
    falsey family, so a Misleading voter against a False consensus is
    directional agreement."""
    bundle = _bundle_with_panel_labels(
        [VerdictLabel.FALSE, VerdictLabel.FALSE,
         VerdictLabel.MISLEADING, VerdictLabel.MISLEADING],
        fine_consensus=VerdictLabel.FALSE,
    )
    html = _claim_card(bundle, idx=1, total=1, rel="../", standalone=True)
    assert "4 of 4</span> agree" in html
    assert 'class="model dissent"' not in html


def test_panel_with_unverifiable_voter_against_truthy_consensus_flags_dissent() -> None:
    """Unverifiable stays its own family. A defensive Unverifiable voter
    against a confident truthy consensus must surface as dissent so
    readers see the disagreement."""
    bundle = _bundle_with_panel_labels(
        [VerdictLabel.TRUE, VerdictLabel.TRUE,
         VerdictLabel.MOSTLY_TRUE, VerdictLabel.UNVERIFIABLE],
        fine_consensus=VerdictLabel.MOSTLY_TRUE,
    )
    html = _claim_card(bundle, idx=1, total=1, rel="../", standalone=True)
    assert 'class="model dissent"' in html
    assert "3 of 4</span> agree" in html


