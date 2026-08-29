"""Regression tests for the Phase 0 display-integrity fixes (P67.4 / PR-1).

Each test pins one bug from the 2026-07-21 external audit, using the
reproduced (not the audit's headline) numbers where they differed:

* F1 — index "100% Model Consensus" over reports recorded at 0.466/0.784:
  the site-wide figure is now the claim-weighted mean of per-report
  ``model_agreement_rate``, never the per-claim pseudo-model agreement.
* F2 — verdict bars omitted the Models-split bucket (sums 169/178, 105/111)
  and the anecdote footnote could exceed the Unverifiable segment it sat
  under: one bucketing now feeds bars, rails, footnote — and sums to
  claim_count.
* F3 — header chips used an all-claims denominator with True-only /
  False-only families: chips now share the headline's families + decided
  denominator, and the mascot mood derives from the headline band.
* F4 — vestigial model-insights page: retired behind a redirect to About.
* T0.8 — the consistency checker catches hand-typed figures at build time.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from truthbot.models import VerdictLabel, VerdictProvenance
from truthbot.publish.site import (
    AGGREGATE_BAR_ORDER,
    SitePublisher,
    _render_about,
    _render_index,
    _render_model_insights_redirect,
    _report_card,
    _verdict_bar_html,
    _verdict_panel,
)
from tests.test_site_render_aggregates import _make_bundle, _make_site_report


# ── F1: site-wide consensus is claim-weighted from reports.json ──────────────


def _stats_for(reports, claims=None):
    return SitePublisher(site_root="/nonexistent")._compute_stats(reports, claims)


def test_avg_consensus_is_claim_weighted_mean_of_report_rates() -> None:
    reports = [
        {"id": "a", "claim_count": 178, "model_agreement_rate": 0.466},
        {"id": "b", "claim_count": 111, "model_agreement_rate": 0.784},
    ]
    stats = _stats_for(reports)
    expected = (0.466 * 178 + 0.784 * 111) / 289
    assert abs(stats["avg_consensus"] - expected) < 1e-9
    assert abs(stats["model_agreement_rate"] - expected) < 1e-9


def test_avg_consensus_ignores_pca_pseudo_model_agreement() -> None:
    """The reconciled hydramind-pca summary matches consensus by construction;
    it must not drag the site-wide figure to 100% (the F1 failure mode)."""
    reports = [{"id": "a", "claim_count": 2, "model_agreement_rate": 0.5}]
    claims = [
        {"report_id": "a", "consensus_verdict": "True",
         "model_verdicts_summary": [{"adapter": "hydramind-pca", "label": "True"}]},
        {"report_id": "a", "consensus_verdict": "False",
         "model_verdicts_summary": [{"adapter": "hydramind-pca", "label": "False"}]},
    ]
    stats = _stats_for(reports, claims)
    assert stats["avg_consensus"] == 0.5  # not 1.0


def test_index_renders_weighted_consensus_not_100() -> None:
    reports = [
        {"id": "a", "claim_count": 178, "model_agreement_rate": 0.466,
         "speaker": "A", "url": "reports/a.html", "verdict_distribution": {}},
        {"id": "b", "claim_count": 111, "model_agreement_rate": 0.784,
         "speaker": "B", "url": "reports/b.html", "verdict_distribution": {}},
    ]
    stats = _stats_for(reports)
    html = _render_index(reports, stats)
    assert '>59<span class="unit">%</span>' in html
    assert '>100<span class="unit">%</span>' not in html


# ── T0.7: canonical claim count comes from the claims index ──────────────────


def test_total_claims_prefers_claims_index_length() -> None:
    reports = [{"id": "a", "claim_count": 5, "model_agreement_rate": 1.0}]
    claims = [{"report_id": "a", "consensus_verdict": "True",
               "model_verdicts_summary": []} for _ in range(4)]
    stats = _stats_for(reports, claims)
    assert stats["total_claims"] == 4  # claims.json is canonical, drift logged


# ── F2: one bucketing — split renders, everything sums to claim_count ────────


def test_aggregate_bar_order_contains_models_split() -> None:
    assert "Models split" in AGGREGATE_BAR_ORDER


def test_verdict_bar_renders_models_split_segment_and_sums() -> None:
    dist = {"True": 45, "Unverifiable": 18, "Models split": 9,
            "Misleading": 68, "False": 38}
    html = _verdict_bar_html(dist, order=AGGREGATE_BAR_ORDER, family_rail=True)
    assert 'title="Models split: 9"' in html
    assert 'v-split' in html
    import re
    seg_counts = [int(m.group(1)) for m in re.finditer(r'title="[^":]+: (\d+)"', html)]
    assert sum(seg_counts) == 178
    # Family rail: split joins the abstain cell, decided stays 151.
    assert "27 claims not decided" in html
    assert "106 of 151 decided claims false-leaning" in html


def test_report_card_bar_includes_models_split() -> None:
    r = {
        "id": "a", "speaker": "Test", "url": "reports/a.html", "claim_count": 4,
        "verdict_distribution": {},
        "verdict_distribution_strict": {"True": 2, "False": 1, "Models split": 1},
        "verdict_distribution_lenient": {"True": 2, "False": 1, "Models split": 1},
    }
    html = _report_card(r)
    assert "v-split" in html
    assert "Models split" in html


# Readability pass (site.py Section 4): the homepage card's source-tier
# chip was removed from ``_report_card`` entirely (that detail now lives
# only on the report page) — test_report_card_source_chips_include_other_bucket
# pinned a chip that no longer exists by design and has been removed.


def test_anecdote_footnote_reconciles_with_split_bucket() -> None:
    """A split anecdote sits in the Models-split bar bucket; the footnote must
    not count it against the Unverifiable segment (the 19-of-18 audit bug)."""
    def _anecdote(bundle):
        bundle.consensus.provenance = VerdictProvenance(
            layer_a_label="check-worthy", layer_a_source="A2",
            layer_a_claim_type="personal-anecdote")
        return bundle

    uv_anec = _anecdote(_make_bundle(
        VerdictLabel.UNVERIFIABLE, coarse_lenient="Unverifiable",
        coarse_strict="Unverifiable"))
    split_anec = _anecdote(_make_bundle(
        VerdictLabel.UNVERIFIABLE, coarse_lenient="Models split",
        coarse_strict="Models split", consensus_verdict="Models split"))
    plain_uv = _make_bundle(VerdictLabel.UNVERIFIABLE,
                            coarse_lenient="Unverifiable",
                            coarse_strict="Unverifiable")
    sr = _make_site_report([uv_anec, split_anec, plain_uv])
    html = _verdict_panel(sr)
    assert ("1 of the 2 Unverifiable claims is a guest anecdote, "
            "plus 1 more among the Models-split claims") in html


# ── F3: mascot mood follows the headline band ────────────────────────────────


def test_mascot_is_sad_on_largely_false_report() -> None:
    bundles = (
        [_make_bundle(VerdictLabel.FALSE, coarse_lenient="False", coarse_strict="False")] * 8
        + [_make_bundle(VerdictLabel.TRUE, coarse_lenient="True", coarse_strict="True")] * 2
    )
    html = _verdict_panel(_make_site_report(bundles))
    assert 'data-mood="sad"' in html
    assert "Mixed signals" not in html


def test_mascot_is_iffy_only_on_mixed_band() -> None:
    bundles = (
        [_make_bundle(VerdictLabel.FALSE, coarse_lenient="False", coarse_strict="False")] * 5
        + [_make_bundle(VerdictLabel.TRUE, coarse_lenient="True", coarse_strict="True")] * 5
    )
    html = _verdict_panel(_make_site_report(bundles))
    assert 'data-mood="iffy"' in html


# ── F4: model-insights retired behind a redirect ─────────────────────────────


def test_model_insights_redirect_stub() -> None:
    html = _render_model_insights_redirect()
    assert 'http-equiv="refresh"' in html
    assert 'url=./about.html' in html
    assert "Hydramind" not in html


def test_publisher_writes_redirect_at_model_insights(tmp_path: Path) -> None:
    pub = SitePublisher(site_root=tmp_path)
    sr = _make_site_report([_make_bundle(
        VerdictLabel.TRUE, coarse_lenient="True", coarse_strict="True")])
    pub.publish(sr)
    text = (tmp_path / "model-insights.html").read_text()
    assert 'http-equiv="refresh"' in text
    index_html = (tmp_path / "index.html").read_text()
    assert "insights-strip" not in index_html


# ── T0.5 / T0.6: tagline + About prose guards ────────────────────────────────


def test_index_has_no_primary_sources_tagline() -> None:
    reports = [{"id": "a", "claim_count": 1, "model_agreement_rate": 1.0,
                "speaker": "A", "url": "reports/a.html",
                "verdict_distribution": {"True": 1}}]
    html = _render_index(reports, _stats_for(reports))
    assert "primary sources" not in html


def test_about_prose_fixed() -> None:
    html = _render_about()
    assert "comparable accuracy" not in html
    assert "never silently broken" not in html
    assert "withheld as metadata" in html
    assert "may still identify the speaker" in html
    assert "Models split" in html  # display-convention disclosure


# ── T0.8: consistency checker catches hand-typed figures ─────────────────────


def _publish_minimal_site(tmp_path: Path) -> Path:
    pub = SitePublisher(site_root=tmp_path)
    bundles = (
        [_make_bundle(VerdictLabel.TRUE, coarse_lenient="True", coarse_strict="True")] * 2
        + [_make_bundle(VerdictLabel.FALSE, coarse_lenient="False", coarse_strict="False")] * 7
        + [_make_bundle(VerdictLabel.UNVERIFIABLE, coarse_lenient="Unverifiable",
                        coarse_strict="Unverifiable")]
    )
    pub.publish(_make_site_report(bundles))
    return tmp_path


def test_consistency_checker_passes_on_freshly_published_site(tmp_path: Path) -> None:
    from truthbot.publish.consistency import check_site
    site = _publish_minimal_site(tmp_path)
    assert check_site(site) == []


def test_consistency_checker_flags_hand_typed_consensus(tmp_path: Path) -> None:
    from truthbot.publish.consistency import check_site
    site = _publish_minimal_site(tmp_path)
    index = site / "index.html"
    html = index.read_text()
    import re
    tampered = re.sub(
        r'<div class="num">\d+<span class="unit">%</span></div>'
        r'<div class="lbl">Model Consensus',
        '<div class="num">42<span class="unit">%</span></div>'
        '<div class="lbl">Model Consensus',
        html)
    assert tampered != html
    index.write_text(tampered)
    assert any("Model Consensus" in v for v in check_site(site))


def test_consistency_checker_flags_claim_count_drift(tmp_path: Path) -> None:
    from truthbot.publish.consistency import check_site
    site = _publish_minimal_site(tmp_path)
    data = site / "data" / "reports.json"
    reports = json.loads(data.read_text())
    reports[0]["claim_count"] += 3
    data.write_text(json.dumps(reports))
    violations = check_site(site)
    assert violations, "inflated claim_count must be caught"


def test_committed_site_passes_consistency_check() -> None:
    """The T0.8 CI gate proper: every quantitative figure on the committed
    site must derive from data/*.json — hand-typed numbers fail the build."""
    import pytest
    from truthbot.publish.consistency import check_site
    site = Path(__file__).resolve().parents[1] / "site-pca"
    if not (site / "data" / "reports.json").exists():
        pytest.skip("no committed site in this checkout")
    # committed tree predates the remediation-v2 regen (cards rendered
    # without the political Sources bucket); Phase-2 regen flips this to
    # strict_buckets=True — the fresh-render strict pass is pinned by
    # tests/test_site_consistency.py::test_fresh_render_passes_strict_lints.
    assert check_site(site, strict_buckets=False) == []
