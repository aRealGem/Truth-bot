"""Phase 4 tests (P67.10 / PR-7, remediation T4.1-T4.2).

The acceptance fixture is the audit's F10/F11 numbers as REPRODUCED on
2026-07-21 (the external audit's own override counts were exactly 2x the
artifacts): they are asserted against the COMMITTED site data, so the
insights math and the published corpus can't drift apart silently.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from truthbot.publish.seat_insights import compute_seat_insights

REPO = Path(__file__).resolve().parents[2]
SITE = REPO / "site-pca"


def _site_claims():
    p = SITE / "data" / "claims.json"
    if not p.exists():
        pytest.skip("no committed site in this checkout")
    return json.loads(p.read_text()), json.loads(
        (SITE / "data" / "reports.json").read_text())


def test_acceptance_fixture_f10_f11_reproduced_from_committed_site() -> None:
    claims, reports = _site_claims()
    by_speaker = {r["speaker"]: r["id"] for r in reports}
    ins = compute_seat_insights(claims)

    trump = ins[by_speaker["Donald Trump"]]
    assert trump.n_claims == 178
    assert trump.seats["critic"].rate("False") == pytest.approx(0.500, abs=1e-3)
    assert trump.escalation_rate == pytest.approx(95 / 178, abs=1e-4)
    assert trump.arbiter_sided["proposer"] == 61
    assert trump.arbiter_sided["critic"] == 20
    assert trump.overrides == {"Misleading→False": 7, "False→Misleading": 4,
                               "Disagreement→Misleading": 5}

    biden = ins[by_speaker["Joe Biden"]]
    assert biden.n_claims == 111
    assert biden.seats["critic"].rate("False") == pytest.approx(0.135, abs=1e-3)
    assert biden.escalation_rate == pytest.approx(24 / 111, abs=1e-4)
    assert biden.arbiter_sided["proposer"] == 14
    assert biden.overrides == {"Misleading→False": 1}


def test_insights_v2_page_renders_seats_no_hydramind() -> None:
    from truthbot.publish.site import _render_model_insights_v2
    claims, reports = _site_claims()
    html = _render_model_insights_v2(reports, claims)
    assert "Model panel insights" in html
    assert "panel_by_role" in html
    assert "Hydramind" not in html
    assert "Proposer" in html and "Critic" in html and "Arbiter" in html
    assert "Severity-Classifier stage-2 overrides" in html
    # F11 escalation figures render
    assert "53.4%" in html and "21.6%" in html


def test_publisher_writes_v2_page_when_seat_data_exists(tmp_path: Path) -> None:
    from tests.test_site_render_aggregates import _make_bundle, _make_site_report
    from truthbot.models import VerdictLabel, VerdictProvenance
    from truthbot.publish.site import SitePublisher

    b = _make_bundle(VerdictLabel.TRUE, coarse_lenient="True", coarse_strict="True")
    b.consensus.provenance = VerdictProvenance(
        layer_a_label="check-worthy", panel_votes={"True": 2},
        panel_by_role={"proposer": ["True"], "critic": ["True"]})
    pub = SitePublisher(site_root=tmp_path)
    pub.publish(_make_site_report([b]))
    text = (tmp_path / "model-insights.html").read_text()
    assert "Model panel insights" in text
    assert 'http-equiv="refresh"' not in text
    # index footer links it
    assert 'model-insights.html">Panel insights' in (tmp_path / "index.html").read_text()


def test_publisher_falls_back_to_redirect_without_seat_data(tmp_path: Path) -> None:
    from tests.test_site_render_aggregates import _make_bundle, _make_site_report
    from truthbot.models import VerdictLabel
    from truthbot.publish.site import SitePublisher

    pub = SitePublisher(site_root=tmp_path)
    pub.publish(_make_site_report([_make_bundle(
        VerdictLabel.TRUE, coarse_lenient="True", coarse_strict="True")]))
    assert 'http-equiv="refresh"' in (tmp_path / "model-insights.html").read_text()


def test_report_meta_carries_roster_and_triage_count(tmp_path: Path) -> None:
    from tests.test_site_render_aggregates import _make_bundle, _make_site_report
    from truthbot.models import VerdictLabel
    from truthbot.publish.site import SitePublisher

    sr = _make_site_report([_make_bundle(
        VerdictLabel.TRUE, coarse_lenient="True", coarse_strict="True")])
    sr.panel_roster = {"name": "dev", "seats": {"proposer": ["mistral"]}}
    sr.characterization = [{"sid": "x:0001", "text": "Thanks."}] * 3
    pub = SitePublisher(site_root=tmp_path)
    pub.publish(sr)
    meta = json.loads((tmp_path / "data" / "reports.json").read_text())[0]
    assert meta["panel_roster"]["seats"]["proposer"] == ["mistral"]
    assert meta["triage_count"] == 3


def test_falsifiability_note_is_derived_and_framed_as_genre() -> None:
    from truthbot.publish.site import _falsifiability_note_html
    html = _falsifiability_note_html(178, 605)
    assert "178 of the 783 sentences" in html
    assert "22.7%" in html
    assert "genre" in html
    assert _falsifiability_note_html(0, 10) == ""


def test_dot_int_classifies_as_government() -> None:
    from truthbot.models import SourceTier
    from truthbot.verify.sources.brave import classify_tier
    assert classify_tier("https://www.nato.int/cps/en/natohq/text.htm") == SourceTier.GOVERNMENT
    from truthbot.publish.site import _tier_bucket
    assert _tier_bucket("https://www.nato.int/x") == "gov"
