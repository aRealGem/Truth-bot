"""About-page copy reconciliation pins (remediation v2, 1.9).

The 2026-08 audit found about.html asserting things the pipeline does not
do: a six-item pack cap (PACK_CAP_V2 is 10), retrieval "plus fact-check
databases" (fact-check sources are EXCLUDED from packs — invariant S-1),
a hand-typed tier table missing the shipped T7·Pol tier, and a "Panel
split" label the site renders as "Models split". These tests pin the
reconciled copy; ``consistency.check_site`` carries matching banned-phrase
lints for fresh renders.
"""
from __future__ import annotations

import re

import pytest

from truthbot.publish.site import _render_about, _tier_table_rows
from truthbot.verify.source_tiers import TIER_DISPLAY


@pytest.fixture(scope="module")
def about_html() -> str:
    return _render_about()


def test_pack_cap_copy_matches_consolidator_constant(about_html: str) -> None:
    from truthbot.verdict.consolidator import PACK_CAP_V2
    # The shipped cap is ten; the copy renders from the constant, so if the
    # constant moves this test tells us to extend the number-word map.
    assert PACK_CAP_V2 == 10
    assert "capped at ten" in about_html
    assert "capped at six" not in about_html


def test_step2_copy_never_claims_fact_check_databases(about_html: str) -> None:
    """Fact-check sources are excluded from evidence packs (S-1); retrieval
    uses web-search connectors. The step-2 area (and the whole page) must
    not claim otherwise."""
    step2 = about_html[about_html.index("2 · Evidence retrieval")
                       : about_html.index("3 · The verdict panel")]
    assert "fact-check databases" not in step2
    assert "fact-check databases" not in about_html
    assert "web-search connectors" in step2
    assert "excluded from the candidate pool" in step2


def test_tier_table_derives_from_shipped_ladder(about_html: str) -> None:
    """All seven registry tiers render — T7·Pol included — with codes taken
    from TIER_DISPLAY, plus the exclusion footnote."""
    rows = _tier_table_rows()
    assert len(rows) == len(TIER_DISPLAY) == 7
    for _tier, (code, _css) in TIER_DISPLAY.items():
        assert code in about_html, f"tier code {code} missing from About"
    assert "T7·Pol" in about_html
    # Approved footnote: fact-check exclusion + political-tier limit.
    assert "excluded from evidence packs" in about_html
    assert "can never decide a verdict" in about_html


def test_domain_sentence_mentions_path_class(about_html: str) -> None:
    assert "path class" in about_html
    assert "press-release path ranks T7·Pol" in about_html


def test_split_label_unified_on_models_split(about_html: str) -> None:
    assert "Panel split" not in about_html
    assert "Models split" in about_html


def test_anecdote_footnote_behavior_documented(about_html: str) -> None:
    """The anecdote count ships as a footnote beneath each report's
    aggregate bar — documented as shipped behavior (no new pill)."""
    assert re.search(r"anecdote count ships as a footnote beneath each "
                     r"report&#x27;s\s*aggregate verdict bar",
                     about_html) or (
        "anecdote count ships as a footnote" in about_html)


def test_no_lens_toggle_described(about_html: str) -> None:
    """1.8 follow-through: the About page describes the single strict
    presentation + %-True headline; no toggle is offered to the reader."""
    assert "Lens chip" not in about_html
    assert "One presentation" in about_html
    assert "%-True" in about_html
