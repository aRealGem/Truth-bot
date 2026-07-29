"""S5 political-communications tier (Claim Eval v3 PR-A / D7).

jackie's ruling: partisan government press releases are admissible only to show
a claim was MADE, never to prove it TRUE, so they must rank at the bottom.
Before this, tiering was domain-only — every ``.gov`` host classified
Government, entering at the rubric's maximum trust weight and able to trigger
the automatic-FALSE override on a contradiction.

The load-bearing test in here is :func:`test_bls_news_release_stays_government`.
BLS publishes its headline series at ``bls.gov/news.release/*``, so a naive
"``/news`` means demote" rule would gut exactly the sources truth-bot most
needs. That case is why the statistical-agency carve-out exists.
"""
from __future__ import annotations

import pytest

from truthbot.models import Evidence, SourceTier, VerdictLabel
from truthbot.publish.site import _tier_badge, _tier_bucket
from truthbot.scoring.rubric import TIER_WEIGHTS
from truthbot.verdict.consolidator import _T13, _TIER_RANK
from truthbot.verify.source_tiers import classify_tier


# ── the carve-out (the reason a blanket /news rule is wrong) ──────────────────

def test_bls_news_release_stays_government():
    """REGRESSION: real data on a press-looking path must stay S1.

    ``bls.gov/news.release/empsit.nr0.htm`` is the monthly jobs report — the
    single most-cited economic series in US political claims.
    """
    assert classify_tier(
        "https://www.bls.gov/news.release/empsit.nr0.htm"
    ) == SourceTier.GOVERNMENT


@pytest.mark.parametrize("url", [
    "https://www.bls.gov/data/",
    "https://www.bea.gov/data/gdp",
    "https://www.census.gov/data/tables/2024.html",
    "https://www.cbo.gov/publication/60870",
    "https://fred.stlouisfed.org/series/UNRATE",
])
def test_statistical_agency_data_pages_are_s1(url):
    assert classify_tier(url) == SourceTier.GOVERNMENT


def test_statistical_agency_press_shop_is_s3():
    """A nonpartisan agency's own press page is S3 — demoted, not condemned."""
    assert classify_tier("https://www.bls.gov/bls/newsrels.htm") == SourceTier.ESTABLISHED


@pytest.mark.parametrize("url", [
    "https://www.congress.gov/congressional-record/2026/01/01/senate-section",
    "https://www.supremecourt.gov/opinions/25pdf/24-1234_abcd.pdf",
    "https://clerk.house.gov/Votes/202612",
    "https://www.cdc.gov/mmwr/volumes/75/wr/mm7501a1.htm",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/",
    "https://stacks.cdc.gov/view/cdc/123456",
])
def test_nonpartisan_primary_sources_survive_the_quarantine(url):
    """REGRESSION, found by measuring the rule against 4,844 stored URLs.

    D7's carve-out named statistical agencies, but the quarantine as first
    written also demoted Supreme Court opinions, Congressional Record text,
    House roll-call votes, CDC/MMWR and PubMed Central papers to 'political
    communications' — the same gutting the BLS case warned about, one scope
    level up. None of these is a press shop.
    """
    assert classify_tier(url) == SourceTier.GOVERNMENT


@pytest.mark.parametrize("url", [
    "https://bidenwhitehouse.archives.gov/briefing-room/statements-releases/x",
    "https://trumpwhitehouse.archives.gov/remarks/y",
])
def test_archived_white_house_stays_political(url):
    """archives.gov is a nonpartisan carve-out host, but the archived White
    House sites under it are still White House communications. The political
    rules are checked FIRST precisely so the carve-out cannot promote them."""
    assert classify_tier(url) == SourceTier.POLITICAL


# ── S5: political communications ──────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "https://www.whitehouse.gov/briefing-room/statements-releases/2026/01/01/x",
    "https://www.whitehouse.gov/fact-sheets/2026/economy",
    # whitehouse.gov is S5 on EVERY path — even one that looks like data.
    "https://www.whitehouse.gov/data/budget",
])
def test_whitehouse_is_political_on_all_paths(url):
    assert classify_tier(url) == SourceTier.POLITICAL


@pytest.mark.parametrize("url", [
    "https://www.energy.gov/articles/secretary-announces-x",
    "https://www.hhs.gov/press-releases/2026/01/01/statement.html",
    "https://www.treasury.gov/newsroom/press-releases/jy1234",
    "https://www.justice.gov/news/press-release-x",
])
def test_agency_press_paths_are_political(url):
    assert classify_tier(url) == SourceTier.POLITICAL


@pytest.mark.parametrize("url", [
    "https://democrats.org/news/statement",
    "https://www.gop.com/platform",
    "https://donaldjtrump.com/news/x",
])
def test_party_and_campaign_domains_are_political(url):
    assert classify_tier(url) == SourceTier.POLITICAL


def test_unmapped_gov_path_is_quarantined():
    """D7 quarantine: an unmapped .gov path fails CLOSED, so a newly-invented
    press path cannot leak into the top tier merely by not being listed."""
    assert classify_tier("https://www.example.gov/some-new-messaging-path") == SourceTier.POLITICAL


@pytest.mark.parametrize("url", [
    "https://www.congress.gov/bill/119th-congress/house-bill/1",
    "https://www.federalregister.gov/documents/2026/01/01/rule",
    "https://www.gao.gov/reports/gao-26-1",
])
def test_substantive_gov_paths_survive_quarantine(url):
    assert classify_tier(url) == SourceTier.GOVERNMENT


@pytest.mark.parametrize("url", [
    "https://www.nato.int/cps/en/natohq/official_texts.htm",
    "https://www.un.int/some/unmapped/path",
    "https://www.army.mil/some/unmapped/path",
])
def test_quarantine_is_scoped_to_dot_gov(url):
    """REGRESSION: the quarantine must not reach .int or .mil.

    An earlier revision applied it to the whole government class and demoted
    nato.int/cps/en/natohq/* to S5. ``.int`` is treaty-established
    intergovernmental orgs — a primary-source class with no US partisan press
    shop to guard against. D7's quarantine is about .gov.
    """
    assert classify_tier(url) == SourceTier.GOVERNMENT


# ── the tier must not be able to decide a verdict ─────────────────────────────

def test_political_cannot_credit_the_decided_verdict_quota():
    """S5 must be absent from _T13 — it may show a claim was made, never that
    it is true, so it cannot be one of the MIN_BEARING_T13 qualifying items."""
    assert SourceTier.POLITICAL not in _T13


def test_political_ranks_below_other():
    assert _TIER_RANK[SourceTier.POLITICAL] > _TIER_RANK[SourceTier.OTHER]
    assert TIER_WEIGHTS[SourceTier.POLITICAL] < TIER_WEIGHTS[SourceTier.OTHER]


def _ev(tier: SourceTier, supports: bool) -> Evidence:
    return Evidence(
        claim_id="c1",
        source_name="src",
        source_url="https://example.test/x",
        source_tier=tier,
        snippet="...",
        supports_claim=supports,
        relevance_score=1.0,
    )


def test_political_contradiction_does_not_force_false():
    """The automatic-FALSE override fires on a GOVERNMENT/WIRE contradiction.
    A press release must not be able to force that verdict on its own — which
    it could before, since every .gov host classified GOVERNMENT.

    Ratio 0.3 is chosen to separate the two branches: it is below the
    override's 0.5 cutoff but above the 0.25 EXAGGERATED floor, so the override
    is the *only* thing that can produce FALSE here.
    """
    from truthbot.scoring.rubric import ScoringRubric

    rubric = ScoringRubric()
    gov = rubric._label_from_ratio(0.3, 1.0, [_ev(SourceTier.GOVERNMENT, False)])
    pol = rubric._label_from_ratio(0.3, 1.0, [_ev(SourceTier.POLITICAL, False)])

    assert gov == VerdictLabel.FALSE          # override trips on S1
    assert pol == VerdictLabel.EXAGGERATED    # identical evidence at S5 does not
    assert pol != gov


# ── I5: tier is guarded provenance, so the value must be truthy ───────────────

def test_political_tier_value_is_truthy():
    """hydramind.invariants.check_i5_provenance uses a FALSY test on 'tier'.
    A tier whose value were '' would silently fail I5 for every item."""
    assert bool(SourceTier.POLITICAL.value)


# ── one implementation: pipeline and renderer must agree ──────────────────────

def test_site_badges_the_political_tier():
    assert "T7·Pol" in _tier_badge("https://www.whitehouse.gov/briefing-room/x")
    assert _tier_bucket("https://www.whitehouse.gov/briefing-room/x") == "political"


@pytest.mark.parametrize("url", [
    "https://fred.stlouisfed.org/series/UNRATE",
    "https://www.federalreserve.gov/data/x",
])
def test_renderer_no_longer_drifts_from_the_pipeline(url):
    """REGRESSION: site.py kept its own domain list omitting federalreserve.gov
    and stlouisfed.org, so a FRASER/FRED source was Government in its I5
    provenance record and badged bottom-tier T6 on the published page."""
    assert classify_tier(url) == SourceTier.GOVERNMENT
    assert _tier_bucket(url) == "gov"
    assert "T1·Gov" in _tier_badge(url)
