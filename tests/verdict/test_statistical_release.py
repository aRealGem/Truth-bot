"""D16(α) statistical-release — the three conditions, the flag, the quota effect.

Each of the three conditions is exercised ON ITS OWN, so a failure names the
condition that broke:

  1. FUNCTION — the fail-closed statistical-agency allowlist,
  2. DATA PERIOD — a parseable month/quarter/fiscal-year/year-noun/anchor
     reference at or before the utterance, with publication dates masked out,
  3. CAP — still inside the S-2 fair-game window, which D16 does not move.

The fixtures are REAL rows from the five rebuilt runs, with their actual urls,
snippets and dates, so the measurement in ``scripts/measure_d16.py`` and the
assertions here cannot disagree about what the corpus contains.

The flag is the load-bearing assertion in this file: with
``TRUTHBOT_D16_STATISTICAL_RELEASE`` unset — which is production — the
consolidator must behave bit-for-bit as it does today. Design/ratification
note: ``docs/decisions/D16-statistical-release.md``.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from truthbot.models import Evidence, SourceTier
from truthbot.verdict import era_lint, speech_context
from truthbot.verdict import statistical_release as sr
from truthbot.verdict import utterance_record as ur
from truthbot.verdict.consolidator import GATE_INSUFFICIENT, POST_SPEECH_NOTE, consolidate

BUSH_UTT = date(2006, 1, 31)            # gwbush_2006 SOTU
CLINTON_UTT = date(1998, 1, 27)         # clinton_1998 SOTU
BIDEN_UTT = date(2022, 3, 1)            # biden_2022 SOTU

# ── the real corpus rows ────────────────────────────────────────────────────
BLS_EMPSIT_URL = ("https://www.bls.gov/news.release/archives/"
                  "empsit_02032006.pdf")
BLS_EMPSIT_SNIPPET = ("[2006-02-03] BLS Employment Situation (Jan 2006) — "
                      "monthly nonfarm payroll series and charts covering "
                      "Feb 2003–Jan 2006.")
BLS_EMPSIT_PUB = date(2006, 2, 3)       # 3 days after the speech

CBO_OUTLOOK_URL = ("https://www.cbo.gov/sites/default/files/"
                   "105th-congress-1997-1998/reports/eb01-98.pdf")
CBO_OUTLOOK_SNIPPET = ("[1998-01-28] CBO's Jan 1998 Economic & Budget Outlook "
                       "projects a small single-digit FY1998 deficit heading "
                       "to balance/surplus — the source projection behind the "
                       "claim.")
CBO_OUTLOOK_PUB = date(1998, 1, 28)

# The two motivating examples the reviewer rejected the blanket rule over.
ONDCP_JUSTICE_URL = "https://www.justice.gov/archive/olp/pdf/ndcs06.pdf"
ONDCP_ERIC_URL = "https://files.eric.ed.gov/fulltext/ED503096.pdf"
ONDCP_ERIC_SNIPPET = ("[2006-02-01] White House ONDCP National Drug Control "
                      "Strategy (Feb 2006) states current illicit drug use "
                      "among youth dropped 19% since 2001 — likely near/after "
                      "cutoff.")
BUDGET_GPO_URL = ("https://www.gpo.gov/fdsys/pkg/BUDGET-1999-BUD/pdf/"
                  "BUDGET-1999-BUD.pdf")
BUDGET_GPO_SNIPPET = ("[1998-02-02] Official FY1999 President's Budget "
                      "explicitly states funding for dislocated worker "
                      "training has more than doubled since 1993 and gives "
                      "program figures.")

# Post-speech NEWS from an allowlisted host: the SPR release announced a week
# after the speech. No data period — exactly what must stay excluded.
EIA_SPR_URL = "https://www.eia.gov/todayinenergy/detail.php?id=51538"
EIA_SPR_SNIPPET = ("[2022-03-08] EIA summary: U.S. committed to release 30 "
                   "million barrels from the SPR; other IEA members add up to "
                   "60 million.")


# ── CONDITION 2, on its own: the data-period parser ─────────────────────────

@pytest.mark.parametrize("text,rule,start", [
    ("BLS Employment Situation (Jan 2006) payrolls", sr.RULE_MONTH, date(2006, 1, 1)),
    ("the January 2026 employment report", sr.RULE_MONTH, date(2026, 1, 1)),
    ("real GDP rose in Q4 2025", sr.RULE_QUARTER, date(2025, 10, 1)),
    ("exports in the fourth quarter of 1997", sr.RULE_QUARTER, date(1997, 10, 1)),
    ("a small single-digit FY1998 deficit", sr.RULE_FISCAL_YEAR, date(1997, 10, 1)),
    ("fiscal year 2009 outlays", sr.RULE_FISCAL_YEAR, date(2008, 10, 1)),
    ("cites the 2005 MTF survey tabulations", sr.RULE_YEAR_DATA, date(2005, 1, 1)),
    ("estimates for 2013 were revised", sr.RULE_YEAR_DATA, date(2013, 1, 1)),
    ("drug use dropped 19% since 2001", sr.RULE_ANCHOR_YEAR, date(2001, 1, 1)),
    ("charts covering 2003–2006", sr.RULE_YEAR_RANGE, date(2003, 1, 1)),
])
def test_each_period_family_parses(text, rule, start) -> None:
    hits = sr.data_periods(text)
    assert hits, text
    assert (hits[0].rule, hits[0].start) == (rule, start)


def test_a_bare_publication_date_prefix_is_NOT_a_data_period() -> None:
    """The single most important negative in the module. The connectors stamp
    ``[YYYY-MM-DD]`` onto every snippet; if that counted, every post-speech
    item on an allowlisted host would pass condition 2 for free."""
    assert sr.data_periods("[2006-02-01] National Drug Control Strategy") == []
    assert sr.data_periods("published 2006-02-01 by the agency") == []
    assert sr.data_periods("CDC Weekly Review (Mar 4, 2022) reports") == []
    assert sr.data_periods("released 4 March 2022") == []
    assert sr.data_periods("dated 3/4/2022") == []


def test_the_replaced_bare_year_heuristic_does_not_survive() -> None:
    """My earlier heuristic was "any 4-digit year ≤ the utterance year anywhere
    in the snippet". It passed gwbush_2006:0217 on its own PUBLICATION year,
    "2006". A bare year, a year in prose, and a year that is really a quantity
    must all parse to nothing."""
    for text in ("National Drug Control Strategy 2006",
                 "the 2006 Strategy was released",
                 "published in 2006",
                 "$1,400 billion and 3200 units"):
        assert sr.data_periods(text) == [], text


def test_a_period_that_had_not_started_yet_does_not_qualify() -> None:
    """"At or before the utterance" is measured on the period's START."""
    assert sr.pre_utterance_period("the February 2006 outlook", BUSH_UTT) is None
    assert sr.pre_utterance_period("the January 2006 report", BUSH_UTT) is not None
    # FY1999 had not begun on 27 January 1998 (it starts 1 October 1998).
    assert sr.pre_utterance_period("the FY1999 request", CLINTON_UTT) is None
    assert sr.pre_utterance_period("FY1998 amounts", CLINTON_UTT) is not None


def test_the_url_is_read_through_the_same_patterns_never_looser() -> None:
    """Statistical agencies name the reference period in the slug more reliably
    than the retriever's one-line snippet does."""
    assert "january 2026" in sr.url_words(
        "https://www.bea.gov/news/2026/personal-income-and-outlays-january-2026")
    assert sr.pre_utterance_period(
        sr.url_words("https://www.bea.gov/news/2026/personal-income-and-"
                     "outlays-january-2026"), date(2026, 2, 24)) is not None
    # ...but a run-together date stamp stays unparseable: no separator between
    # month and year means no match.
    assert sr.data_periods(sr.url_words(BLS_EMPSIT_URL)) == []


# ── CONDITION 3, on its own: the S-2 cap, untouched ─────────────────────────

def test_the_band_is_exactly_the_fair_game_window() -> None:
    assert sr.FAIR_GAME_DAYS == era_lint.FAIR_GAME_DAYS
    fg = era_lint.fair_game_end(BUSH_UTT)
    assert sr.in_post_speech_band(fg, BUSH_UTT) is True
    assert sr.in_post_speech_band(fg + __import__("datetime").timedelta(1),
                                  BUSH_UTT) is False
    assert sr.in_post_speech_band(BUSH_UTT, BUSH_UTT) is False     # not "post"
    assert sr.in_post_speech_band(None, BUSH_UTT) is False
    assert sr.in_post_speech_band(BUSH_UTT, None) is False


def test_a_pre_utterance_item_matches_nothing_here() -> None:
    """It already credits; D16 has no business touching it."""
    assert sr.statistical_release_rule(
        BLS_EMPSIT_URL, BLS_EMPSIT_SNIPPET, utterance=BUSH_UTT,
        item_date=date(2006, 1, 6)) == ""


def test_an_item_past_the_cap_matches_nothing_here() -> None:
    """D16 releases items already inside the band; it never widens the band."""
    assert sr.statistical_release_rule(
        BLS_EMPSIT_URL, BLS_EMPSIT_SNIPPET, utterance=BUSH_UTT,
        item_date=date(2006, 3, 3)) == ""


# ── the three conditions, combined ──────────────────────────────────────────

def test_a_genuine_bls_series_with_a_pre_utterance_period_qualifies() -> None:
    """The motivating positive: the January 2006 Employment Situation,
    published 3 February 2006, measuring a month that ended before the
    31 January speech."""
    assert sr.statistical_release_rule(
        BLS_EMPSIT_URL, BLS_EMPSIT_SNIPPET, utterance=BUSH_UTT,
        item_date=BLS_EMPSIT_PUB) == sr.RULE_MONTH
    detail = sr.statistical_release_detail(
        BLS_EMPSIT_URL, BLS_EMPSIT_SNIPPET, utterance=BUSH_UTT,
        item_date=BLS_EMPSIT_PUB)
    assert detail["agency"] == "BLS" and detail["period_start"] == "2006-01-01"


def test_a_fred_series_qualifies_on_the_same_three_conditions() -> None:
    assert sr.statistical_release_rule(
        "https://fred.stlouisfed.org/series/PAYEMS",
        "Total nonfarm payrolls, monthly series through December 2005.",
        utterance=BUSH_UTT, item_date=BLS_EMPSIT_PUB) == sr.RULE_MONTH


def test_condition_1_fails_alone_allowlist_miss() -> None:
    """Same snippet, same dates — only the host changes."""
    assert sr.statistical_release_rule(
        "https://www.gpo.gov/whatever.pdf", BLS_EMPSIT_SNIPPET,
        utterance=BUSH_UTT, item_date=BLS_EMPSIT_PUB) == ""


def test_condition_2_fails_alone_fail_closed_on_an_unparseable_period() -> None:
    """An allowlisted host publishing post-speech NEWS: the March 2022 SPR
    release announced a week after the speech names no data period, so it stays
    context-only. Fail closed is the intended failure mode."""
    assert sr.statistical_release_rule(
        EIA_SPR_URL, EIA_SPR_SNIPPET, utterance=BIDEN_UTT,
        item_date=date(2022, 3, 8)) == ""
    # ...and neither does a real statistical product whose reference period
    # carries no year (biden_2022:0266's CDC Weekly Review, "Feb 23–Mar 1").
    assert sr.statistical_release_rule(
        "https://stacks.cdc.gov/view/cdc/115111/cdc_115111_DS1.pdf",
        "[2022-03-04] CDC Weekly Review (Mar 4, 2022) reports new hospital "
        "admissions 7-day average (Feb 23–Mar 1) = 4,243.",
        utterance=BIDEN_UTT, item_date=date(2022, 3, 4)) == ""


# ── the two motivating examples STAY EXCLUDED ───────────────────────────────

def test_gwbush_2006_0217_stays_excluded_by_ALLOWLIST_MISS_not_period_miss() -> None:
    """The ONDCP National Drug Control Strategy, served from justice.gov and
    files.eric.ed.gov.

    Named explicitly because WHY it is excluded is the design: its snippet
    DOES carry a valid pre-utterance data period ("since 2001"), so it PASSES
    condition 2 on its own. It is excluded solely by condition 1 — neither host
    is a statistical agency — which is exactly what lets D16(α) keep the
    principal's own executive documents out with NO document-class detector
    (that remains deferred, D17-candidate).
    """
    from truthbot.verify.statistical_agency import classify_ex

    # Condition 2 PASSES on its own — this is the assertion that proves the
    # exclusion is not an accident of the period parser.
    period = sr.pre_utterance_period(ONDCP_ERIC_SNIPPET, BUSH_UTT)
    assert period is not None and period.rule == sr.RULE_ANCHOR_YEAR

    # Condition 1 FAILS — the whole reason the item is still excluded.
    assert classify_ex(ONDCP_ERIC_URL) == (False, "not-listed")
    assert classify_ex(ONDCP_JUSTICE_URL) == (False, "not-listed")

    for url in (ONDCP_JUSTICE_URL, ONDCP_ERIC_URL):
        assert sr.statistical_release_rule(
            url, ONDCP_ERIC_SNIPPET, utterance=BUSH_UTT,
            item_date=date(2006, 2, 1)) == ""


def test_clinton_1998_0101_stays_excluded_by_ALLOWLIST_MISS() -> None:
    """The FY1999 President's Budget, served from gpo.gov.

    Excluded by condition 1: gpo.gov is a printing and distribution office, not
    a statistical agency. (Its snippet would also fail condition 2 — "FY1999"
    had not begun and "since 1993" is quoting the CLAIM — but the load-bearing
    reason is the allowlist, so the exclusion survives any future loosening of
    the period parser.)"""
    from truthbot.verify.statistical_agency import classify_ex

    assert classify_ex(BUDGET_GPO_URL) == (False, "not-listed")
    assert sr.statistical_release_rule(
        BUDGET_GPO_URL, BUDGET_GPO_SNIPPET, utterance=CLINTON_UTT,
        item_date=date(1998, 2, 2)) == ""


# ── the flag ────────────────────────────────────────────────────────────────

def test_flag_is_off_by_default_and_reads_the_env_at_call_time(monkeypatch) -> None:
    monkeypatch.delenv(sr.FLAG_ENV, raising=False)
    assert sr.flag_enabled() is False
    for off in ("", "0", "false", "no", "off", "maybe"):
        monkeypatch.setenv(sr.FLAG_ENV, off)
        assert sr.flag_enabled() is False
    for on in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv(sr.FLAG_ENV, on)
        assert sr.flag_enabled() is True


def test_the_flag_name_is_the_d16_analogue_of_d15s() -> None:
    assert sr.FLAG_ENV == "TRUTHBOT_D16_STATISTICAL_RELEASE"
    assert ur.FLAG_ENV == "TRUTHBOT_D15_UTTERANCE_RECORD"


# ── consolidator effect ─────────────────────────────────────────────────────

def _ev(url: str, tier: SourceTier, snippet: str, when: date,
        supports: bool | None = True) -> Evidence:
    return Evidence(claim_id="c", source_name="R1", source_url=url,
                    source_tier=tier, snippet=snippet, supports_claim=supports,
                    published_at=datetime(when.year, when.month, when.day,
                                          tzinfo=timezone.utc))


@pytest.fixture()
def bush_stat_pack():
    """One pre-utterance outside source plus TWO post-speech BLS releases: the
    shape that decides whether the era rule silences the jobs report."""
    return [
        _ev("https://www.npr.org/2006/01/10/payrolls",
            SourceTier.ESTABLISHED, "NPR on December payrolls.",
            date(2006, 1, 10)),
        _ev(BLS_EMPSIT_URL, SourceTier.GOVERNMENT, BLS_EMPSIT_SNIPPET,
            BLS_EMPSIT_PUB),
        _ev("https://www.bls.gov/opub/ted/2006/feb/wk1/art02.htm",
            SourceTier.GOVERNMENT,
            "[2006-02-07] BLS Economics Daily summary of January 2006 payrolls "
            "(+193,000) and industry job gains.", date(2006, 2, 7)),
    ]


def _consolidate(pack, **kw):
    speech_context.register_speech_date("gwbush_2006", BUSH_UTT)
    return consolidate("gwbush_2006:0134", [("stored", pack)],
                       utterance=BUSH_UTT,
                       window=(date(2004, 1, 1), date(2006, 4, 30)), **kw)


def test_flag_off_leaves_the_gate_exactly_where_it_is(bush_stat_pack,
                                                      monkeypatch) -> None:
    monkeypatch.delenv(sr.FLAG_ENV, raising=False)
    res = _consolidate(bush_stat_pack)
    assert res.quota_met is False and res.gate_code == GATE_INSUFFICIENT
    assert res.statistical_releases == []
    assert [it.stat_release_rule for it in res.items] == ["", "", ""]
    assert [it.to_payload_v2().get("era_note") for it in res.items] == [
        None, POST_SPEECH_NOTE, POST_SPEECH_NOTE]


def test_flag_on_gives_the_statistical_releases_their_credit_back(
        bush_stat_pack, monkeypatch) -> None:
    monkeypatch.setenv(sr.FLAG_ENV, "1")
    res = _consolidate(bush_stat_pack)
    assert res.quota_met is True and res.gate_code == ""
    assert [r["agency"] for r in res.statistical_releases] == ["BLS", "BLS"]
    assert [r["rule"] for r in res.statistical_releases] == [sr.RULE_MONTH,
                                                             sr.RULE_MONTH]


def test_a_released_item_stops_claiming_to_be_context_only(bush_stat_pack,
                                                           monkeypatch) -> None:
    """After the release the old note would be a false statement about an item
    the gate actually spent."""
    monkeypatch.setenv(sr.FLAG_ENV, "1")
    res = _consolidate(bush_stat_pack)
    notes = [it.to_payload_v2().get("era_note") for it in res.items]
    assert notes == [None, sr.RELEASE_NOTE, sr.RELEASE_NOTE]


def test_the_explicit_argument_is_the_same_switch(bush_stat_pack,
                                                  monkeypatch) -> None:
    """``statistical_release=`` overrides the env in BOTH directions, so the $0
    measurement and the tests never depend on ambient environment."""
    monkeypatch.delenv(sr.FLAG_ENV, raising=False)
    assert _consolidate(bush_stat_pack, statistical_release=True).quota_met is True
    monkeypatch.setenv(sr.FLAG_ENV, "1")
    assert _consolidate(bush_stat_pack, statistical_release=False).quota_met is False


def test_a_denied_host_in_the_same_band_is_not_released(monkeypatch) -> None:
    """clinton_1998:0101's own pack: the FY1999 Budget and the administration's
    budget release sit in the same post-speech band and stay context-only."""
    monkeypatch.setenv(sr.FLAG_ENV, "1")
    speech_context.register_speech_date("clinton_1998", CLINTON_UTT)
    pack = [
        _ev(BUDGET_GPO_URL, SourceTier.GOVERNMENT, BUDGET_GPO_SNIPPET,
            date(1998, 2, 2)),
        _ev("https://clintonwhitehouse3.archives.gov/WH/New/99Budget/"
            "education.html", SourceTier.POLITICAL,
            "[1998-02-02] Administration's own FY99 budget release.",
            date(1998, 2, 2)),
    ]
    res = consolidate("clinton_1998:0101", [("stored", pack)],
                      utterance=CLINTON_UTT,
                      window=(date(1996, 1, 1), date(1998, 3, 1)))
    assert res.statistical_releases == []
    assert res.quota_met is False and res.gate_code == GATE_INSUFFICIENT


def test_d15_wins_where_both_rules_apply(monkeypatch) -> None:
    """A record of the utterance credits nothing on any branch. If a transcript
    ever landed on an allowlisted host with a pre-utterance period in its
    snippet, D16 must not hand it back the credit D15 took away."""
    monkeypatch.setenv(sr.FLAG_ENV, "1")
    monkeypatch.setenv(ur.FLAG_ENV, "1")
    speech_context.register_speech_date("clinton_1998", CLINTON_UTT)
    pack = [
        _ev("https://www.bls.gov/content/pkg/WCPD-1998-02-02/pdf/x.pdf",
            SourceTier.GOVERNMENT,
            "Weekly Compilation covering January 1998 remarks.",
            date(1998, 2, 2)),
    ]
    res = consolidate("clinton_1998:0101", [("stored", pack)],
                      utterance=CLINTON_UTT,
                      window=(date(1996, 1, 1), date(1998, 3, 1)))
    assert [it.utterance_rule for it in res.items] == [ur.RULE_WCPD]
    assert [it.stat_release_rule for it in res.items] == [""]
    assert res.statistical_releases == []
    assert res.quota_met is False


def test_the_role_aware_quota_sees_the_release_too(monkeypatch) -> None:
    """The D11.2 branches count corroborants and primary records with their own
    post_speech tests; the release must reach every branch, not just one."""
    from truthbot.verify.principals import PrincipalRelation

    monkeypatch.setenv(sr.FLAG_ENV, "1")
    pack = [
        _ev(BLS_EMPSIT_URL, SourceTier.GOVERNMENT, BLS_EMPSIT_SNIPPET,
            BLS_EMPSIT_PUB),
        _ev("https://www.bls.gov/opub/ted/2006/feb/wk1/art02.htm",
            SourceTier.GOVERNMENT,
            "[2006-02-07] BLS Economics Daily summary of January 2006 "
            "payrolls (+193,000).", date(2006, 2, 7)),
    ]
    res = _consolidate(pack, claim_shape="c-count",
                       relation_of=lambda ev: PrincipalRelation.INDEPENDENT)
    assert res.quota_met is True
    assert [it.role for it in res.items] == ["normal", "normal"]


def test_lenient_era_mode_is_unaffected_by_d16(monkeypatch) -> None:
    """Lenient mode is a PRE-WEB policy about retrospective items; the
    post-speech band and its release are orthogonal, and turning D16 on must
    not disturb a lenient pack that contains no post-speech statistical
    record."""
    monkeypatch.setenv(sr.FLAG_ENV, "1")
    pack = [_ev("https://www.bls.gov/opub/mlr/1974/01/art1full.pdf",
                SourceTier.GOVERNMENT, "1973 BLS table.", date(1974, 1, 5),
                supports=None)]
    kw = dict(utterance=date(1974, 1, 30),
              window=(date(1972, 1, 1), date(1974, 4, 30)), era_mode="lenient")
    on = consolidate("nixon_1974:0001", [("stored", pack)], **kw)
    off = consolidate("nixon_1974:0001", [("stored", pack)],
                      statistical_release=False, **kw)
    assert on.quota_met == off.quota_met
    assert on.statistical_releases == []
