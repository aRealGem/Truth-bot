"""The D16 statistical-agency allowlist — load-time guards and the deny order.

This registry is the ONLY door through which post-speech evidence can reach a
verdict, so the tests here are mostly about what it REFUSES: every structural
exclusion the design promised is asserted by name, and the two motivating
examples are pinned as allowlist misses rather than left to inference.
"""
from __future__ import annotations

import pytest
import yaml

from truthbot.verify import statistical_agency as sa
from truthbot.verify.tier_registry import load_registry as load_tier_registry


@pytest.fixture()
def reg():
    return sa.load_registry()


# ── the file loads, and says what it claims to say ──────────────────────────

def test_registry_loads_with_the_expected_schema(reg) -> None:
    assert reg.schema == sa.SCHEMA
    assert reg.version
    assert reg.entries_by_domain


def test_every_entry_carries_an_agency_and_a_rationale(reg) -> None:
    """Extend consciously: a host with no stated function is not a function
    test, it is a hole."""
    for e in reg.entries():
        assert e.agency and e.rationale and e.date, e.domain


def test_the_seeded_agencies_are_all_present(reg) -> None:
    """The seed the directive named, function by function."""
    assert reg.agencies() >= {"BLS", "BEA", "Census", "CBO", "GAO", "CRS",
                              "FRED", "ALFRED", "EIA", "NCES",
                              "NCHS/CDC-statistical", "USDA-NASS"}


def test_press_prefixes_are_inherited_from_the_tier_registry(reg) -> None:
    """One list, one meaning. The tier registry's ``stat_press_prefixes``
    already encode that bls.gov/news.release/* is the jobs report while
    bls.gov/newsroom/* is the press shop; a second copy would drift."""
    assert set(reg.press_prefixes) >= set(load_tier_registry().stat_press_prefixes)


# ── load-time guards ────────────────────────────────────────────────────────

def test_a_wildcard_domain_is_refused() -> None:
    """No patterns here, deliberately — an allowlist that matches a CLASS is an
    allowlist nobody reads."""
    with pytest.raises(ValueError, match="bare host suffix"):
        sa._parse_entry({"domain": "*.stlouisfed.org", "agency": "X",
                         "rationale": "r"})


def test_an_entry_without_a_rationale_is_refused() -> None:
    with pytest.raises(ValueError, match="agency and"):
        sa._parse_entry({"domain": "example.gov", "agency": "X"})


def test_a_path_that_is_not_a_prefix_is_refused() -> None:
    with pytest.raises(ValueError, match="must be a path prefix"):
        sa._parse_entry({"domain": "cdc.gov", "agency": "X", "rationale": "r",
                         "allow_paths": ["nchs"]})


def test_a_host_cannot_be_both_allowed_and_denied(tmp_path, monkeypatch) -> None:
    """A deny would win silently and the entry would be a lie on disk."""
    doc = {
        "schema": sa.SCHEMA, "version": "test",
        "deny": {"domains": [{"domain": "bls.gov", "rationale": "r"}]},
        "entries": [{"domain": "bls.gov", "agency": "BLS", "rationale": "r"}],
    }
    p = tmp_path / "reg.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    monkeypatch.setattr(sa, "_REGISTRY_PATH", p)
    sa.load_registry.cache_clear()
    try:
        with pytest.raises(ValueError, match="BOTH entries and deny"):
            sa.load_registry()
    finally:
        sa.load_registry.cache_clear()


# ── the allowlist ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("url,reason", [
    ("https://www.bls.gov/news.release/archives/empsit_02032006.pdf", "entry:bls.gov"),
    ("https://www.bea.gov/news/1998/personal-income-and-outlays-december-1997", "entry:bea.gov"),
    ("https://www.cbo.gov/publication/45010", "entry:cbo.gov"),
    ("https://www.eia.gov/todayinenergy/detail.php?id=15131", "entry:eia.gov"),
    ("https://www.census.gov/data/tables/2005/demo/income.html", "entry:census.gov"),
    ("https://www.gao.gov/products/gao-06-123", "entry:gao.gov"),
    ("https://crsreports.congress.gov/product/pdf/R/R45242", "entry:crsreports.congress.gov"),
    ("https://fred.stlouisfed.org/series/PAYEMS", "entry:fred.stlouisfed.org"),
    ("https://nces.ed.gov/programs/digest/d05/tables/dt05_001.asp", "entry:nces.ed.gov"),
    ("https://quickstats.nass.usda.gov/results/1234", "entry:nass.usda.gov"),
    ("https://stacks.cdc.gov/view/cdc/115111/cdc_115111_DS1.pdf", "entry:stacks.cdc.gov"),
])
def test_allowlisted_statistical_records(url, reason) -> None:
    assert sa.classify_ex(url) == (True, reason)


def test_bls_news_release_is_the_jobs_report_not_a_press_page() -> None:
    """The single most important non-exclusion: BLS publishes the Employment
    Situation under /news.release/, and a naive press-path rule would delete
    the entire motivating case for D16."""
    allowed, reason = sa.classify_ex(
        "https://www.bls.gov/news.release/archives/empsit_03042022.htm")
    assert allowed is True and reason == "entry:bls.gov"


def test_bea_publishes_its_statistical_releases_under_slash_news() -> None:
    """Same trap, different agency: bea.gov/news/<year>/<release> IS the GDP
    release. Only the narrower stat_press_prefixes may apply here."""
    assert sa.is_statistical_agency(
        "https://www.bea.gov/news/2014/gross-domestic-product-4th-quarter-and-"
        "annual-2013-advance-estimate") is True


def test_a_statistical_agencys_own_newsroom_is_still_excluded() -> None:
    allowed, reason = sa.classify_ex("https://www.bls.gov/newsroom/spotlight.htm")
    assert allowed is False and reason == "deny:press-prefix:/newsroom"


# ── the structural exclusions (the point of the design) ─────────────────────

@pytest.mark.parametrize("url", [
    "https://www.whitehouse.gov/omb/budget/",
    "https://obamawhitehouse.archives.gov/the-press-office/2014/01/28/remarks",
    "https://clintonwhitehouse3.archives.gov/WH/New/99Budget/education.html",
    "https://georgewbush-whitehouse.archives.gov/onbc/2006/",
])
def test_anything_whitehouse_is_denied_structurally(url) -> None:
    assert sa.classify_ex(url) == (False, "deny:host-substring:whitehouse")


@pytest.mark.parametrize("host,label", [
    ("https://omb.gov/reports/", "omb"),
    ("https://ondcp.gov/strategy/", "ondcp"),
    ("https://cea.eop.gov/report/", "cea"),
])
def test_executive_office_of_the_president_units_are_denied(host, label) -> None:
    """OMB, ONDCP and CEA author the President's Budget, the National Drug
    Control Strategy and the Economic Report of the President. Matched on an
    EXACT dot-label, never a substring."""
    assert sa.classify_ex(host) == (False, f"deny:host-label:{label}")


def test_a_label_deny_is_not_a_substring_deny() -> None:
    """"omb" must not swallow "ombudsman" — the deny is structural, not
    lexical."""
    assert sa.classify_ex("https://ombudsman.example.gov/data/") == (
        False, "not-listed")


def test_fraser_is_denied_even_though_it_serves_real_bls_releases() -> None:
    """FRASER reprints the January 2006 Employment Situation (gwbush_2006:0133)
    AND the OMB budget appendix (gwbush_2006:0155) AND the CEA's Economic
    Report of the President (clinton_1998:0167). A host that serves both cannot
    be a function test."""
    for url in (
        "https://fraser.stlouisfed.org/files/docs/releases/bls/bls_employnews_200601.pdf",
        "https://fraser.stlouisfed.org/title/economic-report-president-45/1998-8097/fulltext",
    ):
        assert sa.classify_ex(url) == (False, "deny:domain:fraser.stlouisfed.org")


def test_cdc_is_path_scoped_to_its_statistical_products() -> None:
    """CDC runs a large press shop, so the entry is an allow_paths entry: the
    biden_2022:0268 media statement is out, NCHS and MMWR are in."""
    assert sa.classify_ex("https://www.cdc.gov/nchs/data/nvsr/nvsr70.pdf") == (
        True, "entry:cdc.gov/path:/nchs")
    assert sa.classify_ex(
        "https://archive.cdc.gov/www_cdc_gov/media/releases/2022/"
        "s0303-covid-19-community-levels.html") == (
        False, "deny:path-not-allowed:archive.cdc.gov")


def test_unlisted_hosts_fail_closed() -> None:
    """The default answer is NO — including for perfectly reputable government
    and wire hosts. This is an allowlist, not a classifier."""
    for url in ("https://apnews.com/article/abc", "https://www.army.mil/news",
                "https://www.energy.gov/articles/x", "https://example.com/"):
        assert sa.classify_ex(url)[0] is False
    assert sa.classify_ex("")[1] == "no-host"


# ── the two motivating examples, pinned by NAME ─────────────────────────────

def test_gwbush_2006_0217_ondcp_strategy_is_an_ALLOWLIST_miss() -> None:
    """gwbush_2006:0217 — the ONDCP National Drug Control Strategy, served from
    justice.gov and files.eric.ed.gov.

    It is excluded because NEITHER HOST IS A STATISTICAL AGENCY — an allowlist
    miss, not a period miss. That distinction is the whole D16(α) design: the
    reviewer rejected the blanket rule precisely because this document reads
    "independent" to ``principal_relation`` (which keys on host) and would need
    a document-class detector to catch. The function test needs none, and to
    prove the exclusion does not depend on the period parser, note that this
    document's snippet DOES carry a valid pre-utterance period ("since 2001")
    — see the companion assertion in tests/verdict/test_statistical_release.py.
    """
    assert sa.classify_ex(
        "https://www.justice.gov/archive/olp/pdf/ndcs06.pdf") == (
        False, "not-listed")
    assert sa.classify_ex(
        "https://files.eric.ed.gov/fulltext/ED503096.pdf") == (
        False, "not-listed")


def test_files_eric_ed_gov_is_not_reachable_from_the_nces_entry() -> None:
    """ERIC is an IES-funded literature clearinghouse sharing ed.gov with NCES.
    Scoping the entry to nces.ed.gov rather than ed.gov is what keeps the ONDCP
    Strategy out — asserted so a future 'simplification' to ed.gov fails here
    instead of in production."""
    assert "ed.gov" not in sa.load_registry().entries_by_domain
    assert sa.is_statistical_agency("https://nces.ed.gov/pubs2006/tables/") is True
    assert sa.is_statistical_agency("https://files.eric.ed.gov/fulltext/x.pdf") is False


def test_clinton_1998_0101_fy1999_budget_is_an_ALLOWLIST_miss() -> None:
    """clinton_1998:0101 — the FY1999 President's Budget, served from gpo.gov,
    plus the third-party CRS mirror in the same pack.

    Both are allowlist misses. gpo.gov is a printing/distribution office, not a
    statistical agency; everycrsreport.com is not the Congressional Research
    Service, which is why the CRS entry names the official crsreports.congress
    .gov host only.
    """
    assert sa.classify_ex(
        "https://www.gpo.gov/fdsys/pkg/BUDGET-1999-BUD/pdf/BUDGET-1999-BUD.pdf"
    ) == (False, "not-listed")
    assert sa.classify_ex(
        "https://www.everycrsreport.com/reports/98-203.html") == (
        False, "not-listed")
    # ...and its administration-authored companion, by the structural rule.
    assert sa.classify_ex(
        "https://clintonwhitehouse3.archives.gov/WH/New/99Budget/education.html"
    ) == (False, "deny:host-substring:whitehouse")


def test_congress_gov_as_a_whole_is_not_the_crs() -> None:
    """congress.gov also serves the Congressional Record, which D15 classifies
    as a record of the utterance itself."""
    assert sa.is_statistical_agency(
        "https://www.congress.gov/congressional-report/105th-congress/"
        "senate-report/58") is False
    assert sa.is_statistical_agency(
        "https://crsreports.congress.gov/product/pdf/R/R45242") is True


def test_agency_for_names_the_function_or_nothing() -> None:
    assert sa.agency_for("https://www.bls.gov/data/") == "BLS"
    assert sa.agency_for("https://www.gpo.gov/anything") == ""
