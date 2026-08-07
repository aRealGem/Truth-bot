"""Tier-registry regression harness (remediation v2, 1.2 / DC-2a).

``tier_registry.yaml`` is the versioned source of tier truth. This suite pins
the DC-2a-approved behavior deltas (jackie, 2026-08-02) and — via the frozen
corpus in ``tests/fixtures/tier_snapshot_pre_registry.json`` (a
``{url: tier}`` dump of the PRE-registry ``classify_tier`` over every census
example URL plus every evidence URL in the five published artifacts) — proves
that NOTHING ELSE changed classification.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from truthbot.domains import host_matches, url_host
from truthbot.models import SourceTier
from truthbot.verify.source_tiers import classify_tier
from truthbot.verify.tier_registry import (QUARANTINE_REASON, classify_tier_ex,
                                           load_registry)

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = Path(__file__).resolve().parent / "fixtures" / "tier_snapshot_pre_registry.json"

NINE_MIRRORS = (
    "whitehouse.gov",
    "obamawhitehouse.archives.gov",
    "bidenwhitehouse.archives.gov",
    "trumpwhitehouse.archives.gov",
    "georgewbush-whitehouse.archives.gov",
    "clintonwhitehouse3.archives.gov",
    "clintonwhitehouse4.archives.gov",
    "clintonwhitehouse5.archives.gov",
    "clintonwhitehouse6.archives.gov",
)

#: EVERY whitehouse host the corpus can produce — live, archived, and the
#: mirrors the Phase-1 registry never enumerated because they appeared once or
#: twice (mirror 1) or not at all (mirror 2 was in stored evidence but not in
#: the census the registry was seeded from). Phase A (A3) parity lint: they all
#: classify the same, because they ARE the same thing — a presidential press
#: shop — and which mirror number NARA assigned is not a tiering fact.
ALL_WHITEHOUSE_HOSTS = NINE_MIRRORS + tuple(
    f"clintonwhitehouse{n}.archives.gov" for n in (1, 2, 7))


# ── (a) parity: all nine executive mirrors classify POLITICAL ─────────────────

@pytest.mark.parametrize("host", NINE_MIRRORS)
def test_all_nine_executive_mirrors_are_political(host):
    """DC-2a delta 1: presidential press shop — ONE tier across ALL
    administrations (georgewbush- and the four clintonwhitehouseN mirrors used
    to fail open to Government)."""
    tier, reason = classify_tier_ex(f"https://{host}/some/page")
    assert tier == SourceTier.POLITICAL
    assert reason == f"entry:{host}"
    # …and on data-looking paths too — political entries win on every path.
    assert classify_tier(f"https://{host}/data/budget") == SourceTier.POLITICAL


# ── (a′) Phase A / A3: the CLASS, not just the enumerated instances ──────────

@pytest.mark.parametrize("host", ALL_WHITEHOUSE_HOSTS)
@pytest.mark.parametrize("path", ["/some/page", "/data/budget",
                                  "/WH/New/html/19970630-16084.htm",
                                  "/Initiatives/OneAmerica/about.html"])
def test_every_whitehouse_host_resolves_to_the_same_tier(host, path):
    """CI LINT (A3): live and archive, every *whitehouse* host is POLITICAL on
    every path shape. Before the pattern rule, clintonwhitehouse1 and 2 were
    GOVERNMENT — they had no entry of their own, so the longest-suffix match
    fell through to archives.gov (the National Archives, correctly Government)
    and a presidential press page inherited the top evidence tier while
    mirrors 3-6 sat at POLITICAL. Same publisher, same content, opposite
    tiers, decided by which mirror numbers happened to appear in the corpus
    the registry was seeded from."""
    assert classify_tier(f"https://{host}{path}") == SourceTier.POLITICAL


def test_no_registry_rule_puts_a_whitehouse_host_anywhere_but_political():
    """Belt to the parity braces: scan the registry DATA, so a future entry or
    pattern naming a whitehouse host cannot diverge without failing here."""
    reg = load_registry()
    for domain, entry in reg.entries_by_domain.items():
        if "whitehouse" in domain:
            assert entry.tier_name == "political", domain
    for rule in reg.patterns:
        if "whitehouse" in rule.pattern:
            assert rule.entry.tier_name == "political", rule.pattern


def test_the_clinton_mirrors_are_one_pattern_rule_not_an_enumeration():
    """A3 records WHY: enumeration fixes the instances you happened to
    observe, a pattern fixes the class. One rule covers every mirror number,
    including ones NARA has not minted yet."""
    reg = load_registry()
    clinton = [r for r in reg.patterns if "clintonwhitehouse" in r.pattern]
    assert len(clinton) == 1
    assert clinton[0].pattern == "clintonwhitehouse*.archives.gov"
    assert "OBSERVED" in clinton[0].entry.rationale
    for n in (1, 2, 7, 42):
        host = f"clintonwhitehouse{n}.archives.gov"
        assert classify_tier(f"https://{host}/x") == SourceTier.POLITICAL


# ── (a″) pattern-rule precedence + fail-closed guards ───────────────────────

def test_an_exact_entry_beats_a_pattern_that_also_matches():
    """Precedence rule 1/2: explicit entries keep working untouched — a
    pattern never overrides a decision someone made about a specific host."""
    tier, reason = classify_tier_ex("https://clintonwhitehouse3.archives.gov/x")
    assert tier == SourceTier.POLITICAL
    assert reason == "entry:clintonwhitehouse3.archives.gov"


def test_a_pattern_beats_a_shorter_suffix_entry():
    """Precedence rule 1: 3 labels beat archives.gov's 2. This is the fix —
    the mirror class must not inherit the National Archives' Government tier."""
    tier, reason = classify_tier_ex(
        "https://clintonwhitehouse2.archives.gov/WH/New/html/19970630-16084.htm")
    assert tier == SourceTier.POLITICAL
    assert reason == "pattern:clintonwhitehouse*.archives.gov"
    # …and the parent host itself is untouched: NARA is still Government.
    assert classify_tier("https://www.archives.gov/research/x") == SourceTier.GOVERNMENT


def test_patterns_do_not_widen_the_unknown_policy():
    """Precedence rule 3: a host matching no entry and no pattern is exactly
    as it was — unmapped gov-class still fails CLOSED to quarantine, unmapped
    non-gov is still OTHER. Pattern rules are additive only."""
    tier, reason = classify_tier_ex("https://made-up-agency.gov/some/page")
    assert tier == SourceTier.POLITICAL and reason == QUARANTINE_REASON
    tier, reason = classify_tier_ex("https://example.com/story")
    assert tier == SourceTier.OTHER and reason == "unmapped-non-gov"


@pytest.mark.parametrize("bad,why", [
    ({"pattern": "clintonwhitehouse2.archives.gov", "tier": "political"},
     "no wildcard — that is an entry, not a class"),
    ({"pattern": "*.gov", "tier": "government"},
     "would claim a whole TLD"),
    ({"pattern": "*", "tier": "government"}, "would claim everything"),
    ({"pattern": "foo.*.gov", "tier": "government"},
     "wildcard inside the last two labels"),
])
def test_loader_refuses_a_pattern_that_would_widen_the_registry(bad, why):
    """FAIL-CLOSED authoring guard: a pattern names a class, never the
    internet. Rejected at LOAD, so a bad rule can never be live."""
    from truthbot.verify.tier_registry import _parse_pattern

    with pytest.raises(ValueError):
        _parse_pattern({**bad, "rationale": "x", "date": "2026-08-07"})


# ── (b) protected table: DC-2a delta 2, real URLs from the census ────────────

@pytest.mark.parametrize("url", [
    # fully protected hosts (press framing never demotes)
    "https://www.bls.gov/bls/news-release/empsit.htm",
    "https://bea.gov/news/2006/gross-domestic-product-fourth-quarter-2005-advance-estimates",
    "https://www.cbo.gov/ftpdocs/60xx/doc6060/01-25-BudgetOutlook.pdf",
    "https://govinfo.gov/content/pkg/CREC-1997-10-21/html/CREC-1997-10-21-pt1-PgH8909.htm",
    "https://www.census.gov/newsroom/press-releases/2024/income-poverty.html",
    "https://cdc.gov/mmwr/volumes/71/wr/mm7104e4.htm",
    "https://stacks.cdc.gov/view/cdc/123456",
    "https://archive.cdc.gov/some/page",
    "https://ucr.fbi.gov/crime-in-the-u.s",
    "https://clerk.house.gov/Votes/202612",
    "https://www.uscourts.gov/news/2024/judiciary-report",
    "https://www.supremecourt.gov/opinions/25pdf/24-1234_abcd.pdf",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240101a.htm",
    "https://www.federalregister.gov/documents/2026/01/01/rule",
    "https://nces.ed.gov/programs/coe/indicator/cma",
    "https://pubmed.ncbi.nlm.nih.gov/12345678/",
    # path-scoped protections
    "https://www.fbi.gov/cjis/ucr/crime-in-the-u.s",
    "https://www.cms.gov/oact/tr/2024",
    "https://www.cms.gov/data-research/statistics-trends-and-reports",
    "https://www.irs.gov/statistics/soi-tax-stats",
    "https://www.irs.gov/pub/irs-pdf/p5307.pdf",
    "https://home.treasury.gov/system/files/136/some-analysis.pdf",
    "https://home.treasury.gov/policy-issues/financial-sanctions",
    "https://home.treasury.gov/data/troubled-asset-relief-program",
    "https://www.senate.gov/legislative/LIS/roll_call_votes/vote1191/vote_119_1_00042.htm",
])
def test_protected_statistical_record_functions_are_government(url):
    """DC-2a delta 2: protected statistical/record functions -> GOVERNMENT
    regardless of press framing. The demote criterion is partisan-principal
    AND comms-function; these fail the first prong."""
    assert classify_tier(url) == SourceTier.GOVERNMENT


@pytest.mark.parametrize("url", [
    "https://www.fbi.gov/news/press-releases/some-arrest",       # delta 2: fbi /news stays demoted
    "https://home.treasury.gov/news/press-releases/sb0301",      # delta 2: /news stays demoted
    "https://www.senate.gov/newsroom/press-releases/x",          # delta 2: keeps its demotion
    "https://www.justice.gov/opa/pr/some-announcement",          # delta 4: /opa/pr covered
    "https://www.justice.gov/usao-dc/pr/violent-crime-dc-hits-30-year-low",
])
def test_press_carveouts_still_demote(url):
    assert classify_tier(url) == SourceTier.POLITICAL


def test_jec_is_political():
    """DC-2a delta 3: congressional committee majority/minority-staff
    publications."""
    tier, reason = classify_tier_ex(
        "https://www.jec.senate.gov/public/_cache/files/infrastructure.pdf")
    assert tier == SourceTier.POLITICAL
    assert reason == "entry:jec.senate.gov"


# ── (c) unknown policy ───────────────────────────────────────────────────────

def test_unmapped_non_gov_host_is_other():
    tier, reason = classify_tier_ex("https://made-up-example-site.example/x")
    assert tier == SourceTier.OTHER
    assert reason == "unmapped-non-gov"


@pytest.mark.parametrize("url", [
    "https://made-up-agency.gov/some/page",
    "https://made-up-base.mil/some/page",
    "https://made-up-org.int/some/page",
])
def test_unmapped_gov_mil_int_quarantines_with_distinct_reason(url):
    """S-6 fail closed: unmapped .gov/.mil/.int -> POLITICAL for pack
    semantics, but with the reason "quarantine-unmapped-gov" so telemetry can
    tell a quarantine from a mapped political rule (.mil/.int used to fail
    OPEN to Government — DC-2a delta 3)."""
    tier, reason = classify_tier_ex(url)
    assert tier == SourceTier.POLITICAL
    assert reason == QUARANTINE_REASON


# ── (d) registry hygiene: every entry documented ─────────────────────────────

def test_every_registry_entry_has_rationale_and_date():
    reg = load_registry()
    assert reg.entries(), "registry loaded empty"
    for e in reg.entries():
        assert e.rationale.strip(), f"{e.domain}: empty rationale"
        assert e.date.strip(), f"{e.domain}: empty date"


# ── (e) frozen-corpus regression ─────────────────────────────────────────────
#
# The snapshot was dumped from the PRE-registry classify_tier BEFORE any
# registry code landed (see module docstring). Every URL must classify
# identically under the registry UNLESS an approved DC-2a delta predicate
# explains the change.

def _entry_for(host: str):
    reg = load_registry()
    labels = host.split(".")
    for i in range(len(labels) - 1):
        e = reg.entries_by_domain.get(".".join(labels[i:]))
        if e is not None:
            return e
    return None


def _approved_delta(url: str, old: str, new: str, reason: str) -> bool:
    """The documented DC-2a delta predicates — the ONLY allowed changes.

    A. delta 1: the nine executive mirrors -> POLITICAL.
    B. delta 3: jec.senate.gov -> POLITICAL.
    C. delta 3: unmapped .mil/.int fail closed -> quarantine POLITICAL
       (reason "quarantine-unmapped-gov"; the host has no registry entry).
    D. delta 2: protected promotion — a press-protected entry or protected
       path_class lifts a press-demoted/established URL to GOVERNMENT.
    E. deltas 3/4: de-quarantine — a census-enumerated host now carries a
       government-class entry, so URLs the blanket .gov quarantine used to
       bottom-tier classify GOVERNMENT through the mapping.
    F. deltas 2/4 press tightening on mapped hosts: the widened generic press
       classes (/opa/pr, /news-releases prefixes; pr/press-release(s)/
       newsreleases segments; explicit political path_classes like
       ed.gov//about/news) demote press content that previously slipped
       through to GOVERNMENT on a substantive-path technicality.
    G. Phase A (A3): a pattern rule classifies a host class the enumeration
       missed — here the unenumerated Clinton mirrors, which fell through to
       archives.gov and inherited GOVERNMENT while their identical siblings
       sat at POLITICAL.
    """
    host = url_host(url)
    if reason.startswith("pattern:"):                                       # G
        return new == "Political" and "whitehouse" in host
    if any(host_matches(host, m) for m in NINE_MIRRORS):                    # A
        return new == "Political"
    if host_matches(host, "jec.senate.gov"):                                # B
        return new == "Political"
    entry = _entry_for(host)
    if entry is None:                                                       # C
        return (host.rsplit(".", 1)[-1] in ("mil", "int")
                and old == "Government" and new == "Political"
                and reason == QUARANTINE_REASON)
    if new == "Government" and old in ("Political", "Established"):         # D+E
        return entry.tier_name in ("government", "quarantine")
    if new == "Political" and old == "Government":                          # F
        return (reason.startswith("press-prefix:")
                or reason.startswith("press-segment:")
                or (reason.startswith("entry:") and "/path:" in reason))
    return False


def test_frozen_corpus_only_approved_deltas_changed():
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert len(snap) > 4000, "snapshot suspiciously small"
    unexplained: list[str] = []
    changed = 0
    for url, old in snap.items():
        new, reason = classify_tier_ex(url)
        if new.value == old:
            continue
        changed += 1
        if not _approved_delta(url, old, new.value, reason):
            unexplained.append(f"{url}: {old} -> {new.value} ({reason})")
    assert not unexplained, (
        f"{len(unexplained)} classification changes outside the approved "
        f"DC-2a deltas:\n" + "\n".join(unexplained[:40]))
    # DC-2a verification number, measured 2026-08-02 over 4,381 frozen URLs:
    # 469. Phase A (A3) adds exactly ONE more — the single
    # clintonwhitehouse1.archives.gov URL in the frozen corpus, Government ->
    # Political via the mirror pattern rule. Mirror 2 is not in this snapshot
    # at all (it appears only in stored run evidence, twice), which is the
    # whole reason enumeration missed it.
    assert changed == 470, (
        f"frozen-corpus delta count drifted: {changed} != 470 — re-derive "
        "and re-approve before changing this pin")


# ── quarantine telemetry threads through the consolidator ────────────────────

def test_consolidator_journals_quarantined_urls():
    from datetime import date, datetime, timezone

    from truthbot.models import Evidence
    from truthbot.verdict.consolidator import consolidate

    def _ev(url):
        return Evidence(claim_id="c1", source_name="src", source_url=url,
                        source_tier=classify_tier(url), snippet="[2026-02-20] x",
                        supports_claim=True, relevance_score=1.0,
                        published_at=datetime(2026, 2, 20, tzinfo=timezone.utc))

    quarantined_url = "https://made-up-agency.gov/some/page"
    res = consolidate(
        "s1", [("r1", [_ev("https://apnews.com/article/x"),
                       _ev(quarantined_url),
                       _ev("https://www.bls.gov/data/series")])],
        utterance=date(2026, 2, 24))
    assert res.quarantined == [quarantined_url]
