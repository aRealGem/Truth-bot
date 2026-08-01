"""Unit contract for ``verify.principals`` (PR-A2.1).

The load-bearing property is I3's relational/conditional line: the relation is
computed by ONE total function, identically for every speaker, with all
person-naming facts in ``principals.json``. The symmetry tests below are the
regression for that — the same URL on the same date must flip SELF/INDEPENDENT
purely by which speaker it is evaluated *against*, and every gap (unknown
speaker, missing date, out-of-era) must fail OPEN to INDEPENDENT.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from truthbot.verify.principals import PrincipalRelation, principal_relation

_OBAMA_SOTU = date(2014, 1, 28)


# ── Era-scoped SELF ───────────────────────────────────────────────────────────


def test_whitehouse_is_self_for_the_sitting_president() -> None:
    rel = principal_relation(
        "https://www.whitehouse.gov/the-press-office/2014/01/28/fact-sheet",
        "Barack Obama", _OBAMA_SOTU)
    assert rel is PrincipalRelation.SELF


def test_archives_mirror_is_self_for_its_administration() -> None:
    # Retrieval for a 2014 claim returns the archival mirror, not live
    # whitehouse.gov — the mirror must carry the same relation.
    rel = principal_relation(
        "https://obamawhitehouse.archives.gov/the-press-office/2014/01/28/x",
        "Barack Obama", _OBAMA_SOTU)
    assert rel is PrincipalRelation.SELF


def test_party_and_campaign_domains_are_self() -> None:
    for url in ("https://democrats.org/news/x", "https://barackobama.com/y"):
        assert principal_relation(url, "Barack Obama", _OBAMA_SOTU) is \
            PrincipalRelation.SELF


def test_out_of_era_is_independent() -> None:
    # Same speaker, same domain, but the utterance predates the presidency.
    rel = principal_relation("https://www.whitehouse.gov/briefing/x",
                             "Barack Obama", date(2007, 6, 1))
    assert rel is PrincipalRelation.INDEPENDENT


def test_era_boundaries_are_inclusive_start_exclusive_end() -> None:
    wh = "https://www.whitehouse.gov/x"
    assert principal_relation(wh, "Barack Obama", date(2009, 1, 20)) is \
        PrincipalRelation.SELF
    assert principal_relation(wh, "Barack Obama", date(2017, 1, 20)) is \
        PrincipalRelation.INDEPENDENT
    # …because on that date the SAME domain belongs to the next principal:
    assert principal_relation(wh, "Donald Trump", date(2017, 1, 20)) is \
        PrincipalRelation.SELF


# ── Relational symmetry (the I3 regression) ───────────────────────────────────


def test_same_url_same_date_flips_by_speaker_only() -> None:
    url = "https://www.whitehouse.gov/the-press-office/2014/01/28/fact-sheet"
    assert principal_relation(url, "Barack Obama", _OBAMA_SOTU) is \
        PrincipalRelation.SELF
    for other in ("Donald Trump", "Joe Biden", "Mitt Romney"):
        assert principal_relation(url, other, _OBAMA_SOTU) is \
            PrincipalRelation.INDEPENDENT


def test_another_speakers_archive_is_independent_evidence() -> None:
    # Obama-administration records cited against a Trump-era claim are not
    # Trump self-sourcing.
    rel = principal_relation("https://obamawhitehouse.archives.gov/x",
                             "Donald Trump", date(2018, 1, 30))
    assert rel is PrincipalRelation.INDEPENDENT


# ── Fail-open gaps ────────────────────────────────────────────────────────────


def test_unknown_speaker_is_independent_for_everything() -> None:
    rel = principal_relation("https://www.whitehouse.gov/x",
                             "Abraham Lincoln", date(1863, 11, 19))
    assert rel is PrincipalRelation.INDEPENDENT


def test_missing_or_bad_date_and_url_fail_open() -> None:
    wh = "https://www.whitehouse.gov/x"
    assert principal_relation(wh, "Barack Obama", None) is \
        PrincipalRelation.INDEPENDENT
    assert principal_relation(wh, "Barack Obama", "not-a-date") is \
        PrincipalRelation.INDEPENDENT
    assert principal_relation("", "Barack Obama", _OBAMA_SOTU) is \
        PrincipalRelation.INDEPENDENT


# ── Input tolerance (bundle date_str strings, datetimes, aliases, casing) ─────


def test_accepts_date_strings_datetimes_and_aliases() -> None:
    wh = "https://whitehouse.gov/x"
    assert principal_relation(wh, "Barack Obama", "2014-01-28") is \
        PrincipalRelation.SELF
    assert principal_relation(
        wh, "barack  obama",
        datetime(2014, 1, 28, 21, 0, tzinfo=timezone.utc)) is \
        PrincipalRelation.SELF
    assert principal_relation(wh, "President Barack Obama", _OBAMA_SOTU) is \
        PrincipalRelation.SELF


def test_host_matching_is_suffix_not_substring() -> None:
    # govtech-class regression from source_tiers: a domain merely CONTAINING
    # a principal domain must not match.
    rel = principal_relation("https://notwhitehouse.gov.example.com/x",
                             "Barack Obama", _OBAMA_SOTU)
    assert rel is PrincipalRelation.INDEPENDENT
