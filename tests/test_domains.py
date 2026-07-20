"""Registered-domain matching (truthbot.domains) — the tier-classification fix.

Regression target: substring rules classified www.govtech.com as Government
because the URL contains ".gov", so a trade magazine won cap-6 evidence-pack
slots (P67 Round B item 1). Matching must be host-suffix on label boundaries.
"""
from __future__ import annotations

import pytest

from truthbot.domains import host_matches, url_host, url_matches_any
from truthbot.models import SourceTier
from truthbot.publish.site import _tier_badge, _tier_bucket
from truthbot.verify.sources.brave import BraveSearchConnector


# ── url_host ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url,host", [
    ("https://www.bls.gov/data/unemployment", "www.bls.gov"),
    ("http://apnews.com/article", "apnews.com"),
    ("https://BLS.GOV:443/x", "bls.gov"),           # port + case normalized
    ("bls.gov/data", "bls.gov"),                     # scheme-less
    ("https://user:pw@nytimes.com/x", "nytimes.com"),  # userinfo stripped
    ("", ""),
    ("not a url at all", ""),
])
def test_url_host(url, host):
    assert url_host(url) == host


# ── host_matches ──────────────────────────────────────────────────────────────

def test_host_matches_exact_and_subdomain():
    assert host_matches("apnews.com", "apnews.com")
    assert host_matches("www.apnews.com", "apnews.com")
    assert not host_matches("notapnews.com", "apnews.com")     # no label boundary
    assert not host_matches("apnews.com.evil.io", "apnews.com")  # suffix, not infix


def test_host_matches_tld_rule():
    assert host_matches("www.bls.gov", ".gov")
    assert host_matches("cbo.gov", ".gov")
    assert not host_matches("www.govtech.com", ".gov")   # THE regression case
    assert not host_matches("gov.example.com", ".gov")


def test_url_matches_any():
    assert url_matches_any("https://www.reuters.com/x", ("apnews.com", "reuters.com"))
    assert not url_matches_any("https://reuters.com.fake.io/x", ("reuters.com",))
    # a path mentioning a trusted domain must not count
    assert not url_matches_any("https://blog.example.com/apnews.com-review", ("apnews.com",))


# ── the three consumers stay in sync on the regression case ───────────────────

def test_brave_classify_tier_rejects_lookalike_gov():
    conn = BraveSearchConnector(api_key="fake")
    assert conn._classify_tier("https://www.govtech.com/some-article") == SourceTier.OTHER
    assert conn._classify_tier("https://www.bls.gov/data") == SourceTier.GOVERNMENT
    assert conn._classify_tier("https://apnews.com.spoof.io/x") == SourceTier.OTHER


def test_site_tier_bucket_rejects_lookalike_gov():
    assert _tier_bucket("https://www.govtech.com/some-article") == "other"
    assert _tier_bucket("https://www.cbo.gov/publication/1") == "gov"
    assert _tier_bucket("https://www.bbc.co.uk/news/x") == "news"


def test_site_tier_badge_rejects_lookalike_gov():
    assert "T6" in _tier_badge("https://www.govtech.com/some-article")
    assert "T1·Gov" in _tier_badge("https://www.census.gov/data")
