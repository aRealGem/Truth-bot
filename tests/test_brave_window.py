"""Brave time-scoping — search_windowed passes a date-range freshness.

Offline: httpx.get is monkeypatched, so no network and no API key is needed."""
from __future__ import annotations

from datetime import date

from truthbot.models import Claim, SourceTier
from truthbot.verify.sources.brave import (
    BraveSearchConnector, _clean_snippet, _freshness_for, _result_date,
)


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _patch_httpx(monkeypatch, capture):
    import httpx

    def fake_get(url, headers=None, params=None, timeout=None):
        capture["params"] = params
        return _FakeResp({"web": {"results": [
            {"url": "https://bls.gov/x", "description": "d", "page_age": "2026-03-16T00:00:00",
             "meta_url": {"hostname": "bls.gov"}},
        ]}})

    monkeypatch.setattr(httpx, "get", fake_get)


def test_freshness_for_window_is_date_range():
    assert _freshness_for((date(2024, 1, 1), date(2026, 5, 1))) == "2024-01-01to2026-05-01"


def test_freshness_for_none_is_past_year():
    assert _freshness_for(None) == "py"


def test_clean_snippet_strips_html_and_unescapes():
    assert _clean_snippet("<strong>Trump said &quot;zero&quot;</strong>") == 'Trump said "zero"'
    assert _clean_snippet("") == ""


def test_result_date_extracts_date_part():
    assert _result_date({"page_age": "2026-03-16T00:00:00"}) == "2026-03-16"
    assert _result_date({"age": "3 months ago"}) == ""      # human age is not machine-usable
    assert _result_date({}) == ""


def test_search_windowed_sends_date_range(monkeypatch):
    cap = {}
    _patch_httpx(monkeypatch, cap)
    conn = BraveSearchConnector(api_key="k")
    ev = conn.search_windowed(Claim(transcript_id="t", text="c"),
                              (date(2024, 1, 1), date(2026, 5, 1)))
    assert cap["params"]["freshness"] == "2024-01-01to2026-05-01"
    # publication date is folded into the snippet so it survives into the payload
    assert ev[0].snippet.startswith("[2026-03-16]")
    assert ev[0].source_tier == SourceTier.GOVERNMENT
    # ...and captured structurally for the pack-build era filter (P67 Round B #60)
    from datetime import datetime
    assert ev[0].published_at == datetime(2026, 3, 16)


def test_brave_drops_homepage_and_listing_results(monkeypatch):
    """P67 2026-07-20 (#67): homepages and listing indexes are not evidence —
    the Brave result loop must skip them, same as FactCheck and pack build."""
    import httpx

    def fake_get(url, headers=None, params=None, timeout=None):
        return _FakeResp({"web": {"results": [
            {"url": "https://www.snopes.com/", "description": "definitive reference",
             "meta_url": {"hostname": "snopes.com"}},
            {"url": "https://www.snopes.com/fact-check/?pagenum=3", "description": "rumors",
             "meta_url": {"hostname": "snopes.com"}},
            {"url": "https://bls.gov/news/article", "description": "data",
             "meta_url": {"hostname": "bls.gov"}},
        ]}})

    monkeypatch.setattr(httpx, "get", fake_get)
    conn = BraveSearchConnector(api_key="k")
    ev = conn.search(Claim(transcript_id="t", text="c"))
    assert [e.source_url for e in ev] == ["https://bls.gov/news/article"]


def test_search_without_window_uses_past_year(monkeypatch):
    cap = {}
    _patch_httpx(monkeypatch, cap)
    conn = BraveSearchConnector(api_key="k")
    conn.search(Claim(transcript_id="t", text="c"))
    assert cap["params"]["freshness"] == "py"


def test_unavailable_connector_returns_empty(monkeypatch):
    conn = BraveSearchConnector(api_key="")     # no key
    assert conn.search_windowed(Claim(transcript_id="t", text="c"),
                                (date(2024, 1, 1), date(2026, 5, 1))) == []
