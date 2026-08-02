"""FactCheck connector time-scoping + pack-build era filter (P67 Round B item 2).

Regression target: FactCheckConnector had no search_windowed override, so a
PolitiFact piece published 2026-02-09 landed in a Biden-2022 evidence pack.
Offline: httpx.get is monkeypatched, no network or key needed.
"""
from __future__ import annotations

from datetime import date, datetime

from truthbot.models import Claim, Evidence, SourceTier
from truthbot.verdict.evidence_pack import build_evidence_pack
from truthbot.verify.evidence_provider import EvidenceProvider
from truthbot.verify.sources.base import TimeWindow
from truthbot.verify.sources.factcheck import FactCheckConnector

WINDOW = (date(2020, 1, 1), date(2022, 6, 1))   # biden_2022-style era window


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _patch_httpx(monkeypatch, capture, results):
    import httpx

    def fake_get(url, headers=None, params=None, timeout=None):
        capture["params"] = params
        return _FakeResp({"web": {"results": results}})

    monkeypatch.setattr(httpx, "get", fake_get)


def _claim():
    return Claim(transcript_id="t", text="The federal government spends about $600B a year.")


def test_search_windowed_sends_date_range(monkeypatch):
    cap = {}
    _patch_httpx(monkeypatch, cap, [])
    conn = FactCheckConnector(brave_api_key="k")
    conn.search_windowed(_claim(), WINDOW)
    assert cap["params"]["freshness"] == "2020-01-01to2022-06-01"


def test_search_without_window_sends_no_freshness(monkeypatch):
    cap = {}
    _patch_httpx(monkeypatch, cap, [])
    conn = FactCheckConnector(brave_api_key="k")
    conn.search(_claim())
    assert "freshness" not in cap["params"]


def test_results_filtered_to_registered_factcheck_domains(monkeypatch):
    cap = {}
    _patch_httpx(monkeypatch, cap, [
        {"url": "https://www.politifact.com/factchecks/2022/x", "description": "d"},
        {"url": "https://notpolitifact.com/x", "description": "d"},
        {"url": "https://evil.io/politifact.com/x", "description": "d"},
        {"url": "https://apnews.com/hub/ap-fact-check/article", "description": "d"},
        {"url": "https://apnews.com/sports/game-recap", "description": "d"},
    ])
    conn = FactCheckConnector(brave_api_key="k")
    got = [e.source_url for e in conn.search_windowed(_claim(), WINDOW)]
    assert got == ["https://www.politifact.com/factchecks/2022/x",
                   "https://apnews.com/hub/ap-fact-check/article"]


def test_result_carries_published_at_and_dated_snippet(monkeypatch):
    cap = {}
    _patch_httpx(monkeypatch, cap, [
        {"url": "https://www.snopes.com/fact-check/x", "description": "<b>ruling</b>",
         "page_age": "2022-03-05T12:00:00"},
    ])
    conn = FactCheckConnector(brave_api_key="k")
    ev = conn.search_windowed(_claim(), WINDOW)[0]
    assert ev.published_at == datetime(2022, 3, 5)
    assert ev.snippet.startswith("[2022-03-05]")
    assert "<b>" not in ev.snippet
    assert ev.source_tier == SourceTier.FACTCHECK
    assert ev.source_name == "Snopes"


# ── pack-build era filter ─────────────────────────────────────────────────────

class _Provider(EvidenceProvider):
    def __init__(self, evidence):
        self._evidence = evidence

    def get_evidence(self, claim: Claim, *, window: TimeWindow = None) -> list[Evidence]:
        return list(self._evidence)


def _ev(url, published_at=None):
    return Evidence(claim_id="c", source_name="s", source_url=url, snippet="snip",
                    published_at=published_at)


def test_pack_drops_dated_items_outside_era_window():
    # biden_2022 window is (2020-01-01, 2022-06-01); a 2026 ruling must not enter.
    p = _Provider([
        _ev("https://politifact.com/2026-piece", datetime(2026, 2, 9)),
        _ev("https://apnews.com/2022-piece", datetime(2022, 2, 15)),
        _ev("https://example.com/undated"),
    ])
    pack = build_evidence_pack("biden_2022:0139", "c", p)
    urls = [it.source_url for it in pack.items]
    assert "https://politifact.com/2026-piece" not in urls
    assert "https://apnews.com/2022-piece" in urls
    assert "https://example.com/undated" in urls          # undated items pass


def test_pack_fails_closed_when_speech_date_unknown():
    # Remediation v2 (1.3): an unregistered speech date used to silently
    # disable era gating; the build now fails closed unless the caller
    # explicitly declares the build dateless.
    import pytest

    from truthbot.verdict.era_lint import EraLintError

    p = _Provider([_ev("https://a/x", datetime(2026, 2, 9))])
    with pytest.raises(EraLintError, match="no utterance date registered"):
        build_evidence_pack("mystery_2099:1", "c", p)
    pack = build_evidence_pack("mystery_2099:1", "c", p, era_exempt=True)
    assert [it.source_url for it in pack.items] == ["https://a/x"]


def test_pack_drops_homepage_and_listing_urls():
    # trump_2026:0107 (jackie): snopes.com homepage + a ?pagenum listing page
    # occupied 2 of 6 pack slots. Non-substantive URLs never enter the pack.
    p = _Provider([
        _ev("https://www.snopes.com/"),
        _ev("https://www.snopes.com/fact-check/?pagenum=3"),
        _ev("https://www.snopes.com/fact-check/texas-flood-camp/"),
    ])
    pack = build_evidence_pack("trump_2026:0107", "c", p)
    assert [it.source_url for it in pack.items] == [
        "https://www.snopes.com/fact-check/texas-flood-camp/"]


def test_factcheck_connector_drops_listing_results(monkeypatch):
    cap = {}
    _patch_httpx(monkeypatch, cap, [
        {"url": "https://www.snopes.com/", "description": "The definitive reference."},
        {"url": "https://www.snopes.com/fact-check/?pagenum=3", "description": "Rumors."},
        {"url": "https://www.snopes.com/fact-check/real-article/", "description": "d"},
    ])
    conn = FactCheckConnector(brave_api_key="k")
    got = [e.source_url for e in conn.search_windowed(_claim(), WINDOW)]
    assert got == ["https://www.snopes.com/fact-check/real-article/"]
