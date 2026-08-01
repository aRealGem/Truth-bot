"""R4 NYT-archive retriever tests (P132 / D12) — fully offline.

The HTTP layer and the cheap-LLM assists are injectable, so these pin the
lane's contract without a key or a network: query generation with fallback,
doc→Evidence mapping (S3 tier, dated snippet), era windowing (including the
lenient-mode context-date recovery), dedup + cap, stance scoring, and the
fail-soft behaviors (no key, HTTP errors, budget).
"""
from __future__ import annotations

from datetime import date

from truthbot.models import SourceTier
from truthbot.verify.archive_retriever import NytArchiveRetriever, _era_window

_UTT = date(1974, 1, 30)   # Nixon SOTU — the corpus this lane exists for


def _doc(url: str, headline: str = "Headline", abstract: str = "Abstract.",
         pub: str = "1974-01-15") -> dict:
    return {"web_url": url, "headline": {"main": headline},
            "abstract": abstract, "pub_date": f"{pub}T00:00:00+0000"}


def _retriever(docs_by_call: list[list[dict]], llm=None) -> NytArchiveRetriever:
    calls = {"urls": []}

    def http_get(url: str) -> dict:
        calls["urls"].append(url)
        idx = min(len(calls["urls"]) - 1, len(docs_by_call) - 1)
        return {"response": {"docs": docs_by_call[idx]}}

    r = NytArchiveRetriever(api_key="test-key", llm=llm if llm is not None else False,
                            http_get=http_get)
    r._test_calls = calls  # type: ignore[attr-defined]
    return r


# ── era windowing ─────────────────────────────────────────────────────────────


def test_explicit_window_wins_and_strict_params_pass_through() -> None:
    win = (date(1972, 1, 1), date(1974, 2, 6))
    assert _era_window(_UTT, win, "") == win


def test_lenient_mode_recovers_the_date_from_the_era_brief() -> None:
    ctx = ("HISTORICAL CLAIM from a speech given on 1974-01-30. Ideal sources "
           "are archival originals…")
    from datetime import timedelta
    win = _era_window(None, None, ctx)
    assert win is not None
    assert win[0] == _UTT - timedelta(days=730)
    assert win[1] == date(1974, 2, 6)      # utterance + 7-day fair game


def test_no_date_anywhere_means_unbounded_search() -> None:
    assert _era_window(None, None, "no date here") is None


# ── shortlist contract ────────────────────────────────────────────────────────


def test_shortlist_maps_docs_to_s3_evidence_with_dated_snippets() -> None:
    r = _retriever([[_doc("https://www.nytimes.com/1974/01/15/a.html",
                          "Energy Crisis Eases", "Federal data show supply up.")]])
    evs = r.shortlist("The energy crisis has eased.", utterance=_UTT,
                      window=(date(1972, 1, 1), date(1974, 2, 6)))
    assert len(evs) == 1
    ev = evs[0]
    assert ev.source_tier is SourceTier.ESTABLISHED          # nytimes.com = S3
    assert ev.source_name == "R4-nyt-archive"
    assert ev.snippet.startswith("[1974-01-15] Energy Crisis Eases — ")
    assert ev.published_at.date() == date(1974, 1, 15)
    assert ev.supports_claim is None                         # unscored w/o llm
    # The API call carried the era window.
    url = r._test_calls["urls"][0]
    assert "begin_date=19720101" in url and "end_date=19740206" in url


def test_shortlist_dedups_and_caps() -> None:
    dup = _doc("https://www.nytimes.com/1974/01/15/a.html")
    many = [dup] + [_doc(f"https://www.nytimes.com/1974/01/{i:02d}/x.html")
                    for i in range(1, 10)]
    r = _retriever([many, many])
    evs = r.shortlist("claim", utterance=_UTT)
    urls = [e.source_url for e in evs]
    assert len(urls) == len(set(urls))
    assert len(evs) <= 8


def test_llm_assists_are_used_when_available() -> None:
    def llm(system: str, user: str) -> dict:
        if "search queries" in system:
            return {"queries": ["federal energy supply 1974", "oil embargo end"]}
        return {"scores": [{"i": 1, "relevance": 0.9, "supports": True}]}

    r = _retriever([[_doc("https://www.nytimes.com/1974/01/15/a.html")]], llm=llm)
    evs = r.shortlist("The energy crisis has eased.", utterance=_UTT)
    assert evs[0].supports_claim is True                     # stance attached
    assert evs[0].relevance_score == 0.9
    # Query-gen output (not the raw claim) reached the API.
    assert "federal+energy+supply+1974" in r._test_calls["urls"][0]


# ── fail-soft behaviors ───────────────────────────────────────────────────────


def test_no_key_returns_empty_shortlist() -> None:
    r = NytArchiveRetriever(api_key="", llm=False,
                            http_get=lambda url: {"response": {"docs": []}})
    import os
    old = os.environ.pop("NYT_API_KEY", None)
    try:
        assert r.shortlist("claim", utterance=_UTT) == []
    finally:
        if old is not None:
            os.environ["NYT_API_KEY"] = old


def test_http_failure_is_soft() -> None:
    def boom(url: str) -> dict:
        raise OSError("network down")
    r = NytArchiveRetriever(api_key="k", llm=False, http_get=boom)
    assert r.shortlist("claim", utterance=_UTT) == []


def test_daily_budget_guard() -> None:
    r = _retriever([[_doc("https://www.nytimes.com/1974/01/15/a.html")]])
    r.requests_made = 10_000
    assert r.shortlist("claim", utterance=_UTT) == []
