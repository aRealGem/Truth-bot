"""Relevance middle step (P67 Round B item 3) — offline, injectable LlmFn fakes.

Covers: cheap-model query generation (with fallback), in-place relevance /
supports scoring (fail-soft), the RelevanceProvider fetch flow, tolerant JSON
parsing, and the pack's relevance-then-tier ranking.
"""
from __future__ import annotations

from datetime import date

import pytest

from truthbot.models import Claim, Evidence, SourceTier
from truthbot.verdict.evidence_pack import _dedup_rank_cap
from truthbot.verify.relevance import (
    RelevanceProvider,
    generate_queries,
    parse_json_loosely,
    score_evidence,
)
from truthbot.verify.sources.brave import BraveSearchConnector
from truthbot.verify.sources.factcheck import FactCheckConnector

WINDOW = (date(2020, 1, 1), date(2022, 6, 1))


def _claim(text="The federal government spends about $600 Billion a year to keep the country safe and secure."):
    return Claim(transcript_id="t", text=text)


def _ev(url, tier=SourceTier.OTHER, snippet="snip", relevance=0.5):
    return Evidence(claim_id="c", source_name="s", source_url=url,
                    source_tier=tier, snippet=snippet, relevance_score=relevance)


# ── parse_json_loosely ────────────────────────────────────────────────────────

def test_parse_json_loosely_plain_and_fenced():
    assert parse_json_loosely('{"a": 1}') == {"a": 1}
    assert parse_json_loosely('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_loosely("no json here")


# ── generate_queries ──────────────────────────────────────────────────────────

def test_generate_queries_dedupes_and_caps():
    def llm(system, user):
        assert "era" in user
        return {"queries": ["federal defense budget FY2022",
                            "Federal Defense Budget FY2022",   # dupe (case)
                            "homeland security appropriations 2022",
                            "", 42, "extra query beyond n"]}
    qs = generate_queries(llm, "claim", window=WINDOW, n=3)
    assert qs == ["federal defense budget FY2022",
                  "homeland security appropriations 2022",
                  "extra query beyond n"]


def test_generate_queries_failure_returns_empty():
    def llm(system, user):
        raise RuntimeError("proxy down")
    assert generate_queries(llm, "claim", window=WINDOW) == []


# ── score_evidence ────────────────────────────────────────────────────────────

def test_score_evidence_populates_fields_in_place():
    evs = [_ev("https://a"), _ev("https://b"), _ev("https://c")]
    def llm(system, user):
        return {"scores": [{"i": 1, "relevance": 0.9, "supports": True},
                           {"i": 2, "relevance": 2.5, "supports": False},   # clamped
                           {"i": 3, "relevance": "bad", "supports": "yes"}]}  # ignored
    score_evidence(llm, "claim", evs)
    assert evs[0].relevance_score == 0.9 and evs[0].supports_claim is True
    assert evs[1].relevance_score == 1.0 and evs[1].supports_claim is False
    assert evs[2].relevance_score == 0.5 and evs[2].supports_claim is None


def test_score_evidence_failure_keeps_defaults():
    evs = [_ev("https://a")]
    def llm(system, user):
        raise RuntimeError("proxy down")
    score_evidence(llm, "claim", evs)
    assert evs[0].relevance_score == 0.5 and evs[0].supports_claim is None


# ── RelevanceProvider flow ────────────────────────────────────────────────────

class _FakeBrave(BraveSearchConnector):
    def __init__(self, by_query):
        super().__init__(api_key="k")
        self.by_query = by_query
        self.queries_seen = []
        self.windowed_calls = 0

    def search_query(self, claim, query, window=None):
        self.queries_seen.append((query, window))
        return list(self.by_query.get(query, []))

    def search_windowed(self, claim, window=None):
        self.windowed_calls += 1
        return [_ev("https://legacy")]


class _FakeFactCheck(FactCheckConnector):
    def __init__(self, evidence):
        super().__init__(brave_api_key="k")
        self._evidence = evidence

    def search_windowed(self, claim, window=None):
        return list(self._evidence)


def test_provider_fetches_per_generated_query_and_scores():
    brave = _FakeBrave({
        "q1": [_ev("https://a"), _ev("https://dup")],
        "q2": [_ev("https://dup"), _ev("https://b")],
    })
    fc = _FakeFactCheck([_ev("https://fc", tier=SourceTier.FACTCHECK)])
    calls = {"n": 0}

    def llm(system, user):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"queries": ["q1", "q2"]}
        return {"scores": [{"i": 1, "relevance": 0.2, "supports": None},
                           {"i": 2, "relevance": 0.9, "supports": True},
                           {"i": 3, "relevance": 0.7, "supports": None},
                           {"i": 4, "relevance": 0.8, "supports": False}]}

    provider = RelevanceProvider(brave, [fc], llm)
    out = provider.get_evidence(_claim(), window=WINDOW)
    # dedup: a, dup, b, fc — in fetch order
    assert [e.source_url for e in out] == ["https://a", "https://dup", "https://b", "https://fc"]
    assert [(q, w) for q, w in brave.queries_seen] == [("q1", WINDOW), ("q2", WINDOW)]
    assert brave.windowed_calls == 0
    assert out[1].relevance_score == 0.9 and out[1].supports_claim is True


def test_provider_falls_back_to_legacy_query_when_generation_fails():
    brave = _FakeBrave({})
    def llm(system, user):
        raise RuntimeError("proxy down")
    provider = RelevanceProvider(brave, [], llm)
    out = provider.get_evidence(_claim(), window=WINDOW)
    assert brave.windowed_calls == 1
    assert [e.source_url for e in out] == ["https://legacy"]


# ── pack ranking: relevance-then-tier ─────────────────────────────────────────

def test_rank_relevance_beats_tier():
    gov_offtopic = _ev("https://pompeo-speech.gov", tier=SourceTier.GOVERNMENT, relevance=0.2)
    other_ontopic = _ev("https://ontopic.example.com", tier=SourceTier.OTHER, relevance=0.9)
    ranked = _dedup_rank_cap([gov_offtopic, other_ontopic], 6)
    assert [e.source_url for e in ranked] == ["https://ontopic.example.com",
                                              "https://pompeo-speech.gov"]


def test_rank_ties_on_relevance_fall_back_to_tier():
    other = _ev("https://other", tier=SourceTier.OTHER)
    gov = _ev("https://gov", tier=SourceTier.GOVERNMENT)
    ranked = _dedup_rank_cap([other, gov], 6)
    assert [e.source_url for e in ranked] == ["https://gov", "https://other"]
