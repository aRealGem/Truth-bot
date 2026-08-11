"""B1b: the relevance/stance scorer injected into ``build_evidence_pack_v2``.

The v2 path never scored relevance or stance — R1/R2/R3 shortlists reached
``consolidate`` untouched, so every item kept the 0.5 relevance default and the
20-30%% of items carrying a null stance could not credit ``MIN_BEARING_T13``.
Packs holding perfectly good evidence therefore gate-forced Unverifiable. These
tests pin the fix: the scorer runs BEFORE consolidate, its stance moves the
quota, and leaving it out changes nothing.

Offline and $0 by construction: every scorer here is a STUB. No proxy key, no
``build_proxy_llm``, no model call — the production flags that would spend are
asserted to default OFF.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from truthbot.models import Evidence, SourceTier
from truthbot.verdict import evidence_pack_v2 as v2
from truthbot.verdict.consolidator import GATE_INSUFFICIENT, MIN_BEARING_T13
from truthbot.verdict.evidence_pack_v2 import build_evidence_pack_v2
from truthbot.verdict.speech_context import register_speech_date

SID = "pytest_v2scored:0001"
register_speech_date("pytest_v2scored", datetime(2026, 2, 24).date())


def _ev(url, *, tier=SourceTier.GOVERNMENT, supports=None, day=20):
    """A candidate exactly as the unscored v2 path really emits one: stance
    ``None`` (retrievers.py maps stance "context" → None) and relevance on the
    untouched 0.5 pydantic default."""
    return Evidence(claim_id="", source_name="R", source_url=url,
                    source_tier=tier, snippet="stat page",
                    supports_claim=supports,
                    published_at=datetime(2026, 2, day, tzinfo=timezone.utc))


class _Retriever:
    """Scripted retriever: pops one shortlist per call."""

    def __init__(self, label, shortlists):
        self.label = label
        self.shortlists = list(shortlists)
        self.calls = 0

    def shortlist(self, claim_text, *, context="", utterance=None, window=None):
        self.calls += 1
        return self.shortlists.pop(0) if self.shortlists else []


def _beckstrom():
    """The observed failure shape (verified on trump_2026:0469): ONE bearing
    Tier-1..3 item — NPR — while AP, NBC and two govinfo records sit UNSCORED.
    One bearing item is below MIN_BEARING_T13=2, so the T2.4 gate fires even
    though four qualifying sources are sitting right there."""
    return [
        _ev("https://npr.org/a", tier=SourceTier.ESTABLISHED, supports=True),
        _ev("https://apnews.com/b", tier=SourceTier.WIRE),
        _ev("https://nbcnews.com/c", tier=SourceTier.ESTABLISHED),
        _ev("https://govinfo.gov/d"),
        _ev("https://govinfo.gov/e"),
    ]


def _stub_scorer(log=None, *, stance=True, relevance=0.9):
    """A scorer with realistic effects and ZERO spend: it fills in the stance
    and relevance a real Haiku call would, entirely locally."""
    def scorer(claim_text, evidence):
        if log is not None:
            log.append(("score", claim_text, len(evidence)))
        for ev in evidence:
            if ev.supports_claim is None:
                ev.supports_claim = stance
            ev.relevance_score = relevance
    return scorer


# ── (a) the scorer runs BEFORE consolidate ──────────────────────────────────

def test_scorer_runs_before_consolidate(monkeypatch):
    log: list = []
    real_consolidate = v2.consolidate

    def spy(*a, **kw):
        log.append(("consolidate",))
        return real_consolidate(*a, **kw)

    monkeypatch.setattr(v2, "consolidate", spy)
    r1 = _Retriever("R1", [_beckstrom()])
    build_evidence_pack_v2(SID, "beckstrom", (r1,), scorer=_stub_scorer(log))

    assert [e[0] for e in log] == ["score", "consolidate"]
    # …and it saw the real claim text plus every deduped candidate.
    assert log[0][1] == "beckstrom" and log[0][2] == 5


def test_scorer_sees_each_candidate_once_across_the_retry(monkeypatch):
    """The T2.4 retry scores only its NEW candidates — a rescued claim must
    never pay twice for a page the first round already scored."""
    log: list = []
    first = [_ev("https://govinfo.gov/dup")]
    # Retry re-serves the same URL plus one new one (retrievers overlap a lot).
    retry = [_ev("https://govinfo.gov/dup"), _ev("https://bls.gov/new")]
    r1 = _Retriever("R1", [first, retry])
    # A scorer that scores nothing keeps the quota unmet, forcing the retry.
    build_evidence_pack_v2(SID, "c", (r1,),
                           scorer=lambda t, evs: log.append([e.source_url for e in evs]))
    assert r1.calls == 2                                    # the retry did run
    assert log == [["https://govinfo.gov/dup"], ["https://bls.gov/new"]]


# ── (b) stance from the scorer changes the quota outcome ────────────────────

def test_beckstrom_shape_gates_without_scorer_and_passes_with_it():
    r1 = _Retriever("R1", [_beckstrom(), []])          # retry finds nothing new
    gated = build_evidence_pack_v2(SID, "beckstrom", (r1,))
    assert gated.gate_code == GATE_INSUFFICIENT
    assert gated.scoring["relevance_scored"] == 0
    # Exactly the pathology: plenty of Tier-1..3 items, only ONE of them bearing.
    assert gated.scoring["stance_supports"] == 1
    assert gated.scoring["stance_null"] >= MIN_BEARING_T13

    r2 = _Retriever("R1", [_beckstrom(), []])          # same shape, fresh objects
    passed = build_evidence_pack_v2(SID, "beckstrom", (r2,),
                                    scorer=_stub_scorer())
    assert passed.gate_code == ""                      # no longer gate-forced
    assert r2.calls == 1                               # quota met first pass
    assert passed.scoring["stance_null"] == 0


def test_scored_pack_reports_scored_items_and_is_fit_to_gate():
    """A1 telemetry needs no wiring of its own — it reads the same mutated
    Evidence the scorer wrote through — but the whole point of B1b is that it
    now reports scored>0, so verify it end to end against the fitness lint."""
    from truthbot.publish.consistency import is_fit_to_gate

    r1 = _Retriever("R1", [_beckstrom()])
    pack = build_evidence_pack_v2(SID, "beckstrom", (r1,), scorer=_stub_scorer())
    assert pack.scoring["relevance_scored"] == len(pack.items) > 0
    assert pack.scoring["relevance_default"] == 0

    artifact = {"evidence": {SID: [
        {"source_url": it.source_url, "relevance_score": it.relevance_score,
         "supports_claim": it.supports_claim} for it in pack.items]}}
    fit, reason = is_fit_to_gate(artifact)
    assert fit, reason


def test_refuting_stance_also_credits_the_quota():
    """_bearing accepts True OR False — a pack of refutations must decide, not
    gate. (A scorer that only ever said True would pass the test above while
    quietly breaking every FALSE verdict.)"""
    r1 = _Retriever("R1", [_beckstrom(), []])
    pack = build_evidence_pack_v2(SID, "beckstrom", (r1,),
                                  scorer=_stub_scorer(stance=False))
    assert pack.gate_code == ""
    assert pack.scoring["stance_refutes"] >= MIN_BEARING_T13


# ── (c) scorer=None is today's behaviour, byte for byte ─────────────────────

def _comparable(pack):
    """Pack contents minus ``retrieved_at``, which is a fresh wall-clock stamp
    on every Evidence and so differs between any two builds."""
    d = dataclasses.asdict(pack)
    for key in ("items", "pool"):
        for item in d.get(key) or []:
            item.pop("retrieved_at", None)
    return d


def test_default_scorer_none_is_byte_identical_to_the_old_path():
    a = build_evidence_pack_v2(SID, "beckstrom",
                               (_Retriever("R1", [_beckstrom(), []]),))
    b = build_evidence_pack_v2(SID, "beckstrom",
                               (_Retriever("R1", [_beckstrom(), []]),),
                               scorer=None)
    assert _comparable(a) == _comparable(b)
    assert a.gate_code == GATE_INSUFFICIENT          # unchanged: still gated
    assert a.scoring["relevance_scored"] == 0


def test_scorer_exceptions_propagate_rather_than_being_swallowed():
    """Unlike a dead retriever (soft), the scorer is where a budget breaker
    lives — swallowing its halt would defeat the cap."""
    class Halt(RuntimeError):
        pass

    def boom(claim_text, evidence):
        raise Halt("budget")

    with pytest.raises(Halt):
        build_evidence_pack_v2(SID, "c", (_Retriever("R1", [_beckstrom()]),),
                               scorer=boom)


# ── (d) the production flags default OFF ────────────────────────────────────

def _capture_parser(build_main):
    """Grab the real argparse parser a CLI builds, without running the CLI."""
    holder = {}

    class _Caught(Exception):
        pass

    def fake_parse_args(self, *a, **kw):
        holder["parser"] = self
        raise _Caught

    orig = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = fake_parse_args
    try:
        with pytest.raises(_Caught):
            build_main()
    finally:
        argparse.ArgumentParser.parse_args = orig
    return holder["parser"], orig


def test_publish_cli_score_evidence_defaults_off():
    from truthbot import pipeline

    parser, parse = _capture_parser(pipeline.main)
    base = ["publish", "--transcript", "t.txt", "--speaker", "S",
            "--date", "2026-02-24"]
    assert parse(parser, base).score_evidence is False
    assert parse(parser, base + ["--score-evidence"]).score_evidence is True


def test_phase3_rebuild_score_evidence_defaults_off():
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "phase3_rebuild_scoring", repo / "scripts" / "phase3_rebuild.py")
    p3 = importlib.util.module_from_spec(spec)
    sys.modules["phase3_rebuild_scoring"] = p3
    spec.loader.exec_module(p3)

    parser, parse = _capture_parser(p3.main)
    base = ["--speech", "gwbush_2006"]
    assert parse(parser, base).score_evidence is False
    assert parse(parser, base + ["--score-evidence"]).score_evidence is True


def test_pack_builder_default_does_not_score():
    """The injection point itself: both defaults are the free, unscored path."""
    import inspect

    from truthbot import pipeline

    assert (inspect.signature(build_evidence_pack_v2)
            .parameters["scorer"].default is None)
    assert (inspect.signature(pipeline._build_v2_pack_builder)
            .parameters["score_evidence"].default is False)
