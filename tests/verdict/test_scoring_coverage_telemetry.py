"""Scoring-coverage telemetry (remediation v2 Phase A, A1) — offline, $0.

WHAT THIS EXISTS TO MAKE VISIBLE: the v2 evidence path never scores relevance
or stance. ``verify.relevance.score_evidence`` is the only writer of
``relevance_score`` / ``supports_claim``, and it is reachable ONLY from the
legacy v1 provider (``pipeline._build_open_book_provider``) and the R4 archive
retriever — ``build_evidence_pack_v2`` wires R1/R2/R3 straight into
``consolidate()``. Every v2 item therefore keeps the 0.5 pydantic default
(models.py Evidence.relevance_score) and whatever stance the retriever's own
JSON claimed, with retrievers.py mapping "context" → None.

Nothing recorded that. These tests pin the telemetry that now does: computed
per pack, carried on ``EvidencePack.scoring``, journaled next to excluded_fc /
quarantined, and recomputable from a STORED artifact so the lint can run over
runs that predate the field.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from truthbot.models import Evidence, SourceTier
from truthbot.verdict import publish_pipeline as pp
from truthbot.verdict.consolidator import (DEFAULT_RELEVANCE_SCORE,
                                           scoring_telemetry,
                                           scoring_telemetry_from_artifact)
from truthbot.verdict.evidence_pack_v2 import build_evidence_pack_v2
from truthbot.verdict.speech_context import register_speech_date

SID = "pytest_a1:0001"
UTT = datetime(2026, 2, 24).date()
register_speech_date("pytest_a1", UTT)


def _ev(url, *, tier=SourceTier.GOVERNMENT, supports=True, relevance=None,
        day=20):
    ev = Evidence(claim_id="", source_name="R", source_url=url,
                  source_tier=tier, snippet="stat page",
                  supports_claim=supports,
                  published_at=datetime(2026, 2, day, tzinfo=timezone.utc))
    if relevance is not None:
        ev.relevance_score = relevance
    return ev


class _Retriever:
    def __init__(self, label, shortlists):
        self.label = label
        self.shortlists = list(shortlists)

    def shortlist(self, claim_text, *, context="", utterance=None, window=None):
        return self.shortlists.pop(0) if self.shortlists else []


# ── the telemetry function itself ────────────────────────────────────────────

def test_default_relevance_constant_matches_the_pydantic_default():
    """If models.Evidence's default ever moves, the telemetry must move with
    it — otherwise 'unscored' silently starts counting as 'scored'."""
    assert Evidence.model_fields["relevance_score"].default == DEFAULT_RELEVANCE_SCORE


def test_telemetry_counts_default_relevance_and_the_stance_split():
    tel = scoring_telemetry([
        _ev("https://a/1", supports=True),                      # default 0.5
        _ev("https://a/2", supports=False),                     # default 0.5
        _ev("https://a/3", supports=None),                      # default 0.5
        _ev("https://a/4", supports=True, relevance=0.9),       # scored
    ])
    assert tel == {"items": 4, "relevance_scored": 1, "relevance_default": 3,
                   "stance_supports": 2, "stance_refutes": 1, "stance_null": 1}


def test_telemetry_treats_a_none_relevance_as_unscored_too():
    """PackItem.relevance_score defaults to None (no relevance layer ran at
    all) while Evidence defaults to 0.5 — both mean 'never scored'."""
    tel = scoring_telemetry([{"relevance_score": None, "supports_claim": None}])
    assert tel["relevance_default"] == 1 and tel["relevance_scored"] == 0


def test_telemetry_reads_dicts_and_objects_identically():
    """One implementation serves live packs and stored artifacts — if these
    could diverge, the lint over old runs would stop describing new ones."""
    objs = [_ev("https://a/1", supports=None), _ev("https://a/2", relevance=0.8)]
    dicts = [{"relevance_score": o.relevance_score,
              "supports_claim": o.supports_claim} for o in objs]
    assert scoring_telemetry(objs) == scoring_telemetry(dicts)


# ── the v2 pack path carries it ──────────────────────────────────────────────

def test_v2_pack_reports_zero_scored_relevance_on_every_item():
    """THE FINDING, pinned: a pack built the way production builds them has
    100% default relevance, because no scoring step exists on this path."""
    r1 = _Retriever("R1", [[_ev("https://bls.gov/a"),
                            _ev("https://bea.gov/b", supports=False)]])
    r2 = _Retriever("R2", [[_ev("https://apnews.com/c", tier=SourceTier.WIRE,
                                supports=None)]])
    pack = build_evidence_pack_v2(SID, "unemployment fell", (r1, r2))
    assert pack.scoring["items"] == len(pack.items) == 3
    assert pack.scoring["relevance_scored"] == 0
    assert pack.scoring["relevance_default"] == 3
    assert pack.scoring["stance_supports"] == 1
    assert pack.scoring["stance_refutes"] == 1
    assert pack.scoring["stance_null"] == 1


def test_v2_pack_telemetry_covers_the_capped_pack_not_the_pool():
    """The quota and the panel see the CAPPED pack; telemetry must describe
    the same set, or the fitness number would flatter a truncated pack."""
    many = [_ev(f"https://gov{i}.gov/p", supports=None) for i in range(14)]
    pack = build_evidence_pack_v2(SID, "c", (_Retriever("R1", [many, []]),))
    assert pack.scoring["items"] == len(pack.items) == 10   # PACK_CAP_V2
    assert len(pack.pool) == 14                             # pre-cap pool kept


def test_packs_journal_carries_scoring_additively(tmp_path):
    """Same additive shape as excluded_fc / quarantined: present when there is
    something to say, absent otherwise — old journal readers keep working."""
    r1 = _Retriever("R1", [[_ev("https://bls.gov/a"),
                            _ev("https://bea.gov/b", supports=False)]])
    pack = build_evidence_pack_v2(SID, "c", (r1,))
    path = tmp_path / "packs.jsonl"
    pp.append_packs_journal(path, SID, pack)
    rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert rec["scoring"]["items"] == 2
    assert rec["scoring"]["relevance_scored"] == 0

    # A v1-shaped pack (no telemetry) writes no key at all.
    from truthbot.verdict.evidence_pack import EvidencePack
    pp.append_packs_journal(path, "x:0002",
                            EvidencePack(sid="x:0002", window=None, items=[]))
    assert "scoring" not in json.loads(
        path.read_text(encoding="utf-8").splitlines()[1])


# ── the stored-artifact helper ───────────────────────────────────────────────

def test_artifact_helper_sums_packs_and_derives_the_rates():
    tel = scoring_telemetry_from_artifact({
        "s:1": [{"relevance_score": 0.5, "supports_claim": True},
                {"relevance_score": 0.5, "supports_claim": None}],
        "s:2": [{"relevance_score": 0.5, "supports_claim": False},
                {"relevance_score": 0.5, "supports_claim": None}],
    })
    assert tel["packs"] == 2 and tel["items"] == 4
    assert tel["stance_null"] == 2 and tel["stance_null_rate"] == 0.5
    assert tel["relevance_scored"] == 0 and tel["scored_rate"] == 0.0


def test_artifact_helper_is_empty_safe():
    tel = scoring_telemetry_from_artifact({})
    assert tel["items"] == 0
    assert tel["scored_rate"] == 0.0 and tel["stance_null_rate"] == 0.0
