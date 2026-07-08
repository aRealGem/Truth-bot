"""Layer B tests: verdict parse/contract (parse_verdict + normalize) and pipeline
routing. All offline — the live P→C→A panel is gated behind -m live."""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))            # hydramind (repo root)
sys.path.insert(0, str(_ROOT / "src"))    # truthbot

from hydramind import ItemResult, StrategyResultKind
from truthbot.verdict import adjudicator, pipeline, prompts


def _resolved(sid, verdict, citations=(), conf=0.8, **ag):
    agreement = {"votes": {verdict: 2}, "split": False, "escalated": False}
    agreement.update(ag)
    return ItemResult(sid, StrategyResultKind.RESOLVED,
                      {"verdict": verdict, "citations": list(citations), "confidence": conf},
                      agreement)


# ── parse_verdict: per-call, fail-safe ────────────────────────────────────────

def test_parse_verdict_coerces_and_casts():
    assert adjudicator.parse_verdict({"verdict": "true", "confidence": "0.9"}) == {
        "verdict": "TRUE", "confidence": 0.9, "citations": []}
    # unrecognized label is a fail-SAFE UNVERIFIABLE vote, not a raise
    assert adjudicator.parse_verdict({"verdict": "kinda-true"})["verdict"] == "UNVERIFIABLE"
    assert adjudicator.parse_verdict({"confidence": None})["confidence"] is None
    assert adjudicator.parse_verdict({"verdict": "FALSE", "citations": ["x"]})["citations"] == ["x"]


# ── normalize: per-item, fail-closed ──────────────────────────────────────────

def test_normalize_resolved():
    row = adjudicator.normalize(_resolved("c1", "TRUE"))
    assert row["status"] == "resolved" and row["verdict"] == "TRUE"
    assert row["citations"] == [] and row["confidence"] == 0.8
    assert row["sid"] == "c1"


def test_normalize_bad_verdict_fails_closed():
    with pytest.raises(ValueError):
        adjudicator.normalize(_resolved("c1", "PROBABLY_TRUE"))


def test_normalize_closed_book_citation_is_i4_violation():
    with pytest.raises(ValueError):
        adjudicator.normalize(_resolved("c1", "TRUE", citations=["http://x"]))
    # open-book: the same row normalizes fine and keeps the citation
    row = adjudicator.normalize(_resolved("c1", "TRUE", citations=["http://x"]), closed_book=False)
    assert row["citations"] == ["http://x"]


def test_normalize_disagreement_carries_status_not_verdict():
    tie = ItemResult("c2", StrategyResultKind.DISAGREEMENT_FLAGGED,
                     {"labels": {"TRUE": 1, "FALSE": 1}},
                     {"votes": {"TRUE": 1, "FALSE": 1}, "split": True})
    row = adjudicator.normalize(tie)
    assert row["status"] == "disagreement" and row["verdict"] is None
    assert row["split"] is True and row["votes"] == {"TRUE": 1, "FALSE": 1}


def test_normalize_no_labels():
    empty = ItemResult("c3", StrategyResultKind.DISAGREEMENT_FLAGGED, {"reason": "no_labels"}, {})
    assert adjudicator.normalize(empty)["status"] == "no_label"


# ── pipeline: routing, no-lane parking ────────────────────────────────────────

def test_pipeline_parks_when_no_lane():
    claims = [{"sid": "c1", "text": "Inflation fell to 1.7%.", "context": ""}]
    res = pipeline.run_layer_b(claims, verdict_fn=None)
    assert res.n_claims == 1
    assert res.verdicts[0]["status"] == "needs_verdict" and res.verdicts[0]["verdict"] is None


def test_pipeline_routes_with_fake_verdict_fn():
    claims = [{"sid": "c1", "text": "A.", "context": ""},
              {"sid": "c2", "text": "B.", "context": ""}]

    def fake(rows):
        return [adjudicator.normalize(_resolved(c["sid"], "FALSE")) for c in rows]

    res = pipeline.run_layer_b(claims, verdict_fn=fake)
    assert {r["sid"] for r in res.verdicts} == {"c1", "c2"}
    assert all(r["status"] == "resolved" and r["verdict"] == "FALSE" for r in res.verdicts)


def test_pipeline_empty():
    assert pipeline.run_layer_b([]).verdicts == []


# ── prompts: I3 speaker-blindness (lint at import) + closed-book contract ──────

def test_prompts_speaker_blind_and_contract():
    # importing prompts already lint-checked every seat prompt (I3); assert the
    # closed-book contract is present in each.
    for role in ("proposer", "critic", "arbiter"):
        assert "citations must be []" in prompts.PROMPTS[role]
        assert "speaker" not in prompts.PROMPTS[role].lower()
    assert "UNVERIFIABLE" in prompts.VERDICTS
