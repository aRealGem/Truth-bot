"""Tests for the Layer C → publish bridge (PR-B).

Pure/offline: rows + packs are hand-built; no lane, no spend.
"""
from __future__ import annotations

import pytest

from truthbot.models import Confidence, VerdictBundle, VerdictLabel
from truthbot.verdict import bridge
from truthbot.verdict.evidence_pack import EvidencePack, PackItem
from truthbot.models import SourceTier


# ── fixtures ──────────────────────────────────────────────────────────────────

def _pack(sid, *items):
    """items: (pack_id, url, tier, snippet) tuples."""
    pis = [
        PackItem(pack_id=pid, source_name=f"src-{pid}", source_url=url,
                 tier=tier, snippet=snip, retrieved_at="2026-01-01T00:00:00+00:00",
                 sha256="deadbeef")
        for (pid, url, tier, snip) in items
    ]
    return EvidencePack(sid=sid, window=None, items=pis)


def _row(sid, status="resolved", verdict=None, confidence=None, citations=(),
         reasoning="", votes=None, **extra):
    row = {
        "sid": sid, "status": status, "verdict": verdict,
        "confidence": confidence, "citations": list(citations),
        "reasoning": reasoning, "votes": votes or {},
        "split": False, "escalated": False,
    }
    row.update(extra)
    return row


def _claim(sid, text="A claim.", **extra):
    c = {"sid": sid, "text": text}
    c.update(extra)
    return c


# ── helper units ──────────────────────────────────────────────────────────────

def test_confidence_from_float_thresholds():
    assert bridge.confidence_from_float(0.95) is Confidence.HIGH
    assert bridge.confidence_from_float(0.85) is Confidence.HIGH
    assert bridge.confidence_from_float(0.84) is Confidence.MEDIUM
    assert bridge.confidence_from_float(0.55) is Confidence.MEDIUM
    assert bridge.confidence_from_float(0.54) is Confidence.LOW
    assert bridge.confidence_from_float(0.0) is Confidence.LOW
    assert bridge.confidence_from_float(None) is Confidence.LOW


def test_strength_from_votes():
    assert bridge.strength_from_votes({"TRUE": 3}) == "strong"
    assert bridge.strength_from_votes({"TRUE": 2, "FALSE": 1}) == "weak"
    assert bridge.strength_from_votes({"TRUE": 2}) == "weak"
    assert bridge.strength_from_votes({"TRUE": 1}) == "single"
    assert bridge.strength_from_votes({"TRUE": 1, "FALSE": 1, "MISLEADING": 1}) == "none"
    assert bridge.strength_from_votes({}) == "none"
    assert bridge.strength_from_votes(None) == "none"


# ── resolved rows ───────────────────────────────────────────────────────────

def test_resolved_true_closed_book():
    row = _row("t1:0", verdict="TRUE", confidence=0.9, reasoning="matches BLS",
               votes={"TRUE": 3})
    out = bridge.bridge([row], [_claim("t1:0", "Unemployment fell.")])
    assert len(out.bundles) == 1
    b = out.bundles[0]
    assert isinstance(b, VerdictBundle)
    assert b.claim.text == "Unemployment fell."
    c = b.consensus
    assert c.consensus_label is VerdictLabel.TRUE
    assert c.consensus_verdict == "True"
    assert c.confidence is Confidence.HIGH
    assert c.consensus_strength == "strong"
    assert c.agreement is True
    assert c.explanation == "matches BLS"
    # coarse projection reuses the canonical tables
    assert c.coarse_lenient_label == "True"
    assert c.coarse_strict_label == "True"
    assert c.coarse_lenient_strength == "strong"
    # single reconciled model verdict, no evidence in closed-book
    assert len(b.model_verdicts) == 1
    assert b.model_verdicts[0].adapter_name == bridge.PANEL_ADAPTER
    assert b.model_verdicts[0].web_sources == []
    assert b.evidence_count == 0
    assert out.evidence["t1:0"] == []
    # agreeing_models property lines up (mv.label == consensus_label)
    assert b.agreeing_models == [bridge.PANEL_ADAPTER]
    assert b.dissenting_models == []


def test_resolved_open_book_wires_citations_and_evidence():
    pack = _pack(
        "t1:1",
        ("E1", "https://bls.gov/a", SourceTier.GOVERNMENT, "gov snippet"),
        ("E2", "https://ap.org/b", SourceTier.WIRE, "wire snippet"),
        ("E3", "https://blog.example/c", SourceTier.OTHER, "other snippet"),
    )
    row = _row("t1:1", verdict="FALSE", confidence=0.7, citations=["E1", "E2"],
               reasoning="contradicted by BLS", votes={"FALSE": 2, "MISLEADING": 1})
    out = bridge.bridge([row], [_claim("t1:1")], {"t1:1": pack})
    b = out.bundles[0]
    mv = b.model_verdicts[0]
    # only cited ids become web_sources, in citation order
    assert mv.web_sources == ["https://bls.gov/a", "https://ap.org/b"]
    assert b.consensus.consensus_label is VerdictLabel.FALSE
    assert b.consensus.confidence is Confidence.MEDIUM
    assert b.consensus.consensus_strength == "weak"
    assert b.consensus.agreement is False
    # coarse projection of FALSE
    assert b.consensus.coarse_lenient_label == "False"
    # evidence_count = full pack; evidence corpus = every pack item
    assert b.evidence_count == 3
    ev = out.evidence["t1:1"]
    assert [e.source_url for e in ev] == [
        "https://bls.gov/a", "https://ap.org/b", "https://blog.example/c"]
    assert ev[0].source_tier is SourceTier.GOVERNMENT
    assert all(e.claim_id == "t1:1" for e in ev)


def test_misleading_projection():
    row = _row("m:0", verdict="MISLEADING", confidence=0.6, votes={"MISLEADING": 3})
    out = bridge.bridge([row], [_claim("m:0")])
    c = out.bundles[0].consensus
    assert c.consensus_label is VerdictLabel.MISLEADING
    # Lenient and Strict both send MISLEADING → Falsey
    assert c.coarse_lenient_label == "Falsey"
    assert c.coarse_strict_label == "Falsey"


def test_crm114_override_annotated_and_final_label_wins():
    # stage-1 votes plurality was MISLEADING; CRM-114 flipped to FALSE.
    row = _row("c:0", verdict="FALSE", confidence=0.8,
               reasoning="core assertion contradicted",
               votes={"MISLEADING": 2, "FALSE": 1},
               crm114={"stage1": "MISLEADING", "final": "FALSE"})
    out = bridge.bridge([row], [_claim("c:0")])
    b = out.bundles[0]
    # final (resolved) label is authoritative, not the vote plurality
    assert b.consensus.consensus_label is VerdictLabel.FALSE
    assert b.model_verdicts[0].label is VerdictLabel.FALSE
    # override is surfaced in the explanation for audit
    assert "CRM-114: MISLEADING→FALSE" in b.consensus.explanation
    # strength still reflects the raw panel (2 agreed) → weak
    assert b.consensus.consensus_strength == "weak"


# ── non-resolved rows ───────────────────────────────────────────────────────

def test_disagreement_becomes_unverifiable_split():
    row = _row("d:0", status="disagreement", votes={"TRUE": 1, "FALSE": 1})
    out = bridge.bridge([row], [_claim("d:0")])
    c = out.bundles[0].consensus
    assert c.consensus_label is VerdictLabel.UNVERIFIABLE
    assert c.consensus_verdict == "Models split"
    assert c.coarse_lenient_label == "Models split"
    assert c.consensus_strength == "none"
    assert out.bundles[0].model_verdicts == []


def test_no_label_becomes_no_verdict():
    row = _row("n:0", status="no_label")
    out = bridge.bridge([row], [_claim("n:0")])
    c = out.bundles[0].consensus
    assert c.consensus_label is VerdictLabel.UNVERIFIABLE
    assert c.consensus_verdict == "No verdict"
    assert out.bundles[0].model_verdicts == []


# ── batch behavior ──────────────────────────────────────────────────────────

def test_bridge_preserves_order_and_tolerates_missing_claim():
    rows = [
        _row("a:0", verdict="TRUE", confidence=0.9, votes={"TRUE": 3}),
        _row("b:0", status="disagreement"),
    ]
    # only a claim for a:0 — b:0 must still bridge with a minimal claim
    out = bridge.bridge(rows, [_claim("a:0", "First.")])
    assert [b.consensus.claim_id for b in out.bundles] == ["a:0", "b:0"]
    assert out.bundles[0].claim.text == "First."
    assert out.bundles[1].claim.text == "(claim text unavailable)"


def test_transcript_id_and_speaker_wired_from_claim():
    row = _row("spx:2", verdict="TRUE", confidence=0.9, votes={"TRUE": 3})
    out = bridge.bridge(
        [row],
        [_claim("spx:2", "Text.", speaker="Speaker X", date_str="2026-03-04")],
    )
    b = out.bundles[0]
    assert b.claim.transcript_id == "spx"
    assert b.claim.speaker == "Speaker X"
    assert b.speaker == "Speaker X"
    assert b.date_str == "2026-03-04"


def test_out_of_contract_resolved_raises():
    # normalize() fails closed upstream, but the bridge guards too.
    row = _row("x:0", verdict="MOSTLY_TRUE", confidence=0.9, votes={"MOSTLY_TRUE": 3})
    with pytest.raises(ValueError):
        bridge.bridge([row], [_claim("x:0")])
