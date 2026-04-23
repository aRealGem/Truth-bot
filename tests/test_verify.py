"""TB-05t: Verification engine — verdict schema, stubs, confidence scoring."""

from __future__ import annotations

import pytest

from truthbot.models import (
    Claim,
    Confidence,
    ConsensusVerdict,
    Evidence,
    SourceTier,
    Verdict,
    VerdictLabel,
)
from truthbot.verify.engine import VerificationEngine
from truthbot.verify.sources.base import SourceConnector


class MockConnector(SourceConnector):
    """Returns a fixed list of evidence for testing."""

    source_name = "MockSource"
    tier = SourceTier.WIRE

    def __init__(self, evidence_list=None):
        super().__init__()
        self._evidence = evidence_list or []

    def search(self, claim):
        return [
            Evidence(
                claim_id=claim.id,
                source_name=self.source_name,
                source_url="https://example.com",
                source_tier=self.tier,
                snippet=ev_text,
                supports_claim=supports,
            )
            for ev_text, supports in self._evidence
        ]


class ErrorConnector(SourceConnector):
    """Always raises an exception."""

    source_name = "ErrorSource"
    tier = SourceTier.OTHER

    def search(self, claim):
        raise RuntimeError("Simulated connector failure")


class ExplodingConnector(SourceConnector):
    """Fails if search is invoked — used to assert prefetch stays stubbed."""

    source_name = "ExplodingSource"
    tier = SourceTier.OTHER

    def search(self, claim):
        raise AssertionError("connector search must not run when prefetch is stubbed")


class TestVerificationEngine:
    def test_verify_returns_tuple(self, sample_claim):
        engine = VerificationEngine(connectors=[], api_key="")
        evidence, verdict = engine.verify(sample_claim)
        assert isinstance(evidence, list)
        assert isinstance(verdict, ConsensusVerdict)

    def test_verdict_has_required_fields(self, sample_claim):
        engine = VerificationEngine(connectors=[], api_key="")
        _, verdict = engine.verify(sample_claim)
        assert verdict.claim_id == sample_claim.id
        assert isinstance(verdict.consensus_label, VerdictLabel)
        assert isinstance(verdict.confidence, Confidence)
        assert verdict.explanation

    def test_no_connectors_returns_unverifiable(self, sample_claim):
        engine = VerificationEngine(connectors=[], api_key="")
        _, verdict = engine.verify(sample_claim)
        assert verdict.consensus_label == VerdictLabel.UNVERIFIABLE

    def test_connector_error_is_handled(self, sample_claim, monkeypatch):
        monkeypatch.setenv("TRUTHBOT_EVIDENCE_SOURCE", "connectors")
        engine = VerificationEngine(connectors=[ErrorConnector()], api_key="")
        # Should not raise — errors in connectors are caught
        evidence, verdict = engine.verify(sample_claim)
        assert isinstance(verdict, ConsensusVerdict)

    def test_gather_evidence_stubbed_by_default(self, sample_claim, monkeypatch):
        monkeypatch.delenv("TRUTHBOT_PREFETCH_EVIDENCE", raising=False)
        monkeypatch.delenv("TRUTHBOT_EVIDENCE_SOURCE", raising=False)
        engine = VerificationEngine(connectors=[ExplodingConnector()], api_key="")
        assert engine._gather_evidence(sample_claim) == []

    def test_gather_evidence_calls_connectors_when_prefetch_on(self, sample_claim, monkeypatch):
        monkeypatch.delenv("TRUTHBOT_EVIDENCE_SOURCE", raising=False)
        monkeypatch.setenv("TRUTHBOT_PREFETCH_EVIDENCE", "1")
        engine = VerificationEngine(
            connectors=[MockConnector([("A snippet.", True)])],
            api_key="",
        )
        ev = engine._gather_evidence(sample_claim)
        assert len(ev) == 1
        assert ev[0].snippet == "A snippet."

    def test_verify_many_returns_all(self, sample_claim):
        engine = VerificationEngine(connectors=[], api_key="")
        claim2 = Claim(
            transcript_id=sample_claim.transcript_id,
            text="The deficit has been cut in half.",
            speaker=sample_claim.speaker,
            is_checkable=True,
        )
        results = engine.verify_many([sample_claim, claim2])
        assert len(results) == 2

    def test_uncheckable_claim_gets_unverifiable(self):
        engine = VerificationEngine(connectors=[], api_key="")
        claim = Claim(
            transcript_id="t1",
            text="America is the greatest country in the world.",
            speaker="Test",
            is_checkable=False,
        )
        results = engine.verify_many([claim])
        assert len(results) == 1
        _, evidence, verdict = results[0]
        assert verdict.consensus_label == VerdictLabel.UNVERIFIABLE
        assert evidence == []

    def test_stub_verdict_is_unverifiable(self, sample_claim):
        engine = VerificationEngine(connectors=[], api_key="")
        verdict = engine._stub_verdict(sample_claim, [])
        assert verdict.label == VerdictLabel.UNVERIFIABLE
        assert verdict.confidence == Confidence.LOW
