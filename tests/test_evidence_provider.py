"""EvidenceProvider factory and DataHoover stub."""

from __future__ import annotations

from truthbot.models import Claim, Evidence, SourceTier
from truthbot.verify.evidence_provider import (
    DataHooverEvidenceProvider,
    NoOpEvidenceProvider,
    build_evidence_provider,
)
from truthbot.verify.sources.base import SourceConnector


class _StubConn(SourceConnector):
    source_name = "Stub"
    tier = SourceTier.OTHER

    def search(self, claim: Claim):
        return [
            Evidence(
                claim_id=claim.id,
                source_name="x",
                source_url="https://a",
                source_tier=SourceTier.OTHER,
                snippet="hi",
            )
        ]


def test_build_none_returns_empty(sample_claim, monkeypatch):
    monkeypatch.delenv("TRUTHBOT_EVIDENCE_SOURCE", raising=False)
    p = build_evidence_provider(source="none", connectors=[_StubConn()])
    assert isinstance(p, NoOpEvidenceProvider)
    assert p.get_evidence(sample_claim) == []


def test_build_connectors_invokes_search(sample_claim):
    p = build_evidence_provider(source="connectors", connectors=[_StubConn()])
    ev = p.get_evidence(sample_claim)
    assert len(ev) == 1
    assert ev[0].snippet == "hi"


def test_datahoover_stub_always_empty(sample_claim, monkeypatch):
    monkeypatch.setenv("TRUTHBOT_DATAHOOVER_URL", "https://hoover.example")
    monkeypatch.setenv("TRUTHBOT_DATAHOOVER_MANIFEST", "/tmp/m.json")
    p = DataHooverEvidenceProvider()
    assert p.get_evidence(sample_claim) == []


def test_settings_evidence_source_explicit_beats_prefetch(monkeypatch):
    from truthbot.config import Settings

    monkeypatch.delenv("TRUTHBOT_PREFETCH_EVIDENCE", raising=False)
    monkeypatch.setenv("TRUTHBOT_EVIDENCE_SOURCE", "none")
    s = Settings()
    assert s.evidence_source == "none"

    monkeypatch.setenv("TRUTHBOT_PREFETCH_EVIDENCE", "1")
    assert s.evidence_source == "none"

    monkeypatch.setenv("TRUTHBOT_EVIDENCE_SOURCE", "connectors")
    assert s.evidence_source == "connectors"


def test_settings_legacy_prefetch_when_source_unset(monkeypatch):
    from truthbot.config import Settings

    monkeypatch.delenv("TRUTHBOT_EVIDENCE_SOURCE", raising=False)
    monkeypatch.delenv("TRUTHBOT_PREFETCH_EVIDENCE", raising=False)
    assert Settings().evidence_source == "none"

    monkeypatch.setenv("TRUTHBOT_PREFETCH_EVIDENCE", "1")
    assert Settings().evidence_source == "connectors"
