"""Open-book wiring — adjudicate.build_items payload shape + open-book prompts.

Offline: build_items only touches the (fake) provider, never a live lane."""
from __future__ import annotations

from truthbot.models import Claim, Evidence, SourceTier
from truthbot.verify.evidence_provider import EvidenceProvider
from truthbot.verify.sources.base import TimeWindow
from truthbot.verdict import adjudicator, prompts


class _FakeProvider(EvidenceProvider):
    def get_evidence(self, claim: Claim, *, window: TimeWindow = None):
        return [Evidence(claim_id=claim.id, source_name="BLS",
                         source_url="https://bls.gov/x", source_tier=SourceTier.GOVERNMENT,
                         snippet="unemployment 4.4% in Feb 2026")]


_CLAIMS = [{"sid": "trump_2026:3", "text": "Unemployment is at a record low.", "context": ""}]


def test_build_items_closed_book_has_empty_pack_ids():
    items, packs = adjudicator.build_items(_CLAIMS)   # no provider → closed-book
    assert packs == {}
    p = items[0]["payload"]
    assert p["evidence_pack_ids"] == [] and "evidence" not in p
    assert "TEMPORAL CONTEXT" in p["context"]          # preamble still prefixed


def test_build_items_open_book_injects_pack_and_ids():
    items, packs = adjudicator.build_items(_CLAIMS, evidence_provider=_FakeProvider())
    p = items[0]["payload"]
    assert p["evidence_pack_ids"] == ["E1"]
    assert p["evidence"][0]["id"] == "E1"
    assert p["evidence"][0]["tier"] == "Government"
    assert "unemployment" in p["evidence"][0]["snippet"]
    # pack retained for telemetry, and citations are a subset of the pack ids
    assert packs["trump_2026:3"].ids == ["E1"]


def test_open_book_prompts_are_speaker_blind_and_cite_by_id():
    for role in ("proposer", "critic", "arbiter"):
        tmpl = prompts.OPEN_BOOK_PROMPTS[role]
        assert "speaker" not in tmpl.lower()
        assert '"citations"' in tmpl and "provided ids" in tmpl
        # open-book must NOT carry the closed-book "cite nothing" instruction
        assert "citations must be []" not in tmpl
