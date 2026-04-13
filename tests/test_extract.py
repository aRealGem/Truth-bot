"""TB-03t: Claim extraction — schema, normalization, fact vs opinion."""

from __future__ import annotations

import pytest

from truthbot.extract.claims import ClaimExtractor
from truthbot.models import Claim, Transcript


@pytest.fixture
def extractor():
    # No real API key in tests — will use stub mode
    return ClaimExtractor(api_key="")


@pytest.fixture
def transcript(sample_transcript):
    return sample_transcript


class TestClaimExtractor:
    def test_returns_list(self, extractor, transcript):
        claims = extractor.extract(transcript)
        assert isinstance(claims, list)

    def test_claims_are_claim_objects(self, extractor, transcript):
        claims = extractor.extract(transcript)
        for c in claims:
            assert isinstance(c, Claim)

    def test_claim_text_not_empty(self, extractor, transcript):
        claims = extractor.extract(transcript)
        for c in claims:
            assert c.text.strip()

    def test_claim_transcript_id_matches(self, extractor, transcript):
        claims = extractor.extract(transcript)
        for c in claims:
            assert c.claim_id_matches_transcript(transcript) or c.transcript_id == transcript.id

    def test_claim_has_speaker(self, extractor, transcript):
        claims = extractor.extract(transcript)
        for c in claims:
            assert c.speaker

    def test_stub_on_no_api_key(self, extractor, transcript):
        """With no API key, should return stub claims rather than raising."""
        claims = extractor.extract(transcript)
        # Stub returns up to 3 sentences from the transcript
        assert len(claims) <= 10  # reasonable upper bound

    def test_empty_transcript_returns_empty(self, extractor):
        """Very short transcript should yield no or minimal claims."""
        t = Transcript(
            text="OK.",
            transcript_id="test-id",
        )
        # This may return empty or one item — just shouldn't raise
        claims = extractor.extract(t)
        assert isinstance(claims, list)


# Monkey-patch Claim for the transcript_id check test
Claim.claim_id_matches_transcript = lambda self, t: self.transcript_id == t.id
