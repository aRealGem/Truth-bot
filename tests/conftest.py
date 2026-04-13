"""
Shared pytest fixtures for truth-bot tests.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Make sure API keys are set to dummy values so Settings doesn't blow up
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("BRAVE_API_KEY", "test-brave-key")

from truthbot.models import (
    Claim,
    Confidence,
    Evidence,
    Report,
    SourceTier,
    Transcript,
    Verdict,
    VerdictLabel,
)


@pytest.fixture
def sample_transcript() -> Transcript:
    return Transcript(
        text=(
            "Unemployment is at a 50-year low. "
            "We've created 10 million new jobs in the last two years. "
            "The deficit has been cut in half."
        ),
        speaker="Test Politician",
        date=datetime(2025, 1, 20),
        venue="Test Venue",
    )


@pytest.fixture
def sample_claim(sample_transcript) -> Claim:
    return Claim(
        transcript_id=sample_transcript.id,
        text="Unemployment is at a 50-year low.",
        speaker="Test Politician",
        context="Unemployment is at a 50-year low.",
        category="economy",
        is_checkable=True,
    )


@pytest.fixture
def sample_evidence(sample_claim) -> Evidence:
    return Evidence(
        claim_id=sample_claim.id,
        source_name="BLS",
        source_url="https://bls.gov/news.release/empsit.nr0.htm",
        source_tier=SourceTier.GOVERNMENT,
        snippet="The unemployment rate fell to 3.4% in January 2023, the lowest since 1969.",
        supports_claim=True,
        relevance_score=0.95,
    )


@pytest.fixture
def contradicting_evidence(sample_claim) -> Evidence:
    return Evidence(
        claim_id=sample_claim.id,
        source_name="BLS",
        source_url="https://bls.gov/news.release/empsit.nr0.htm",
        source_tier=SourceTier.GOVERNMENT,
        snippet="Unemployment rate is 4.1%, higher than previous year.",
        supports_claim=False,
        relevance_score=0.90,
    )


@pytest.fixture
def sample_verdict(sample_claim) -> Verdict:
    return Verdict(
        claim_id=sample_claim.id,
        label=VerdictLabel.MOSTLY_TRUE,
        confidence=Confidence.HIGH,
        explanation="The claim is approximately correct; the rate hit a low in 2023.",
        support_count=3,
        contradict_count=0,
        primary_source_tier=SourceTier.GOVERNMENT,
    )


@pytest.fixture
def sample_report(sample_transcript, sample_claim, sample_evidence, sample_verdict) -> Report:
    return Report(
        transcript=sample_transcript,
        claims=[sample_claim],
        evidence=[sample_evidence],
        verdicts=[sample_verdict],
    )


@pytest.fixture
def tmp_dir() -> Path:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
