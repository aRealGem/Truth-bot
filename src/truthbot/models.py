"""
Core Pydantic data models for truth-bot.

All data that flows through the pipeline lives in one of these types.
Keeping models centralized prevents circular imports and makes serialization
(JSON, disk cache) straightforward.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────


class VerdictLabel(str, Enum):
    """The six possible verdict labels in the truth-bot rubric."""

    TRUE = "True"
    MOSTLY_TRUE = "Mostly True"
    MISLEADING = "Misleading"
    EXAGGERATED = "Exaggerated"
    FALSE = "False"
    UNVERIFIABLE = "Unverifiable"


class Confidence(str, Enum):
    """Confidence in the verdict: how much evidence supports the rating."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class SourceTier(str, Enum):
    """
    Trust hierarchy for evidence sources (descending trust).

    Used by the scoring rubric to weight conflicting evidence.
    """

    GOVERNMENT = "Government"       # BLS, FRED, CBO, Census, etc.
    WIRE = "Wire"                   # AP, Reuters
    ESTABLISHED = "Established"     # NYT, WaPo, BBC, etc.
    ACADEMIC = "Academic"           # Peer-reviewed, major NGOs
    FACTCHECK = "FactCheck"         # PolitiFact, FactCheck.org, Snopes
    OTHER = "Other"


# ── Core models ───────────────────────────────────────────────────────────────


class Transcript(BaseModel):
    """A raw speech or statement ingested into the pipeline."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., description="Full normalized transcript text")
    speaker: str = Field(default="Unknown", description="Name or title of the speaker")
    date: Optional[datetime] = Field(None, description="Date of the speech/statement")
    venue: Optional[str] = Field(None, description="Location or context (e.g. 'State of the Union')")
    source_url: Optional[str] = Field(None, description="URL where the transcript was obtained")
    word_count: int = Field(default=0)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Transcript text cannot be empty")
        return v.strip()

    def model_post_init(self, __context: Any) -> None:
        if self.word_count == 0:
            self.word_count = len(self.text.split())


class Claim(BaseModel):
    """
    An atomic, verifiable factual claim extracted from a transcript.

    A claim should be a single, specific assertion that can in principle
    be checked against evidence — not an opinion, not a value judgment.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    transcript_id: str = Field(..., description="ID of the source transcript")
    text: str = Field(..., description="The claim as a self-contained sentence")
    speaker: str = Field(default="Unknown")
    context: Optional[str] = Field(None, description="Surrounding text for context")
    category: Optional[str] = Field(
        None,
        description="Subject category e.g. 'economy', 'immigration', 'healthcare'",
    )
    is_checkable: bool = Field(
        True, description="False if LLM judged it an opinion or value statement"
    )
    extracted_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Claim text cannot be empty")
        return v.strip()


class Evidence(BaseModel):
    """A single piece of evidence retrieved to evaluate a claim."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_id: str = Field(..., description="The claim this evidence relates to")
    source_name: str = Field(..., description="Publication or dataset name")
    source_url: str = Field(..., description="Direct URL to the evidence")
    source_tier: SourceTier = Field(SourceTier.OTHER)
    snippet: str = Field(..., description="Relevant excerpt or summary")
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)
    supports_claim: Optional[bool] = Field(
        None,
        description="True=supports, False=contradicts, None=ambiguous",
    )
    relevance_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="0–1 relevance to the claim",
    )


class Verdict(BaseModel):
    """The final verdict on a single claim."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_id: str = Field(...)
    label: VerdictLabel = Field(...)
    confidence: Confidence = Field(...)
    explanation: str = Field(..., description="Human-readable explanation of the verdict")
    evidence_ids: list[str] = Field(default_factory=list)
    scored_at: datetime = Field(default_factory=datetime.utcnow)
    # Raw scores used internally by the rubric
    support_count: int = Field(default=0, description="Number of supporting evidence items")
    contradict_count: int = Field(default=0, description="Number of contradicting evidence items")
    primary_source_tier: Optional[SourceTier] = Field(
        None,
        description="Highest-trust tier among the evidence",
    )


class Report(BaseModel):
    """
    A complete fact-check report for one transcript.

    Contains the original transcript, all extracted claims, the evidence
    gathered, and the final verdicts — ready to publish.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    transcript: Transcript
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    verdicts: list[Verdict] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None
    report_url: Optional[str] = None
    bluesky_thread_url: Optional[str] = None
    rss_feed_url: Optional[str] = None

    # Convenience counts
    @property
    def total_claims(self) -> int:
        return len(self.claims)

    @property
    def checkable_claims(self) -> int:
        return sum(1 for c in self.claims if c.is_checkable)

    @property
    def verdict_summary(self) -> dict[str, int]:
        """Count of each verdict label across all verdicts."""
        counts: dict[str, int] = {label.value: 0 for label in VerdictLabel}
        for v in self.verdicts:
            counts[v.label.value] += 1
        return counts

    def verdict_for(self, claim_id: str) -> Optional[Verdict]:
        """Look up the verdict for a specific claim ID."""
        for v in self.verdicts:
            if v.claim_id == claim_id:
                return v
        return None
