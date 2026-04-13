"""
Verdict scoring rubric.

Implements the truth-bot taxonomy:
  True / Mostly True / Misleading / Exaggerated / False / Unverifiable

Each verdict has a numeric score (0–10) for sorting/display, and the rubric
can accept a set of evidence items and return a recommended verdict + confidence
based on evidence counts, source tiers, and support ratios.

This module does NOT call any external APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from truthbot.models import Claim, Confidence, Evidence, SourceTier, Verdict, VerdictLabel

# ── Verdict metadata ──────────────────────────────────────────────────────────

# Score represents "accuracy" on a 0–10 scale for ranking/sorting
VERDICT_SCORES: dict[VerdictLabel, float] = {
    VerdictLabel.TRUE: 10.0,
    VerdictLabel.MOSTLY_TRUE: 7.5,
    VerdictLabel.MISLEADING: 4.0,
    VerdictLabel.EXAGGERATED: 5.0,
    VerdictLabel.FALSE: 0.0,
    VerdictLabel.UNVERIFIABLE: 5.0,  # neutral
}

# Human-readable descriptions of each verdict
VERDICT_DESCRIPTIONS: dict[VerdictLabel, str] = {
    VerdictLabel.TRUE: (
        "Accurate and supported by primary sources. The claim is factually correct."
    ),
    VerdictLabel.MOSTLY_TRUE: (
        "Accurate but missing nuance or important context that changes the picture."
    ),
    VerdictLabel.MISLEADING: (
        "Technically accurate framing that implies something false or creates a "
        "false impression."
    ),
    VerdictLabel.EXAGGERATED: (
        "Directionally correct but substantially overstated. The underlying fact "
        "exists but the degree or scope is inflated."
    ),
    VerdictLabel.FALSE: (
        "Contradicted by credible evidence. The claim is factually incorrect."
    ),
    VerdictLabel.UNVERIFIABLE: (
        "Insufficient evidence exists to confirm or deny this claim with confidence."
    ),
}

# Source tier trust weights (higher = more weight in scoring)
TIER_WEIGHTS: dict[SourceTier, float] = {
    SourceTier.GOVERNMENT: 1.0,
    SourceTier.WIRE: 0.85,
    SourceTier.ESTABLISHED: 0.70,
    SourceTier.ACADEMIC: 0.80,
    SourceTier.FACTCHECK: 0.75,
    SourceTier.OTHER: 0.40,
}


@dataclass
class VerdictScore:
    """The output of the rubric's scoring logic."""

    label: VerdictLabel
    confidence: Confidence
    numeric_score: float
    support_weight: float
    contradict_weight: float
    evidence_count: int
    best_tier: Optional[SourceTier]


class ScoringRubric:
    """
    Assign a verdict based on evidence quality and quantity.

    The rubric scores evidence by:
    1. Summing weighted support/contradict signals (weights = tier trust)
    2. Checking for high-tier contradictions (these override weak support)
    3. Mapping the support ratio to a verdict label
    4. Deriving confidence from evidence count + tier quality

    This is the heuristic fallback when LLM synthesis is unavailable.
    The VerificationEngine typically uses LLM synthesis instead, but the
    rubric is always available for testing and offline scoring.
    """

    def score(self, claim: Claim, evidence: list[Evidence]) -> VerdictScore:
        """
        Compute a verdict for a claim from its evidence set.

        Parameters
        ----------
        claim:
            The claim being scored.
        evidence:
            Evidence items retrieved for this claim.

        Returns
        -------
        VerdictScore
            Detailed scoring result.
        """
        if not evidence:
            return VerdictScore(
                label=VerdictLabel.UNVERIFIABLE,
                confidence=Confidence.LOW,
                numeric_score=VERDICT_SCORES[VerdictLabel.UNVERIFIABLE],
                support_weight=0.0,
                contradict_weight=0.0,
                evidence_count=0,
                best_tier=None,
            )

        support_weight = 0.0
        contradict_weight = 0.0
        best_tier: Optional[SourceTier] = None
        tier_order = list(SourceTier)  # index 0 = highest trust

        for ev in evidence:
            w = TIER_WEIGHTS.get(ev.source_tier, 0.4) * ev.relevance_score
            if ev.supports_claim is True:
                support_weight += w
            elif ev.supports_claim is False:
                contradict_weight += w
            # Track best (most trusted) source tier
            if best_tier is None or tier_order.index(ev.source_tier) < tier_order.index(best_tier):
                best_tier = ev.source_tier

        total = support_weight + contradict_weight
        support_ratio = support_weight / total if total > 0 else 0.5

        label = self._label_from_ratio(support_ratio, contradict_weight, evidence)
        confidence = self._confidence_from_evidence(evidence, total)
        numeric_score = VERDICT_SCORES[label]

        return VerdictScore(
            label=label,
            confidence=confidence,
            numeric_score=numeric_score,
            support_weight=support_weight,
            contradict_weight=contradict_weight,
            evidence_count=len(evidence),
            best_tier=best_tier,
        )

    def to_verdict(self, claim: Claim, score: VerdictScore, explanation: str = "") -> Verdict:
        """
        Convert a VerdictScore to a Verdict model.

        Parameters
        ----------
        claim:
            The claim this verdict is for.
        score:
            The computed VerdictScore.
        explanation:
            Optional human-readable explanation. Auto-generated if empty.
        """
        if not explanation:
            explanation = VERDICT_DESCRIPTIONS.get(score.label, "No explanation available.")

        return Verdict(
            claim_id=claim.id,
            label=score.label,
            confidence=score.confidence,
            explanation=explanation,
            support_count=round(score.support_weight),
            contradict_count=round(score.contradict_weight),
            primary_source_tier=score.best_tier,
        )

    # ── Private scoring logic ─────────────────────────────────────────────────

    def _label_from_ratio(
        self,
        support_ratio: float,
        contradict_weight: float,
        evidence: list[Evidence],
    ) -> VerdictLabel:
        """
        Map a support ratio to a verdict label.

        Rules (applied in order):
        1. High-trust contradiction (gov/wire) → automatic Mostly False or False
        2. ratio >= 0.85 → True
        3. ratio >= 0.65 → Mostly True
        4. ratio >= 0.45 → Misleading (ambiguous)
        5. ratio >= 0.25 → Exaggerated
        6. ratio < 0.25 → False
        7. No signal either way → Unverifiable
        """
        # Check for authoritative contradiction
        has_gov_contradict = any(
            e.supports_claim is False and e.source_tier in (SourceTier.GOVERNMENT, SourceTier.WIRE)
            for e in evidence
        )
        if has_gov_contradict and support_ratio < 0.5:
            return VerdictLabel.FALSE

        if support_ratio >= 0.85:
            return VerdictLabel.TRUE
        elif support_ratio >= 0.65:
            return VerdictLabel.MOSTLY_TRUE
        elif support_ratio >= 0.45:
            return VerdictLabel.MISLEADING
        elif support_ratio >= 0.25:
            return VerdictLabel.EXAGGERATED
        else:
            # No clear signal
            if contradict_weight < 0.1 and support_ratio == 0.5:
                return VerdictLabel.UNVERIFIABLE
            return VerdictLabel.FALSE

    def _confidence_from_evidence(
        self, evidence: list[Evidence], total_weight: float
    ) -> Confidence:
        """
        Derive confidence from evidence volume and tier quality.

        High: >= 3 weighted evidence items with at least one gov/wire source
        Medium: >= 2 evidence items or one high-tier item
        Low: only weak / few evidence items
        """
        count = len(evidence)
        has_high_tier = any(
            e.source_tier in (SourceTier.GOVERNMENT, SourceTier.WIRE, SourceTier.ACADEMIC)
            for e in evidence
        )

        if count >= 3 and has_high_tier:
            return Confidence.HIGH
        elif count >= 2 or (count >= 1 and has_high_tier):
            return Confidence.MEDIUM
        else:
            return Confidence.LOW

    @staticmethod
    def describe(label: VerdictLabel) -> str:
        """Return the human-readable description for a verdict label."""
        return VERDICT_DESCRIPTIONS.get(label, "")

    @staticmethod
    def numeric_score(label: VerdictLabel) -> float:
        """Return the numeric score (0–10) for a verdict label."""
        return VERDICT_SCORES.get(label, 5.0)
