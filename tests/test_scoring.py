"""TB-07t: Scoring rubric — consistency, source hierarchy, edge cases."""

from __future__ import annotations

import pytest

from truthbot.models import Claim, Confidence, Evidence, SourceTier, VerdictLabel
from truthbot.scoring.rubric import ScoringRubric, VERDICT_SCORES, TIER_WEIGHTS


@pytest.fixture
def rubric():
    return ScoringRubric()


def make_evidence(claim, supports, tier=SourceTier.ESTABLISHED, relevance=0.8):
    return Evidence(
        claim_id=claim.id,
        source_name="Test",
        source_url="https://test.com",
        source_tier=tier,
        snippet="Evidence snippet.",
        supports_claim=supports,
        relevance_score=relevance,
    )


class TestScoringRubric:
    def test_no_evidence_returns_unverifiable(self, rubric, sample_claim):
        score = rubric.score(sample_claim, [])
        assert score.label == VerdictLabel.UNVERIFIABLE
        assert score.confidence == Confidence.LOW

    def test_all_supporting_is_true(self, rubric, sample_claim):
        evidence = [make_evidence(sample_claim, True) for _ in range(3)]
        score = rubric.score(sample_claim, evidence)
        assert score.label in (VerdictLabel.TRUE, VerdictLabel.MOSTLY_TRUE)

    def test_all_contradicting_is_false(self, rubric, sample_claim):
        evidence = [make_evidence(sample_claim, False) for _ in range(3)]
        score = rubric.score(sample_claim, evidence)
        assert score.label in (VerdictLabel.FALSE, VerdictLabel.EXAGGERATED)

    def test_gov_contradiction_forces_false(self, rubric, sample_claim):
        evidence = [
            make_evidence(sample_claim, False, tier=SourceTier.GOVERNMENT),
            make_evidence(sample_claim, True, tier=SourceTier.OTHER),
        ]
        score = rubric.score(sample_claim, evidence)
        assert score.label == VerdictLabel.FALSE

    def test_confidence_high_with_gov_source(self, rubric, sample_claim):
        evidence = [
            make_evidence(sample_claim, True, tier=SourceTier.GOVERNMENT),
            make_evidence(sample_claim, True, tier=SourceTier.WIRE),
            make_evidence(sample_claim, True, tier=SourceTier.ESTABLISHED),
        ]
        score = rubric.score(sample_claim, evidence)
        assert score.confidence == Confidence.HIGH

    def test_confidence_low_with_single_weak_source(self, rubric, sample_claim):
        evidence = [make_evidence(sample_claim, True, tier=SourceTier.OTHER, relevance=0.3)]
        score = rubric.score(sample_claim, evidence)
        assert score.confidence == Confidence.LOW

    def test_best_tier_tracking(self, rubric, sample_claim):
        evidence = [
            make_evidence(sample_claim, True, tier=SourceTier.OTHER),
            make_evidence(sample_claim, True, tier=SourceTier.GOVERNMENT),
        ]
        score = rubric.score(sample_claim, evidence)
        assert score.best_tier == SourceTier.GOVERNMENT

    def test_numeric_scores_consistent(self):
        assert VERDICT_SCORES[VerdictLabel.TRUE] > VERDICT_SCORES[VerdictLabel.MOSTLY_TRUE]
        assert VERDICT_SCORES[VerdictLabel.FALSE] == 0.0
        assert VERDICT_SCORES[VerdictLabel.TRUE] == 10.0

    def test_tier_weights_ordered(self):
        assert TIER_WEIGHTS[SourceTier.GOVERNMENT] >= TIER_WEIGHTS[SourceTier.WIRE]
        assert TIER_WEIGHTS[SourceTier.WIRE] >= TIER_WEIGHTS[SourceTier.OTHER]

    def test_to_verdict(self, rubric, sample_claim):
        from truthbot.models import Verdict
        evidence = [make_evidence(sample_claim, True) for _ in range(3)]
        score = rubric.score(sample_claim, evidence)
        verdict = rubric.to_verdict(sample_claim, score)
        assert isinstance(verdict, Verdict)
        assert verdict.claim_id == sample_claim.id
        assert verdict.explanation

    def test_describe_returns_string(self, rubric):
        desc = rubric.describe(VerdictLabel.FALSE)
        assert isinstance(desc, str)
        assert len(desc) > 10

    def test_numeric_score_static(self):
        assert ScoringRubric.numeric_score(VerdictLabel.TRUE) == 10.0
        assert ScoringRubric.numeric_score(VerdictLabel.FALSE) == 0.0
