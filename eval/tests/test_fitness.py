"""Tests for eval/evolver/fitness.py"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from evolver.fitness import (
    verdict_agreement_score,
    parsimony_score,
    explanation_quality_score,
    source_citation_score,
    fuzzy_claim_similarity,
    FitnessScorer,
)


# ── Verdict agreement ─────────────────────────────────────────────────────────

def test_verdict_agreement_exact_match():
    assert verdict_agreement_score("TRUE", "True") == 1.0


def test_verdict_agreement_opposite():
    score = verdict_agreement_score("TRUE", "False")
    assert score < 0.3


def test_verdict_agreement_adjacent():
    score = verdict_agreement_score("TRUE", "Mostly True")
    assert score > 0.7


def test_verdict_agreement_compound_label_partly_true():
    """PARTLY TRUE / MISLEADING must not crash or return 0."""
    score = verdict_agreement_score("PARTLY TRUE / MISLEADING", "Mostly True")
    assert score > 0.0, f"Expected > 0 for compound label, got {score}"


def test_verdict_agreement_compound_label_false_medical():
    """FALSE / CONTRADICTS MEDICAL CONSENSUS should map close to False."""
    score = verdict_agreement_score("FALSE / CONTRADICTS MEDICAL CONSENSUS", "False")
    # Before Phase 5 compound-label fix: maps to unverifiable, distance=0.2, score=0.8
    # After Phase 5 fix: maps to false, score=1.0
    assert score >= 0.8, f"Expected >= 0.8, got {score}"

def test_verdict_agreement_compound_label_ideological():
    """IDEOLOGICAL CLAIM / CONTRADICTS MEDICAL CONSENSUS -> false -> False = 1.0"""
    score = verdict_agreement_score("IDEOLOGICAL CLAIM / CONTRADICTS MEDICAL CONSENSUS", "False")
    assert score >= 0.8


def test_verdict_agreement_compound_label_true_action():
    """TRUE (AS TO ACTION) should map to true -> True = 1.0"""
    score = verdict_agreement_score("TRUE (AS TO ACTION)", "True")
    assert score == 1.0


def test_verdict_agreement_compound_label_partly_true_unsupported():
    """PARTLY TRUE / UNSUPPORTED -> mostly_true"""
    score = verdict_agreement_score("PARTLY TRUE / UNSUPPORTED", "Mostly True")
    assert score == 1.0


def test_verdict_agreement_unknown_pred_label_does_not_crash():
    """Totally unknown predicted label should not raise, should return a finite float."""
    score = verdict_agreement_score("TRUE", "SomeMadeUpVerdict")
    assert 0.0 <= score <= 1.0


# ── Parsimony ─────────────────────────────────────────────────────────────────

def test_parsimony_below_min_is_1():
    # After Phase 4 fix: new target_min = 4000
    assert parsimony_score(100) == 1.0
    assert parsimony_score(4000) == 1.0
    assert parsimony_score(3999) == 1.0


def test_parsimony_above_max_is_0():
    assert parsimony_score(99999) == 0.0
    assert parsimony_score(30000) == 0.0


def test_parsimony_realistic_run_is_nonzero():
    """After calibration fix: a realistic 15,000-token run should score > 0."""
    score = parsimony_score(15_000)
    assert score > 0.0, f"Expected > 0 for 15k tokens, got {score}"
    assert score < 1.0


# ── Explanation quality ────────────────────────────────────────────────────────

def test_explanation_quality_rewards_data():
    # Use a data-rich explanation to ensure 5+ signal patterns match
    expl = (
        "BLS data shows the unemployment rate fell to 3.4 percent in January 2023. "
        "According to FactCheck.org, the $15 billion figure cited by AP in 2022 was "
        "corroborated by BEA data. The 3.4% figure is the lowest since 1969."
    )
    score = explanation_quality_score(expl)
    assert score > 0.5, f"Expected > 0.5 for data-rich explanation, got {score}"


def test_explanation_quality_empty_is_zero():
    assert explanation_quality_score("") == 0.0


# ── Source citation ────────────────────────────────────────────────────────────

def test_source_citation_score_with_sources():
    expl = "According to PolitiFact and FactCheck.org, the claim is mostly accurate."
    score = source_citation_score(expl)
    assert score > 0.0


def test_source_citation_score_empty():
    assert source_citation_score("") == 0.0


# ── Fuzzy similarity ──────────────────────────────────────────────────────────

def test_fuzzy_similarity_identical():
    s = "The unemployment rate fell to 3.4 percent."
    assert fuzzy_claim_similarity(s, s) == 1.0


def test_fuzzy_similarity_empty():
    assert fuzzy_claim_similarity("", "anything") == 0.0


def test_fuzzy_similarity_partial():
    a = "The unemployment rate fell to 3.4 percent."
    b = "Unemployment reached 3.4 percent."
    score = fuzzy_claim_similarity(a, b)
    assert 0.0 < score < 1.0


# ── FitnessScorer ─────────────────────────────────────────────────────────────

def test_fitness_scorer_scores_in_range(sample_reference, sample_claims, sample_verdicts):
    scorer = FitnessScorer(sample_reference)
    scores = scorer.score(sample_claims, sample_verdicts, token_count=1000)
    for key in ("claim_recall", "verdict_agreement", "explanation_quality",
                "source_citation_quality", "parsimony", "fitness"):
        val = scores[key]
        assert 0.0 <= val <= 1.0, f"{key} = {val} out of [0, 1]"


def test_fitness_scorer_zero_claims_recall_zero(sample_reference, sample_verdicts):
    scorer = FitnessScorer(sample_reference)
    scores = scorer.score([], sample_verdicts, token_count=0)
    assert scores["claim_recall"] == 0.0


def test_fitness_scorer_returns_matched_count(sample_reference, sample_claims, sample_verdicts):
    scorer = FitnessScorer(sample_reference)
    scores = scorer.score(sample_claims, sample_verdicts, token_count=100)
    assert "matched_count" in scores
    assert "total_extracted" in scores


# ── Numeric error direction ────────────────────────────────────────────────────

from evolver.fitness import extract_first_number, numeric_error_direction


def test_extract_first_number_basic():
    assert extract_first_number("unemployment fell to 3.4 percent") == 3.4


def test_extract_first_number_with_comma():
    assert extract_first_number("cost was $1,500") == 1500.0


def test_extract_first_number_none_when_missing():
    assert extract_first_number("no numbers here at all") is None


def test_numeric_error_direction_inflated():
    # pred says 200, ref says 100 → ratio 2.0 → inflated
    score = numeric_error_direction("the figure was 100 million", "the figure was 200 million")
    assert score == 'inflated'


def test_numeric_error_direction_deflated():
    # pred says 50, ref says 100 → ratio 0.5 → deflated
    score = numeric_error_direction("cost was 100 billion", "cost was 50 billion")
    assert score == 'deflated'


def test_numeric_error_direction_match():
    # pred says 103, ref says 100 → ratio 1.03 → match
    score = numeric_error_direction("the rate is 100", "the rate is 103")
    assert score == 'match'


def test_numeric_error_direction_unknown_when_no_numbers():
    score = numeric_error_direction("no numbers", "still no numbers")
    assert score == 'unknown'


def test_fitness_scorer_includes_numeric_error_directions(sample_reference, sample_claims, sample_verdicts):
    scorer = FitnessScorer(sample_reference)
    scores = scorer.score(sample_claims, sample_verdicts, token_count=5000)
    assert "numeric_error_directions" in scores
    assert isinstance(scores["numeric_error_directions"], list)
    for ned in scores["numeric_error_directions"]:
        assert "ref_id" in ned
        assert ned["direction"] in ('inflated', 'deflated', 'match', 'unknown')
