"""Tests for the 5-bucket coarse-axis verdict-agreement metric.

The scorer accepts ``axis="coarse_lenient"`` / ``"coarse_strict"`` to score
verdict agreement on the 5-bucket Truthy scale (True / Truthy / Unverifiable
/ Falsey / False) instead of the historical 6-bucket fine scale. The
projections mirror ``LENIENT_PROJECTION`` / ``STRICT_PROJECTION`` in
``src/truthbot/verify/engine.py``:

  * Lenient: Mostly True + Exaggerated → Truthy, Misleading → Falsey
  * Strict:  Mostly True → Truthy, Exaggerated + Misleading → Falsey

The reference set (``eval/sotu-2026/reference.json``) is already roughly
coarse-axis-shaped, so coarse-axis scoring removes Mostly-True vs
Exaggerated label drift between the four-adapter consensus and the
human-curated reference.

Coverage:
  * Identity (each axis individually)
  * Lenient lifts MT/Excg agreement; Strict separates them
  * Symmetric distance and max-distance preserved across axes
  * "Models split" production state is treated as unverifiable on every axis
  * ``axis="fine"`` round-trips byte-identically to the pre-axis-param form
  * Unknown axis raises (fail loud, not silent)
  * ``FitnessScorer.score(axis=...)`` returns the requested axis label and
    actually re-scores verdict_agreement (not just decorates the dict)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from evolver.fitness import (  # noqa: E402
    FitnessScorer,
    _COARSE_LABEL_ORDER,
    _LABEL_ORDER,
    _LENIENT_PROJECTION_FOR_SCORING,
    _STRICT_PROJECTION_FOR_SCORING,
    _project_for_axis,
    verdict_agreement_score,
    verdict_distance,
)


# ── Mapping invariants ────────────────────────────────────────────────────────


def test_lenient_and_strict_share_neutral_buckets() -> None:
    """True / Mostly True / False / Unverifiable map identically across lenses."""
    for label in ("true", "mostly_true", "false", "unverifiable"):
        assert (
            _LENIENT_PROJECTION_FOR_SCORING[label]
            == _STRICT_PROJECTION_FOR_SCORING[label]
        )


def test_lenient_vs_strict_diverge_only_on_exaggerated() -> None:
    assert _LENIENT_PROJECTION_FOR_SCORING["exaggerated"] == "truthy"
    assert _STRICT_PROJECTION_FOR_SCORING["exaggerated"] == "falsey"
    # Misleading is Falsey for both.
    assert _LENIENT_PROJECTION_FOR_SCORING["misleading"] == "falsey"
    assert _STRICT_PROJECTION_FOR_SCORING["misleading"] == "falsey"


def test_coarse_label_set_is_exactly_the_truthy_scale() -> None:
    expected = {"true", "truthy", "unverifiable", "falsey", "false"}
    assert set(_LENIENT_PROJECTION_FOR_SCORING.values()) == expected
    assert set(_STRICT_PROJECTION_FOR_SCORING.values()) == expected
    assert set(_COARSE_LABEL_ORDER) == expected


def test_project_for_axis_fine_is_identity() -> None:
    for label in _LABEL_ORDER:
        assert _project_for_axis(label, "fine") == label


def test_project_for_axis_unknown_axis_raises() -> None:
    with pytest.raises(ValueError):
        _project_for_axis("true", "coarse_galaxybrain")


# ── verdict_distance / verdict_agreement_score ────────────────────────────────


@pytest.mark.parametrize("axis", ["fine", "coarse_lenient", "coarse_strict"])
def test_identity_distance_zero_on_every_axis(axis: str) -> None:
    """Same label on both sides → distance 0, agreement 1.0, regardless of axis."""
    assert verdict_distance("TRUE", "True", axis=axis) == 0.0
    assert verdict_agreement_score("TRUE", "True", axis=axis) == 1.0
    assert verdict_distance("FALSE", "False", axis=axis) == 0.0


def test_lenient_lifts_partly_true_vs_exaggerated_to_perfect_agreement() -> None:
    """The headline coarse-axis win: PARTLY TRUE (reference) vs Exaggerated
    (prediction) is partial credit on the fine axis but perfect agreement
    under Lenient (both → Truthy)."""
    fine = verdict_agreement_score("PARTLY TRUE", "Exaggerated", axis="fine")
    lenient = verdict_agreement_score(
        "PARTLY TRUE", "Exaggerated", axis="coarse_lenient"
    )
    assert lenient == 1.0
    assert fine < 1.0
    assert lenient > fine


def test_strict_separates_partly_true_from_exaggerated() -> None:
    """Same pair under Strict: PARTLY TRUE → Truthy, Exaggerated → Falsey."""
    strict = verdict_agreement_score(
        "PARTLY TRUE", "Exaggerated", axis="coarse_strict"
    )
    # Truthy ↔ Falsey is 2 steps on the 5-bucket axis (max=4), so score = 0.5.
    assert strict == pytest.approx(0.5)
    # Strict is strictly worse than Lenient here.
    lenient = verdict_agreement_score(
        "PARTLY TRUE", "Exaggerated", axis="coarse_lenient"
    )
    assert strict < lenient


def test_misleading_vs_partly_true_is_falsey_vs_truthy_on_both_coarse_axes() -> None:
    """Misleading projects to Falsey on both Lenient and Strict; PARTLY TRUE
    projects to Truthy. So this pair scores identically across the two coarse
    lenses — the coarse-axis disagreement is genuine, not lens-dependent."""
    lenient = verdict_agreement_score(
        "PARTLY TRUE", "Misleading", axis="coarse_lenient"
    )
    strict = verdict_agreement_score(
        "PARTLY TRUE", "Misleading", axis="coarse_strict"
    )
    assert lenient == strict


@pytest.mark.parametrize("axis", ["fine", "coarse_lenient", "coarse_strict"])
def test_distance_is_symmetric(axis: str) -> None:
    """dist(a, b) == dist(b, a) for every axis. The reference normalize map
    only handles uppercase, so we send each side through both directions of
    the same canonical pair."""
    pairs = [
        ("TRUE", "True"),
        ("PARTLY TRUE", "Mostly True"),
        ("PARTLY TRUE", "Exaggerated"),
        ("MISLEADING", "Misleading"),
        ("FALSE", "False"),
        ("UNSUPPORTED", "Unverifiable"),
    ]
    for ref, pred in pairs:
        ab = verdict_distance(ref, pred, axis=axis)
        # Swap roles: now treat what was 'pred' as 'ref' (uppercase) and vice
        # versa (titlecase). Both should normalize to the same canonical
        # bucket, so the distance should be unchanged regardless of which
        # side fed which normalize map.
        ba = verdict_distance(pred.upper(), ref.title(), axis=axis)
        assert ab == pytest.approx(ba), f"asymmetric on {axis}: {ref} vs {pred}"


@pytest.mark.parametrize("axis", ["fine", "coarse_lenient", "coarse_strict"])
def test_true_vs_false_is_max_distance_on_every_axis(axis: str) -> None:
    """The maximum disagreement (true ↔ false) is 1.0 on every axis, so
    verdict_agreement_score is comparable across axes."""
    assert verdict_distance("TRUE", "False", axis=axis) == pytest.approx(1.0)
    assert verdict_agreement_score("TRUE", "False", axis=axis) == pytest.approx(0.0)


# ── Models-split production state ─────────────────────────────────────────────


@pytest.mark.parametrize("axis", ["fine", "coarse_lenient", "coarse_strict"])
def test_models_split_treated_as_unverifiable(axis: str) -> None:
    """The production consensus emits 'Models split' when adapters disagree
    and no plurality emerges. Existing _TRUTHBOT_LABEL_NORMALIZE collapses
    it to 'unverifiable'; the coarse projections must preserve that mapping
    so split bundles don't score as max-disagreement on any axis."""
    score = verdict_agreement_score("UNSUPPORTED", "Models split", axis=axis)
    # Both sides land on 'unverifiable' on every axis → identity.
    assert score == 1.0


# ── Backward compatibility: fine-axis byte-identical round-trip ───────────────


def test_fine_axis_default_matches_explicit_fine() -> None:
    """Calls without an axis argument must score byte-identically to
    axis='fine' calls. Pins the default-arg behavior so a future refactor
    that flips the default doesn't silently change every existing eval."""
    pairs = [
        ("TRUE", "True"),
        ("TRUE", "Mostly True"),
        ("TRUE", "False"),
        ("PARTLY TRUE", "Exaggerated"),
        ("PARTLY TRUE / MISLEADING", "Mostly True"),
        ("UNSUPPORTED", "Unverifiable"),
        ("FALSE", "Misleading"),
    ]
    for ref, pred in pairs:
        assert verdict_distance(ref, pred) == verdict_distance(
            ref, pred, axis="fine"
        )
        assert verdict_agreement_score(ref, pred) == verdict_agreement_score(
            ref, pred, axis="fine"
        )


# ── FitnessScorer integration ─────────────────────────────────────────────────


def _scorer_with_minimal_reference() -> tuple[FitnessScorer, list[dict], list[dict]]:
    """Build a FitnessScorer with a 1-claim synthetic reference + matching
    extracted/verdict pair, so we can read verdict_agreement directly out of
    score() under different axes without leaning on the SOTU corpus."""
    reference = [
        {
            "id": "ref-1",
            "claim": "Unemployment is at a 50-year low.",
            "verdict": "PARTLY TRUE",
            "explanation": "Reference explanation.",
            "expected_sources": [],
        },
    ]
    extracted = [
        {"text": "Unemployment is at a 50-year low.", "is_checkable": True},
    ]
    # Prediction = "Exaggerated": the editorial-tension cohort.
    verdicts = [
        {
            "claim_text": "Unemployment is at a 50-year low.",
            "label": "Exaggerated",
            "explanation": "Model said Exaggerated.",
        },
    ]
    return FitnessScorer(reference=reference), extracted, verdicts


def test_scorer_axis_fine_default_matches_explicit_fine() -> None:
    scorer, extracted, verdicts = _scorer_with_minimal_reference()
    default = scorer.score(extracted, verdicts, token_count=100)
    explicit = scorer.score(extracted, verdicts, token_count=100, axis="fine")
    assert default["verdict_agreement"] == explicit["verdict_agreement"]
    assert default["fitness"] == explicit["fitness"]
    # Default should not declare its axis as anything other than 'fine'.
    assert explicit["axis"] == "fine"


def test_scorer_lenient_lifts_verdict_agreement_for_editorial_tension_pair() -> None:
    """PARTLY TRUE / Exaggerated is partial credit on fine but perfect under
    Lenient. Verdict agreement under Lenient must be strictly greater than
    fine, confirming the 16pp-ish lift visible at the consensus level
    surfaces through the scorer too."""
    scorer, extracted, verdicts = _scorer_with_minimal_reference()
    fine = scorer.score(extracted, verdicts, token_count=100, axis="fine")
    lenient = scorer.score(
        extracted, verdicts, token_count=100, axis="coarse_lenient"
    )
    strict = scorer.score(extracted, verdicts, token_count=100, axis="coarse_strict")

    assert fine["verdict_agreement"] < 1.0
    assert lenient["verdict_agreement"] == pytest.approx(1.0)
    assert lenient["verdict_agreement"] > fine["verdict_agreement"]
    # Strict is editorial-tougher: it should be no better than Lenient on
    # this pair, and on this specific pair it should be worse than Lenient.
    assert strict["verdict_agreement"] < lenient["verdict_agreement"]
    # Recall + parsimony + explanation/source quality components are
    # axis-independent, so fitness should move only via verdict_agreement.
    for k in (
        "claim_recall",
        "explanation_quality",
        "source_citation_quality",
        "parsimony",
    ):
        assert fine[k] == lenient[k] == strict[k]


def test_scorer_returns_axis_label_in_result_dict() -> None:
    scorer, extracted, verdicts = _scorer_with_minimal_reference()
    for axis in ("fine", "coarse_lenient", "coarse_strict"):
        result = scorer.score(extracted, verdicts, token_count=100, axis=axis)
        assert result["axis"] == axis


def test_scorer_unknown_axis_raises() -> None:
    scorer, extracted, verdicts = _scorer_with_minimal_reference()
    with pytest.raises(ValueError):
        scorer.score(extracted, verdicts, token_count=100, axis="coarse_galaxybrain")
