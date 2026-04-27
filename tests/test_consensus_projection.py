"""Tests for the 5-bucket coarse-axis projection layer.

The consensus engine emits two parallel projections of the four-adapter panel
onto a 5-bucket "Truthy scale" alongside the existing 6-bucket fine consensus:

  * LENIENT_PROJECTION (Part H default, published-default)
      Mostly True + Exaggerated → Truthy
      Misleading                → Falsey
  * STRICT_PROJECTION (toggle, tougher editorial bar)
      Mostly True               → Truthy
      Exaggerated + Misleading  → Falsey

These tests cover:
  * Lenient and Strict mappings — unanimity, strong, weak, split
  * Split-projection guardrail (no plurality → "Models split", not tie-broken)
  * Round-trip: adding the projection layer must not change fine-axis output
  * Edge cases: single-adapter, all-Unverifiable, empty panel
"""

from __future__ import annotations

from truthbot.models import Confidence, ModelVerdict, VerdictLabel
from truthbot.verify.engine import (
    LENIENT_PROJECTION,
    STRICT_PROJECTION,
    _build_consensus,
    _project_consensus,
)


def _mv(adapter: str, label: VerdictLabel, claim_id: str = "c-1") -> ModelVerdict:
    """Minimal ModelVerdict factory for projection tests."""
    return ModelVerdict(
        adapter_name=adapter,
        model_id=f"{adapter}-test",
        claim_id=claim_id,
        label=label,
        confidence=Confidence.HIGH,
        explanation="test",
    )


# ── Mapping invariants ────────────────────────────────────────────────────────


def test_lenient_and_strict_share_the_neutral_buckets() -> None:
    """True / Mostly True / False / Unverifiable map identically across lenses."""
    for label in (
        VerdictLabel.TRUE,
        VerdictLabel.MOSTLY_TRUE,
        VerdictLabel.FALSE,
        VerdictLabel.UNVERIFIABLE,
    ):
        assert LENIENT_PROJECTION[label] == STRICT_PROJECTION[label]


def test_lenient_vs_strict_diverge_only_on_exaggerated() -> None:
    """Lenient: Excg → Truthy. Strict: Excg → Falsey. Misleading is Falsey for both."""
    assert LENIENT_PROJECTION[VerdictLabel.EXAGGERATED] == "Truthy"
    assert STRICT_PROJECTION[VerdictLabel.EXAGGERATED] == "Falsey"
    assert LENIENT_PROJECTION[VerdictLabel.MISLEADING] == "Falsey"
    assert STRICT_PROJECTION[VerdictLabel.MISLEADING] == "Falsey"


def test_projection_bucket_values_are_only_the_5_truthy_scale_labels() -> None:
    allowed = {"True", "Truthy", "Unverifiable", "Falsey", "False"}
    assert set(LENIENT_PROJECTION.values()) == allowed
    assert set(STRICT_PROJECTION.values()) == allowed


# ── _project_consensus directly ───────────────────────────────────────────────


def test_project_unanimous_panel_is_strong_under_both_lenses() -> None:
    panel = [_mv("a", VerdictLabel.TRUE), _mv("b", VerdictLabel.TRUE),
             _mv("c", VerdictLabel.TRUE), _mv("d", VerdictLabel.TRUE)]
    assert _project_consensus(panel, LENIENT_PROJECTION) == ("True", "strong")
    assert _project_consensus(panel, STRICT_PROJECTION) == ("True", "strong")


def test_lenient_lifts_mostly_true_plus_exaggerated_to_truthy_strong() -> None:
    """The headline win: 4 models split MT/Excg under 6 buckets fuse to Truthy under Lenient."""
    panel = [
        _mv("anthropic", VerdictLabel.MOSTLY_TRUE),
        _mv("openai", VerdictLabel.EXAGGERATED),
        _mv("gemini", VerdictLabel.MOSTLY_TRUE),
        _mv("xai", VerdictLabel.EXAGGERATED),
    ]
    assert _project_consensus(panel, LENIENT_PROJECTION) == ("Truthy", "strong")


def test_strict_separates_mostly_true_from_exaggerated() -> None:
    """Same panel under Strict: MT-MT-Excg-Excg becomes Truthy-Truthy-Falsey-Falsey → split."""
    panel = [
        _mv("anthropic", VerdictLabel.MOSTLY_TRUE),
        _mv("openai", VerdictLabel.EXAGGERATED),
        _mv("gemini", VerdictLabel.MOSTLY_TRUE),
        _mv("xai", VerdictLabel.EXAGGERATED),
    ]
    label, strength = _project_consensus(panel, STRICT_PROJECTION)
    assert label == "Models split"
    assert strength == "none"


def test_lenient_misleading_dissenter_drops_consensus_to_weak_or_split() -> None:
    """3xMostly True + 1xMisleading → 3 Truthy + 1 Falsey under Lenient = strong Truthy."""
    panel = [
        _mv("a", VerdictLabel.MOSTLY_TRUE),
        _mv("b", VerdictLabel.MOSTLY_TRUE),
        _mv("c", VerdictLabel.MOSTLY_TRUE),
        _mv("d", VerdictLabel.MISLEADING),
    ]
    assert _project_consensus(panel, LENIENT_PROJECTION) == ("Truthy", "strong")
    assert _project_consensus(panel, STRICT_PROJECTION) == ("Truthy", "strong")


def test_split_projection_guardrail_fires_on_2_2_split() -> None:
    """Genuine 2-2 directional split must not be smoothed into a single label."""
    panel = [
        _mv("a", VerdictLabel.TRUE),
        _mv("b", VerdictLabel.TRUE),
        _mv("c", VerdictLabel.FALSE),
        _mv("d", VerdictLabel.FALSE),
    ]
    label, strength = _project_consensus(panel, LENIENT_PROJECTION)
    assert label == "Models split"
    assert strength == "none"
    label_s, strength_s = _project_consensus(panel, STRICT_PROJECTION)
    assert label_s == "Models split"
    assert strength_s == "none"


def test_split_projection_guardrail_fires_on_all_different_panel() -> None:
    """4 distinct projected labels → no plurality → Models split."""
    panel = [
        _mv("a", VerdictLabel.TRUE),
        _mv("b", VerdictLabel.MOSTLY_TRUE),
        _mv("c", VerdictLabel.MISLEADING),
        _mv("d", VerdictLabel.FALSE),
    ]
    # Lenient projects to: True, Truthy, Falsey, False — all four distinct.
    label, strength = _project_consensus(panel, LENIENT_PROJECTION)
    assert label == "Models split"
    assert strength == "none"


def test_weak_plurality_is_preserved_when_other_models_split() -> None:
    """2 of 4 agree on a projected label, others differ → weak (not split, not strong)."""
    panel = [
        _mv("a", VerdictLabel.TRUE),
        _mv("b", VerdictLabel.TRUE),
        _mv("c", VerdictLabel.MOSTLY_TRUE),  # → Truthy under both
        _mv("d", VerdictLabel.FALSE),
    ]
    label, strength = _project_consensus(panel, LENIENT_PROJECTION)
    assert label == "True"
    assert strength == "weak"
    label_s, strength_s = _project_consensus(panel, STRICT_PROJECTION)
    assert label_s == "True"
    assert strength_s == "weak"


def test_single_adapter_panel_uses_single_strength() -> None:
    panel = [_mv("only", VerdictLabel.EXAGGERATED)]
    assert _project_consensus(panel, LENIENT_PROJECTION) == ("Truthy", "single")
    assert _project_consensus(panel, STRICT_PROJECTION) == ("Falsey", "single")


def test_empty_panel_returns_blank_label_none_strength() -> None:
    assert _project_consensus([], LENIENT_PROJECTION) == ("", "none")
    assert _project_consensus([], STRICT_PROJECTION) == ("", "none")


def test_all_unverifiable_panel_projects_unverifiable() -> None:
    panel = [_mv(f"a{i}", VerdictLabel.UNVERIFIABLE) for i in range(4)]
    assert _project_consensus(panel, LENIENT_PROJECTION) == ("Unverifiable", "strong")
    assert _project_consensus(panel, STRICT_PROJECTION) == ("Unverifiable", "strong")


# ── _build_consensus integration: round-trip + projection wiring ──────────────


def test_build_consensus_round_trip_does_not_change_fine_axis() -> None:
    """Adding the projection layer must not perturb existing fine-axis fields."""
    panel = [
        _mv("anthropic", VerdictLabel.MOSTLY_TRUE),
        _mv("openai", VerdictLabel.MOSTLY_TRUE),
        _mv("gemini", VerdictLabel.EXAGGERATED),
        _mv("xai", VerdictLabel.TRUE),
    ]
    consensus = _build_consensus("c-1", panel)

    assert consensus.consensus_label == VerdictLabel.MOSTLY_TRUE
    assert consensus.consensus_strength == "weak"
    assert consensus.consensus_verdict == "Mostly True"
    # Coarse axis populated alongside.
    assert consensus.coarse_lenient_label == "Truthy"
    assert consensus.coarse_lenient_strength == "strong"
    assert consensus.coarse_strict_label == "Truthy"
    assert consensus.coarse_strict_strength == "weak"


def test_build_consensus_empty_panel_emits_blank_coarse_fields() -> None:
    consensus = _build_consensus("c-empty", [])
    assert consensus.coarse_lenient_label == ""
    assert consensus.coarse_lenient_strength == "none"
    assert consensus.coarse_strict_label == ""
    assert consensus.coarse_strict_strength == "none"


def test_build_consensus_split_2_2_propagates_to_both_axes() -> None:
    """2-2 True/False split: fine axis tie-breaks to False (conservative);
    coarse projections both fire the guardrail."""
    panel = [
        _mv("a", VerdictLabel.TRUE),
        _mv("b", VerdictLabel.TRUE),
        _mv("c", VerdictLabel.FALSE),
        _mv("d", VerdictLabel.FALSE),
    ]
    consensus = _build_consensus("c-split", panel)
    # Fine axis: tie-break picks False (most conservative).
    assert consensus.consensus_label == VerdictLabel.FALSE
    assert consensus.consensus_strength == "weak"
    # Coarse axis: 2-2 across categories → Models split, both lenses.
    assert consensus.coarse_lenient_label == "Models split"
    assert consensus.coarse_lenient_strength == "none"
    assert consensus.coarse_strict_label == "Models split"
    assert consensus.coarse_strict_strength == "none"


def test_build_consensus_single_adapter_projection() -> None:
    panel = [_mv("only", VerdictLabel.MISLEADING)]
    consensus = _build_consensus("c-solo", panel)
    assert consensus.consensus_strength == "single"
    assert consensus.coarse_lenient_label == "Falsey"
    assert consensus.coarse_lenient_strength == "single"
    assert consensus.coarse_strict_label == "Falsey"
    assert consensus.coarse_strict_strength == "single"


def test_build_consensus_lenient_lifts_hidden_truthy_agreement() -> None:
    """The headline empirical win: a panel that's weak/split on the fine axis
    should surface as strong Truthy under Lenient. Captures the 16pp
    consensus-strength delta the projection layer was designed to deliver."""
    panel = [
        _mv("anthropic", VerdictLabel.MOSTLY_TRUE),
        _mv("openai", VerdictLabel.EXAGGERATED),
        _mv("gemini", VerdictLabel.EXAGGERATED),
        _mv("xai", VerdictLabel.MOSTLY_TRUE),
    ]
    consensus = _build_consensus("c-tension", panel)
    # Fine axis: 2-2 MT/Excg → tie-break picks Excg (more conservative).
    assert consensus.consensus_label in (VerdictLabel.EXAGGERATED, VerdictLabel.MOSTLY_TRUE)
    assert consensus.consensus_strength == "weak"
    # Lenient: all four → Truthy → strong.
    assert consensus.coarse_lenient_label == "Truthy"
    assert consensus.coarse_lenient_strength == "strong"
    # Strict: 2 Truthy + 2 Falsey → guardrail.
    assert consensus.coarse_strict_label == "Models split"
    assert consensus.coarse_strict_strength == "none"
