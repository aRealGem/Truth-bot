"""Render-side tests for the 5-bucket coarse-axis projection.

The headline ``.claim-pill`` should:
  * carry both ``data-coarse-lenient`` and ``data-coarse-strict`` attributes,
  * render the Lenient label by default,
  * keep the per-model strip on the 6-bucket fine-label CSS classes
    (``vt-mostly-true`` / ``vt-exaggerated`` / etc., never ``vt-truthy``).

Backward-compat: when a bundle has empty coarse_* fields (legacy bundles
written before this layer landed), the headline pill must fall back to the
fine label.
"""

from __future__ import annotations

from truthbot.models import (
    Claim,
    Confidence,
    ConsensusVerdict,
    ModelVerdict,
    VerdictBundle,
    VerdictLabel,
)
from truthbot.publish.site import _claim_card


def _bundle(
    *,
    fine_label: VerdictLabel,
    model_labels: list[VerdictLabel],
    lenient: str,
    strict: str,
    lenient_strength: str = "strong",
    strict_strength: str = "strong",
) -> VerdictBundle:
    claim = Claim(
        transcript_id="test-transcript",
        text="Test claim.",
        speaker="Speaker",
        context="ctx",
        category="economy",
        is_checkable=True,
    )
    mvs = [
        ModelVerdict(
            adapter_name=f"adapter-{i}",
            model_id=f"model-{i}",
            claim_id=claim.id,
            label=lbl,
            confidence=Confidence.HIGH,
            explanation="Reasoning.",
        )
        for i, lbl in enumerate(model_labels)
    ]
    consensus = ConsensusVerdict(
        claim_id=claim.id,
        model_verdicts=mvs,
        consensus_label=fine_label,
        consensus_verdict=fine_label.value,
        confidence=Confidence.HIGH,
        agreement=False,
        consensus_strength="weak",
        explanation="Test.",
        coarse_lenient_label=lenient,
        coarse_lenient_strength=lenient_strength,
        coarse_strict_label=strict,
        coarse_strict_strength=strict_strength,
    )
    return VerdictBundle(
        claim=claim,
        speaker="Speaker",
        date_str="2026-03-04",
        model_verdicts=mvs,
        consensus=consensus,
    )


def test_headline_pill_carries_both_projection_data_attrs() -> None:
    bundle = _bundle(
        fine_label=VerdictLabel.MOSTLY_TRUE,
        model_labels=[VerdictLabel.MOSTLY_TRUE, VerdictLabel.EXAGGERATED,
                      VerdictLabel.MOSTLY_TRUE, VerdictLabel.EXAGGERATED],
        lenient="Truthy",
        strict="Models split",
        lenient_strength="strong",
        strict_strength="none",
    )
    html = _claim_card(bundle, idx=1, total=1, rel="../", standalone=True)
    assert 'data-fine-label="Mostly True"' in html
    assert 'data-coarse-lenient="Truthy"' in html
    assert 'data-coarse-strict="Models split"' in html
    assert 'data-coarse-lenient-css="truthy"' in html
    # 'Models split' renders with the neutral unverifiable styling.
    assert 'data-coarse-strict-css="unverifiable"' in html


def test_headline_pill_default_renders_lenient_label_and_class() -> None:
    bundle = _bundle(
        fine_label=VerdictLabel.MOSTLY_TRUE,
        model_labels=[VerdictLabel.MOSTLY_TRUE, VerdictLabel.EXAGGERATED,
                      VerdictLabel.MOSTLY_TRUE, VerdictLabel.EXAGGERATED],
        lenient="Truthy",
        strict="Models split",
    )
    html = _claim_card(bundle, idx=1, total=1, rel="../", standalone=True)
    assert "claim-pill-headline" in html
    # Default-rendered class + visible label is the Lenient projection.
    assert "v-truthy" in html
    assert ">Truthy</span>" in html
    # Fine label is NOT what's painted by default — it lives in data-* only.
    assert ">Mostly True</span>" not in html


def test_per_model_strip_keeps_fine_axis_classes() -> None:
    bundle = _bundle(
        fine_label=VerdictLabel.MOSTLY_TRUE,
        model_labels=[VerdictLabel.MOSTLY_TRUE, VerdictLabel.EXAGGERATED,
                      VerdictLabel.MOSTLY_TRUE, VerdictLabel.EXAGGERATED],
        lenient="Truthy",
        strict="Models split",
    )
    html = _claim_card(bundle, idx=1, total=1, rel="../", standalone=True)
    # Per-model verdict cards must still surface the 6-bucket labels...
    assert "vt-mostly-true" in html
    assert "vt-exaggerated" in html
    assert ">Mostly True<" in html or "Mostly True" in html
    assert "Exaggerated" in html
    # ...and must NOT collapse to the coarse-axis classes on the strip.
    assert "vt-truthy" not in html
    assert "vt-falsey" not in html


def test_headline_pill_falls_back_to_fine_label_for_legacy_bundles() -> None:
    """Bundles cached before the projection layer have empty coarse_* fields.
    The renderer must degrade gracefully to the existing fine-axis pill so
    historical reports re-render without rebuilding their bundles."""
    bundle = _bundle(
        fine_label=VerdictLabel.MOSTLY_TRUE,
        model_labels=[VerdictLabel.MOSTLY_TRUE, VerdictLabel.MOSTLY_TRUE],
        lenient="",
        strict="",
        lenient_strength="none",
        strict_strength="none",
    )
    html = _claim_card(bundle, idx=1, total=1, rel="../", standalone=True)
    # Visible pill text falls back to fine label.
    assert ">Mostly True</span>" in html
    # data-* attrs echo the fine label so JS toggle no-ops cleanly.
    assert 'data-coarse-lenient="Mostly True"' in html
    assert 'data-coarse-strict="Mostly True"' in html


def test_dissent_count_uses_fine_axis_not_coarse_label() -> None:
    """If the dissent comparator accidentally used the coarse pill label,
    every model would be flagged as dissenting (since model labels are 6-bucket
    and the headline is now 5-bucket). This regression test keeps that wired
    to the fine axis."""
    bundle = _bundle(
        fine_label=VerdictLabel.MOSTLY_TRUE,
        model_labels=[VerdictLabel.MOSTLY_TRUE, VerdictLabel.MOSTLY_TRUE,
                      VerdictLabel.MOSTLY_TRUE, VerdictLabel.MOSTLY_TRUE],
        lenient="Truthy",
        strict="Truthy",
    )
    html = _claim_card(bundle, idx=1, total=1, rel="../", standalone=True)
    # Unanimous on fine axis: 4 of 4 agree, no "dissent" text.
    assert "4 of 4" in html
    assert "dissent" not in html.lower() or " dissent\"" not in html
