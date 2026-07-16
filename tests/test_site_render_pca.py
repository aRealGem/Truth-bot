"""Reconciled-judge (PCA) claim-card rendering.

The PCA bridge emits one reconciled ``ModelVerdict`` (or none, for a split), so the
legacy "N of M agree" per-adapter vocabulary reads as a vacuous "1 of 1 agree" and
split claims render a blank model strip. These tests pin the reconciled-judge mode:
panel-vote summary + Layer A→panel→CRM-114 provenance strip, split claims included.
"""

from __future__ import annotations

from truthbot.verdict import bridge
from truthbot.publish import site


def _card(row, claim):
    b = bridge.bridge([row], [claim]).bundles[0]
    b.claim.is_checkable = True
    return site._claim_card(b, 0, 5, standalone=True)


def _claim(sid, text, source):
    return {"sid": sid, "text": text, "speaker": "X", "date_str": "2026-02-24",
            "layer_a": {"label": "check-worthy", "source": source}}


def test_resolved_pca_card_speaks_panel_vote_vocabulary_with_provenance():
    row = {"sid": "s:0", "status": "resolved", "verdict": "FALSE", "confidence": 0.8,
           "citations": [], "reasoning": "Contradicted by BLS.",
           "votes": {"MISLEADING": 2, "FALSE": 1}, "split": False, "escalated": True,
           "crm114": {"stage1": "MISLEADING", "final": "FALSE"}}
    html = _card(row, _claim("s:0", "Inflation is the highest ever.", "A2"))

    assert "Reconciled judgment" in html
    assert "2 of 3</span> seats agree" in html          # not "1 of 1 agree"
    assert "1 of 1" not in html
    # provenance chain surfaced (Layer A + tally + CRM-114 override)
    assert "Layer A: check-worthy (A2)" in html
    assert "PCA panel: Misleading ×2, False ×1" in html
    assert "CRM-114: MISLEADING→FALSE" in html


def test_split_pca_card_shows_tally_not_blank_strip():
    row = {"sid": "s:1", "status": "disagreement", "verdict": None, "confidence": None,
           "citations": [], "reasoning": "", "votes": {"TRUE": 1, "FALSE": 1},
           "split": True, "escalated": True}
    html = _card(row, _claim("s:1", "The border is fully secure.", "A1"))

    assert "Panel split" in html
    assert "False ×1, True ×1" in html
    assert "No single verdict" in html                  # the empty-grid placeholder
    assert "0 of 0" not in html                         # the old vacuous tally


def test_legacy_multi_adapter_card_unchanged():
    # >1 model verdict + empty provenance => classic "Model consensus" path.
    from datetime import datetime, timezone
    from truthbot.models import (
        Claim, ConsensusVerdict, Confidence, ModelVerdict, VerdictBundle, VerdictLabel,
    )
    claim = Claim(transcript_id="t", text="A claim.", speaker="X", is_checkable=True)
    mvs = [
        ModelVerdict(adapter_name=f"a{i}", model_id=f"m{i}", claim_id=claim.id,
                     label=VerdictLabel.TRUE, confidence=Confidence.HIGH, explanation="r")
        for i in range(3)
    ]
    consensus = ConsensusVerdict(
        claim_id=claim.id, model_verdicts=mvs, consensus_label=VerdictLabel.TRUE,
        consensus_verdict="True", confidence=Confidence.HIGH, agreement=True,
        consensus_strength="strong", explanation="x",
    )
    b = VerdictBundle(claim=claim, speaker="X", date_str="2026-02-24",
                      model_verdicts=mvs, consensus=consensus)
    html = site._claim_card(b, 0, 5, standalone=True)
    assert "Model consensus" in html
    assert "3 of 3" in html
    assert "Reconciled judgment" not in html
    assert "pca-provenance" not in html
