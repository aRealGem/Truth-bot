"""
Layer B — normalize the PCA (proposer→critic→arbiter) panel output to the verdict
label contract. This is the parse_a2 analogue for verdicts (mirrors the parse
discipline in checkworthy/classifier.py), split across the two points where a raw
model output has to be trusted:

  * parse_verdict — per model CALL (transport boundary). One seat's raw JSON → the
    fields pca reads. An unrecognized label is a fail-SAFE UNVERIFIABLE vote: it is
    one vote among the panel and reduce handles disagreement, so a single odd seat
    output must not crash the run.
  * normalize — per ITEM (post-reduce). ItemResult → verdict-contract row. Fails
    CLOSED like parse_a2: a RESOLVED verdict outside the contract, or a closed-book
    citation (I4), raises rather than silently mislabel. A DISAGREEMENT_FLAGGED item
    is never coerced into a verdict — it carries an explicit status (I2: never a
    silent tie-break).

Closed-book 4-label contract: TRUE | FALSE | MISLEADING | UNVERIFIABLE. Full
6-bucket VerdictLabel mapping is deferred to Layer C (evidence-grounded).
"""
from __future__ import annotations

from typing import Optional

from hydramind import HydraMind, ItemResult, StrategyResultKind

from .prompts import PROMPTS

_VALID_VERDICTS = {"TRUE", "FALSE", "MISLEADING", "UNVERIFIABLE"}


def parse_verdict(raw: dict) -> dict:
    """Per-call response parser: one seat's raw JSON → {verdict, confidence, citations}.
    Fail-SAFE — an unrecognized verdict becomes an UNVERIFIABLE vote (see module note)."""
    v = (raw.get("verdict") or "").strip().upper()
    if v not in _VALID_VERDICTS:
        v = "UNVERIFIABLE"
    c = raw.get("confidence")
    try:
        c = float(c)
    except (TypeError, ValueError):
        c = None
    return {"verdict": v, "confidence": c, "citations": list(raw.get("citations", []))}


def normalize(item: ItemResult, *, closed_book: bool = True) -> dict:
    """Per-item normalizer: ItemResult → verdict-contract row. Fail-CLOSED (see note).

    Row: {sid, status, verdict, confidence, citations, votes, split, escalated}
    where status ∈ {resolved, disagreement, no_label} and verdict is None unless
    status == resolved."""
    ag = item.agreement or {}
    row = {
        "sid": item.item_id,
        "status": None,
        "verdict": None,
        "confidence": None,
        "citations": [],
        "votes": ag.get("votes", {}),
        "split": bool(ag.get("split", False)),
        "escalated": bool(ag.get("escalated", False)),
    }
    if item.kind is StrategyResultKind.RESOLVED:
        v = (item.value.get("verdict") or "").strip().upper()
        if v not in _VALID_VERDICTS:
            raise ValueError(
                f"pca resolved an out-of-contract verdict {item.value.get('verdict')!r} "
                f"for {item.item_id}")
        citations = list(item.value.get("citations", []))
        if closed_book and citations:
            raise ValueError(
                f"closed-book verdict for {item.item_id} carries citations "
                f"{citations!r} — I4 violation")
        row.update(status="resolved", verdict=v,
                   confidence=item.value.get("confidence"), citations=citations)
        return row

    # DISAGREEMENT_FLAGGED — distinguish "no seat produced a label" from a material tie.
    row["status"] = "no_label" if item.value.get("reason") == "no_labels" else "disagreement"
    return row


def adjudicate(hm: HydraMind, claims: list[dict], *, roster: str = "dev",
               tune: Optional[dict] = None, rc_id: Optional[str] = None):
    """claims: [{"sid","text","context"}]. Returns (rows, manifest, notes).

    Runs each claim through the pca panel closed-book (evidence_pack_ids=[] ⇒ I4
    requires citations==[]) and normalizes to verdict-contract rows. Requires a live
    L-P/L-B lane (proxy virtual key from repo .env). Mirrors classifier.classify.

    Pass rc_id ONLY for a scored heldout pass (I6 read-once); leave None for TRAIN
    iteration."""
    items = [{"item_id": c["sid"],
              "payload": {"claim": c["text"], "context": c.get("context", ""),
                          "evidence_pack_ids": []}}
             for c in claims]
    run_tune = {"prompts": PROMPTS}
    run_tune.update(tune or {})
    result, manifest = hm.run("verdict", items, "pca", roster=roster,
                              tune=run_tune, rc_id=rc_id)
    rows = [normalize(r) for r in result.items]
    return rows, manifest, result.notes
