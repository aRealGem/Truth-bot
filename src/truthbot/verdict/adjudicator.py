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

from datetime import date
from typing import Optional

from hydramind import HydraMind, ItemResult, StrategyResultKind

from . import evidence_pack, speech_context
from .evidence_pack import DEFAULT_MAX_ITEMS, EvidencePack
from .prompts import OPEN_BOOK_PROMPTS, PROMPTS
from truthbot.verify.evidence_provider import EvidenceProvider

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


def build_items(claims: list[dict], *, evidence_provider: Optional[EvidenceProvider] = None,
                today: Optional[date] = None, max_items: int = DEFAULT_MAX_ITEMS
                ) -> tuple[list[dict], dict[str, EvidencePack]]:
    """Build the pca payload items for a claim batch (pure; no network unless a
    provider is given). Returns (items, packs).

    Each item's context is prefixed with the speaker-blind temporal preamble. When
    ``evidence_provider`` is supplied (OPEN-BOOK), a time-scoped evidence pack is
    fetched per claim and attached to the payload as ``evidence`` (model-facing) +
    ``evidence_pack_ids`` (the ids I4 checks citations against); ``packs`` maps sid →
    pack for telemetry. Without a provider (CLOSED-BOOK) ``evidence_pack_ids`` stays
    empty so I4 requires ``citations == []``."""
    items: list[dict] = []
    packs: dict[str, EvidencePack] = {}
    for c in claims:
        preamble = speech_context.build_temporal_preamble(
            c["sid"], reference_period=c.get("reference_period"), today=today)
        payload = {"claim": c["text"],
                   "context": preamble + c.get("context", ""),
                   "evidence_pack_ids": []}
        if evidence_provider is not None:
            pack = evidence_pack.build_evidence_pack(
                c["sid"], c["text"], evidence_provider,
                today=today, max_items=max_items, context=c.get("context", ""))
            packs[c["sid"]] = pack
            payload["evidence"] = pack.to_payload()
            payload["evidence_pack_ids"] = pack.ids
        items.append({"item_id": c["sid"], "payload": payload})
    return items, packs


def adjudicate(hm: HydraMind, claims: list[dict], *, roster: str = "dev",
               tune: Optional[dict] = None, rc_id: Optional[str] = None,
               evidence_provider: Optional[EvidenceProvider] = None,
               max_items: int = DEFAULT_MAX_ITEMS, today=None):
    """claims: [{"sid","text","context"}]. Returns (rows, manifest, notes).

    Runs each claim through the pca panel and normalizes to verdict-contract rows.
    Requires a live L-P/L-B lane (proxy virtual key from repo .env). Mirrors
    classifier.classify.

    Mode is set by ``evidence_provider``:
      * None (default) → CLOSED-BOOK: no evidence, evidence_pack_ids=[] ⇒ I4 requires
        citations==[]; the panel judges from parametric knowledge.
      * an EvidenceProvider → OPEN-BOOK (Layer C): a time-scoped, provenanced evidence
        pack is fetched per claim and injected; the panel must ground its verdict in
        that pack and cite the ids it used (I4 enforces citations ⊆ pack), and
        normalize keeps those citations (closed_book=False).

    Charter (temporal veracity): each claim's context is prefixed with a speaker-blind
    temporal preamble (utterance date + expected evidence window + today-authoritative)
    so the panel judges as-of when the claim was made and does not dismiss post-cutoff
    events. A claim may carry an optional 'reference_period' for the span it is about.

    Pass rc_id ONLY for a scored heldout pass (I6 read-once); leave None for TRAIN
    iteration."""
    open_book = evidence_provider is not None
    items, packs = build_items(claims, evidence_provider=evidence_provider,
                               today=today, max_items=max_items)
    run_tune = {"prompts": OPEN_BOOK_PROMPTS if open_book else PROMPTS}
    run_tune.update(tune or {})
    result, manifest = hm.run("verdict", items, "pca", roster=roster,
                              tune=run_tune, rc_id=rc_id)
    rows = [normalize(r, closed_book=not open_book) for r in result.items]
    notes = dict(result.notes or {})
    notes["open_book"] = open_book
    if open_book:
        notes["evidence_counts"] = {sid: len(p.items) for sid, p in packs.items()}
    return rows, manifest, notes
