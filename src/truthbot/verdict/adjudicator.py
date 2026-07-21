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

from . import discriminator, evidence_pack, speech_context
from .evidence_pack import DEFAULT_MAX_ITEMS, EvidencePack
from .prompts import CALIBRATED_OPEN_BOOK_PROMPTS, PROMPTS
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
    return {"verdict": v, "confidence": c, "citations": list(raw.get("citations", [])),
            "reasoning": (raw.get("reasoning") or "").strip()}


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
        "reasoning": "",
        "votes": ag.get("votes", {}),
        "by_role": ag.get("by_role", {}),   # role → [labels]; critics may be a panel
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
                   confidence=item.value.get("confidence"), citations=citations,
                   reasoning=(item.value.get("reasoning") or "").strip())
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
               max_items: int = DEFAULT_MAX_ITEMS, two_stage: bool = True,
               disc_tier: str = "standard", today=None):
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
    # Open-book default is the CALIBRATED prompt set (adopted 2026-07-19, P67 Track B:
    # +0.06 decided-acc over plain at equal cost; plain remains available via tune).
    run_tune = {"prompts": CALIBRATED_OPEN_BOOK_PROMPTS if open_book else PROMPTS}
    run_tune.update(tune or {})
    result, manifest = hm.run("verdict", items, "pca", roster=roster,
                              tune=run_tune, rc_id=rc_id)
    rows = [normalize(r, closed_book=not open_book) for r in result.items]
    notes = dict(result.notes or {})
    notes["open_book"] = open_book
    # T2.7: record the retrieval stack explicitly. The current live path is
    # v1; shared_pack_v2 lands with the PR-5 retrievers + consolidator.
    from truthbot.verdict.evidence_mode import EvidenceMode
    notes["evidence_mode"] = EvidenceMode.infer_legacy(open_book).value

    # Stage 2 (CRM-114): re-decide the FALSE-vs-MISLEADING boundary on the adverse
    # bucket with a focused binary discriminator, on the SAME evidence packs. Open-book
    # only — the discriminator judges on evidence, not parametric knowledge.
    if two_stage and open_book:
        # Unanimous-FALSE guard (P67 Phase 4): a row every seat voted FALSE is settled —
        # the binary discriminator could only confirm it (no-op) or soften it, and the
        # one soften-flip ever observed (trump_2026:0556, 35-row scoring 2026-07-19)
        # undid a correct unanimous panel FALSE. Skipping the row entirely both blocks
        # that override and saves the stage-2 call. Split rows (FALSE in a mixed tally)
        # still route — there the discriminator adds real information.
        def _unanimous_false(r):
            return r["verdict"] == "FALSE" and set(r.get("votes") or {}) == {"FALSE"}
        adverse = {r["sid"] for r in rows
                   if r["status"] == "resolved" and r["verdict"] in ("FALSE", "MISLEADING")
                   and not _unanimous_false(r)}
        # TIE_ABSTAIN routing (P67 Phase 3, closes the F1 bypass): a DISAGREEMENT-
        # flagged row whose vote set is within {FALSE, MISLEADING, UNVERIFIABLE} is an
        # adverse-severity tie, not a genuine can't-decide — a correct FALSE vote was
        # dying in the tie because stage 2 only saw resolved rows. Route it to the
        # discriminator; ties involving a TRUE vote stay flagged (binary F/M would be
        # the wrong question). The override is recorded on the row (crm114 + status),
        # so this is an explicit stage-2 adjudication, not a silent tie-break (I2).
        tie_routed = {r["sid"] for r in rows
                      if r["status"] == "disagreement" and r["votes"]
                      and set(r["votes"]) <= {"FALSE", "MISLEADING", "UNVERIFIABLE"}}
        disc_items = [it for it in items if it["item_id"] in adverse | tie_routed]
        disc = discriminator.discriminate(hm, disc_items, tier=disc_tier)
        discriminator.apply_discrimination(rows, disc)
        discriminator.apply_tie_routing(rows, disc)
        notes["two_stage"] = True
        notes["disc_tier"] = disc_tier
        notes["crm114_tie_routed"] = sorted(tie_routed)
        notes["crm114_unanimous_false_skipped"] = sorted(
            r["sid"] for r in rows if r["status"] == "resolved" and _unanimous_false(r))
        notes["crm114_overrides"] = {r["sid"]: r["crm114"] for r in rows if r.get("crm114")}

    if open_book:
        notes["evidence_counts"] = {sid: len(p.items) for sid, p in packs.items()}
        notes["packs"] = packs   # sid → EvidencePack, for the publish bridge (in-process)
    return rows, manifest, notes
