"""Layer C → publish bridge (PR-B).

The HydraMind PCA verdict stack (Layer B/C) speaks in *adjudicate rows* — the
verdict-contract dicts ``adjudicator.normalize`` emits — plus an in-process
``notes["packs"]`` map of ``sid → EvidencePack``. The publisher
(``publish/site.py``) speaks in ``VerdictBundle`` objects (the output unit of the
legacy ``VerificationEngine``). This module is the pure, **offline** adapter
between the two: rows + packs + the originating claims → ``VerdictBundle`` list,
with the cited evidence surfaced alongside. No network, no LLM, no spend — it is
fully unit-testable and carries no live dependency.

Design decisions (why this shape and not another):

  * **PCA is one reconciled judge.** The panel is proposer→critic→arbiter
    *internally*; a row is its single reconciled verdict, not four independent
    provider verdicts. So each bundle gets ONE ``ModelVerdict``
    (``adapter_name = "hydramind-pca"``) carrying the resolved label, the panel's
    one-clause reasoning, and the URLs it cited. We do NOT fabricate one model
    card per seat-vote — the panel's internal agreement is a scalar
    (``consensus_strength``), which is exactly how the row exposes it (``votes``).

  * **The resolved label is authoritative.** ``consensus_label`` and the coarse
    projection come from the row's ``verdict`` (post-CRM-114 stage-2 override),
    NOT from the raw ``votes`` plurality. ``votes`` drives only
    ``consensus_strength``. This keeps the published top-line identical to the
    verdict the pipeline actually resolved, even when CRM-114 flipped the
    stage-1 plurality (FALSE↔MISLEADING).

  * **Coarse projection reuses the canonical tables.** We map the resolved label
    through ``engine.LENIENT_PROJECTION`` / ``STRICT_PROJECTION`` (the Truthy
    5-bucket scale). The PCA 4-label contract (TRUE/FALSE/MISLEADING/
    UNVERIFIABLE) maps 1:1 onto distinct coarse buckets — MOSTLY_TRUE /
    EXAGGERATED are unused, so no two labels collide into one bucket and the
    projection is deterministic from the label. That means the coarse strength
    equals the fine strength (no hidden-agreement collapse to reconcile), so we
    reuse the single ``consensus_strength`` for both axes.

  * **Non-resolved rows are kept, not dropped.** A ``disagreement`` / ``no_label``
    / ``needs_verdict`` row becomes an UNVERIFIABLE bundle with an explicit
    "Models split" / "No verdict" consensus. Silently dropping claims from a
    published report is a correctness bug, not a simplification.

Mapping summary (plan PR-B):
  4-label ``verdict``     → ``VerdictLabel``      (direct; MOSTLY_TRUE/EXAGGERATED unused)
  float ``confidence``    → ``Confidence``        (>=0.85 High, >=0.55 Medium, else Low)
  ``reasoning``           → ``ConsensusVerdict.explanation`` / ``ModelVerdict.explanation``
  ``votes``               → ``consensus_strength``
  cited pack items        → ``ModelVerdict.web_sources`` (+ full pack → ``Evidence``)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from truthbot.models import (
    Claim,
    Confidence,
    ConsensusVerdict,
    Evidence,
    ModelVerdict,
    VerdictBundle,
    VerdictLabel,
    VerdictProvenance,
)
from truthbot.verdict.evidence_pack import EvidencePack
from truthbot.verify.engine import LENIENT_PROJECTION, STRICT_PROJECTION

logger = logging.getLogger(__name__)

# PCA closed-book/open-book 4-label contract → the 6-bucket VerdictLabel enum.
# MOSTLY_TRUE and EXAGGERATED are never produced by the PCA panel, so they have
# no source label here (Layer C intentionally judges on 4 buckets).
_LABEL_MAP: dict[str, VerdictLabel] = {
    "TRUE": VerdictLabel.TRUE,
    "FALSE": VerdictLabel.FALSE,
    "MISLEADING": VerdictLabel.MISLEADING,
    "UNVERIFIABLE": VerdictLabel.UNVERIFIABLE,
}

# The single reconciled-judge identity every PCA bundle presents to the publisher.
PANEL_ADAPTER = "hydramind-pca"
PANEL_MODEL_ID = "pca"


@dataclass
class BridgeOutput:
    """Result of bridging one adjudicate batch to the publish layer.

    ``bundles`` is the primary output (one per input row, in input order).
    ``evidence`` maps ``sid → [Evidence]`` for the *full* retrieved pack (not just
    the cited subset) so PR-C can assemble a ``Report.evidence`` corpus; the
    cited subset is already reflected on each bundle's ``ModelVerdict.web_sources``.
    """

    bundles: list[VerdictBundle] = field(default_factory=list)
    evidence: dict[str, list[Evidence]] = field(default_factory=dict)


def confidence_from_float(c: Optional[float]) -> Confidence:
    """Map a 0–1 confidence to the coarse ``Confidence`` enum.

    Thresholds are the inverse of ``verify.triage.confidence_numeric`` anchors
    (High=1.0, Medium=0.7, Low=0.4): the High/Medium cut sits at 0.85 and the
    Medium/Low cut at 0.55. A missing confidence fails to Low."""
    if c is None:
        return Confidence.LOW
    if c >= 0.85:
        return Confidence.HIGH
    if c >= 0.55:
        return Confidence.MEDIUM
    return Confidence.LOW


def strength_from_votes(votes: Optional[dict]) -> str:
    """Panel agreement → ``consensus_strength`` string.

    Mirrors ``engine._build_consensus``: single (1 seat) / strong (≥3 agree on the
    top label) / weak (exactly 2 agree) / none (no plurality — a genuine split)."""
    if not votes:
        return "none"
    counts = [int(v) for v in votes.values()]
    total = sum(counts)
    if total == 0:
        return "none"
    if total == 1:
        return "single"
    top = max(counts)
    if top >= 3:
        return "strong"
    if top == 2:
        return "weak"
    return "none"


def _cited_urls(row: dict, pack: Optional[EvidencePack]) -> list[str]:
    """URLs for the pack ids the panel cited (``E1``…), in citation order.

    Unknown ids (shouldn't happen post-I4) are skipped rather than raised — the
    bridge is a display adapter, not an invariant checkpoint."""
    if pack is None:
        return []
    by_id = {it.pack_id: it for it in pack.items}
    urls: list[str] = []
    for cid in row.get("citations", []) or []:
        it = by_id.get(cid)
        if it is not None and it.source_url:
            urls.append(it.source_url)
    return urls


def _pack_sources(pack: Optional[EvidencePack]) -> list[dict]:
    """Full retrieved pack -> render-ready source dicts (all items, not just cited)."""
    if pack is None:
        return []
    return [
        {"id": it.pack_id, "source": it.source_name, "url": it.source_url,
         "tier": it.tier.value, "snippet": it.snippet,
         "supports_claim": it.supports_claim, "relevance_score": it.relevance_score,
         "role": getattr(it, "role", "") or ""}
        for it in pack.items
    ]


def _pack_to_evidence(sid: str, pack: Optional[EvidencePack]) -> list[Evidence]:
    """Full retrieved pack → ``Evidence`` list (provenance preserved)."""
    if pack is None:
        return []
    out: list[Evidence] = []
    for it in pack.items:
        ev = Evidence(
            claim_id=sid,
            source_name=it.source_name,
            source_url=it.source_url,
            source_tier=it.tier,
            snippet=it.snippet,
            supports_claim=it.supports_claim,
            # P67.5: round-trip the publication date into the artifact —
            # it used to serialize as null, leaving the era lint only the
            # [YYYY-MM-DD] snippet prefix to work from.
            published_at=(datetime.fromisoformat(it.published_at)
                          if getattr(it, "published_at", None) else None),
        )
        # relevance_score has a non-None model default (0.5); only override when the
        # pack item actually carries a score, so undated/unscored packs round-trip clean.
        if it.relevance_score is not None:
            ev.relevance_score = it.relevance_score
        out.append(ev)
    return out


def _build_claim(sid: str, claim_src: Optional[dict]) -> Claim:
    """Reconstruct the ``Claim`` for a bundle from the originating claim dict.

    ``transcript_id`` is the sid prefix before the first ':' (the same convention
    ``evidence_pack.build_evidence_pack`` uses). Falls back to a minimal claim if
    the source dict is missing (keeps the bridge total)."""
    src = claim_src or {}
    text = (src.get("text") or "").strip() or "(claim text unavailable)"
    return Claim(
        transcript_id=sid.split(":", 1)[0],
        text=text,
        speaker=src.get("speaker") or "Unknown",
        context=src.get("context") or None,
        category=src.get("category"),
        speech_date=src.get("date") or src.get("speech_date"),
    )


def _normalize_votes(votes: Optional[dict]) -> dict[str, int]:
    """PCA seat tally with keys normalized to ``VerdictLabel`` display values.

    Vote keys arrive as the 4-label uppercase contract (``TRUE``/``FALSE``/…); map
    them to the enum's display casing (``True``/``False``/…) so the tally reads the
    same as the verdict pills. Unmappable keys pass through verbatim rather than being
    dropped — a stray key is a display curiosity, not worth losing a vote over."""
    out: dict[str, int] = {}
    for k, v in (votes or {}).items():
        lbl = _LABEL_MAP.get(str(k).strip().upper())
        key = lbl.value if lbl is not None else str(k)
        try:
            out[key] = out.get(key, 0) + int(v)
        except (TypeError, ValueError):
            continue
    return out


def _seat_rationales(row: dict) -> list[dict]:
    """The row's per-seat rationales, label-normalized for display (R-3).

    Pass-through of stored text: the reasoning strings are copied VERBATIM and
    only the verdict label is mapped into the published vocabulary. Seats that
    returned no text are kept — a seat that voted and said nothing is itself
    a fact about the panel, and dropping it would overstate agreement."""
    out: list[dict] = []
    for seat in row.get("seat_rationales") or []:
        if not isinstance(seat, dict):
            continue
        lbl = _LABEL_MAP.get(str(seat.get("verdict") or "").strip().upper())
        out.append({
            "role": str(seat.get("role") or ""),
            "verdict": lbl.value if lbl is not None else str(seat.get("verdict") or ""),
            "confidence": seat.get("confidence"),
            "reasoning": str(seat.get("reasoning") or ""),
            "citations": [str(c) for c in (seat.get("citations") or [])],
        })
    return out


def split_rationales(row: dict) -> list[dict]:
    """The seat rationales a PUBLISHED SPLIT must show — one per distinct
    verdict the seats reached, with text (0462 ruling, 2026-08-10).

    A persistent models-split is a legitimate published outcome, not a failure,
    and publishing it as the bare line "Panel split — no consensus verdict"
    tells a reader that the panel disagreed without telling them what about.
    This selects the seats whose reasons a split page has to carry: the first
    seat with text for each distinct label, in seat order."""
    seen: set[str] = set()
    out: list[dict] = []
    for seat in _seat_rationales(row):
        label = seat["verdict"]
        if label in seen or not seat["reasoning"].strip():
            continue
        seen.add(label)
        out.append(seat)
    return out


def _build_provenance(row: dict, claim_src: Optional[dict]) -> VerdictProvenance:
    """Capture the pipeline provenance the reconciled-judge collapse would discard.

    Everything here already exists on the adjudication row (``votes``/``split``/
    ``escalated``/``crm114``) or on the originating claim dict (``layer_a``, threaded
    by ``run_pca_verify`` from the check-worthy queue). We record it structurally so
    per-claim agreement and the Layer A→panel→CRM-114 chain survive into the bundle."""
    src = claim_src or {}
    la = src.get("layer_a") or {}
    crm = row.get("crm114") or {}
    by_role: dict[str, list[str]] = {}
    for role, labels in (row.get("by_role") or {}).items():
        norm = []
        for raw in labels if isinstance(labels, list) else [labels]:
            lbl = _LABEL_MAP.get(str(raw).strip().upper())
            norm.append(lbl.value if lbl is not None else str(raw))
        by_role[str(role)] = norm
    prov = VerdictProvenance(
        layer_a_label=str(la.get("label") or ""),
        layer_a_source=str(la.get("source") or ""),
        layer_a_claim_type=str(la.get("claim_type") or ""),
        layer_a_claim_shape=str(la.get("claim_shape") or ""),
        panel_votes=_normalize_votes(row.get("votes")),
        panel_split=bool(row.get("split", False)),
        panel_escalated=bool(row.get("escalated", False)),
        crm114_stage1=str(crm.get("stage1") or ""),
        crm114_final=str(crm.get("final") or ""),
        panel_by_role=by_role,
        correction_note=str((row.get("corrected") or {}).get("note") or ""),
        # T2.4 gate wiring fix (PR-A2.1): the adjudicator's _forced_uv_row has
        # always written the gate marker as ``provenance_code`` while this
        # field read only ``evidence_gate`` — so no published bundle ever
        # carried the gate. Accept both; ``evidence_gate`` wins if a future
        # writer uses the model field's own name.
        evidence_gate=str(row.get("evidence_gate")
                          or row.get("provenance_code") or ""),
        # Standing agreed-verdict audit (remediation v2, 1.12): stamped onto
        # the row by publish_pipeline's audit stage just before bridging.
        audit_flags=[str(f) for f in (row.get("audit_flags") or [])],
        audit_queue=bool(row.get("audit_queue") or False),
        # R-3 (2026-08-10): the seats' own rationale text, and — when the
        # published rationale was ADOPTED rather than authored — the attribution
        # that says so. Both are pass-through; the bridge adds no words.
        panel_seat_rationales=_seat_rationales(row),
        rationale_provenance=dict(row.get("rationale_provenance") or {}),
    )
    # Computed exhibit (A8 / R-2): the adjudication row is where it is stamped
    # (scripts/wave_adjudicate.py attaches the ratified exhibit to the claims
    # it was built for), and this is the ONE place it crosses into a published
    # bundle. It goes through ``computed_exhibit.attach`` rather than a plain
    # assignment so the admissibility rule — never on a C-EVAL judgment — is
    # enforced on the way in as well as at render time. An inadmissible or
    # malformed exhibit is DROPPED with a warning, never raised: the renderer
    # already refuses to draw one, so failing the whole publish here would
    # trade an identical page for an outage.
    exhibit = row.get("computed_exhibit") or {}
    if exhibit:
        from truthbot.publish import computed_exhibit as _ce
        try:
            _ce.attach(prov, dict(exhibit),
                       claim_shape=prov.layer_a_claim_shape)
        except _ce.InadmissibleExhibit as exc:
            logger.warning("computed exhibit dropped for %s: %s",
                           row.get("sid"), exc)
    return prov


def _consensus_and_panel(
    sid: str, row: dict, cited_urls: list[str], claim_src: Optional[dict] = None,
) -> tuple[ConsensusVerdict, list[ModelVerdict]]:
    """Build the ConsensusVerdict + the single reconciled ModelVerdict for a row."""
    provenance = _build_provenance(row, claim_src)
    status = row.get("status")
    votes = row.get("votes") or {}
    strength = strength_from_votes(votes)
    reasoning = (row.get("reasoning") or "").strip()
    conf = confidence_from_float(row.get("confidence"))

    if status == "resolved":
        raw = (row.get("verdict") or "").strip().upper()
        label = _LABEL_MAP.get(raw)
        if label is None:
            # normalize() fails closed on out-of-contract verdicts, so a resolved
            # row is guaranteed in-contract; guard anyway rather than KeyError.
            raise ValueError(f"bridge: resolved row {sid} carries unmapped verdict {raw!r}")

        # CRM-114 stage-2 may have flipped the label away from the vote plurality.
        crm = row.get("crm114")
        if crm and crm.get("stage1") == "DISAGREEMENT":
            # Tie-routed row: the panel reached NO plurality and the stage-2
            # discriminator resolved the adverse-severity tie — say that, rather
            # than the false "PCA panel resolved X."
            expl = reasoning or (
                f"Panel split with no plurality; the Severity Classifier resolved "
                f"{label.value} on the same evidence pack.")
            expl = f"{expl} (CRM-114: DISAGREEMENT→{crm.get('final')})".strip()
        else:
            expl = reasoning or f"PCA panel resolved {label.value}."
            if crm:
                expl = f"{expl} (CRM-114: {crm.get('stage1')}→{crm.get('final')})".strip()

        mv = ModelVerdict(
            adapter_name=PANEL_ADAPTER,
            model_id=PANEL_MODEL_ID,
            claim_id=sid,
            label=label,
            confidence=conf,
            explanation=expl,
            web_sources=list(cited_urls),
            model_reported_sources=list(cited_urls),
        )
        agreement = len(votes) == 1  # one distinct label across the panel
        # PCA carries the coarse fields as the FINE label itself (2026-07-19
        # review): the Truthy/Falsey umbrellas exist to fold the legacy 6-bucket
        # scale (Mostly True/Exaggerated) — the PCA 4-label set has nothing to
        # fold, and projecting Misleading into a "Falsey" chip overstated the
        # panel's own call on every downstream surface (chip, toggle attrs,
        # distribution, claims.json).
        coarse_lenient = label.value
        coarse_strict = label.value
        consensus = ConsensusVerdict(
            claim_id=sid,
            model_verdicts=[mv],
            consensus_label=label,
            consensus_verdict=label.value,
            confidence=conf,
            agreement=agreement,
            consensus_strength=strength,
            explanation=expl,
            coarse_lenient_label=coarse_lenient,
            coarse_lenient_strength=strength,
            coarse_strict_label=coarse_strict,
            coarse_strict_strength=strength,
            provenance=provenance,
        )
        return consensus, [mv]

    # ── Non-resolved: disagreement / no_label / needs_verdict ────────────────
    # Kept in the report as an explicit UNVERIFIABLE, never silently dropped.
    if status == "disagreement":
        verdict_text = "Models split"
        # 0462 ruling (2026-08-10): a persistent split PUBLISHES, with both
        # sides' reasons shown. The seats' own text is joined verbatim and
        # attributed by role; nothing is written on the panel's behalf. Runs
        # that predate seat-rationale capture have no text and fall back to the
        # original line, so an old bundle renders exactly as it did.
        sides = split_rationales(row)
        if sides:
            expl = "Panel split — no consensus verdict. " + " ".join(
                f"{s['role'].capitalize()} ({s['verdict']}): {s['reasoning']}"
                for s in sides)
        else:
            expl = reasoning or "Panel split — no consensus verdict."
    else:  # no_label / needs_verdict / anything unexpected
        verdict_text = "No verdict"
        expl = reasoning or "No verdict produced for this claim."

    consensus = ConsensusVerdict(
        claim_id=sid,
        model_verdicts=[],
        consensus_label=VerdictLabel.UNVERIFIABLE,
        consensus_verdict=verdict_text,
        confidence=Confidence.LOW,
        agreement=False,
        consensus_strength="none",
        explanation=expl,
        coarse_lenient_label="Models split",
        coarse_lenient_strength="none",
        coarse_strict_label="Models split",
        coarse_strict_strength="none",
        provenance=provenance,
    )
    return consensus, []


def row_to_bundle(
    row: dict,
    *,
    claim_src: Optional[dict] = None,
    pack: Optional[EvidencePack] = None,
) -> VerdictBundle:
    """Bridge one adjudicate row (+ its claim dict and evidence pack) to a bundle."""
    sid = row["sid"]
    claim = _build_claim(sid, claim_src)
    cited = _cited_urls(row, pack)
    consensus, model_verdicts = _consensus_and_panel(sid, row, cited, claim_src)

    return VerdictBundle(
        claim=claim,
        speaker=(claim_src or {}).get("speaker") or claim.speaker or "",
        date_str=(claim_src or {}).get("date_str") or "",
        model_verdicts=model_verdicts,
        consensus=consensus,
        evidence_count=len(pack.items) if pack is not None else 0,
        sources_consulted=_pack_sources(pack),
        cache_hit=False,
    )


def bridge(
    rows: list[dict],
    claims: list[dict],
    packs: Optional[dict[str, EvidencePack]] = None,
) -> BridgeOutput:
    """Bridge an ``adjudicate`` batch to the publish layer.

    Args:
      rows:   verdict-contract rows from ``adjudicator.adjudicate`` (or the
              Layer B pipeline). Each carries at least ``sid`` and ``status``.
      claims: the originating claim dicts (``{"sid","text","context",...}``) —
              the source of each bundle's ``Claim``. Indexed by ``sid``; a row
              with no matching claim still bridges (minimal claim).
      packs:  ``notes["packs"]`` (``sid → EvidencePack``) from an open-book run,
              or ``None``/empty for closed-book (no evidence, no citations).

    Returns a ``BridgeOutput`` (bundles in row order + per-sid evidence corpus).
    Pure and offline — safe to call in unit tests with hand-built rows.
    """
    packs = packs or {}
    claim_by_sid = {c["sid"]: c for c in claims if "sid" in c}

    out = BridgeOutput()
    for row in rows:
        sid = row["sid"]
        pack = packs.get(sid)
        bundle = row_to_bundle(row, claim_src=claim_by_sid.get(sid), pack=pack)
        out.bundles.append(bundle)
        out.evidence[sid] = _pack_to_evidence(sid, pack)
    return out
