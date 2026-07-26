"""B1 phase-split (P120): build the evidence packs for the WHOLE check-worthy
queue BEFORE the panel runs (Phase R), so retrieval spend is journaled at build
time and crash-proof, and the panel loop (Phase P) consumes prebuilt packs by
lookup instead of retrieving mid-chunk.

Retrieval is serial here; P120 PR-2 swaps in an adaptive parallel pool behind the
same ``build_packs_phase`` seam (the resource-aware worker pool), so this module
is the structural boundary the parallelism lands on. Design note:
wiki ``projects:truthbot:batch-lanes-design`` §3 B1.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from truthbot.verdict.evidence_pack import EvidencePack

logger = logging.getLogger(__name__)

# Same shape as adjudicator.PackBuilder: (sid, text, context) -> EvidencePack.
PackBuilder = Callable[[str, str, str], EvidencePack]


def _gated_empty(sid: str) -> EvidencePack:
    """An empty, gate-failed pack — adjudicate forces it Unverifiable (T2.4)."""
    from truthbot.verdict.consolidator import GATE_INSUFFICIENT
    return EvidencePack(sid=sid, window=None, items=[], gate_code=GATE_INSUFFICIENT)


def packs_only_builder(packs: dict) -> PackBuilder:
    """A ``pack_builder`` that RETRIEVES NOTHING: it returns the Phase R pack for a
    sid. Wire this into the Phase P adjudicate lane so the panel consumes prebuilt
    packs. On a miss — which should never happen, Phase R builds every todo sid —
    it logs loudly and returns a gate-forced empty pack, so the claim becomes
    Unverifiable rather than crashing the chunk (same fail-closed shape as a thin
    pack, never a silent verdict)."""
    def _lookup(sid: str, text: str, context: str) -> EvidencePack:
        pack = packs.get(sid)
        if pack is None:
            logger.warning("phase-split: no Phase R pack for %s — forcing "
                           "Unverifiable (this is a bug, not a thin pack)", sid)
            return _gated_empty(sid)
        return pack
    return _lookup


def build_packs_phase(
    claims: list[dict],
    pack_builder: PackBuilder,
    *,
    journal_path=None,
    resume_packs: Optional[dict] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Build (and journal) an ``EvidencePack`` for every claim — Phase R, serial.

    Args:
      claims:       adjudicate-shaped dicts ``[{"sid","text","context",…}]``
                    (from ``publish_pipeline.claims_from_queue``).
      pack_builder: the real retriever-trio builder (``_build_v2_pack_builder``).
      resume_packs: sid → EvidencePack already built and journaled by a prior run;
                    their sids are skipped (retrieval spend already banked).
      journal_path: when set, each freshly built pack is appended to the Phase R
                    packs journal immediately, so a mid-phase crash loses at most
                    the in-flight claim.
      on_progress:  optional ``callback(i, n, sid)`` per built pack, for CLI logging.

    Returns sid → EvidencePack for the full claim set (resumed + freshly built).
    A ``pack_builder`` exception propagates (matching the inline path, where it
    surfaces via the chunk's partial-result channel) — everything built before it
    is already journaled, so the run resumes from there.
    """
    from truthbot.verdict import publish_pipeline as pp

    packs: dict = dict(resume_packs or {})
    todo = [c for c in claims if c["sid"] not in packs]
    n = len(todo)
    for i, c in enumerate(todo, 1):
        sid = c["sid"]
        pack = pack_builder(sid, c["text"], c.get("context", ""))
        packs[sid] = pack
        if journal_path is not None:
            pp.append_packs_journal(journal_path, sid, pack)
        if on_progress is not None:
            on_progress(i, n, sid)
    return packs
