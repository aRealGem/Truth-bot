"""Evidence-mode enum (P67.7 / remediation T2.7).

Three retrieval stacks, recorded per run so ablations stay comparable:

* ``closed_book`` — no evidence; the panel judges from parametric knowledge
  (I4 requires empty citations).
* ``shared_pack_v1`` — the Round-B stack: Brave + FactCheck connectors,
  relevance middle step, 6-item packs, reserved fact-check slot. Retained
  as the ablation baseline; fact-check rulings ARE in model context here
  (the audited F5 behavior).
* ``shared_pack_v2`` — the evidence-v2 stack: retriever shortlists through
  the deterministic consolidator, fact-checker exclusion, fair-game era
  filter, tier quotas, 10-item cap, quality gate.
"""
from __future__ import annotations

from enum import Enum


class EvidenceMode(str, Enum):
    CLOSED_BOOK = "closed_book"
    SHARED_PACK_V1 = "shared_pack_v1"
    SHARED_PACK_V2 = "shared_pack_v2"

    @classmethod
    def infer_legacy(cls, open_book: bool) -> "EvidenceMode":
        """Mode for runs that predate the enum: open-book runs were v1."""
        return cls.SHARED_PACK_V1 if open_book else cls.CLOSED_BOOK
