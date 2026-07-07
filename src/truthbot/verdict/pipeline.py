"""
Layer B pipeline (spec §5): take Layer A's check-worthy queue and adjudicate each
claim through the PCA panel, producing verdict-contract rows.

Mirrors checkworthy/pipeline.py — a pure router with an injected `verdict_fn`, so it
is unit-testable with no live lane. If verdict_fn is None (no lane) claims are parked
as `needs_verdict` rather than guessed, the same discipline as Layer A parking
ambiguous items.

Input rows come from LayerAResult.check_worthy_queue and must carry {"sid","text"}.
Note the A1-pass rows carry `text`, but A2-derived check-worthy rows currently drop
it (classifier.classify returns parse_a2 output keyed by sid only) — carrying the
claim text through A2 is a small Layer A follow-up before an end-to-end A→B wiring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

# verdict_fn signature: (list[{"sid","text","context"}]) -> list[verdict-row]
VerdictFn = Callable[[list[dict]], list[dict]]


@dataclass
class LayerBResult:
    verdicts: list[dict] = field(default_factory=list)   # verdict-contract rows
    n_claims: int = 0


def run_layer_b(claims: list[dict], verdict_fn: Optional[VerdictFn] = None) -> LayerBResult:
    """claims: Layer A check-worthy queue rows, each with at least {"sid","text"}."""
    res = LayerBResult(n_claims=len(claims))
    if not claims:
        return res
    if verdict_fn is None:
        res.verdicts = [{"sid": c["sid"], "status": "needs_verdict", "verdict": None,
                         "confidence": None, "citations": [], "votes": {},
                         "split": False, "escalated": False} for c in claims]
        return res
    res.verdicts = list(verdict_fn(claims))
    return res
