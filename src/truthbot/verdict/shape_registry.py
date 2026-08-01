"""Per-run claim-shape registry (PR-A2.3 wiring).

The consolidator's evidential-role quota needs each claim's Layer A
``claim_shape``, but the ``pack_builder`` hook signature is fixed at
``(sid, text, context)`` across three call sites (adjudicator, retrieval
pool, retrieval phase). Rather than widen every hook, the run registers its
claims here right after Layer A builds them — the same per-process-run
pattern as ``speech_context.register_speech_date`` — and the pack builder
looks the shape up by sid.

Speaker-free by construction (shapes come from the speaker-blind A2
classifier), so I3 is untouched. Unregistered sids return "" = legacy
behavior, bit-for-bit.
"""
from __future__ import annotations

_SHAPES: dict[str, str] = {}


def register_claim_shapes(claims: list[dict]) -> int:
    """Record ``sid → layer_a.claim_shape`` for a run's claim dicts
    (``claims_from_queue`` output). Returns how many non-empty shapes were
    registered. Last write wins; a CLI process adjudicates one speech."""
    n = 0
    for c in claims:
        sid = c.get("sid")
        if not sid:
            continue
        shape = (c.get("layer_a") or {}).get("claim_shape") or ""
        _SHAPES[sid] = shape
        n += bool(shape)
    return n


def shape_for(sid: str) -> str:
    """The registered claim shape for ``sid``, or "" (legacy) when unknown."""
    return _SHAPES.get(sid, "")


def clear() -> None:
    """Test hook: reset the registry."""
    _SHAPES.clear()
