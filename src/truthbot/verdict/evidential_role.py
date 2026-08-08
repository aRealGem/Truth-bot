"""Evidential role — the claim-relative second axis (PR-A2.3, D11-approved).

``role = f(claim_shape, principal_relation)``. The tier ladder stays global
and speaker-neutral; this axis captures the evidence-law distinction between a
record of an official act (admissible, with corroboration) and a self-serving
assertion (attribution only). Design note: wiki
``projects:truthbot:evidential-role-design`` (D11 sign-off 2026-08-01).

Effects live in the consolidator quota and the pack payload; this module is
the pure table.
"""
from __future__ import annotations

from enum import Enum

from truthbot.verdict.utterance_record import ROLE as _UTTERANCE_RECORD
from truthbot.verify.principals import PrincipalRelation

#: Ministerial shapes — records of the speaker's own official acts/quantities.
MINISTERIAL_SHAPES = {"c-exist", "c-count"}


class EvidentialRole(str, Enum):
    PRIMARY_RECORD = "primary-record"      # c-exist/c-count × SELF: ≤1 quota slot
    CORROBORANT = "corroborant"            # c-exist/c-count × PARTICIPANT: fills independent slot
    ATTRIBUTION_ONLY = "attribution-only"  # c-eval × SELF: weight 0, satisfies nothing
    PLAIN_S5 = "plain-s5"                  # c-third × SELF: no special role
    NORMAL = "normal"                      # everything else: tier handling unchanged
    # D15 (PROPOSED, flag-gated — docs/decisions/D15-utterance-derivative.md).
    # A record of the UTTERANCE ITSELF: the transcript, the Congressional
    # Record of the day, the archive copy. NOT a product of the D11.2 table
    # below — it is f(url, snippet, speech_date), so it is assigned by
    # ``verdict.utterance_record`` detection and is NEVER returned by
    # :func:`evidential_role`. Quota credit 0; display allowed as provenance.
    UTTERANCE_RECORD = _UTTERANCE_RECORD    # "utterance-record"


def evidential_role(claim_shape: str | None,
                    relation: PrincipalRelation) -> EvidentialRole:
    """The D11.2 role table, encoded exactly. Unknown/legacy shapes ('' / None)
    map to NORMAL — pre-shape claims keep today's behavior bit-for-bit."""
    if relation is PrincipalRelation.INDEPENDENT or not claim_shape:
        return EvidentialRole.NORMAL
    if claim_shape in MINISTERIAL_SHAPES:
        return (EvidentialRole.PRIMARY_RECORD
                if relation is PrincipalRelation.SELF
                else EvidentialRole.CORROBORANT)
    if relation is PrincipalRelation.SELF:
        return (EvidentialRole.ATTRIBUTION_ONLY if claim_shape == "c-eval"
                else EvidentialRole.PLAIN_S5)
    # PARTICIPANT on a non-ministerial shape earns nothing special.
    return EvidentialRole.NORMAL
