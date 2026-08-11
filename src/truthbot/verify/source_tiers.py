"""Canonical source-tier classification (Claim Eval v3 PR-A / D7).

**This is the single API surface.** It used to be two implementations:
``classify_tier`` in the Brave connector decided the tier the pipeline stored,
and ``_tier_bucket`` in :mod:`truthbot.publish.site` re-derived a tier from the
URL at render time to draw its badges. They shared a *matcher*
(:mod:`truthbot.domains`) but kept separate *domain lists*, and had already
drifted. Both now call in here.

That drift matters more than a wrong badge. ``tier`` is one of exactly four
fields invariant **I5** requires on every evidence item
(``hydramind.invariants._REQUIRED_PROVENANCE``), so it is part of the integrity
record — the site must not contradict it. See ``docs/integrity-invariants.md``.

As of remediation v2 item 1.2 (DC-2a, 2026-08-02) this module is a **facade**
over :mod:`truthbot.verify.tier_registry`, whose ``tier_registry.yaml`` is the
versioned source of tier truth (rules are config, not code — one file, every
entry with a rationale and date). The old ``source_tiers.json`` is fully
migrated there and moved to ``quarantine/2026-08-02T2030Z/``.

Why the path rules exist (jackie's ruling, 2026-07-29): partisan government
press releases are admissible only to confirm a claim was **made**, never to
prove it **true**. ``SourceTier.POLITICAL`` is the tier jackie's design note
calls **S5**; the site badges it **T7** because that is its actual rank (last).
S5 and T7 are the same tier under two numbering schemes.
"""
from __future__ import annotations

from truthbot.models import SourceTier
from truthbot.verify.tier_registry import classify_tier_ex


def classify_tier(url: str) -> SourceTier:
    """Assign a trust tier to ``url``.

    Host-suffix matching (:mod:`truthbot.domains`), never substring — a
    substring rule once made ``www.govtech.com`` rank Government because it
    contains ``.gov``, letting a trade magazine win pack slots. Paths are
    consulted only for government-class hosts and explicit path classes.

    Delegates to :func:`truthbot.verify.tier_registry.classify_tier_ex`;
    callers that need to know *why* (e.g. quarantine telemetry) call the
    ``_ex`` form directly.
    """
    return classify_tier_ex(url)[0]


#: Display metadata per tier, consumed by the site renderer. ``code`` is the
#: badge's rank label — POLITICAL is T7 because it ranks last; the design note
#: calls the same tier S5.
TIER_DISPLAY: dict[SourceTier, tuple[str, str]] = {
    SourceTier.GOVERNMENT: ("T1·Gov", "tier-gov"),
    SourceTier.WIRE: ("T2·Wire", "tier-news"),
    SourceTier.ESTABLISHED: ("T3·News", "tier-news"),
    SourceTier.ACADEMIC: ("T4·Acad", "tier-news"),
    SourceTier.FACTCHECK: ("T5·FC", "tier-fc"),
    SourceTier.OTHER: ("T6", "tier-other"),
    SourceTier.POLITICAL: ("T7·Pol", "tier-political"),
}

#: Coarse bucket key per tier, for the site's per-report source tallies.
TIER_BUCKET: dict[SourceTier, str] = {
    SourceTier.GOVERNMENT: "gov",
    SourceTier.WIRE: "wire",
    SourceTier.ESTABLISHED: "news",
    SourceTier.ACADEMIC: "news",
    SourceTier.FACTCHECK: "fc",
    SourceTier.OTHER: "other",
    SourceTier.POLITICAL: "political",
}
