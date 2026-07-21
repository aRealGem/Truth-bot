"""Deterministic evidence consolidator — shared_pack_v2 (P67.7 / T2.3-T2.4).

Design note: wiki ``projects:truthbot:evidence-v2-design`` (published before
this code, per the wiki-first rule). The consolidator takes the retriever
shortlists (PR-5: R1 Opus/Lane-Worker native search, R2 GPT-5.5 browsing,
R3 pending D1) and assembles the pack with NO small-model re-scoring — every
step is pure code, so the same shortlists always produce the same pack:

  1. round-robin merge of the shortlists (each retriever's own order is
     preserved; no retriever can dominate the head of the pack),
  2. URL dedup (first drawn wins),
  3. era filter — the originally-coded window AND the fair-game window
     (utterance + 7 days), via :mod:`truthbot.verdict.era_lint`,
  4. fact-checker exclusion (T2.1) — enforced here even though the
     retrievers filter too (a retriever that forgets is caught),
  5. junk-URL filter (homepages / listing pages),
  6. tier quotas: at least ``MIN_BEARING_T13`` Tier-1..3 items bearing on
     the core assertion (stance supports/refutes), at most ``MAX_T6`` OTHER
     items; excess OTHER items are dropped lowest-priority-first,
  7. cap at ``PACK_CAP_V2`` (10) items, ordered by (draw round, tier rank).

Quality gate (T2.4): quota unmet → the caller performs ONE targeted
re-retrieval and consolidates again; still unmet → the claim's verdict is
FORCED Unverifiable with provenance code ``insufficient-qualifying-evidence``
(``GATE_INSUFFICIENT``). No silent thin-pack verdicts.

evidence_pack v2.0 payload per item: {url, date, tier, stance, one_line_why}.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Sequence

from truthbot.domains import is_substantive_url
from truthbot.models import Evidence, SourceTier
from truthbot.verify.factcheck_exclusion import is_excluded_factchecker

from . import era_lint

PACK_CAP_V2 = 10
MIN_BEARING_T13 = 2          # ≥2 Tier-1..3 items bearing on the core assertion
MAX_T6 = 2                   # ≤2 OTHER-tier items
GATE_INSUFFICIENT = "insufficient-qualifying-evidence"

SCHEMA_VERSION = "evidence_pack v2.0"

_TIER_RANK = {
    SourceTier.GOVERNMENT: 0,
    SourceTier.WIRE: 1,
    SourceTier.ESTABLISHED: 2,
    SourceTier.ACADEMIC: 3,
    SourceTier.FACTCHECK: 4,   # unreachable in v2 (excluded), kept for totality
    SourceTier.OTHER: 5,
}

_T13 = {SourceTier.GOVERNMENT, SourceTier.WIRE, SourceTier.ESTABLISHED}


@dataclass(frozen=True)
class ConsolidatedItem:
    """One v2 pack item plus its merge provenance."""
    evidence: Evidence
    draw_round: int          # which round-robin round drew it
    retriever: str           # shortlist label it came from

    def to_payload_v2(self) -> dict:
        ev = self.evidence
        stance = {True: "supports", False: "refutes"}.get(ev.supports_claim, "context")
        return {
            "url": ev.source_url,
            "date": ev.published_at.date().isoformat() if ev.published_at else None,
            "tier": ev.source_tier.value,
            "stance": stance,
            "one_line_why": (ev.snippet or "").strip()[:200],
        }


@dataclass
class ConsolidationResult:
    sid: str
    items: list[ConsolidatedItem] = field(default_factory=list)
    quota_met: bool = False
    gate_code: str = ""              # GATE_INSUFFICIENT when forced-UV applies
    dropped: dict[str, int] = field(default_factory=dict)  # reason -> count

    @property
    def schema_version(self) -> str:
        return SCHEMA_VERSION

    def to_payload(self) -> list[dict]:
        return [it.to_payload_v2() for it in self.items]


def _bearing(ev: Evidence) -> bool:
    """An item 'bears on the core assertion' when the stance layer classified
    it as supports or refutes (not ambiguous context)."""
    return ev.supports_claim is True or ev.supports_claim is False


def _round_robin(shortlists: Sequence[tuple[str, Sequence[Evidence]]]):
    """Yield (draw_round, retriever_label, evidence) interleaving the lists."""
    idx = 0
    while True:
        emitted = False
        for label, items in shortlists:
            if idx < len(items):
                emitted = True
                yield idx, label, items[idx]
        if not emitted:
            return
        idx += 1


def consolidate(
    sid: str,
    shortlists: Sequence[tuple[str, Sequence[Evidence]]],
    *,
    utterance: Optional[date],
    window: Optional[tuple[date, date]] = None,
    max_items: int = PACK_CAP_V2,
) -> ConsolidationResult:
    """Assemble a shared_pack_v2 pack from retriever shortlists.

    ``shortlists`` is an ordered sequence of ``(retriever_label, items)``;
    each retriever's list is in ITS preference order. Deterministic: no
    randomness, no model calls, stable ordering throughout."""
    result = ConsolidationResult(sid=sid)
    seen: set[str] = set()
    kept: list[ConsolidatedItem] = []

    def _drop(reason: str) -> None:
        result.dropped[reason] = result.dropped.get(reason, 0) + 1

    for draw_round, label, ev in _round_robin(shortlists):
        url = (ev.source_url or "").strip()
        if not url:
            _drop("empty-url")
            continue
        key = url.rstrip("/").lower()
        if key in seen:
            _drop("duplicate-url")
            continue
        seen.add(key)
        if is_excluded_factchecker(url) or ev.source_tier == SourceTier.FACTCHECK:
            _drop("factcheck-excluded")
            continue
        if not is_substantive_url(url):
            _drop("non-substantive-url")
            continue
        d = era_lint.item_date(ev.published_at, ev.snippet or "")
        if d is not None:
            if window is not None and not (window[0] <= d <= window[1]):
                _drop("outside-coded-window")
                continue
            if utterance is not None and d > era_lint.fair_game_end(utterance):
                _drop("after-fair-game-window")
                continue
        kept.append(ConsolidatedItem(evidence=ev, draw_round=draw_round, retriever=label))

    # T6 quota: keep at most MAX_T6 OTHER items, dropping the lowest-priority
    # (latest-drawn, then worst-tier — here all OTHER, so latest-drawn) first.
    others = [it for it in kept if it.evidence.source_tier == SourceTier.OTHER]
    if len(others) > MAX_T6:
        to_drop = {id(it) for it in others[MAX_T6:]}
        kept = [it for it in kept if id(it) not in to_drop]
        result.dropped["t6-quota"] = len(others) - MAX_T6

    # Final order: draw round, then tier rank — the round-robin merge is the
    # primary ranking (T2.3), tier breaks ties within a round.
    kept.sort(key=lambda it: (it.draw_round, _TIER_RANK[it.evidence.source_tier]))
    result.items = kept[:max_items]
    if len(kept) > max_items:
        result.dropped["pack-cap"] = len(kept) - max_items

    bearing_t13 = sum(
        1 for it in result.items
        if it.evidence.source_tier in _T13 and _bearing(it.evidence))
    result.quota_met = bearing_t13 >= MIN_BEARING_T13
    if not result.quota_met:
        result.gate_code = GATE_INSUFFICIENT
    return result
