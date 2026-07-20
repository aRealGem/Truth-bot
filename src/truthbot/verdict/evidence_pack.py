"""Layer C evidence packs — the structural point where retrieved evidence enters
the verdict panel (design §3.3, invariant I5).

Closed-book Layer B judges from parametric knowledge alone; that collapses on
current events (confident staleness / coverage gaps). Layer C fetches evidence
TIME-SCOPED to the claim's era and hands the panel a small, provenanced pack so it
can commit — open-book — with citations back to the pack.

This module owns four things:

  * ``window_for`` — derive the retrieval window from a sid, reusing the SAME rule
    (``expected_claim_window``) the temporal preamble uses, so retrieval and the
    preamble share one notion of "the claim's era".
  * ``build_evidence_pack`` — fetch via the ``EvidenceProvider`` port, dedup, rank
    by source trust, cap, assign stable pack ids (``E1..En``), and stamp provenance.
  * I5 enforcement — every pack item must carry ``url/retrieved_at/sha256/tier``
    (``check_i5_provenance``); malformed evidence fails closed here, at entry, so it
    can never reach a verdict.
  * payload / render helpers — the model-facing evidence list (JSON payload) and a
    human-readable block for logs and tests.

Speaker-blind (I3): nothing here conditions on who made the claim — only its sid
(which encodes the utterance DATE) and text.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from hydramind.invariants import check_i5_provenance
from truthbot.domains import is_substantive_url
from truthbot.models import Claim, Evidence, SourceTier
from truthbot.verify.evidence_provider import EvidenceProvider
from truthbot.verify.sources.base import TimeWindow

from . import speech_context

# Source-trust order (design: Government > Wire > Established > Academic >
# FactCheck > Other). Lower rank = more trusted → surfaced first in the pack.
_TIER_RANK: dict[SourceTier, int] = {
    SourceTier.GOVERNMENT: 0,
    SourceTier.WIRE: 1,
    SourceTier.ESTABLISHED: 2,
    SourceTier.ACADEMIC: 3,
    SourceTier.FACTCHECK: 4,
    SourceTier.OTHER: 5,
}

DEFAULT_MAX_ITEMS = 6


def window_for(sid: str, *, today: Optional[date] = None) -> TimeWindow:
    """Retrieval window for a sid, or ``None`` when the utterance date is unknown.

    Shares ``expected_claim_window`` with the temporal preamble so evidence is
    scoped to the same era the panel is told to judge as-of."""
    utt = speech_context.speech_date_for(sid)
    if utt is None:
        return None
    return speech_context.expected_claim_window(utt)


@dataclass(frozen=True)
class PackItem:
    """One provenanced evidence item in a pack, addressable by ``pack_id``."""

    pack_id: str            # E1, E2, ... — the id the panel cites (I4)
    source_name: str
    source_url: str
    tier: SourceTier
    snippet: str
    retrieved_at: str       # ISO8601
    sha256: str             # content hash of url+snippet (integrity / dedup)

    def provenance(self) -> dict:
        """The I5 provenance record (``url/retrieved_at/sha256/tier`` required)."""
        return {
            "url": self.source_url,
            "retrieved_at": self.retrieved_at,
            "sha256": self.sha256,
            "tier": self.tier.value,
        }

    def to_payload(self) -> dict:
        """Model-facing view (goes into the JSON payload the panel reads)."""
        return {
            "id": self.pack_id,
            "source": self.source_name,
            "tier": self.tier.value,
            "url": self.source_url,
            "snippet": self.snippet,
        }


@dataclass(frozen=True)
class EvidencePack:
    """A claim's assembled evidence: ordered items + their citable ids."""

    sid: str
    window: TimeWindow
    items: list[PackItem] = field(default_factory=list)

    @property
    def ids(self) -> list[str]:
        return [it.pack_id for it in self.items]

    def to_payload(self) -> list[dict]:
        return [it.to_payload() for it in self.items]

    def render(self) -> str:
        """Human-readable block (logs / tests). Empty string when the pack is empty."""
        if not self.items:
            return ""
        lines = []
        for it in self.items:
            lines.append(f"[{it.pack_id}] {it.source_name} ({it.tier.value}) {it.source_url}")
            lines.append(f"    {it.snippet}")
        return "\n".join(lines)


def _sha256(url: str, snippet: str) -> str:
    return hashlib.sha256(f"{url}\n{snippet}".encode("utf-8")).hexdigest()


def _retrieved_iso(ev: Evidence) -> str:
    ra = ev.retrieved_at or datetime.now(timezone.utc)
    return ra.isoformat()


def _within_window(ev: Evidence, window: TimeWindow) -> bool:
    """Era filter: a DATED item outside the claim's window is dropped; undated
    items pass (the window can't adjudicate them). Belt-and-suspenders behind the
    connectors' freshness scoping — Brave's filter is advisory, and a fact-check
    ruling published years after the utterance must not enter the pack."""
    if window is None or ev.published_at is None:
        return True
    start, end = window
    return start <= ev.published_at.date() <= end


def _dedup_rank_cap(evidence: list[Evidence], max_items: int) -> list[Evidence]:
    """Drop duplicate URLs (first wins), stably rank relevance-then-tier, cap.

    Relevance beats tier (P67 Round B item 3): tier-first is how an off-topic
    .gov speech topped an on-topic pack. Unscored evidence carries the neutral
    default (0.5), so a pack with no relevance layer ties on relevance and
    falls through to the old trust-tier ordering unchanged.

    Items without a URL are dropped — I5 requires a url, and an unaddressable
    snippet cannot be cited or re-verified."""
    seen: set[str] = set()
    unique: list[Evidence] = []
    for ev in evidence:
        url = (ev.source_url or "").strip()
        if not url:
            continue
        if not is_substantive_url(url):
            # A homepage or listing index can never BE evidence — it only
            # points at a site (the snopes.com/?pagenum=3 pack-slot bug).
            continue
        key = url.lower().rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        unique.append(ev)
    unique.sort(key=lambda e: (-(e.relevance_score if e.relevance_score is not None else 0.5),
                               _TIER_RANK.get(e.source_tier, 99)))  # stable
    return unique[:max_items]


def build_evidence_pack(
    sid: str,
    claim_text: str,
    provider: EvidenceProvider,
    *,
    today: Optional[date] = None,
    max_items: int = DEFAULT_MAX_ITEMS,
    context: str = "",
) -> EvidencePack:
    """Fetch, dedup, rank, cap, and provenance-stamp evidence for one claim.

    Retrieval is time-scoped via ``window_for(sid)``. Each surviving item is
    assigned a stable ``E<n>`` id and validated against I5 (``check_i5_provenance``)
    — a provenance gap fails closed here, at evidence entry, not at verdict time."""
    window = window_for(sid, today=today)
    claim = Claim(transcript_id=sid.split(":", 1)[0], text=claim_text, context=context or None)
    raw = provider.get_evidence(claim, window=window)
    raw = [ev for ev in raw if _within_window(ev, window)]
    kept = _dedup_rank_cap(raw, max_items)

    items: list[PackItem] = []
    for i, ev in enumerate(kept, start=1):
        item = PackItem(
            pack_id=f"E{i}",
            source_name=ev.source_name or "Unknown",
            source_url=ev.source_url,
            tier=ev.source_tier,
            snippet=ev.snippet or "",
            retrieved_at=_retrieved_iso(ev),
            sha256=_sha256(ev.source_url, ev.snippet or ""),
        )
        check_i5_provenance(item.provenance())  # I5: fail closed at entry
        items.append(item)
    return EvidencePack(sid=sid, window=window, items=items)
