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
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from hydramind.invariants import check_i5_provenance
from truthbot.domains import is_substantive_url
from truthbot.models import Claim, Evidence, SourceTier
from truthbot.verify.evidence_provider import EvidenceProvider
from truthbot.verify.sources.base import TimeWindow

from . import era_lint, speech_context

logger = logging.getLogger(__name__)

# Source-trust order (design: Government > Wire > Established > Academic >
# FactCheck > Other). Lower rank = more trusted → surfaced first in the pack.
_TIER_RANK: dict[SourceTier, int] = {
    SourceTier.GOVERNMENT: 0,
    SourceTier.WIRE: 1,
    SourceTier.ESTABLISHED: 2,
    SourceTier.ACADEMIC: 3,
    SourceTier.FACTCHECK: 4,
    SourceTier.OTHER: 5,
    SourceTier.POLITICAL: 6,   # S5 — ranks below OTHER (Claim Eval v3 PR-A / D7)
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
    # Relevance-layer signals (P67 Round B.5). Carried through from the source
    # Evidence so the panel can SEE the stance the relevance step already computed
    # — previously dropped here, invisible to the seats. None when no relevance
    # layer ran (the payload then stays byte-identical to the pre-B.5 pack).
    supports_claim: Optional[bool] = None    # True=supports, False=refutes, None=ambiguous/unscored
    relevance_score: Optional[float] = None  # 0–1 relevance to the claim
    # Publication date, ISO YYYY-MM-DD (P67.5). Until this field existed the
    # date survived only as a [YYYY-MM-DD] snippet prefix — artifacts
    # serialized published_at=null and the era lint had nothing to check.
    published_at: Optional[str] = None
    # Evidential role (PR-A2.3): primary-record | corroborant |
    # attribution-only | plain-s5 | "" (legacy / normal). Display + payload
    # metadata; NOT part of the I5 provenance quad.
    role: str = ""
    # Era note (remediation v2, 1.3): "post-speech · context-only" for items
    # dated after the utterance but within fair-game. Display + payload
    # metadata; such items never credit the quota. NOT part of the I5 quad.
    era_note: str = ""
    # D17-c series excerpt (wave 2): the observations behind an arithmetic
    # claim, carried structurally so the panel AND the reader get the rows, the
    # window that produced them, and what the window left out. None for every
    # non-series item. Display + payload metadata; NOT part of the I5 quad —
    # that quad attests to the SOURCE, and these rows are a view of it, so
    # folding them in would blur what the hash covers.
    series_rows: Optional[dict] = None

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
        payload = {
            "id": self.pack_id,
            "source": self.source_name,
            "tier": self.tier.value,
            "url": self.source_url,
            "snippet": self.snippet,
        }
        # Only surface stance when the relevance layer actually classified it
        # (supports/refutes). None is omitted so packs built without a relevance
        # layer — closed-book, --no-relevance A/B — carry the exact prior payload.
        stance = _stance_label(self.supports_claim)
        if stance is not None:
            payload["stance"] = stance
        # Evidential role (PR-A2.3): surfaced so the panel can see that an
        # attribution-only self record is non-probative. Omitted when unset /
        # normal — pre-shape packs carry the exact prior payload.
        if self.role and self.role != "normal":
            payload["role"] = self.role
        # Post-speech band (remediation v2, 1.3): the panel is told the item
        # may inform context but must not decide the verdict.
        if self.era_note:
            payload["era_note"] = self.era_note
        # D17-c series excerpt (wave 2): the actual observations, so a seat can
        # do the arithmetic instead of taking a snippet's word for it. Omitted
        # when unset, so every non-series pack carries the exact prior payload.
        #
        # The window's own limits ride WITH the rows — rows_shown of
        # total_rows_in_full_table, the predicate that chose them, and the
        # mismatch flag where the window does not reach the claim's period. A
        # seat handed 25 rows and not told 779 were withheld is being invited
        # to over-read them.
        if self.series_rows:
            payload["series_rows"] = self.series_rows
        return payload


@dataclass(frozen=True)
class EvidencePack:
    """A claim's assembled evidence: ordered items + their citable ids.

    ``gate_code`` (shared_pack_v2 only, T2.4): non-empty when the pack failed
    the quality gate after its one targeted re-retrieval — the claim's verdict
    is then FORCED Unverifiable upstream of any panel call. v1 packs always
    carry "". The code is banked on the adjudication ROW (which journals), so
    it deliberately does not round-trip through the chunk journal's
    Evidence-list serialization."""

    sid: str
    window: TimeWindow
    items: list[PackItem] = field(default_factory=list)
    gate_code: str = ""
    # Post-filter/post-quota candidate pool BEFORE the pack cap (PR-A2.2),
    # superset of ``items`` in the same order; empty on v1 packs and when the
    # pool fit under the cap. Never part of the panel payload or I4 id space —
    # it exists so the packs journal can persist what the cap discarded and a
    # cap/quota change can be measured offline without re-retrieval.
    pool: list[PackItem] = field(default_factory=list)
    # Fact-check exclusion log (remediation v2, 1.1): what the consolidator
    # dropped as fact-checker content and why — journaled with the pack,
    # never part of the panel payload. Exclusions are never silent.
    excluded_fc: list[dict] = field(default_factory=list)
    # Quarantine telemetry (remediation v2, 1.2 / S-6): URLs of kept items
    # whose POLITICAL tier came from the fail-closed quarantine of an unmapped
    # government-class host ("quarantine-unmapped-gov"), not from a mapped
    # rule — journaled with the pack, never part of the panel payload.
    quarantined: list[str] = field(default_factory=list)
    # Scoring-coverage telemetry (remediation v2 Phase A, A1):
    # ``consolidator.scoring_telemetry`` over this pack's items — how many
    # carry the untouched default relevance vs a real score, and the stance
    # True/False/None split. Set by the v2 builder; empty on v1 packs.
    # Journaled with the pack, never part of the panel payload. This is the
    # number that makes "the v2 path never scores relevance or stance"
    # visible on disk instead of inferable only by reading the call graph.
    scoring: dict = field(default_factory=dict)

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


def _stance_label(supports: Optional[bool]) -> Optional[str]:
    """Legible stance for the panel payload, from the relevance layer's
    supports/refutes classification. None (ambiguous or unscored) → omitted."""
    if supports is True:
        return "supports"
    if supports is False:
        return "refutes"
    return None


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


def _within_fair_game(ev: Evidence, utterance: Optional[date]) -> bool:
    """Fair-game filter (P67.5 / T1.1): a DATED item published after the
    speaker's fair-game window (utterance + 7 days) is dropped — the coded
    window ran to speech-month+3 and let post-utterance world-state (the
    Iran-war price surge, the shutdown resolution) falsify claims the
    audience heard months earlier. Undated items pass, as in _within_window."""
    if utterance is None or ev.published_at is None:
        return True
    keep = ev.published_at.date() <= era_lint.fair_game_end(utterance)
    if not keep:
        logger.info(
            "era gate: dropped %s — dated %s, observed after the speaker's "
            "fair-game window (utterance %s + %d days)",
            ev.source_url, ev.published_at.date(), utterance,
            era_lint.FAIR_GAME_DAYS)
    return keep


def _dedup_rank_cap(evidence: list[Evidence], max_items: int) -> list[Evidence]:
    """Drop duplicate URLs (first wins), stably rank relevance-then-tier, cap.

    Relevance beats tier (P67 Round B item 3): tier-first is how an off-topic
    .gov speech topped an on-topic pack. Unscored evidence carries the neutral
    default (0.5), so a pack with no relevance layer ties on relevance and
    falls through to the old trust-tier ordering unchanged.

    Items without a URL are dropped — I5 requires a url, and an unaddressable
    snippet cannot be cited or re-verified.

    Fact-checker content is EXCLUDED, never reserved (remediation v2 Phase A,
    A2). A "reserved fact-check slot" used to live here: if no FactCheck-tier
    item made the cap but one existed, it displaced the last slot, so a
    PolitiFact or factcheck.org ruling could not be crowded out (the
    biden_2022:0342 case, P67 Round B.5). T2.1 then made the opposite call —
    truth-bot reaches its own verdict from primary sources and must never
    launder another outlet's ruling into its evidence — and the v2 consolidator
    drops fact-checkers outright. The v1 builder kept FORCING one in. Nothing
    on the v2 path called it, so no shipped verdict came from it; it sat here
    as a trap for the next caller of build_evidence_pack, guaranteeing the one
    item current policy most wants out of the pack. Exclusion now matches
    consolidate(): the same ``factcheck_exclusion_reason`` domain/path rules,
    plus the tier itself."""
    from truthbot.verify.factcheck_exclusion import factcheck_exclusion_reason

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
        if factcheck_exclusion_reason(url) or ev.source_tier == SourceTier.FACTCHECK:
            # T2.1, enforced on BOTH pack paths: another outlet's ruling is not
            # our evidence. Same two-part test consolidate() applies (blocklist
            # domain/path rules OR the FACTCHECK tier), so a retriever that
            # tiers a ruling as Established is still caught, and a
            # FACTCHECK-tiered host outside the blocklist is too.
            logger.info("v1 pack: excluded fact-checker content %s", url)
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
    era_exempt: bool = False,
) -> EvidencePack:
    """Fetch, dedup, rank, cap, and provenance-stamp evidence for one claim.

    Retrieval is time-scoped via ``window_for(sid)``. Each surviving item is
    assigned a stable ``E<n>`` id and validated against I5 (``check_i5_provenance``)
    — a provenance gap fails closed here, at evidence entry, not at verdict time."""
    window = window_for(sid, today=today)
    utterance = speech_context.speech_date_for(sid)
    if utterance is None and not era_exempt:
        # Fail CLOSED (remediation v2, 1.3): an unregistered speech date used
        # to silently disable ALL era gating — the Obama-2014 rescue leg
        # shipped 2026-dated evidence into a 2014 speech this way.
        raise era_lint.EraLintError(
            f"no utterance date registered for {sid!r} — the era gate cannot "
            "run. Call speech_context.register_speech_date() first, or pass "
            "era_exempt=True for a deliberately dateless build.")
    claim = Claim(transcript_id=sid.split(":", 1)[0], text=claim_text, context=context or None)
    raw = provider.get_evidence(claim, window=window)
    raw = [ev for ev in raw if _within_window(ev, window)]
    raw = [ev for ev in raw if _within_fair_game(ev, utterance)]
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
            supports_claim=ev.supports_claim,
            relevance_score=ev.relevance_score,
            published_at=(ev.published_at.date().isoformat()
                          if ev.published_at else None),
        )
        check_i5_provenance(item.provenance())  # I5: fail closed at entry
        items.append(item)
    pack = EvidencePack(sid=sid, window=window, items=items)
    # T1.1: the build FAILS on era violations — defense in depth behind the
    # two filters above (a violation here means a filter regressed).
    if not era_exempt:
        era_lint.assert_pack_within_era(pack, utterance)
    return pack
