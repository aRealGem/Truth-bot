"""shared_pack_v2 pack builder — retriever trio → deterministic consolidator
(P67.9, wiring the T2.3/T2.4 core into the publish path; design note at wiki
``projects:truthbot:evidence-v2-design``).

``build_evidence_pack_v2`` is the v2 counterpart of
``evidence_pack.build_evidence_pack``: same output type (``EvidencePack``, so
the panel payload, I4 citation checking, journal and bridge are unchanged), but
the pack is assembled by the R1/R2/R3 retriever shortlists through
``consolidator.consolidate`` instead of connector search + ``_dedup_rank_cap``.
The v1 builder — including its reserved fact-check slot — stays untouched as
the ablation baseline (B.5 handoff: lift only at explicit cutover).

Quality gate (T2.4): quota unmet → exactly ONE targeted re-retrieval (the
retry asks specifically for Tier-1..3 sources bearing on the core assertion)
→ consolidate over the union → still unmet → the returned pack carries
``gate_code == GATE_INSUFFICIENT`` and the caller (adjudicate) FORCES the
verdict Unverifiable without spending a panel call. No silent thin-pack
verdicts.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Callable, Optional, Sequence

from hydramind.invariants import check_i5_provenance
from truthbot.verify.retrievers import Retriever

from . import era_lint, speech_context
from .consolidator import PACK_CAP_V2, consolidate
from .evidence_pack import EvidencePack, PackItem, _retrieved_iso, _sha256, window_for

logger = logging.getLogger(__name__)

# Appended to the claim context on the T2.4 retry — the ONE targeted
# re-retrieval. The retrieval prompt already demands primary sources; the
# retry narrows to what the quota actually needs.
_RETRY_FOCUS = (
    "TARGETED RE-RETRIEVAL: an earlier pass found too few qualifying items. "
    "Return ONLY primary/official (government/agency), wire-service, or "
    "established-outlet pages that DIRECTLY support or refute the claim's "
    "core assertion — no background explainers. ")

# Lenient-mode retrieval guidance (historical-era policy). The strict prompt's
# "later items will be discarded" language would suppress exactly the
# retrospective sources lenient mode admits, so pre-web claims get the era
# brief through the CONTEXT channel instead of the utterance/window params
# (no Retriever interface change).
_HISTORICAL_FOCUS = (
    "HISTORICAL CLAIM from a speech given on {utterance}. Ideal sources are "
    "archival originals carrying their era publication dates (government "
    "statistical releases, FRASER/govinfo/agency archives, newspaper "
    "archives from that period). Reliable retrospective historical sources "
    "about that period are also acceptable. ")

# A shortlist runner fans the per-retriever calls out (serial by default; P120
# PR-2 injects a concurrent one). It takes the retriever pool and a per-retriever
# ``call(retriever) -> list[Evidence]`` and MUST return results in pool order.
ShortlistRunner = Callable[[Sequence[Retriever], Callable[[Retriever], list]], list]


def _serial_runner(pool: Sequence[Retriever],
                   call: Callable[[Retriever], list]) -> list:
    return [call(r) for r in pool]


def build_evidence_pack_v2(
    sid: str,
    claim_text: str,
    retrievers: Sequence[Retriever],
    *,
    retry_retrievers: Optional[Sequence[Retriever]] = None,
    today: Optional[date] = None,
    context: str = "",
    max_items: int = PACK_CAP_V2,
    shortlist_runner: Optional[ShortlistRunner] = None,
) -> EvidencePack:
    """Assemble a shared_pack_v2 ``EvidencePack`` for one claim.

    Time scoping mirrors v1: window from ``window_for(sid)`` (the same rule the
    temporal preamble uses), fair-game era from the sid's utterance date — both
    enforced inside ``consolidate`` and re-asserted on the built pack (T1.1
    defense in depth). Speaker-blind (I3): only sid/text/context flow in.

    ``retry_retrievers`` (jackie, 2026-07-24): the roster for the ONE targeted
    T2.4 retry; defaults to ``retrievers``. Passing a superset implements
    escalation-on-thin-evidence — e.g. R1+R2 primary with grok joining only
    the rescue round, which keeps its lineage diversity exactly where evidence
    is scarce at ~5-15% of its always-on cost."""
    window = window_for(sid, today=today)
    utterance = speech_context.speech_date_for(sid)
    # Historical-era policy (wiki projects:truthbot:historical-era-design):
    # pre-web speeches run lenient — unless the claim is (heuristically) a
    # prediction, which must never be judged with hindsight.
    mode = era_lint.era_mode_for(utterance, claim_text)
    if mode == "lenient":
        context = _HISTORICAL_FOCUS.format(utterance=utterance) + context
        # The strict prompt hard-scopes publication dates; lenient retrieval
        # briefs the era via context instead and leaves the params unset.
        prompt_utterance, prompt_window = None, None
        logger.info("historical-era lenient mode for %s (utterance %s)",
                    sid, utterance)
    else:
        prompt_utterance, prompt_window = utterance, window

    runner = shortlist_runner or _serial_runner

    def _call_one(r: Retriever, ctx: str) -> list:
        try:
            return r.shortlist(claim_text, context=ctx,
                               utterance=prompt_utterance,
                               window=prompt_window)
        except Exception as exc:
            # One dead retriever must not kill the claim — the consolidator quota
            # decides whether what remains is enough (and the gate forces UV when
            # it isn't). Loud in the log, soft in the run.
            logger.warning("v2 retriever %s failed for %s: %s",
                           r.label, sid, exc)
            return []

    def _shortlists(pool: Sequence[Retriever], label_suffix: str, ctx: str):
        # runner controls fan-out (serial by default, concurrent under the P120
        # pool); it returns shortlists in pool order, so labels line up.
        results = runner(pool, lambda r: _call_one(r, ctx))
        return [(r.label + label_suffix, sl) for r, sl in zip(pool, results)]

    shortlists = _shortlists(retrievers, "", context)
    res = consolidate(sid, shortlists, utterance=utterance, window=window,
                      max_items=max_items, era_mode=mode)
    if not res.quota_met:
        retry = _shortlists(retry_retrievers or retrievers, "-retry",
                            _RETRY_FOCUS + context)
        res = consolidate(sid, shortlists + retry, utterance=utterance,
                          window=window, max_items=max_items, era_mode=mode)
        if not res.quota_met:
            logger.info("T2.4 gate: %s pack fails quota after targeted retry "
                        "(%s) — verdict will be forced Unverifiable",
                        sid, res.gate_code)

    def _pack_item(i: int, cit) -> PackItem:
        ev = cit.evidence
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
        check_i5_provenance(item.provenance())   # I5: fail closed at entry
        return item

    items = [_pack_item(i, cit) for i, cit in enumerate(res.items, start=1)]
    # Pre-cap pool (PR-A2.2): persisted alongside the pack when the cap
    # actually discarded candidates, so cap/quota changes can be measured
    # offline without re-retrieval. Same E<n> numbering — the pool's first
    # len(items) entries ARE the pack.
    pool: list[PackItem] = []
    if len(res.pre_cap_items) > len(res.items):
        pool = [_pack_item(i, cit)
                for i, cit in enumerate(res.pre_cap_items, start=1)]
    pack = EvidencePack(sid=sid, window=window, items=items,
                        gate_code=res.gate_code, pool=pool)
    era_lint.assert_pack_within_era(pack, utterance, era_mode=mode)
    return pack
