"""Relevance middle step for Layer C retrieval (P67 Round B item 3).

The agreed middle ground between "Brave decides" and full model-native search:
a cheap model (a) writes 2-3 TARGETED queries per claim (the $600B budget claim
should search "federal defense homeland security budget FY2022", not the claim
sentence verbatim), and (b) scores every fetched candidate for relevance and
supports/refutes — populating the previously dead ``Evidence.relevance_score``
/ ``supports_claim`` fields so the pack can rank relevance-then-tier instead of
tier-only (tier-first is how an off-topic .gov speech topped an on-topic pack).

Brave stays the fetch layer: I5 provenance, era windowing, speaker-blindness
(I3 — prompts here see claim text and era only, never a speaker), and cost
control are all preserved. LLM calls go through the same LiteLLM proxy lane as
the panel (``verdict.proxy_lane`` env conventions) on a cheap tier; both calls
fail SOFT — query-gen failure falls back to the legacy claim-built query, and
score failure leaves the neutral defaults, so retrieval never breaks because
the refinement layer hiccuped.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Callable, Optional

from truthbot.models import Claim, Evidence
from truthbot.verify.evidence_provider import EvidenceProvider
from truthbot.verify.sources.base import SourceConnector, TimeWindow
from truthbot.verify.sources.brave import BraveSearchConnector

logger = logging.getLogger(__name__)

#: (system_prompt, user_payload_json) -> parsed JSON dict. Injectable so the
#: provider is fully testable offline; the live implementation is build_proxy_llm.
LlmFn = Callable[[str, str], dict]

DEFAULT_MODEL = "claude-haiku"
DEFAULT_QUERIES_PER_CLAIM = 3
#: Cap on candidates sent to the scoring call — bounds prompt size and cost.
DEFAULT_SCORE_CAP = 16

_QUERY_SYSTEM = (
    "You write web-search queries for fact-checking. Given a factual claim and "
    "the era it was made in, produce up to N short, targeted queries that would "
    "surface primary data or contemporaneous reporting able to VERIFY the "
    "claim's core assertion. Prefer the specific statistic, program, agency, or "
    "event named in the claim; include the fiscal year or calendar year implied "
    "by the era. Do not include any person's name. "
    'Return JSON only: {"queries": ["...", "..."]}.'
)

_SCORE_SYSTEM = (
    "You rate retrieved evidence for a fact-check. For each numbered item, "
    "judge the snippet against the claim: relevance is 0.0-1.0 (1.0 = directly "
    "addresses the claim's specific quantity/event/assertion; 0.5 = same broad "
    "topic but does not bear on the assertion; 0.0 = unrelated), and supports "
    "is true if the snippet corroborates the claim's core assertion, false if "
    "it contradicts it, null if it does neither or is unclear. "
    'Return JSON only: {"scores": [{"i": 1, "relevance": 0.0, "supports": null}, ...]}.'
)

_JSON_BLOCK_RX = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_loosely(text: str) -> dict:
    """Parse a model reply that should be a JSON object, tolerating code fences
    and prose around the object. Raises ValueError when no object is found."""
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        pass
    m = _JSON_BLOCK_RX.search(text or "")
    if not m:
        raise ValueError(f"no JSON object in model reply: {text!r:.200}")
    return json.loads(m.group(0))


def build_proxy_llm(model: str = DEFAULT_MODEL, *, timeout: float = 30.0) -> Optional[LlmFn]:
    """Live LlmFn over the LiteLLM proxy (verdict.proxy_lane env conventions).

    Returns None when no proxy key is configured, so callers can degrade to the
    unrefined provider loudly rather than silently spending nothing."""
    import os

    from truthbot.verdict import proxy_lane

    key = os.environ.get(proxy_lane.resolve_key_env(), "")
    if not key:
        return None
    url = proxy_lane.base_url().rstrip("/") + "/v1/chat/completions"

    def llm(system: str, user: str) -> dict:
        import httpx

        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return parse_json_loosely(content)

    return llm


def _era_label(window: TimeWindow) -> str:
    if not window:
        return "unknown"
    start, end = window
    return f"{start.isoformat()} to {end.isoformat()}"


def generate_queries(llm: LlmFn, claim_text: str, *, context: str = "",
                     window: TimeWindow = None,
                     n: int = DEFAULT_QUERIES_PER_CLAIM,
                     forbidden_terms: tuple[str, ...] = ()) -> list[str]:
    """Cheap-model query generation. Returns up to ``n`` deduped, non-empty
    queries; [] on any failure (caller falls back to the legacy claim query).

    T2.2 constraint validation (P67.7): queries containing fact-check tokens
    or any caller-supplied ``forbidden_terms`` (speaker-name tokens — checked
    in CODE, never shown to the model, so the generation stays speaker-blind)
    are dropped with a log line rather than sent to retrieval."""
    from truthbot.verify.factcheck_exclusion import query_violates_constraints

    payload = json.dumps({
        "claim": claim_text,
        "context": (context or "")[:500],
        "era": _era_label(window),
        "n": n,
    })
    try:
        out = llm(_QUERY_SYSTEM, payload)
        raw = out.get("queries", [])
    except Exception as exc:
        logger.warning("relevance: query generation failed (%s) — falling back", exc)
        return []
    queries: list[str] = []
    for q in raw:
        if not isinstance(q, str):
            continue
        q = q.strip()[:200]
        if not q:
            continue
        reason = query_violates_constraints(q, forbidden_terms)
        if reason:
            logger.info("relevance: dropped query %r — %s (T2.2)", q, reason)
            continue
        if q.lower() not in {x.lower() for x in queries}:
            queries.append(q)
    return queries[:n]


def score_evidence(llm: LlmFn, claim_text: str, evidence: list[Evidence]) -> None:
    """Cheap-model relevance / supports-refutes scoring, IN PLACE.

    Populates ``relevance_score`` and ``supports_claim`` on each item. Fails
    soft: on any error the items keep their defaults (neutral 0.5), so a
    scoring hiccup degrades to the old tier-only ranking rather than dropping
    evidence."""
    if not evidence:
        return
    items = [{"i": i, "source": ev.source_name, "snippet": (ev.snippet or "")[:400]}
             for i, ev in enumerate(evidence, start=1)]
    payload = json.dumps({"claim": claim_text, "items": items})
    try:
        out = llm(_SCORE_SYSTEM, payload)
        scores = out.get("scores", [])
    except Exception as exc:
        logger.warning("relevance: scoring failed (%s) — keeping defaults", exc)
        return
    by_index = {}
    for s in scores:
        try:
            by_index[int(s["i"])] = s
        except (KeyError, TypeError, ValueError):
            continue
    for i, ev in enumerate(evidence, start=1):
        s = by_index.get(i)
        if not s:
            continue
        try:
            rel = float(s.get("relevance"))
        except (TypeError, ValueError):
            rel = None
        if rel is not None:
            ev.relevance_score = min(1.0, max(0.0, rel))
        supports = s.get("supports")
        if isinstance(supports, bool) or supports is None:
            ev.supports_claim = supports


class RelevanceProvider(EvidenceProvider):
    """Evidence provider with the cheap-model relevance middle step.

    Flow per claim: generate targeted queries → Brave-fetch each (era-windowed)
    → run the remaining connectors (e.g. FactCheck) as before → dedup by URL →
    score every candidate for relevance + supports/refutes. Ranking happens
    downstream in ``evidence_pack`` (relevance-then-tier).
    """

    def __init__(self, brave: BraveSearchConnector,
                 others: list[SourceConnector], llm: LlmFn, *,
                 queries_per_claim: int = DEFAULT_QUERIES_PER_CLAIM,
                 score_cap: int = DEFAULT_SCORE_CAP) -> None:
        self._brave = brave
        self._others = others
        self._llm = llm
        self._queries_per_claim = queries_per_claim
        self._score_cap = score_cap

    def get_evidence(self, claim: Claim, *, window: TimeWindow = None) -> list[Evidence]:
        evidence: list[Evidence] = []
        if self._brave.is_available():
            queries = generate_queries(
                self._llm, claim.text, context=claim.context or "",
                window=window, n=self._queries_per_claim)
            if queries:
                for q in queries:
                    evidence.extend(self._brave.search_query(claim, q, window))
            else:
                # Query generation failed — legacy claim-built query.
                evidence.extend(self._brave.search_windowed(claim, window))
        for connector in self._others:
            if not connector.is_available():
                continue
            try:
                evidence.extend(connector.search_windowed(claim, window))
            except Exception as exc:
                logger.error("Connector %s failed: %s", connector.source_name, exc)

        # Dedup by URL before scoring so the score call doesn't pay for repeats.
        seen: set[str] = set()
        unique: list[Evidence] = []
        for ev in evidence:
            key = (ev.source_url or "").lower().rstrip("/")
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(ev)

        score_evidence(self._llm, claim.text, unique[: self._score_cap])
        return unique


def build_relevance_provider(brave: BraveSearchConnector,
                             others: list[SourceConnector], *,
                             model: str = DEFAULT_MODEL) -> Optional[RelevanceProvider]:
    """Factory for the live wiring: proxy-keyed cheap model, or None when the
    proxy key is absent (caller decides the fallback)."""
    llm = build_proxy_llm(model)
    if llm is None:
        return None
    return RelevanceProvider(brave, others, llm)
