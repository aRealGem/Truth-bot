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
    # ── B2: the data table is not "context" ──────────────────────────────────
    # A primary series was the single largest source of stanceless Tier-1..3
    # items: the scorer read a BLS table as background TO the claim rather than
    # as the measurement OF it, so the best evidence in the pack credited
    # nothing and the gate withheld a verdict it had the data to reach
    # (trump_2026:0054, clinton_1998:0101).
    "A PRIMARY DATA SERIES or OFFICIAL RECORD that CONTAINS the figure at "
    "issue is not background: it is the measurement itself. Score it supports "
    "or refutes according to what its own numbers show, never context. If the "
    "series carries the quantity the claim asserts — an employment level, a "
    "budget line, a case count, an appropriation — read the number and take a "
    "side. Reserve context for genuine background: material on the same topic "
    "that does not carry the quantity at issue, commentary, or an item whose "
    "figures cannot be lined up against the claim at all. "
    "State in one_line_why the COMPARISON YOU ACTUALLY MADE — the claim's "
    "figure against the source's figure, in words, e.g. 'claim says 3.2M new "
    "jobs; table row for Jan 2026 shows 2.9M'. Do not restate the snippet. "
    # ── the arithmetic hinge (reviewer-mandated guard) ───────────────────────
    "Set arithmetic_hinge true when your stance depends on ARITHMETIC YOU "
    "performed over the series rather than on a figure the source states "
    "outright — taking a maximum across a series, computing a ratio or a "
    "share, deflating to real terms, or comparing two periods the source does "
    "not itself compare. A stance reached that way is a hypothesis for the "
    "panel to check, not a settled reading. Set it false when the source "
    "states the comparison itself. "
    'Return JSON only: {"scores": [{"i": 1, "relevance": 0.0, '
    '"supports": null, "one_line_why": "", "arithmetic_hinge": false}, ...]}.'
)

#: Longest snippet (characters) sent per item in the scoring payload. This is
#: the DEFAULT for ``max_snippet_chars``; callers may raise it per call (D17-c
#: series excerpts) but the shared constant does not move for one experiment.
SCORE_SNIPPET_CHARS = 400

#: Appended in place of the characters a cap removed, so a clipped snippet is
#: visibly clipped rather than silently short. Nothing in the corpus has ever
#: hit the 400 cap (n=4,344, max 207), so this marks a failure mode that has
#: not yet occurred — which is exactly when it is cheap to install.
TRUNCATION_MARKER = "… [truncated {n} chars]"

#: Longest stored ``one_line_why``. The pack payload truncates at 200, so this
#: keeps the stored comparison a little richer than the rendered one without
#: letting a runaway reply bloat the artifact.
ONE_LINE_WHY_CHARS = 240

_JSON_BLOCK_RX = re.compile(r"\{.*\}", re.DOTALL)

#: A pack-level scorer: ``(claim_text, evidence) -> None``, mutating the list
#: IN PLACE. This is the injection point ``build_evidence_pack_v2`` takes —
#: distinct from ``LlmFn``, which is the raw transport underneath it.
PackScorer = Callable[[str, list], None]


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


def _clip(text: str, cap: "int | None") -> tuple[str, int]:
    """(snippet as sent, characters removed). ``cap=None`` means no limit.

    A clip is always visible: the marker states how much went missing, so a
    downstream reader can tell a short source from a truncated one."""
    if cap is None or len(text) <= cap:
        return text, 0
    removed = len(text) - cap
    return text[:cap] + TRUNCATION_MARKER.format(n=removed), removed


#: What a series excerpt contributes to the scoring payload. Deliberately a
#: WHITELIST: the golden carries provenance a reader needs (fixture sha, role,
#: claim_sid) that the scorer does not, and shipping the whole record would
#: quietly grow the prompt every time the golden schema gained a field.
_SERIES_ROWS_KEYS = (
    "series_id", "rows", "window_start", "window_end", "rows_shown",
    "total_rows_in_full_table", "vintage_as_of", "units",
    "units_unavailable_because", "full_table", "selection_predicate",
    "window_period_mismatch", "window_period_mismatch_note",
)


def _series_rows_payload(rows: dict) -> dict:
    """The scorer's view of a series excerpt: the rows plus what bounds them.

    ``rows_shown`` of ``total_rows_in_full_table`` and the predicate travel WITH
    the data on purpose — a window is only honest if the reader can see what was
    left out and why it was left out."""
    return {k: rows[k] for k in _SERIES_ROWS_KEYS if rows.get(k) is not None}


def score_payload_ex(
    claim_text: str,
    evidence: list[Evidence],
    max_snippet_chars: "int | None" = SCORE_SNIPPET_CHARS,
) -> tuple[str, list[dict]]:
    """``(payload, per-item {chars_sent, chars_truncated})``.

    The machine-readable half of the truncation contract: callers that must
    prove nothing was clipped (D17-c Stage A asserts ``chars_truncated == 0``)
    read it here rather than re-deriving it from the payload string."""
    items, meta = [], []
    for i, ev in enumerate(evidence, start=1):
        sent, removed = _clip(ev.snippet or "", max_snippet_chars)
        item = {"i": i, "source": ev.source_name, "snippet": sent}
        rows = getattr(ev, "series_rows", None)
        if rows:
            # Structured, NOT folded into the snippet: the rows are data, and
            # the cap governs prose. Clipping a series mid-table would leave a
            # payload that still parses and quietly means something else.
            item["series_rows"] = _series_rows_payload(rows)
        items.append(item)
        meta.append({"i": i, "chars_sent": len(sent), "chars_truncated": removed,
                     "has_series_rows": bool(rows)})
    return json.dumps({"claim": claim_text, "items": items}), meta


def score_payload(
    claim_text: str,
    evidence: list[Evidence],
    max_snippet_chars: "int | None" = SCORE_SNIPPET_CHARS,
) -> str:
    """The EXACT user payload ``score_evidence`` sends for these items.

    Factored out so a $0 cost estimator can measure the real prompt volume of a
    stored pack (scripts/rescore_stored_packs.py --estimate) instead of guessing
    at it — the estimate prices the same bytes the funded run would send.

    ``max_snippet_chars`` defaults to ``SCORE_SNIPPET_CHARS``, so every existing
    caller is byte-unchanged; ``tests/test_score_payload_default_identity.py``
    holds that. D17-c raises it per call to carry series excerpts, which run to
    ~22,000 characters — the shared constant stays at 400."""
    return score_payload_ex(claim_text, evidence, max_snippet_chars)[0]


def score_evidence(
    llm: LlmFn,
    claim_text: str,
    evidence: list[Evidence],
    max_snippet_chars: "int | None" = SCORE_SNIPPET_CHARS,
) -> None:
    """Cheap-model relevance / supports-refutes scoring, IN PLACE.

    Populates ``relevance_score``, ``supports_claim`` and — under the B2
    contract — ``one_line_why`` (the comparison the stance rests on) and
    ``arithmetic_hinge`` (the stance came from arithmetic the SCORER did, so it
    is a hypothesis for the panel rather than proof). Fails soft: on any error
    the items keep their defaults (neutral 0.5), so a scoring hiccup degrades
    to the old tier-only ranking rather than dropping evidence.

    Both new fields are OPTIONAL on the wire. A model that answers with the
    older three-key shape still scores normally — ``one_line_why`` stays None
    and the payload falls back to the snippet — because an unparsed extra field
    must never cost us the stance we paid for."""
    if not evidence:
        return
    payload = score_payload(claim_text, evidence, max_snippet_chars)
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
        why = s.get("one_line_why")
        if isinstance(why, str) and why.strip():
            ev.one_line_why = why.strip()[:ONE_LINE_WHY_CHARS]
        # Only a literal true marks the hinge. A missing or malformed field
        # means "not asserted", and the guard must never be switched ON by
        # accident — but note it is also never switched OFF here, so a merge
        # cannot quietly clear a hinge an earlier pass recorded.
        if s.get("arithmetic_hinge") is True:
            ev.arithmetic_hinge = True


def build_scorer(*, model: str = DEFAULT_MODEL,
                 score_cap: int = DEFAULT_SCORE_CAP,
                 llm: Optional[LlmFn] = None) -> Optional[PackScorer]:
    """Factory for the ``build_evidence_pack_v2(scorer=...)`` injection point.

    Returns a ``PackScorer`` bound to the cheap proxy lane (Haiku via LiteLLM by
    default), or ``None`` when no proxy key is configured — so a caller that
    asked for scoring finds out LOUDLY that the lane is absent instead of
    silently shipping another all-default pack (remediation v2, B1b).

    **Calling the returned scorer is MODEL SPEND.** Every production caller
    therefore keeps it behind an explicit, default-OFF flag (DC-B1); pass
    ``llm`` to drive it from a stub in tests, which spends nothing.

    ``score_cap`` bounds the per-call prompt: only the first ``score_cap`` items
    are sent, matching ``RelevanceProvider``. Stored v2 packs cap at
    ``PACK_CAP_V2`` (10), comfortably under the default 16."""
    llm = llm if llm is not None else build_proxy_llm(model)
    if llm is None:
        return None

    def scorer(claim_text: str, evidence: list) -> None:
        # Slicing shares the Evidence objects, so the in-place writes land on
        # the caller's list.
        score_evidence(llm, claim_text, evidence[:score_cap])

    return scorer


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
