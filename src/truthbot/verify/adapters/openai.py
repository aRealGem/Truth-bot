"""
OpenAI GPT adapter with web search via Responses API or Chat Completions.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Any

from truthbot.metrics.telemetry import get_synthesis_mode, get_telemetry
from truthbot.models import Claim, Confidence, Evidence, ModelVerdict, VerdictLabel
from truthbot.verify.adapters.base import (
    OPENAI_SYNTHESIS_SYSTEM,
    LLMAdapter,
    apply_url_grounding,
    build_multi_user_message,
    build_multi_verdicts,
    build_user_message,
    normalize_verdict_label,
    parse_multi_claim_json,
)
from truthbot.verify.context import apply_temporal_flags


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)

logger = logging.getLogger(__name__)

_FALLBACK_MODEL = "gpt-4o"
# Promoted from gpt-4.1 to gpt-5.4 (Phase 2a of fix-accuracy-sotu-v2).
# Rationale: gpt-4.1 knowledge cutoff ~Oct 2024 produced training-data-only
# verdicts on 2025-2026 current-events claims in the SOTU run (findings C1,
# C2). gpt-5.4 is current flagship, already wired into costs.py and the
# eval harness, and is the required substrate for the Phase 2.5 empirical
# batch-web_search capability test. Fallback to gpt-4o preserves the prior
# stability guarantee if 5.4 returns 500s on a given call.
_PRIMARY_MODEL = "gpt-5.4"


def _parse_verdict_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise json.JSONDecodeError("No valid JSON found in response", text, 0)


def _walk_output_for_urls(output: Any) -> tuple[str, list[str], int]:
    """Walk a Responses API ``output`` list; return ``(text, urls, web_search_call_count)``.

    Pulls URLs from every known surface produced by the GA ``web_search`` tool
    so the Layer 1d ground-truth intersection has a real ``tool_retrieved`` set
    instead of the empty list the legacy parser produced for batch bodies:

      * ``web_search_call.action.url`` — the ``open_page`` action variant
        (the model directly fetched a specific URL). This is the surface
        that was missing in the original parser and caused the 100%
        fabrication-rate readout for OpenAI batch verdicts in the
        ``ed7be4ad-…`` SOTU run.
      * ``web_search_call.action.sources[].url`` — defensive coverage for
        the documented Responses API + ``web_search`` shape that surfaces
        SERP-result URLs alongside the action. Did not appear in the
        observed batch body but is included so a future API revision
        cannot silently regress telemetry.
      * ``message.content[].annotations[].url`` — covers both bare
        ``{url: ...}`` annotations and ``type: 'url_citation'`` shapes the
        live Responses API emits.

    URLs are deduplicated in encounter order. ``text`` is the concatenation
    of every ``output_text`` block, identical to the legacy behavior.
    """
    text = ""
    urls: list[str] = []
    seen: set[str] = set()
    tool_count = 0

    def _add(url: Any) -> None:
        if isinstance(url, str) and url and url not in seen:
            seen.add(url)
            urls.append(url)

    for item in output or []:
        itype = _get(item, "type", "")
        if itype == "web_search_call":
            tool_count += 1
            action = _get(item, "action", None)
            _add(_get(action, "url", None))
            for src in _get(action, "sources", []) or []:
                _add(_get(src, "url", None))
            continue
        if itype == "message":
            for block in _get(item, "content", []) or []:
                if _get(block, "type", "") == "output_text":
                    text += _get(block, "text", "") or ""
                for ann in _get(block, "annotations", []) or []:
                    _add(_get(ann, "url", None))

    return text, urls, tool_count


class OpenAIAdapter(LLMAdapter):
    """OpenAI GPT adapter using Responses API with web search preview."""

    adapter_name = "openai"
    model_id = _PRIMARY_MODEL
    required_env_key = "OPENAI_API_KEY"
    supports_batch = True
    # Conservative first-pass cap; tune up after empirical multi-claim runs.
    max_claims_per_request = 10

    def __init__(self) -> None:
        super().__init__()
        self._api_key = os.environ["OPENAI_API_KEY"]
        self._active_model = self.model_id

    # ── Batch support ─────────────────────────────────────────────────────────

    def build_batch_payload(
        self,
        claim: Claim,
        evidence: list[Evidence],
        *,
        inject_evidence: bool = True,
    ) -> dict:
        """Build a request ``body`` for an OpenAI Batch JSONL line (endpoint=/v1/responses)."""
        user_msg = build_user_message(claim, evidence, inject_evidence=inject_evidence)
        return {
            "model": self.model_id,
            # Phase 2.5a: GA ``web_search`` replaces legacy ``web_search_preview``.
            # Both coexist per OpenAI docs, but ``web_search`` is the
            # recommended post-GA variant and accepts additional flags
            # (e.g. ``external_web_access``). See
            # developers.openai.com/api/docs/guides/tools-web-search.
            "tools": [{"type": "web_search"}],
            "input": [
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": OPENAI_SYNTHESIS_SYSTEM},
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_msg}],
                },
            ],
            "max_tool_calls": 2,
            "max_output_tokens": 8192,
        }

    def parse_batch_response(
        self,
        raw_response: Any,
        claim: Claim,
    ) -> ModelVerdict:
        """
        Parse a completed OpenAI batch row's response body into a ModelVerdict.

        ``raw_response`` is the ``response.body`` field (a Responses API envelope)
        of a single ``status='completed'`` batch result line.
        """
        output = _get(raw_response, "output", []) or []
        text, urls, tool_count = _walk_output_for_urls(output)

        model_id = _get(raw_response, "model", self.model_id)

        try:
            raw = _parse_verdict_json(text)
            label = normalize_verdict_label(raw["label"])
            confidence = Confidence(raw["confidence"])
        except Exception as exc:
            logger.error("OpenAIAdapter batch parse error for claim %s: %s", claim.id, exc)
            return ModelVerdict(
                adapter_name=self.adapter_name,
                model_id=model_id,
                claim_id=claim.id,
                label=VerdictLabel.UNVERIFIABLE,
                confidence=Confidence.LOW,
                explanation=f"Failed to parse batch response: {exc}",
                tier="frontier",
                synthesis_mode="batch",
            )

        usage = _get(raw_response, "usage", None)
        details = _get(usage, "prompt_tokens_details", None)
        cached = _get(details, "cached_tokens", 0) or 0

        ws, mrs, stripped = apply_url_grounding(raw, urls)
        verdict = ModelVerdict(
            adapter_name=self.adapter_name,
            model_id=model_id,
            claim_id=claim.id,
            label=label,
            confidence=confidence,
            explanation=raw.get("explanation", ""),
            caveats=raw.get("caveats", ""),
            web_sources=ws,
            model_reported_sources=mrs,
            stripped_source_count=stripped,
            tier="frontier",
            synthesis_mode="batch",
            cached_input_tokens=int(cached),
            tool_call_count=int(tool_count),
        )
        apply_temporal_flags(verdict, claim)
        return verdict

    # ── Multi-claim batch support ────────────────────────────────────────────

    def build_multi_batch_payload(
        self,
        claims: list[Claim],
        evidence_by_claim: dict[str, list[Evidence]],
        *,
        inject_evidence: bool = True,
        max_evidence_per_claim: int = 5,
    ) -> dict:
        """Build a Responses API body for a multi-claim OpenAI Batch row."""
        user_msg = build_multi_user_message(
            claims,
            evidence_by_claim,
            inject_evidence=inject_evidence,
            max_evidence_per_claim=max_evidence_per_claim,
        )
        n = max(1, len(claims))
        return {
            "model": self.model_id,
            # Phase 2.5a: use GA ``web_search`` for multi-claim batches too.
            "tools": [{"type": "web_search"}],
            "input": [
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": OPENAI_SYNTHESIS_SYSTEM},
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_msg}],
                },
            ],
            # 2 web searches per claim, capped so one bad claim can't burn the budget.
            "max_tool_calls": 2 * n,
            # 2048 tokens headroom + 1024 per claim verdict (explanation + sources).
            "max_output_tokens": 2048 + 1024 * n,
        }

    def parse_multi_batch_response(
        self,
        raw_response: Any,
        claims: list[Claim],
        *,
        batch_call_id: str = "",
    ) -> list[ModelVerdict]:
        """Parse one OpenAI Responses-API body into N multi-claim ModelVerdicts."""
        output = _get(raw_response, "output", []) or []
        text, urls, tool_count = _walk_output_for_urls(output)

        model_id = _get(raw_response, "model", self.model_id)

        try:
            raw_by_claim = parse_multi_claim_json(text, claims)
        except json.JSONDecodeError as exc:
            logger.error(
                "OpenAIAdapter multi-claim parse error (call=%s): %s",
                batch_call_id,
                exc,
            )
            raw_by_claim = {}

        usage = _get(raw_response, "usage", None)
        details = _get(usage, "prompt_tokens_details", None)
        cached = _get(details, "cached_tokens", 0) or 0
        call_usage = {
            "input_tokens": int(
                _get(usage, "input_tokens", 0)
                or _get(usage, "prompt_tokens", 0)
                or 0
            ),
            "output_tokens": int(
                _get(usage, "output_tokens", 0)
                or _get(usage, "completion_tokens", 0)
                or 0
            ),
            "cached_input_tokens": int(cached),
            "tool_call_count": int(tool_count),
        }

        verdicts = build_multi_verdicts(
            claims,
            raw_by_claim,
            adapter_name=self.adapter_name,
            model_id=model_id,
            synthesis_mode="batch",
            tier="frontier",
            call_usage=call_usage,
            batch_call_id=batch_call_id,
            tool_retrieved_urls=urls,
        )
        if verdicts and not verdicts[0].web_sources and not verdicts[0].model_reported_sources:
            verdicts[0].web_sources = urls[:10]
        return verdicts

    # ── Live multi-claim call (Phase 3a — promotion from Batch API) ──────────

    def call_multi(
        self,
        claims: list[Claim],
        evidence_by_claim: dict[str, list[Evidence]],
        *,
        inject_evidence: bool = True,
        max_evidence_per_claim: int = 5,
        telemetry_tier: str = "frontier",
        run_id: str | None = None,
    ) -> list[ModelVerdict]:
        """Call OpenAI live for N claims; mirrors the batch multi-claim payload.

        Phase 3a: when ``settings.openai_live_mode`` is set the pipeline
        routes OpenAI through the sidecar live path instead of the Batch
        API. This trades the 50% batch discount for a sub-minute end-to-end
        completion window (the batch path's 3–24h SLA was unworkable for
        iteration). Token / tool budgets scale linearly with N to match
        ``build_multi_batch_payload``.
        """
        if not claims:
            return []

        import openai

        telemetry = get_telemetry()
        n = len(claims)
        user_msg = build_multi_user_message(
            claims,
            evidence_by_claim,
            inject_evidence=inject_evidence,
            max_evidence_per_claim=max_evidence_per_claim,
        )
        batch_call_id = f"openai-live-multi-{uuid.uuid4().hex[:12]}"
        max_output_tokens = 2048 + 1024 * n
        max_tool_calls = 2 * n

        with telemetry.measure(
            self.adapter_name,
            self._active_model,
            claims[0].id,
            tier=telemetry_tier,
            run_id=run_id,
        ) as td:
            td["claim_count"] = n
            td["batch_call_id"] = batch_call_id

            try:
                client = openai.OpenAI(api_key=self._api_key, timeout=120.0)
                verdict_text, urls, tool_count, usage = self._call_with_search(
                    client,
                    user_msg,
                    max_output_tokens=max_output_tokens,
                    max_tool_calls=max_tool_calls,
                )

                cached = 0
                if usage:
                    td["input_tokens"] = (
                        getattr(usage, "input_tokens", 0)
                        or getattr(usage, "prompt_tokens", 0)
                    )
                    td["output_tokens"] = (
                        getattr(usage, "output_tokens", 0)
                        or getattr(usage, "completion_tokens", 0)
                    )
                    details = getattr(usage, "prompt_tokens_details", None)
                    if details is not None:
                        cached = getattr(details, "cached_tokens", 0) or 0
                        td["openai_cached_prompt_tokens"] = cached
                td["tool_call_count"] = tool_count
                td["retrieved_url_count"] = len(urls)

                try:
                    raw_by_claim = parse_multi_claim_json(verdict_text, claims)
                except json.JSONDecodeError as exc:
                    logger.error(
                        "OpenAIAdapter multi-claim parse error (call=%s, n=%d): %s",
                        batch_call_id, n, exc,
                    )
                    td["status"] = "parse_error"
                    raw_by_claim = {}
                else:
                    td["status"] = "ok"

                call_usage = {
                    "input_tokens": int(td.get("input_tokens", 0) or 0),
                    "output_tokens": int(td.get("output_tokens", 0) or 0),
                    "cached_input_tokens": int(cached),
                    "tool_call_count": int(tool_count),
                }
                verdicts = build_multi_verdicts(
                    claims,
                    raw_by_claim,
                    adapter_name=self.adapter_name,
                    model_id=self._active_model,
                    synthesis_mode=get_synthesis_mode(),
                    tier=telemetry_tier,
                    call_usage=call_usage,
                    batch_call_id=batch_call_id,
                    tool_retrieved_urls=urls,
                )
                if (
                    verdicts
                    and not verdicts[0].web_sources
                    and not verdicts[0].model_reported_sources
                    and urls
                ):
                    verdicts[0].web_sources = urls[:10]
                for v, claim in zip(verdicts, claims):
                    apply_temporal_flags(v, claim)
                return verdicts

            except Exception as exc:
                exc_str = str(exc).lower()
                td["status"] = (
                    "timeout"
                    if ("timeout" in exc_str or "timed out" in exc_str)
                    else "api_error"
                )
                logger.error(
                    "OpenAIAdapter multi-claim API error (call=%s, n=%d): %s",
                    batch_call_id, n, exc,
                )
                return [
                    ModelVerdict(
                        adapter_name=self.adapter_name,
                        model_id=self._active_model,
                        claim_id=c.id,
                        label=VerdictLabel.UNVERIFIABLE,
                        confidence=Confidence.LOW,
                        explanation=f"API error: {exc}",
                        no_response=True,
                        tier=telemetry_tier,
                        synthesis_mode=get_synthesis_mode(),
                        batch_call_id=batch_call_id,
                    )
                    for c in claims
                ]

    def call(
        self,
        claim: Claim,
        evidence: list[Evidence],
        *,
        inject_evidence: bool = True,
        telemetry_tier: str = "frontier",
        run_id: str | None = None,
    ) -> ModelVerdict:
        """Call OpenAI with web search and return a ModelVerdict."""
        import openai

        telemetry = get_telemetry()
        user_msg = self._build_user_message(claim, evidence, inject_evidence=inject_evidence)

        with telemetry.measure(
            self.adapter_name,
            self._active_model,
            claim.id,
            tier=telemetry_tier,
            run_id=run_id,
        ) as td:
            try:
                client = openai.OpenAI(api_key=self._api_key, timeout=60.0)
                verdict_text, urls, tool_count, usage = self._call_with_search(
                    client, user_msg
                )

                if usage:
                    td["input_tokens"] = getattr(usage, "input_tokens", 0) or getattr(
                        usage, "prompt_tokens", 0
                    )
                    td["output_tokens"] = getattr(usage, "output_tokens", 0) or getattr(
                        usage, "completion_tokens", 0
                    )
                    details = getattr(usage, "prompt_tokens_details", None)
                    if details is not None:
                        td["openai_cached_prompt_tokens"] = getattr(
                            details, "cached_tokens", 0
                        ) or 0
                td["tool_call_count"] = tool_count
                td["retrieved_url_count"] = len(urls)
                td["status"] = "ok"

                raw = _parse_verdict_json(verdict_text)
                label = normalize_verdict_label(raw["label"])
                confidence = Confidence(raw["confidence"])

                ws, mrs, stripped = apply_url_grounding(raw, urls)
                td["model_reported_source_count"] = len(mrs)
                td["stripped_source_count"] = stripped
                verdict = ModelVerdict(
                    adapter_name=self.adapter_name,
                    model_id=self._active_model,
                    claim_id=claim.id,
                    label=label,
                    confidence=confidence,
                    explanation=raw.get("explanation", ""),
                    web_sources=ws,
                    model_reported_sources=mrs,
                    stripped_source_count=stripped,
                    caveats=raw.get("caveats", ""),
                    tier=telemetry_tier,
                    synthesis_mode=get_synthesis_mode(),
                    cached_input_tokens=int(td.get("openai_cached_prompt_tokens", 0)),
                    tool_call_count=int(tool_count),
                )
                apply_temporal_flags(verdict, claim)
                return verdict

            except json.JSONDecodeError as exc:
                td["status"] = "parse_error"
                logger.error("OpenAIAdapter parse error for claim %s: %s", claim.id, exc)
                return ModelVerdict(
                    adapter_name=self.adapter_name,
                    model_id=self._active_model,
                    claim_id=claim.id,
                    label=VerdictLabel.UNVERIFIABLE,
                    confidence=Confidence.LOW,
                    explanation=f"Failed to parse model response: {exc}",
                    tier=telemetry_tier,
                    synthesis_mode=get_synthesis_mode(),
                )
            except Exception as exc:
                exc_str = str(exc).lower()
                if "timeout" in exc_str or "timed out" in exc_str:
                    td["status"] = "timeout"
                else:
                    td["status"] = "api_error"
                logger.error("OpenAIAdapter API error for claim %s: %s", claim.id, exc)
                return ModelVerdict(
                    adapter_name=self.adapter_name,
                    model_id=self._active_model,
                    claim_id=claim.id,
                    label=VerdictLabel.UNVERIFIABLE,
                    confidence=Confidence.LOW,
                    explanation=f"API error: {exc}",
                    tier=telemetry_tier,
                    synthesis_mode=get_synthesis_mode(),
                )

    def _call_with_search(
        self,
        client,
        user_msg: str,
        *,
        max_output_tokens: int | None = None,
        max_tool_calls: int = 2,
    ):
        """Try Responses API first, fall back to Chat Completions.

        ``max_output_tokens`` and ``max_tool_calls`` are passed through so
        the live multi-claim path (``call_multi``) can scale both budgets
        with the chunk size N — same scaling rule as
        ``build_multi_batch_payload``.
        """
        import openai

        try:
            if not hasattr(client, "responses"):
                raise AttributeError("responses API not available")

            for model in [self.model_id, _FALLBACK_MODEL]:
                self._active_model = model
                if max_output_tokens is not None:
                    max_out = max_output_tokens
                else:
                    max_out = 4096 if model == _FALLBACK_MODEL else 8192
                input_blocks = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "input_text", "text": OPENAI_SYNTHESIS_SYSTEM},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_msg},
                        ],
                    },
                ]
                try:
                    kwargs = dict(
                        model=model,
                        # Phase 2.5a: live Responses API also uses GA ``web_search``.
                        tools=[{"type": "web_search"}],
                        input=input_blocks,
                        max_tool_calls=max_tool_calls,
                        max_output_tokens=max_out,
                    )
                    try:
                        response = client.responses.create(
                            response_format={"type": "json_object"},
                            **kwargs,
                        )
                    except TypeError:
                        logger.warning("OpenAIAdapter: response_format not supported by SDK; falling back to text output")
                        response = client.responses.create(**kwargs)

                    status = getattr(response, "status", "completed")
                    if status != "completed":
                        details = getattr(response, "incomplete_details", None)
                        reason = getattr(details, "reason", str(details)) if details else "unknown"
                        if reason == "max_output_tokens" and model != _FALLBACK_MODEL:
                            logger.warning(
                                "OpenAIAdapter: model %s hit max_output_tokens, retrying on fallback %s",
                                model,
                                _FALLBACK_MODEL,
                            )
                            continue  # retry with fallback model + higher token budget
                        raise RuntimeError(f"OpenAI Responses status={status}: {details}")

                    text, urls, tool_count = _walk_output_for_urls(
                        getattr(response, "output", [])
                    )
                    usage = getattr(response, "usage", None)
                    return text, urls, tool_count, usage

                except (openai.NotFoundError, openai.BadRequestError) as exc:
                    logger.warning(
                        "OpenAIAdapter: model %s unavailable (%s), trying fallback",
                        model,
                        exc,
                    )
                    if model == _FALLBACK_MODEL:
                        raise

        except AttributeError:
            raise RuntimeError(
                "OpenAI Responses API is unavailable in the installed SDK version. "
                "Web search requires the Responses API (openai>=1.66.0). "
                "Upgrade the SDK or remove the OpenAI adapter. "
                "Falling back to training-data-only Chat Completions is not permitted — "
                "verdicts must be grounded in live web search."
            )
