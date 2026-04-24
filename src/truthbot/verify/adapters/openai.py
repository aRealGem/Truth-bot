"""
OpenAI GPT adapter with web search via Responses API or Chat Completions.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from truthbot.metrics.telemetry import get_synthesis_mode, get_telemetry
from truthbot.models import Claim, Confidence, Evidence, ModelVerdict, VerdictLabel
from truthbot.verify.adapters.base import (
    OPENAI_SYNTHESIS_SYSTEM,
    LLMAdapter,
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
            "tools": [{"type": "web_search_preview"}],
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
        text = ""
        urls: list[str] = []
        tool_count = 0
        for item in output:
            itype = _get(item, "type", "")
            if itype == "web_search_call":
                tool_count += 1
            elif itype == "message":
                for block in _get(item, "content", []) or []:
                    if _get(block, "type", "") == "output_text":
                        text += _get(block, "text", "") or ""
                    for ann in _get(block, "annotations", []) or []:
                        url = _get(ann, "url", None)
                        if url:
                            urls.append(url)

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

        verdict = ModelVerdict(
            adapter_name=self.adapter_name,
            model_id=model_id,
            claim_id=claim.id,
            label=label,
            confidence=confidence,
            explanation=raw.get("explanation", ""),
            caveats=raw.get("caveats", ""),
            web_sources=raw.get("web_sources", urls[:10]),
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
            "tools": [{"type": "web_search_preview"}],
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
        text = ""
        urls: list[str] = []
        tool_count = 0
        for item in output:
            itype = _get(item, "type", "")
            if itype == "web_search_call":
                tool_count += 1
                continue
            if itype == "message":
                for block in _get(item, "content", []) or []:
                    if _get(block, "type", "") == "output_text":
                        text += _get(block, "text", "") or ""
                    for ann in _get(block, "annotations", []) or []:
                        url = _get(ann, "url", None)
                        if url:
                            urls.append(url)

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
        )
        if verdicts and not verdicts[0].web_sources:
            verdicts[0].web_sources = urls[:10]
        return verdicts

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

                verdict = ModelVerdict(
                    adapter_name=self.adapter_name,
                    model_id=self._active_model,
                    claim_id=claim.id,
                    label=label,
                    confidence=confidence,
                    explanation=raw.get("explanation", ""),
                    web_sources=raw.get("web_sources", urls[:10]),
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

    def _call_with_search(self, client, user_msg: str):
        """Try Responses API first, fall back to Chat Completions."""
        import openai

        # Try Responses API
        try:
            if not hasattr(client, "responses"):
                raise AttributeError("responses API not available")

            for model in [self.model_id, _FALLBACK_MODEL]:
                self._active_model = model
                # Use a higher token budget for the fallback model
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
                        tools=[{"type": "web_search_preview"}],
                        input=input_blocks,
                        max_tool_calls=2,
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

                    text = ""
                    urls: list[str] = []
                    tool_count = 0
                    for item in getattr(response, "output", []):
                        itype = getattr(item, "type", "")
                        if itype == "web_search_call":
                            tool_count += 1
                        elif itype == "message":
                            for block in getattr(item, "content", []):
                                if getattr(block, "type", "") == "output_text":
                                    text += getattr(block, "text", "")
                                # Extract citations/URLs
                                for ann in getattr(block, "annotations", []):
                                    url = getattr(ann, "url", None)
                                    if url:
                                        urls.append(url)
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
