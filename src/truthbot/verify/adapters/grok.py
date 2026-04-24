"""
xAI Grok adapter using the Agent Tools API (Responses endpoint) for live web search.

TODO(prompt-cache): xAI does not expose Anthropic-style prompt caching or a documented
Context Caching API as of 2026-04. When available, add a provider hook parallel to
Anthropic/OpenAI/Gemini and wire cache-read token fields into telemetry.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid

from truthbot.metrics.telemetry import get_synthesis_mode, get_telemetry
from truthbot.models import Claim, Confidence, Evidence, ModelVerdict, VerdictLabel
from truthbot.verify.adapters.base import (
    SYNTHESIS_SYSTEM,
    LLMAdapter,
    build_multi_user_message,
    build_multi_verdicts,
    normalize_verdict_label,
    parse_multi_claim_json,
)
from truthbot.verify.context import apply_temporal_flags

logger = logging.getLogger(__name__)

_XAI_BASE_URL = "https://api.x.ai/v1"


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


class GrokAdapter(LLMAdapter):
    """xAI Grok adapter using the Agent Tools Responses API with live web search."""

    adapter_name = "xai"
    model_id = "grok-4"
    required_env_key = "XAI_API_KEY"
    # xAI has no batch API; claim-batching at the live layer is where cost
    # savings live. Conservative cap — leaves headroom for web-search results
    # in the shared response budget. See grok-gemini-live-claim-batching plan.
    max_claims_per_request = 6

    def __init__(self) -> None:
        super().__init__()
        self._api_key = os.environ["XAI_API_KEY"]
        self._active_model = self.model_id

    def call(
        self,
        claim: Claim,
        evidence: list[Evidence],
        *,
        inject_evidence: bool = True,
        telemetry_tier: str = "frontier",
        run_id: str | None = None,
    ) -> ModelVerdict:
        """Call Grok via the Agent Tools Responses API and return a ModelVerdict."""
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
                client = openai.OpenAI(api_key=self._api_key, base_url=_XAI_BASE_URL)
                verdict_text, urls, tool_count, usage = self._call_with_search(client, user_msg)

                if usage:
                    td["input_tokens"] = (
                        getattr(usage, "input_tokens", 0)
                        or getattr(usage, "prompt_tokens", 0)
                    )
                    td["output_tokens"] = (
                        getattr(usage, "output_tokens", 0)
                        or getattr(usage, "completion_tokens", 0)
                    )
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
                    tier=telemetry_tier,
                    synthesis_mode=get_synthesis_mode(),
                )
                apply_temporal_flags(verdict, claim)
                return verdict

            except json.JSONDecodeError as exc:
                td["status"] = "parse_error"
                logger.error("GrokAdapter parse error for claim %s: %s", claim.id, exc)
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
                logger.error("GrokAdapter API error for claim %s: %s", claim.id, exc)
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

    # ── Live multi-claim call (Phase E — Grok/Gemini slice) ──────────────────

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
        """Call Grok once for N claims; amortize SYNTHESIS_SYSTEM across the chunk.

        xAI has no batch API and no ``max_tool_calls`` knob on the Responses
        endpoint, so the web-search budget is enforced via prompt instruction
        in ``build_multi_user_message``. Token budget scales linearly with N.
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
        batch_call_id = f"xai-live-multi-{uuid.uuid4().hex[:12]}"

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
                client = openai.OpenAI(api_key=self._api_key, base_url=_XAI_BASE_URL)
                verdict_text, urls, tool_count, usage = self._call_with_search(
                    client, user_msg, max_output_tokens=2048 + 1024 * n
                )

                if usage:
                    td["input_tokens"] = (
                        getattr(usage, "input_tokens", 0)
                        or getattr(usage, "prompt_tokens", 0)
                    )
                    td["output_tokens"] = (
                        getattr(usage, "output_tokens", 0)
                        or getattr(usage, "completion_tokens", 0)
                    )
                td["tool_call_count"] = tool_count
                td["retrieved_url_count"] = len(urls)

                try:
                    raw_by_claim = parse_multi_claim_json(verdict_text, claims)
                except json.JSONDecodeError as exc:
                    logger.error(
                        "GrokAdapter multi-claim parse error (call=%s, n=%d): %s",
                        batch_call_id, n, exc,
                    )
                    td["status"] = "parse_error"
                    raw_by_claim = {}
                else:
                    td["status"] = "ok"

                call_usage = {
                    "input_tokens": int(td.get("input_tokens", 0) or 0),
                    "output_tokens": int(td.get("output_tokens", 0) or 0),
                    "cached_input_tokens": 0,
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
                )
                if verdicts and not verdicts[0].web_sources and urls:
                    verdicts[0].web_sources = urls[:10]
                return verdicts

            except Exception as exc:
                exc_str = str(exc).lower()
                td["status"] = (
                    "timeout"
                    if ("timeout" in exc_str or "timed out" in exc_str)
                    else "api_error"
                )
                logger.error(
                    "GrokAdapter multi-claim API error (call=%s, n=%d): %s",
                    batch_call_id, n, exc,
                )
                # Contract: multi-claim failures return N UNVERIFIABLE
                # no_response verdicts; the caller (BatchDispatcher sidecar /
                # VerificationEngine) decides whether to per-claim retry.
                return build_multi_verdicts(
                    claims,
                    {},
                    adapter_name=self.adapter_name,
                    model_id=self._active_model,
                    synthesis_mode=get_synthesis_mode(),
                    tier=telemetry_tier,
                    call_usage={"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0},
                    batch_call_id=batch_call_id,
                )

    def _call_with_search(self, client, user_msg: str, *, max_output_tokens: int = 2048):
        """Call via the Agent Tools Responses API; fall back to plain Chat Completions."""
        # Prefer the Responses API with web_search tool (Agent Tools API)
        try:
            if not hasattr(client, "responses"):
                raise AttributeError("Responses API not available in SDK")

            response = client.responses.create(
                model=self._active_model,
                input=[
                    {"role": "system", "content": SYNTHESIS_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                tools=[{"type": "web_search"}],
                max_output_tokens=max_output_tokens,
            )

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
                        for ann in getattr(block, "annotations", []):
                            url = getattr(ann, "url", None)
                            if url:
                                urls.append(url)

            # Some SDK versions surface citations at the top level
            for c in getattr(response, "citations", []) or []:
                if isinstance(c, str):
                    urls.append(c)
                elif hasattr(c, "url"):
                    urls.append(c.url)

            usage = getattr(response, "usage", None)
            return text, urls, tool_count, usage

        except AttributeError:
            logger.warning(
                "GrokAdapter: Responses API not available in installed SDK; "
                "falling back to plain Chat Completions without web search."
            )
        except Exception as exc:
            # Surface errors that aren't about tool/API availability
            exc_lower = str(exc).lower()
            if any(kw in exc_lower for kw in ("responses", "not found", "404", "unknown endpoint")):
                logger.warning(
                    "GrokAdapter: Responses endpoint unavailable (%s); "
                    "falling back to plain Chat Completions.",
                    exc,
                )
            else:
                raise

        # Fallback: plain Chat Completions (no live search)
        logger.warning(
            "GrokAdapter: running without live web search — verdict from training knowledge only."
        )
        resp = client.chat.completions.create(
            model=self._active_model,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=max_output_tokens,
        )
        text = resp.choices[0].message.content or ""
        return text, [], 0, resp.usage
