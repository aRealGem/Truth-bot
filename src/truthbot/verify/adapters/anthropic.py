"""
Anthropic Claude adapter with native web search.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from truthbot.metrics.telemetry import get_synthesis_mode, get_telemetry
from truthbot.models import Claim, Confidence, Evidence, ModelVerdict, VerdictLabel
from truthbot.verify.adapters.base import SYNTHESIS_SYSTEM, AdapterUnavailable, LLMAdapter

logger = logging.getLogger(__name__)

# Ordered list of fallback models (most preferred first)
_FALLBACK_MODELS = [
    "claude-opus-4-7",
    "claude-opus-4-5",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
]


def _parse_verdict_json(text: str) -> dict:
    """Parse JSON from model response, handling markdown wrappers."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Extract first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise json.JSONDecodeError("No valid JSON found in response", text, 0)


class AnthropicAdapter(LLMAdapter):
    """Anthropic Claude adapter using server-side web search."""

    adapter_name = "anthropic"
    model_id = "claude-opus-4-7"
    required_env_key = "ANTHROPIC_API_KEY"

    def __init__(self) -> None:
        super().__init__()
        self._api_key = os.environ["ANTHROPIC_API_KEY"]
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
        """Call Anthropic Claude with web search and return a ModelVerdict."""
        import anthropic

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
                client = anthropic.Anthropic(api_key=self._api_key)
                response = self._call_with_fallback(client, user_msg)

                # Parse content blocks
                tool_call_count = 0
                retrieved_urls: list[str] = []
                verdict_text = ""

                for block in response.content:
                    btype = getattr(block, "type", "")
                    if btype == "server_tool_use":
                        tool_call_count += 1
                    elif btype == "web_search_tool_result":
                        content = getattr(block, "content", [])
                        if isinstance(content, list):
                            for result in content:
                                url = getattr(result, "url", None)
                                if url:
                                    retrieved_urls.append(url)
                    elif btype == "text":
                        verdict_text += getattr(block, "text", "")

                # Update telemetry data
                usage = getattr(response, "usage", None)
                if usage:
                    td["input_tokens"] = getattr(usage, "input_tokens", 0) or 0
                    td["output_tokens"] = getattr(usage, "output_tokens", 0) or 0
                    td["cache_read_input_tokens"] = getattr(usage, "cache_read_input_tokens", 0) or 0
                    td["cache_creation_input_tokens"] = (
                        getattr(usage, "cache_creation_input_tokens", 0) or 0
                    )
                td["tool_call_count"] = tool_call_count
                td["retrieved_url_count"] = len(retrieved_urls)
                td["status"] = "ok"

                raw = _parse_verdict_json(verdict_text)
                label = VerdictLabel(raw["label"])
                confidence = Confidence(raw["confidence"])

                return ModelVerdict(
                    adapter_name=self.adapter_name,
                    model_id=self._active_model,
                    claim_id=claim.id,
                    label=label,
                    confidence=confidence,
                    explanation=raw.get("explanation", ""),
                    caveats=raw.get("caveats", ""),
                    web_sources=raw.get("web_sources", retrieved_urls[:10]),
                    tier=telemetry_tier,
                    synthesis_mode=get_synthesis_mode(),
                    cached_input_tokens=int(td.get("cache_read_input_tokens", 0)),
                )

            except json.JSONDecodeError as exc:
                td["status"] = "parse_error"
                logger.error("AnthropicAdapter parse error for claim %s: %s", claim.id, exc)
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
                logger.error("AnthropicAdapter API error for claim %s: %s", claim.id, exc)
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

    def _call_with_fallback(self, client: Any, user_msg: str) -> Any:
        """Try models in fallback order, returning the first successful response."""
        import anthropic

        last_exc: Exception | None = None
        for model in _FALLBACK_MODELS:
            if model != self.model_id:
                logger.warning(
                    "AnthropicAdapter: model %s not available, trying %s",
                    self._active_model,
                    model,
                )
            self._active_model = model
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=2048,
                    system=[
                        {
                            "type": "text",
                            "text": SYNTHESIS_SYSTEM,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    messages=[{"role": "user", "content": user_msg}],
                )
                return response
            except anthropic.BadRequestError as exc:
                if "model" in str(exc).lower():
                    last_exc = exc
                    continue
                raise
            except anthropic.NotFoundError as exc:
                last_exc = exc
                continue

        # All models failed — try without web search on last fallback
        logger.warning("AnthropicAdapter: all models failed with web search, trying plain completion")
        try:
            response = client.messages.create(
                model=_FALLBACK_MODELS[-1],
                max_tokens=2048,
                system=[
                    {
                        "type": "text",
                        "text": SYNTHESIS_SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_msg}],
            )
            self._active_model = _FALLBACK_MODELS[-1]
            return response
        except Exception:
            pass

        raise last_exc or RuntimeError("All Anthropic model fallbacks exhausted")
