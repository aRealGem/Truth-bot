"""
xAI Grok adapter (OpenAI-compatible API) with live web search.
"""

from __future__ import annotations

import json
import logging
import os
import re

from truthbot.metrics.telemetry import get_telemetry
from truthbot.models import Claim, Confidence, Evidence, ModelVerdict, VerdictLabel
from truthbot.verify.adapters.base import SYNTHESIS_SYSTEM, LLMAdapter

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
    """xAI Grok adapter using OpenAI-compatible API with live search."""

    adapter_name = "xai"
    model_id = "grok-4"
    required_env_key = "XAI_API_KEY"

    def __init__(self) -> None:
        super().__init__()
        self._api_key = os.environ["XAI_API_KEY"]
        self._active_model = self.model_id

    def call(self, claim: Claim, evidence: list[Evidence]) -> ModelVerdict:
        """Call Grok with live search and return a ModelVerdict."""
        import openai

        telemetry = get_telemetry()
        user_msg = self._build_user_message(claim, evidence)

        with telemetry.measure(self.adapter_name, self._active_model, claim.id) as td:
            try:
                client = openai.OpenAI(api_key=self._api_key, base_url=_XAI_BASE_URL)
                verdict_text, urls, usage = self._call_with_search(client, user_msg)

                if usage:
                    td["input_tokens"] = getattr(usage, "prompt_tokens", 0)
                    td["output_tokens"] = getattr(usage, "completion_tokens", 0)
                td["retrieved_url_count"] = len(urls)
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
                    web_sources=raw.get("web_sources", urls[:10]),
                )

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
                )

    def _call_with_search(self, client, user_msg: str):
        """Try with live search first, fall back to plain completion."""
        # Try with search_parameters
        try:
            resp = client.chat.completions.create(
                model=self._active_model,
                messages=[
                    {"role": "system", "content": SYNTHESIS_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=2048,
                extra_body={"search_parameters": {"mode": "auto"}},
            )
            text = resp.choices[0].message.content or ""
            urls = self._extract_citations(resp)
            return text, urls, resp.usage
        except Exception as exc:
            if "search_parameters" in str(exc).lower() or "unknown" in str(exc).lower():
                logger.info("GrokAdapter: live search not available, falling back to plain completion")
            else:
                raise

        # Fall back to plain completion
        resp = client.chat.completions.create(
            model=self._active_model,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=2048,
        )
        text = resp.choices[0].message.content or ""
        return text, [], resp.usage

    def _extract_citations(self, response) -> list[str]:
        """Extract citation URLs from a Grok response if present."""
        urls: list[str] = []
        # Try message-level citations
        for choice in getattr(response, "choices", []):
            msg = getattr(choice, "message", None)
            if msg:
                citations = getattr(msg, "citations", None) or []
                for c in citations:
                    if isinstance(c, str):
                        urls.append(c)
                    elif hasattr(c, "url"):
                        urls.append(c.url)
        return urls
