"""
xAI Grok adapter using the Agent Tools API (Responses endpoint) for live web search.
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
    """xAI Grok adapter using the Agent Tools Responses API with live web search."""

    adapter_name = "xai"
    model_id = "grok-4"
    required_env_key = "XAI_API_KEY"

    def __init__(self) -> None:
        super().__init__()
        self._api_key = os.environ["XAI_API_KEY"]
        self._active_model = self.model_id

    def call(self, claim: Claim, evidence: list[Evidence]) -> ModelVerdict:
        """Call Grok via the Agent Tools Responses API and return a ModelVerdict."""
        import openai

        telemetry = get_telemetry()
        user_msg = self._build_user_message(claim, evidence)

        with telemetry.measure(self.adapter_name, self._active_model, claim.id) as td:
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
                max_output_tokens=2048,
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
            max_tokens=2048,
        )
        text = resp.choices[0].message.content or ""
        return text, [], 0, resp.usage
