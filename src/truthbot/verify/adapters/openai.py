"""
OpenAI GPT adapter with web search via Responses API or Chat Completions.
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

_FALLBACK_MODEL = "gpt-4o"


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
    model_id = "gpt-5.4-pro"
    required_env_key = "OPENAI_API_KEY"

    def __init__(self) -> None:
        super().__init__()
        self._api_key = os.environ["OPENAI_API_KEY"]
        self._active_model = self.model_id

    def call(self, claim: Claim, evidence: list[Evidence]) -> ModelVerdict:
        """Call OpenAI with web search and return a ModelVerdict."""
        import openai

        telemetry = get_telemetry()
        user_msg = self._build_user_message(claim, evidence)

        with telemetry.measure(self.adapter_name, self._active_model, claim.id) as td:
            try:
                client = openai.OpenAI(api_key=self._api_key, timeout=180.0)
                verdict_text, urls, tool_count, usage = self._call_with_search(
                    client, user_msg
                )

                if usage:
                    td["input_tokens"] = getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0)
                    td["output_tokens"] = getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0)
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
                    caveats=raw.get("caveats", ""),
                )

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
                system_text = (
                    SYNTHESIS_SYSTEM
                    + "\n\nOperational constraints (OpenAI): Use at most 3 web searches. "
                    + "Keep reasoning brief. Return ONLY the JSON object."
                )
                input_blocks = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "input_text", "text": system_text},
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
                        max_output_tokens=2500,
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
                except openai.NotFoundError:
                    logger.warning("OpenAIAdapter: model %s not found, trying fallback", model)
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
