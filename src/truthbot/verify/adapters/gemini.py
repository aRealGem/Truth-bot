"""
Google Gemini adapter with Google Search grounding.
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


class GeminiAdapter(LLMAdapter):
    """Google Gemini adapter using Google Search grounding."""

    adapter_name = "gemini"
    model_id = "gemini-2.5-pro"
    required_env_key = "GEMINI_API_KEY"

    def __init__(self) -> None:
        super().__init__()
        self._api_key = os.environ["GEMINI_API_KEY"]
        self._active_model = self.model_id

    def call(self, claim: Claim, evidence: list[Evidence]) -> ModelVerdict:
        """Call Gemini with Google Search grounding and return a ModelVerdict."""
        import google.generativeai as genai

        telemetry = get_telemetry()
        user_msg = self._build_user_message(claim, evidence)

        with telemetry.measure(self.adapter_name, self._active_model, claim.id) as td:
            try:
                genai.configure(api_key=self._api_key)

                model = genai.GenerativeModel(
                    model_name=self._active_model,
                    system_instruction=SYNTHESIS_SYSTEM,
                    tools=["google_search_retrieval"],
                )

                response = model.generate_content(user_msg)

                # Extract grounding metadata
                urls: list[str] = []
                search_query_count = 0

                candidates = getattr(response, "candidates", [])
                for candidate in candidates:
                    gm = getattr(candidate, "grounding_metadata", None)
                    if gm:
                        search_queries = getattr(gm, "search_queries", []) or getattr(gm, "web_search_queries", [])
                        search_query_count += len(search_queries)
                        chunks = getattr(gm, "grounding_chunks", []) or getattr(gm, "grounding_support_chunks", [])
                        for chunk in chunks:
                            web = getattr(chunk, "web", None)
                            if web:
                                url = getattr(web, "uri", None)
                                if url:
                                    urls.append(url)

                # Extract text
                verdict_text = ""
                try:
                    verdict_text = response.text
                except Exception:
                    for candidate in candidates:
                        for part in getattr(getattr(candidate, "content", None), "parts", []):
                            verdict_text += getattr(part, "text", "")

                # Token usage
                usage_meta = getattr(response, "usage_metadata", None)
                if usage_meta:
                    td["input_tokens"] = getattr(usage_meta, "prompt_token_count", 0) or 0
                    td["output_tokens"] = getattr(usage_meta, "candidates_token_count", 0) or 0
                td["tool_call_count"] = search_query_count
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
                logger.error("GeminiAdapter parse error for claim %s: %s", claim.id, exc)
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
                logger.error("GeminiAdapter API error for claim %s: %s", claim.id, exc)
                return ModelVerdict(
                    adapter_name=self.adapter_name,
                    model_id=self._active_model,
                    claim_id=claim.id,
                    label=VerdictLabel.UNVERIFIABLE,
                    confidence=Confidence.LOW,
                    explanation=f"API error: {exc}",
                )
