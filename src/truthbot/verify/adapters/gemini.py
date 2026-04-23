"""
Google Gemini adapter with Google Search grounding.
Migrated from google-generativeai (deprecated) to google-genai SDK.
"""

from __future__ import annotations

import json
import logging
import os
import re

from truthbot.metrics.telemetry import get_synthesis_mode, get_telemetry
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
    """Google Gemini adapter using Google Search grounding (google-genai SDK)."""

    adapter_name = "gemini"
    model_id = "gemini-2.5-pro"
    required_env_key = "GEMINI_API_KEY"
    # Process-wide reuse of Context Caching resource for the static rubric.
    _cached_content_name: str | None = None

    def __init__(self) -> None:
        super().__init__()
        self._api_key = os.environ["GEMINI_API_KEY"]
        self._active_model = self.model_id

    def _get_or_create_cached_content(self, client: object, types: object) -> str | None:
        """Create a Context Caching entry for SYNTHESIS_SYSTEM once per process."""
        if GeminiAdapter._cached_content_name:
            return GeminiAdapter._cached_content_name
        try:
            create_config = types.CreateCachedContentConfig(
                display_name="truthbot-synthesis-rubric",
                system_instruction=SYNTHESIS_SYSTEM,
                ttl="14400s",
            )
            cached = client.caches.create(
                model=self._active_model,
                config=create_config,
            )
            name = getattr(cached, "name", None)
            if name:
                GeminiAdapter._cached_content_name = name
            return name
        except Exception as exc:
            logger.warning("Gemini context caching unavailable, using inline system_instruction: %s", exc)
            return None

    def call(
        self,
        claim: Claim,
        evidence: list[Evidence],
        *,
        inject_evidence: bool = True,
        telemetry_tier: str = "frontier",
        run_id: str | None = None,
    ) -> ModelVerdict:
        """Call Gemini with Google Search grounding and return a ModelVerdict."""
        from google import genai
        from google.genai import types

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
                client = genai.Client(api_key=self._api_key)

                cache_name = self._get_or_create_cached_content(client, types)
                tools = [types.Tool(google_search=types.GoogleSearch())]
                if cache_name:
                    gen_config = types.GenerateContentConfig(
                        cached_content=cache_name,
                        tools=tools,
                    )
                else:
                    gen_config = types.GenerateContentConfig(
                        system_instruction=SYNTHESIS_SYSTEM,
                        tools=tools,
                    )

                response = client.models.generate_content(
                    model=self._active_model,
                    contents=user_msg,
                    config=gen_config,
                )

                # Extract grounding metadata
                urls: list[str] = []
                search_query_count = 0

                candidates = response.candidates or []
                for candidate in candidates:
                    gm = getattr(candidate, "grounding_metadata", None)
                    if gm:
                        search_queries = getattr(gm, "web_search_queries", []) or []
                        search_query_count += len(search_queries)
                        chunks = getattr(gm, "grounding_chunks", []) or []
                        for chunk in chunks:
                            web = getattr(chunk, "web", None)
                            if web:
                                url = getattr(web, "uri", None)
                                if url:
                                    urls.append(url)

                # Extract text
                verdict_text = ""
                try:
                    verdict_text = response.text or ""
                except Exception:
                    for candidate in candidates:
                        content = getattr(candidate, "content", None)
                        for part in getattr(content, "parts", []):
                            verdict_text += getattr(part, "text", "")

                # Token usage
                usage_meta = getattr(response, "usage_metadata", None)
                if usage_meta:
                    td["input_tokens"] = getattr(usage_meta, "prompt_token_count", 0) or 0
                    td["output_tokens"] = getattr(usage_meta, "candidates_token_count", 0) or 0
                    td["gemini_cached_content_tokens"] = (
                        getattr(usage_meta, "cached_content_token_count", 0) or 0
                    )
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
                    tier=telemetry_tier,
                    synthesis_mode=get_synthesis_mode(),
                    cached_input_tokens=int(td.get("gemini_cached_content_tokens", 0)),
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
                    tier=telemetry_tier,
                    synthesis_mode=get_synthesis_mode(),
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
                    tier=telemetry_tier,
                    synthesis_mode=get_synthesis_mode(),
                )
