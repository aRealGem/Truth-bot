"""
Google Gemini adapter with Google Search grounding.
Migrated from google-generativeai (deprecated) to google-genai SDK.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from truthbot.metrics.telemetry import get_synthesis_mode, get_telemetry
from truthbot.models import Claim, Confidence, Evidence, ModelVerdict, VerdictLabel
from truthbot.verify.adapters.base import SYNTHESIS_SYSTEM, LLMAdapter, build_user_message


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)

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
    # Google's batch API (``google.genai.batches``) is a Vertex AI-style batch
    # prediction service: inputs/outputs flow through GCS buckets, it needs a
    # GCP project + service-account credentials (separate from GEMINI_API_KEY),
    # and — critically — it does NOT support tool calls or GoogleSearch
    # grounding. Grounding is only available on synchronous/streaming calls.
    # ``supports_batch=False`` is therefore permanent until Google ships a
    # grounding-capable batch tier. In the meantime the BatchDispatcher routes
    # this adapter to the live-sidecar path: ``GeminiAdapter.call()`` runs live
    # during ``truthbot publish --mode batch`` with full GoogleSearch grounding
    # and context caching (``SYNTHESIS_SYSTEM`` cached process-wide), and
    # verdicts are spooled to ``metrics/batch_sidecar/<run_id>.jsonl`` for
    # reconcile-time merging. This preserves verdict quality at the cost of
    # missing the 50% batch discount that Anthropic/OpenAI enjoy.
    supports_batch = False
    # Multi-claim batching only ships for ``supports_batch=True`` adapters
    # today (batch mode). Keeping this explicit at 1 documents that the live
    # sidecar path is single-claim; raising it will be done alongside the
    # Phase E live-mode multi-claim fan-out backlog item.
    max_claims_per_request = 1
    # Process-wide reuse of Context Caching resource for the static rubric.
    _cached_content_name: str | None = None

    # ── Batch support (payload/parse only; submit/retrieve guarded above) ─────

    def build_batch_payload(
        self,
        claim: Claim,
        evidence: list[Evidence],
        *,
        inject_evidence: bool = True,
    ) -> dict:
        """Provider-agnostic description of a single Gemini batch request."""
        user_msg = build_user_message(claim, evidence, inject_evidence=inject_evidence)
        return {
            "model": self.model_id,
            "contents": user_msg,
            "system_instruction": SYNTHESIS_SYSTEM,
            "tools": [{"google_search": {}}],
        }

    def parse_batch_response(
        self,
        raw_response: Any,
        claim: Claim,
    ) -> ModelVerdict:
        """Parse a Gemini batch result row (shape: GenerateContentResponse-like)."""
        candidates = _get(raw_response, "candidates", []) or []
        verdict_text = ""
        urls: list[str] = []

        for candidate in candidates:
            gm = _get(candidate, "grounding_metadata", None)
            if gm:
                for chunk in _get(gm, "grounding_chunks", []) or []:
                    web = _get(chunk, "web", None)
                    if web:
                        url = _get(web, "uri", None)
                        if url:
                            urls.append(url)
            content = _get(candidate, "content", None)
            for part in _get(content, "parts", []) or []:
                verdict_text += _get(part, "text", "") or ""

        model_id = _get(raw_response, "model_version", self.model_id)

        try:
            raw = _parse_verdict_json(verdict_text)
            label = VerdictLabel(raw["label"])
            confidence = Confidence(raw["confidence"])
        except Exception as exc:
            logger.error("GeminiAdapter batch parse error for claim %s: %s", claim.id, exc)
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

        usage = _get(raw_response, "usage_metadata", None)
        cached = _get(usage, "cached_content_token_count", 0) or 0

        return ModelVerdict(
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
        )

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
