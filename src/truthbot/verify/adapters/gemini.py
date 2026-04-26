"""
Google Gemini adapter with Google Search grounding.
Migrated from google-generativeai (deprecated) to google-genai SDK.
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
    SYNTHESIS_SYSTEM,
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


# Gemini's ``grounding_chunks[].web.uri`` is *always* a Vertex AI
# grounding-redirect URL of the form
# ``https://vertexaisearch.cloud.google.com/grounding-api-redirect/<opaque>``.
# These are useless as standalone citation targets — they require a session
# cookie to follow and return 403 otherwise — so we resolve them to their
# final destination via :func:`resolve_gemini_redirect` before they enter
# the anti-hallucination ground-truth intersection. URLs that fail to
# resolve are dropped.
#
# Phase 3b follow-up: v-p1-p2 had 3/45 URLs in this pattern. See
# eval/sotu-2026/v-p1-p2-followups.md follow-up (B). Phase: anti-
# hallucination Layer 1c.
_GEMINI_OPAQUE_URL_PREFIXES = (
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/",
    "http://vertexaisearch.cloud.google.com/grounding-api-redirect/",
)


def _should_keep_gemini_url(url: str) -> bool:
    """Reject Gemini grounding-redirect URLs (not durable citations).

    Retained for any legacy call sites; new code should call
    :func:`resolve_gemini_redirect` instead, which actively follows the
    redirect rather than dropping it.
    """
    if not isinstance(url, str) or not url:
        return False
    return not url.startswith(_GEMINI_OPAQUE_URL_PREFIXES)


def resolve_gemini_redirect(
    url: str,
    *,
    cache: Any = None,
    timeout: float = 5.0,
) -> "str | None":
    """Resolve a Gemini grounding-redirect URL to its final destination.

    Behavior matrix:
      * ``url`` is empty / not a string → return ``None`` (drop).
      * ``url`` is not a Vertex AI grounding-redirect → return ``url``
        unchanged (passthrough, in case Gemini ever returns a real URL).
      * ``url`` is a grounding-redirect:
          - cache hit with a resolved ``final_url`` → return that.
          - cache hit reporting a failed resolution → return ``None``.
          - cache miss → run :func:`url_validation.check_url`, persist the
            result to the cache (if provided), and return ``final_url`` on
            success or ``None`` on any failure (timeout, 403, no redirect
            chain, etc.).

    The cache argument is typed as ``Any`` to avoid a hard dependency from
    ``gemini.py`` onto ``url_validation`` at import time — pass a
    ``UrlCache`` instance from the caller (e.g. the engine), which is the
    common case.
    """
    if not isinstance(url, str) or not url:
        return None
    if not url.startswith(_GEMINI_OPAQUE_URL_PREFIXES):
        return url

    if cache is not None:
        try:
            cached = cache.get(url)
        except Exception:
            cached = None
        if cached is not None:
            # ``UrlCheckResult.final_url`` is None on resolution failure;
            # we propagate that as a drop signal.
            return cached.final_url

    from truthbot.verify.url_validation import check_url

    try:
        result = check_url(url, timeout=timeout)
    except Exception:
        return None

    if cache is not None:
        try:
            cache.put(result)
        except Exception:
            pass

    return result.final_url


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
    # Live multi-claim batching folds SYNTHESIS_SYSTEM across N claims in a
    # single ``generate_content`` call, amortizing the ~1.5 K-token rubric and
    # (when CachedContent is active) keeping cache hits on the shared prefix.
    # Cap kept conservative at 4 because GoogleSearch grounding quality drops
    # as N grows — grounding is per-request, not per-claim. Raise after
    # smoke-measured grounding accuracy at higher N (tracked in the Phase E
    # plan).
    max_claims_per_request = 4
    # Process-wide reuse of Context Caching resource for the static rubric,
    # keyed by model id. Keying by model is required because Google's API
    # rejects ``generate_content`` calls whose model differs from the one
    # the CachedContent was created against ("Model used by GenerateContent
    # request and CachedContent has to be the same"). Triage and frontier
    # share the rubric prefix but use different models (e.g. flash vs pro),
    # so each gets its own cache entry instead of stomping on a single slot.
    _cached_content_names: dict[str, str] = {}
    # Shared URL cache for grounding-redirect resolution (anti-hallucination
    # Layer 1c). Lazy-loaded from ``metrics/url_cache.jsonl`` on first use,
    # mutated in-place across calls. Persistence is the publish layer's job
    # via the ``truthbot urls`` subcommands; the adapter does not save.
    _url_cache: Any = None

    @classmethod
    def _get_redirect_cache(cls) -> Any:
        """Return a process-wide ``UrlCache`` for redirect resolution.

        Returns ``None`` if the cache module or backing file is unavailable
        (tests, sandboxed runs); ``resolve_gemini_redirect`` handles ``None``
        gracefully and proceeds without persistence.
        """
        if cls._url_cache is not None:
            return cls._url_cache
        try:
            from truthbot.config import settings
            from truthbot.verify.url_validation import UrlCache

            cache_path = settings.metrics_dir / "url_cache.jsonl"
            cls._url_cache = UrlCache.load(cache_path)
        except Exception as exc:
            logger.debug("Gemini redirect cache unavailable: %s", exc)
            cls._url_cache = None
        return cls._url_cache

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
        search_query_count = 0

        for candidate in candidates:
            gm = _get(candidate, "grounding_metadata", None)
            if gm:
                search_query_count += len(
                    _get(gm, "web_search_queries", []) or []
                )
                cache = self._get_redirect_cache()
                for chunk in _get(gm, "grounding_chunks", []) or []:
                    web = _get(chunk, "web", None)
                    if web:
                        url = _get(web, "uri", None)
                        if url:
                            resolved = resolve_gemini_redirect(url, cache=cache)
                            if resolved:
                                urls.append(resolved)
            content = _get(candidate, "content", None)
            for part in _get(content, "parts", []) or []:
                verdict_text += _get(part, "text", "") or ""

        model_id = _get(raw_response, "model_version", self.model_id)

        try:
            raw = _parse_verdict_json(verdict_text)
            label = normalize_verdict_label(raw["label"])
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
            tool_call_count=int(search_query_count),
        )
        apply_temporal_flags(verdict, claim)
        return verdict

    def __init__(self) -> None:
        super().__init__()
        self._api_key = os.environ["GEMINI_API_KEY"]
        self._active_model = self.model_id

    def _get_or_create_cached_content(self, client: object, types: object) -> str | None:
        """Create a Context Caching entry for SYNTHESIS_SYSTEM once per process,
        keyed by ``self._active_model``.

        Google's genai API rejects a ``generate_content`` call that passes
        ``cached_content`` alongside ``system_instruction`` or ``tools`` with:

            CachedContent can not be used with GenerateContent request setting
            system_instruction, tools or tool_config.

        It also requires the request model to match the cache's model:

            Model used by GenerateContent request (models/gemini-2.5-pro) and
            CachedContent (models/gemini-2.5-flash) has to be the same.

        Because ``GeminiAdapter`` is shared by triage (``gemini-2.5-flash``)
        and frontier (``gemini-2.5-pro``) subclasses, the cache map is keyed
        by ``self._active_model``. The Phase 3a calibration broke when both
        tiers shared a single cache slot: triage created the cache against
        flash, and every subsequent frontier call 400'd until the slot
        expired. Keying by model preserves cross-instance reuse within a
        tier while preventing cross-tier contamination.

        The fix for the system_instruction/tools rejection (separate issue)
        remains: bind both ``system_instruction`` AND ``tools`` into the
        ``CachedContent`` at creation time; the per-claim ``GenerateContent``
        call then references the cache and passes neither field.
        """
        model = self._active_model
        cached_name = GeminiAdapter._cached_content_names.get(model)
        if cached_name:
            return cached_name
        try:
            tools = [types.Tool(google_search=types.GoogleSearch())]
            create_config = types.CreateCachedContentConfig(
                display_name=f"truthbot-synthesis-rubric-{model}",
                system_instruction=SYNTHESIS_SYSTEM,
                tools=tools,
                ttl="14400s",
            )
            cached = client.caches.create(
                model=model,
                config=create_config,
            )
            name = getattr(cached, "name", None)
            if name:
                GeminiAdapter._cached_content_names[model] = name
            return name
        except Exception as exc:
            logger.warning(
                "Gemini context caching unavailable for model %s, using inline system_instruction: %s",
                model,
                exc,
            )
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
                if cache_name:
                    # system_instruction + tools live on the CachedContent; passing
                    # either here is a hard API error ("CachedContent can not be
                    # used with GenerateContent request setting system_instruction,
                    # tools or tool_config").
                    gen_config = types.GenerateContentConfig(
                        cached_content=cache_name,
                    )
                else:
                    tools = [types.Tool(google_search=types.GoogleSearch())]
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
                cache = self._get_redirect_cache()
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
                                    resolved = resolve_gemini_redirect(
                                        url, cache=cache
                                    )
                                    if resolved:
                                        urls.append(resolved)

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
                    tier=telemetry_tier,
                    synthesis_mode=get_synthesis_mode(),
                    cached_input_tokens=int(td.get("gemini_cached_content_tokens", 0)),
                    tool_call_count=int(search_query_count),
                )
                apply_temporal_flags(verdict, claim)
                return verdict

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
        """Call Gemini once for N claims; reuses CachedContent on SYNTHESIS_SYSTEM.

        The whole point: with a live CachedContent handle active, one
        ``generate_content`` call over N claims produces one cache-hit on the
        rubric instead of N. ``usage_metadata.cached_content_token_count``
        is per-response, not per-claim, so it lands on index-0 via
        ``build_multi_verdicts``.
        """
        if not claims:
            return []

        from google import genai
        from google.genai import types

        telemetry = get_telemetry()
        n = len(claims)
        user_msg = build_multi_user_message(
            claims,
            evidence_by_claim,
            inject_evidence=inject_evidence,
            max_evidence_per_claim=max_evidence_per_claim,
        )
        batch_call_id = f"gemini-live-multi-{uuid.uuid4().hex[:12]}"

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
                client = genai.Client(api_key=self._api_key)

                cache_name = self._get_or_create_cached_content(client, types)
                if cache_name:
                    # Regression guard: cached_content + system_instruction /
                    # tools is a hard Google API error. See test_gemini_cache.py.
                    gen_config = types.GenerateContentConfig(
                        cached_content=cache_name,
                    )
                else:
                    tools = [types.Tool(google_search=types.GoogleSearch())]
                    gen_config = types.GenerateContentConfig(
                        system_instruction=SYNTHESIS_SYSTEM,
                        tools=tools,
                    )

                response = client.models.generate_content(
                    model=self._active_model,
                    contents=user_msg,
                    config=gen_config,
                )

                urls: list[str] = []
                search_query_count = 0
                candidates = response.candidates or []
                cache = self._get_redirect_cache()
                for candidate in candidates:
                    gm = getattr(candidate, "grounding_metadata", None)
                    if gm:
                        search_query_count += len(
                            getattr(gm, "web_search_queries", []) or []
                        )
                        for chunk in getattr(gm, "grounding_chunks", []) or []:
                            web = getattr(chunk, "web", None)
                            if web:
                                url = getattr(web, "uri", None)
                                if url:
                                    resolved = resolve_gemini_redirect(
                                        url, cache=cache
                                    )
                                    if resolved:
                                        urls.append(resolved)

                verdict_text = ""
                try:
                    verdict_text = response.text or ""
                except Exception:
                    for candidate in candidates:
                        content = getattr(candidate, "content", None)
                        for part in getattr(content, "parts", []) or []:
                            verdict_text += getattr(part, "text", "") or ""

                usage_meta = getattr(response, "usage_metadata", None)
                if usage_meta:
                    td["input_tokens"] = (
                        getattr(usage_meta, "prompt_token_count", 0) or 0
                    )
                    td["output_tokens"] = (
                        getattr(usage_meta, "candidates_token_count", 0) or 0
                    )
                    td["gemini_cached_content_tokens"] = (
                        getattr(usage_meta, "cached_content_token_count", 0) or 0
                    )
                td["tool_call_count"] = search_query_count
                td["retrieved_url_count"] = len(urls)

                try:
                    raw_by_claim = parse_multi_claim_json(verdict_text, claims)
                except json.JSONDecodeError as exc:
                    logger.error(
                        "GeminiAdapter multi-claim parse error (call=%s, n=%d): %s",
                        batch_call_id, n, exc,
                    )
                    td["status"] = "parse_error"
                    raw_by_claim = {}
                else:
                    td["status"] = "ok"

                call_usage = {
                    "input_tokens": int(td.get("input_tokens", 0) or 0),
                    "output_tokens": int(td.get("output_tokens", 0) or 0),
                    "cached_input_tokens": int(
                        td.get("gemini_cached_content_tokens", 0) or 0
                    ),
                    "tool_call_count": int(search_query_count),
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
                return verdicts

            except Exception as exc:
                exc_str = str(exc).lower()
                td["status"] = (
                    "timeout"
                    if ("timeout" in exc_str or "timed out" in exc_str)
                    else "api_error"
                )
                logger.error(
                    "GeminiAdapter multi-claim API error (call=%s, n=%d): %s",
                    batch_call_id, n, exc,
                )
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
