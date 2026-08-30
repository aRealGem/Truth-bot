"""
Configuration — all settings sourced from environment variables.

No secrets should ever be hardcoded here. Load a .env file in development
(e.g. `python-dotenv`), or set vars directly in your shell/process manager.
"""

from __future__ import annotations

import os
from pathlib import Path


def _require(key: str) -> str:
    """Return an env var or raise a clear error if missing."""
    val = os.environ.get(key)
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"See .env.example for setup instructions."
        )
    return val


def _optional(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


class Settings:
    """
    Central settings object. Instantiate once at startup.

    Attributes are lazy — they read from the environment at access time so
    tests can patch os.environ freely.
    """

    # ── LLM ──────────────────────────────────────────────────────────────────
    @property
    def anthropic_api_key(self) -> str:
        return _require("ANTHROPIC_API_KEY")

    @property
    def max_claims(self) -> int:
        return int(_optional("TRUTHBOT_MAX_CLAIMS", "30"))

    @property
    def claims_per_request(self) -> int:
        """
        How many atomic claims to bundle into a single LLM request.

        Default ``5`` enables multi-claim batching out of the box (the user
        message enumerates N claims and the model returns a JSON array of
        verdicts). Set ``TRUTHBOT_CLAIMS_PER_REQUEST=1`` to force the legacy
        one-claim-per-call behaviour. Per-adapter ``max_claims_per_request``
        caps clamp this value at submit time. Honoured in both
        ``--mode batch`` (native batch API multi-claim payloads) and
        ``--mode live`` (Phase E Grok/Gemini slice — live multi-claim via
        ``LLMAdapter.call_multi`` → ``VerificationEngine.verify_bundles_batch``).
        """
        try:
            return max(1, int(_optional("TRUTHBOT_CLAIMS_PER_REQUEST", "5")))
        except ValueError:
            return 5

    @property
    def openai_live_mode(self) -> bool:
        """Run the OpenAI adapter live (sidecar) instead of via the Batch API.

        OpenAI exposes a 50%-discounted Batch API (1× input/output token cost
        becomes 0.5×) but its end-to-end turnaround in practice is hours
        (3–24h SLA), which is unworkable for tight iteration cycles. Setting
        ``TRUTHBOT_OPENAI_LIVE=1`` flips the pipeline to dispatch OpenAI
        through the live Responses API at submit time alongside Gemini/Grok,
        with verdicts persisted in the sidecar. Roughly trades 50% of OpenAI's
        per-run cost for a sub-minute completion window.

        Default ``False`` preserves the legacy batch path so existing
        scheduled/long-running fact-check runs keep their cost discount
        until explicitly opted out.
        """
        return _optional("TRUTHBOT_OPENAI_LIVE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    @property
    def max_evidence_per_claim_in_batch(self) -> int:
        """
        Cap on injected evidence snippets per claim.

        Applied everywhere; single-claim and multi-claim paths share the same
        5-snippet cap by default. Tighten via ``TRUTHBOT_MAX_EVIDENCE_PER_CLAIM_IN_BATCH``
        if multi-claim runs ever push past provider context windows.
        """
        try:
            return max(1, int(_optional("TRUTHBOT_MAX_EVIDENCE_PER_CLAIM_IN_BATCH", "5")))
        except ValueError:
            return 5

    # ── Search ────────────────────────────────────────────────────────────────
    @property
    def brave_api_key(self) -> str:
        return _require("BRAVE_API_KEY")

    @property
    def prefetch_evidence(self) -> bool:
        """
        When False (default), the verification engine does not call source connectors;
        models receive an empty evidence list and rely on their own web search.
        Set TRUTHBOT_PREFETCH_EVIDENCE=1 to enable Brave / FactCheck / Gov prefetch.

        Prefer ``TRUTHBOT_EVIDENCE_SOURCE``; when that is unset, ``prefetch_evidence``
        maps to ``connectors`` vs ``none`` for backward compatibility.
        """
        return _optional("TRUTHBOT_PREFETCH_EVIDENCE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    @property
    def evidence_source(self) -> str:
        """
        ``none`` | ``connectors`` | ``datahoover``.

        If ``TRUTHBOT_EVIDENCE_SOURCE`` is unset, derive from ``TRUTHBOT_PREFETCH_EVIDENCE``
        (truthy → ``connectors``, else ``none``).
        """
        raw = _optional("TRUTHBOT_EVIDENCE_SOURCE", "").strip().lower()
        if raw in ("none", "connectors", "datahoover"):
            return raw
        return "connectors" if self.prefetch_evidence else "none"

    # ── Government APIs ───────────────────────────────────────────────────────
    @property
    def fred_api_key(self) -> str:
        return _optional("FRED_API_KEY", "")

    # ── Bluesky ───────────────────────────────────────────────────────────────
    @property
    def bluesky_handle(self) -> str:
        return _optional("BLUESKY_HANDLE", "")

    @property
    def bluesky_app_password(self) -> str:
        return _optional("BLUESKY_APP_PASSWORD", "")

    @property
    def bluesky_enabled(self) -> bool:
        return bool(self.bluesky_handle and self.bluesky_app_password)

    # ── Storage ───────────────────────────────────────────────────────────────
    @property
    def cache_dir(self) -> Path:
        return Path(_optional("TRUTHBOT_CACHE_DIR", "./truthbot_cache"))

    @property
    def report_dir(self) -> Path:
        return Path(_optional("TRUTHBOT_REPORT_DIR", "./reports"))

    @property
    def metrics_dir(self) -> Path:
        return Path(_optional("TRUTHBOT_METRICS_DIR", "./metrics"))

    @property
    def site_root(self) -> Path:
        return Path(_optional("TRUTHBOT_SITE_ROOT", "/var/www/truthbot"))

    @property
    def site_url(self) -> str:
        """Public base URL of the published static site (no trailing slash).

        Every absolute self-link (canonical/og:url tags, feed.xml entries)
        derives from this one value. Override with ``TRUTHBOT_SITE_URL`` when
        the site moves to a custom domain; the default is GitHub Pages, which
        has served the committed ``site-pca/`` tree since 2026-08-29.

        Must stay equal to ``publish.site._DEFAULT_SITE_URL`` --
        ``tests/publish/test_stable_slugs.py`` asserts it.
        """
        return _optional(
            "TRUTHBOT_SITE_URL",
            "https://arealgem.github.io/Truth-bot/site-pca",
        ).rstrip("/")

    @property
    def openai_api_key(self) -> str:
        return _optional("OPENAI_API_KEY", "")

    @property
    def gemini_api_key(self) -> str:
        return _optional("GEMINI_API_KEY", "")

    @property
    def xai_api_key(self) -> str:
        return _optional("XAI_API_KEY", "")

    # ── Similarity / cache ────────────────────────────────────────────────────
    @property
    def cache_similarity_threshold(self) -> int:
        """Minimum fuzzy match score (0–100) to consider a claim a cache hit."""
        return int(_optional("TRUTHBOT_CACHE_THRESHOLD", "85"))


# Module-level singleton — import and use directly.
settings = Settings()
