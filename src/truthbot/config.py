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

    # ── Search ────────────────────────────────────────────────────────────────
    @property
    def brave_api_key(self) -> str:
        return _require("BRAVE_API_KEY")

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
