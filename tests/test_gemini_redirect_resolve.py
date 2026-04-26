"""Unit tests for ``resolve_gemini_redirect`` (anti-hallucination Layer 1c)."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from truthbot.verify.adapters.gemini import resolve_gemini_redirect
from truthbot.verify.url_validation import UrlCache, UrlCheckResult


GROUNDING_REDIRECT = (
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/"
    "AUZIYQGpUK0yb_tDA6dD892o4NUHTUrshTQPfWDPk9LLOdB"
)
REAL_TARGET = "https://www.bls.gov/news.release/cpi_03122025.htm"


class TestResolveGeminiRedirect:
    def test_non_redirect_url_passthrough(self):
        # Real URLs (in the rare case Gemini ever emits one direct) are
        # returned unchanged. No network call.
        url = "https://www.bls.gov/news.release/cpi.htm"
        with patch("truthbot.verify.url_validation.check_url") as fake:
            assert resolve_gemini_redirect(url) == url
            fake.assert_not_called()

    def test_empty_returns_none(self):
        assert resolve_gemini_redirect("") is None
        assert resolve_gemini_redirect(None) is None  # type: ignore[arg-type]
        assert resolve_gemini_redirect(123) is None  # type: ignore[arg-type]

    def test_redirect_resolves_to_final_url(self):
        # Redirect URL that successfully resolves: helper returns final_url.
        ok = UrlCheckResult(
            url=GROUNDING_REDIRECT,
            reachable=True,
            status=200,
            checked_at=datetime.utcnow().isoformat(),
            final_url=REAL_TARGET,
        )
        with patch(
            "truthbot.verify.url_validation.check_url", return_value=ok
        ) as fake:
            assert resolve_gemini_redirect(GROUNDING_REDIRECT) == REAL_TARGET
            fake.assert_called_once()

    def test_redirect_resolution_failure_returns_none(self):
        # 403 on the redirect endpoint with no final_url → None (drop).
        bad = UrlCheckResult(
            url=GROUNDING_REDIRECT,
            reachable=False,
            status=403,
            error="http-403",
            checked_at=datetime.utcnow().isoformat(),
            final_url=None,
        )
        with patch(
            "truthbot.verify.url_validation.check_url", return_value=bad
        ):
            assert resolve_gemini_redirect(GROUNDING_REDIRECT) is None

    def test_cache_hit_does_not_call_network(self, tmp_path):
        cache = UrlCache()
        cache.put(
            UrlCheckResult(
                url=GROUNDING_REDIRECT,
                reachable=True,
                status=200,
                checked_at=datetime.utcnow().isoformat(),
                final_url=REAL_TARGET,
            )
        )
        with patch("truthbot.verify.url_validation.check_url") as fake:
            assert (
                resolve_gemini_redirect(GROUNDING_REDIRECT, cache=cache)
                == REAL_TARGET
            )
            fake.assert_not_called()

    def test_cache_miss_writes_result(self, tmp_path):
        cache = UrlCache()
        ok = UrlCheckResult(
            url=GROUNDING_REDIRECT,
            reachable=True,
            status=200,
            checked_at=datetime.utcnow().isoformat(),
            final_url=REAL_TARGET,
        )
        with patch(
            "truthbot.verify.url_validation.check_url", return_value=ok
        ) as fake:
            assert (
                resolve_gemini_redirect(GROUNDING_REDIRECT, cache=cache)
                == REAL_TARGET
            )
            fake.assert_called_once()
        # Subsequent call must hit the cache.
        with patch("truthbot.verify.url_validation.check_url") as fake2:
            assert (
                resolve_gemini_redirect(GROUNDING_REDIRECT, cache=cache)
                == REAL_TARGET
            )
            fake2.assert_not_called()

    def test_check_url_exception_returns_none(self):
        with patch(
            "truthbot.verify.url_validation.check_url",
            side_effect=RuntimeError("network down"),
        ):
            assert resolve_gemini_redirect(GROUNDING_REDIRECT) is None
