"""
Unit tests for ``verify.url_validation`` (Phase 3b).

Tests use ``monkeypatch`` on the ``_request`` seam rather than ``respx``
so they run without any httpx network I/O and without adding a new
test-time dependency. The seam is the single point where real HTTP
would happen; patching it exercises every branch of the checker.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import pytest

from truthbot.models import Confidence, ModelVerdict, VerdictLabel
from truthbot.verify import url_validation as uv


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def fake_requests(monkeypatch: pytest.MonkeyPatch) -> Callable[[dict], None]:
    """Factory that installs a fake ``_request`` returning canned responses.

    Pass a mapping of ``url -> (status_code | Exception, final_url)``; any
    URL not in the mapping raises ``ConnectionError`` (simulating DNS fail).
    Exception *instances* are raised to exercise the error path.
    """

    def _install(responses: dict) -> None:
        def _fake(url: str, *, method: str, timeout: float, user_agent: str):
            if url not in responses:
                raise ConnectionError(f"no fake for {url}")
            rv = responses[url]
            if isinstance(rv, BaseException):
                raise rv
            status, final = rv
            return status, final

        monkeypatch.setattr(uv, "_request", _fake)

    return _install


def _verdict(
    *,
    claim_id: str = "c1",
    adapter_name: str = "openai",
    sources: list[str] | None = None,
) -> ModelVerdict:
    return ModelVerdict(
        adapter_name=adapter_name,
        model_id="test",
        claim_id=claim_id,
        label=VerdictLabel.TRUE,
        confidence=Confidence.HIGH,
        explanation="x",
        web_sources=sources or [],
    )


# ── check_url: happy path + variants ────────────────────────────────────


class TestCheckUrl:
    def test_head_200_is_reachable(self, fake_requests) -> None:
        fake_requests({"https://example.com/a": (200, None)})
        r = uv.check_url("https://example.com/a")
        assert r.reachable and r.status == 200 and r.method_used == "HEAD"

    def test_head_301_follow_keeps_reachable(self, fake_requests) -> None:
        """httpx follows redirects by default; final URL is exposed."""
        fake_requests(
            {"https://example.com/old": (200, "https://example.com/new")}
        )
        r = uv.check_url("https://example.com/old")
        assert r.reachable
        assert r.final_url == "https://example.com/new"

    def test_head_404_is_unreachable(self, fake_requests) -> None:
        fake_requests({"https://example.com/404": (404, None)})
        r = uv.check_url("https://example.com/404")
        assert not r.reachable
        assert r.status == 404
        assert r.error == "http-404"

    def test_head_403_retries_with_get(self, fake_requests) -> None:
        """Many .gov servers 403 HEAD but 200 GET; checker must retry."""

        calls: list[str] = []

        def _fake(url, *, method, timeout, user_agent):
            calls.append(method)
            return (403, None) if method == "HEAD" else (200, None)

        pytest.MonkeyPatch().setattr  # no-op to quiet type-checker on param name
        import truthbot.verify.url_validation as _mod

        mp = pytest.MonkeyPatch()
        mp.setattr(_mod, "_request", _fake)
        try:
            r = uv.check_url("https://agency.gov/report")
        finally:
            mp.undo()
        assert calls == ["HEAD", "GET"]
        assert r.reachable and r.method_used == "GET" and r.status == 200

    def test_head_405_retries_with_get_failure_stays_unreachable(
        self, fake_requests
    ) -> None:
        def _fake(url, *, method, timeout, user_agent):
            if method == "HEAD":
                return 405, None
            return 404, None

        mp = pytest.MonkeyPatch()
        mp.setattr(uv, "_request", _fake)
        try:
            r = uv.check_url("https://news.site/x")
        finally:
            mp.undo()
        assert not r.reachable and r.method_used == "GET" and r.status == 404

    def test_connection_error_is_unreachable_not_raised(self, fake_requests) -> None:
        fake_requests({"https://dead.example": TimeoutError("boom")})
        r = uv.check_url("https://dead.example")
        assert not r.reachable
        assert r.error is not None
        assert "TimeoutError" in r.error

    def test_invalid_scheme_is_rejected_without_http(self, fake_requests) -> None:
        fake_requests({})
        r = uv.check_url("javascript:alert(1)")
        assert not r.reachable
        assert r.error == "invalid-scheme"

    def test_empty_url_returns_unreachable(self, fake_requests) -> None:
        fake_requests({})
        r = uv.check_url("")
        assert not r.reachable


# ── Bulk check ──────────────────────────────────────────────────────────


class TestCheckUrlsBulk:
    def test_bulk_deduplicates(self, fake_requests) -> None:
        fake_requests(
            {
                "https://a.com": (200, None),
                "https://b.com": (404, None),
            }
        )
        res = uv.check_urls_bulk(
            ["https://a.com", "https://a.com", "https://b.com"],
            max_workers=2,
        )
        assert set(res.keys()) == {"https://a.com", "https://b.com"}
        assert res["https://a.com"].reachable
        assert not res["https://b.com"].reachable

    def test_bulk_empty_input(self, fake_requests) -> None:
        fake_requests({})
        assert uv.check_urls_bulk([]) == {}

    def test_bulk_uses_cache_when_fresh(self, fake_requests, tmp_path: Path) -> None:
        """Cached URLs must not spawn a worker / issue requests."""
        cache = uv.UrlCache()
        fresh = uv.UrlCheckResult(
            url="https://cached.com",
            reachable=True,
            status=200,
            checked_at=datetime.utcnow().isoformat(),
        )
        cache.put(fresh)

        called: list[str] = []

        def _fake(url, *, method, timeout, user_agent):
            called.append(url)
            return 200, None

        mp = pytest.MonkeyPatch()
        mp.setattr(uv, "_request", _fake)
        try:
            res = uv.check_urls_bulk(
                ["https://cached.com", "https://new.com"], cache=cache
            )
        finally:
            mp.undo()

        assert called == ["https://new.com"]
        assert res["https://cached.com"].reachable
        assert res["https://new.com"].reachable

    def test_bulk_writes_fresh_results_into_cache(self, fake_requests) -> None:
        fake_requests({"https://a.com": (200, None)})
        cache = uv.UrlCache()
        uv.check_urls_bulk(["https://a.com"], cache=cache)
        assert "https://a.com" in cache.entries
        assert cache.entries["https://a.com"].reachable

    def test_bulk_on_result_callback_fires_per_url(self, fake_requests) -> None:
        fake_requests(
            {"https://a.com": (200, None), "https://b.com": (404, None)}
        )
        seen: list[str] = []
        uv.check_urls_bulk(
            ["https://a.com", "https://b.com"],
            max_workers=2,
            on_result=lambda r: seen.append(r.url),
        )
        assert set(seen) == {"https://a.com", "https://b.com"}


# ── Cache I/O ───────────────────────────────────────────────────────────


class TestUrlCache:
    def test_roundtrip(self, tmp_path: Path) -> None:
        cache = uv.UrlCache()
        cache.put(
            uv.UrlCheckResult(
                url="https://a.com",
                reachable=True,
                status=200,
                checked_at=datetime.utcnow().isoformat(),
            )
        )
        p = tmp_path / "cache.jsonl"
        cache.save(p)
        assert p.exists()
        reloaded = uv.UrlCache.load(p)
        assert reloaded.entries["https://a.com"].reachable

    def test_load_tolerates_missing_file(self, tmp_path: Path) -> None:
        cache = uv.UrlCache.load(tmp_path / "nope.jsonl")
        assert cache.entries == {}

    def test_load_tolerates_malformed_row(self, tmp_path: Path) -> None:
        p = tmp_path / "c.jsonl"
        p.write_text("not json\n" + '{"url":"https://ok.com","reachable":true}\n')
        cache = uv.UrlCache.load(p)
        assert set(cache.entries) == {"https://ok.com"}

    def test_ttl_expires_old_entries(self) -> None:
        cache = uv.UrlCache(ttl_days=7)
        stale = uv.UrlCheckResult(
            url="https://stale.com",
            reachable=True,
            status=200,
            checked_at=(datetime.utcnow() - timedelta(days=30)).isoformat(),
        )
        cache.put(stale)
        assert cache.get("https://stale.com") is None

    def test_ttl_keeps_fresh_entries(self) -> None:
        cache = uv.UrlCache(ttl_days=7)
        fresh = uv.UrlCheckResult(
            url="https://fresh.com",
            reachable=True,
            status=200,
            checked_at=datetime.utcnow().isoformat(),
        )
        cache.put(fresh)
        assert cache.get("https://fresh.com") is fresh

    def test_save_is_atomic(self, tmp_path: Path) -> None:
        """``save`` writes via ``.tmp`` + rename — no partial files."""
        cache = uv.UrlCache()
        cache.put(
            uv.UrlCheckResult(
                url="https://a.com",
                reachable=True,
                status=200,
                checked_at=datetime.utcnow().isoformat(),
            )
        )
        p = tmp_path / "sub" / "c.jsonl"
        cache.save(p)
        assert p.exists()
        assert not (p.parent / (p.name + ".tmp")).exists()


# ── Verdict annotation ──────────────────────────────────────────────────


class TestAnnotateVerdicts:
    def test_attaches_reachable_and_unreachable_lists(self, fake_requests) -> None:
        fake_requests(
            {
                "https://live.gov": (200, None),
                "https://dead.news": (404, None),
            }
        )
        v = _verdict(sources=["https://live.gov", "https://dead.news"])
        audit = uv.annotate_verdicts([v])
        a = audit[(v.claim_id, v.adapter_name)]
        assert a.checked == ["https://live.gov", "https://dead.news"]
        assert a.reachable == ["https://live.gov"]
        assert a.unreachable == ["https://dead.news"]

    def test_dedupes_urls_across_verdicts(self, fake_requests) -> None:
        """Two verdicts citing the same URL should only HEAD it once."""
        count = {"n": 0}

        def _fake(url, *, method, timeout, user_agent):
            count["n"] += 1
            return 200, None

        mp = pytest.MonkeyPatch()
        mp.setattr(uv, "_request", _fake)
        try:
            v1 = _verdict(claim_id="c1", sources=["https://shared.com"])
            v2 = _verdict(claim_id="c2", sources=["https://shared.com"])
            audit = uv.annotate_verdicts([v1, v2])
        finally:
            mp.undo()

        assert count["n"] == 1
        assert audit[(v1.claim_id, v1.adapter_name)].reachable == ["https://shared.com"]
        assert audit[(v2.claim_id, v2.adapter_name)].reachable == ["https://shared.com"]

    def test_does_not_mutate_web_sources(self, fake_requests) -> None:
        """Audit trail requirement: never silently strip hallucinated URLs."""
        fake_requests({"https://dead.com": (404, None)})
        v = _verdict(sources=["https://dead.com"])
        before = list(v.web_sources)
        uv.annotate_verdicts([v])
        assert v.web_sources == before

    def test_empty_sources_yields_empty_audit(self, fake_requests) -> None:
        fake_requests({})
        v = _verdict(sources=[])
        audit = uv.annotate_verdicts([v])
        a = audit[(v.claim_id, v.adapter_name)]
        assert a.checked == []
        assert a.reachable == []
        assert a.unreachable == []


# ── Failure classifier (Phase 3b refinement) ────────────────────────────


class TestClassifyFailure:
    def test_reachable_classifies_ok(self) -> None:
        r = uv.UrlCheckResult(url="https://a.com", reachable=True, status=200)
        assert r.failure_class == "ok"
        assert r.likely_real

    def test_gov_403_is_bot_blocked_not_dead(self) -> None:
        """.gov sites commonly 403 HEAD+GET from scripts. Must NOT be
        classified as dead — they're almost certainly real."""
        r = uv.UrlCheckResult(
            url="https://www.bls.gov/cpi.html",
            reachable=False,
            status=403,
            error="http-403",
            method_used="GET",
        )
        assert r.failure_class == "bot-blocked"
        assert r.likely_real

    def test_major_news_403_is_bot_blocked(self) -> None:
        r = uv.UrlCheckResult(
            url="https://www.nytimes.com/article",
            reachable=False,
            status=403,
            error="http-403",
        )
        assert r.failure_class == "bot-blocked"
        assert r.likely_real

    def test_unknown_domain_403_is_not_bot_blocked(self) -> None:
        """Trust-tier gate: a 403 from ``random.example`` is suspicious."""
        r = uv.UrlCheckResult(
            url="https://random.example/x",
            reachable=False,
            status=403,
            error="http-403",
        )
        assert r.failure_class == "unknown"
        assert not r.likely_real

    def test_404_classifies_dead_4xx(self) -> None:
        r = uv.UrlCheckResult(
            url="https://a.com/nope",
            reachable=False,
            status=404,
            error="http-404",
        )
        assert r.failure_class == "dead-4xx"
        assert not r.likely_real

    def test_malformed_scheme_classifies_malformed(self) -> None:
        r = uv.UrlCheckResult(
            url="httpshttps://bad.com",
            reachable=False,
            status=None,
            error="invalid-scheme",
        )
        assert r.failure_class == "malformed"

    def test_dns_failure_classifies_dns(self) -> None:
        r = uv.UrlCheckResult(
            url="https://typo.example",
            reachable=False,
            status=None,
            error="head:ConnectError:[Errno 8] nodename nor servname provided, or not known",
        )
        assert r.failure_class == "dns"

    def test_ssl_failure_classifies_cert_error(self) -> None:
        r = uv.UrlCheckResult(
            url="https://broken-cert.com",
            reachable=False,
            status=None,
            error="head:SSLError:[SSL: CERTIFICATE_VERIFY_FAILED] hostname mismatch",
        )
        assert r.failure_class == "cert-error"

    def test_timeout_classifies_transient(self) -> None:
        r = uv.UrlCheckResult(
            url="https://slow.com",
            reachable=False,
            status=None,
            error="get:ReadTimeout:The read operation timed out",
        )
        assert r.failure_class == "transient"

    def test_500_classifies_transient(self) -> None:
        r = uv.UrlCheckResult(
            url="https://buggy.com",
            reachable=False,
            status=503,
            error="http-503",
        )
        assert r.failure_class == "transient"


# ── ModelVerdict.web_sources sanitizer (Phase 3b follow-up) ─────────────


class TestWebSourcesSanitizer:
    """Regression guards for the ``ModelVerdict.web_sources`` Pydantic
    field validator. Exercised through verdict construction so the
    validator fires as it will in production."""

    def test_double_scheme_gets_normalized(self) -> None:
        """The exact v-p1-p2 bug: ``httpshttps://www.ebc.com/...``."""
        v = _verdict(sources=["httpshttps://www.ebc.com/path"])
        assert v.web_sources == ["https://www.ebc.com/path"]

    def test_mixed_scheme_double_prefix(self) -> None:
        v = _verdict(sources=["httpshttp://example.com"])
        assert v.web_sources == ["http://example.com"]

    def test_well_formed_urls_are_unchanged(self) -> None:
        v = _verdict(sources=["https://good.com/path", "http://plain.example"])
        assert v.web_sources == [
            "https://good.com/path",
            "http://plain.example",
        ]

    def test_non_http_schemes_are_dropped(self) -> None:
        v = _verdict(
            sources=["ftp://weird.com", "javascript:alert(1)", "https://ok.com"]
        )
        assert v.web_sources == ["https://ok.com"]

    def test_whitespace_is_trimmed(self) -> None:
        v = _verdict(sources=["  https://trimmed.com  "])
        assert v.web_sources == ["https://trimmed.com"]

    def test_empty_and_none_are_dropped(self) -> None:
        v = _verdict(sources=["", "   ", None, 42, "https://ok.com"])  # type: ignore[list-item]
        assert v.web_sources == ["https://ok.com"]

    def test_non_list_value_defaults_to_empty(self) -> None:
        v = _verdict(sources=None)  # type: ignore[arg-type]
        assert v.web_sources == []


# ── Gemini grounding-redirect filter (Phase 3b follow-up) ──────────────


class TestGeminiRedirectFilter:
    """Regression guards for the ``_should_keep_gemini_url`` helper.

    The helper is imported from the gemini adapter and exercised directly —
    isolated from the full ``GeminiAdapter`` stack so it doesn't require
    the google-genai mocks.
    """

    def test_rejects_vertex_ai_grounding_redirect_https(self) -> None:
        from truthbot.verify.adapters.gemini import _should_keep_gemini_url

        url = (
            "https://vertexaisearch.cloud.google.com/grounding-api-redirect/"
            "AUZIYQGALX1-ptRaLEJglfXTDy2KY_WlyC7t_bwAIOB-3Pn"
        )
        assert not _should_keep_gemini_url(url)

    def test_rejects_vertex_ai_grounding_redirect_http(self) -> None:
        from truthbot.verify.adapters.gemini import _should_keep_gemini_url

        url = "http://vertexaisearch.cloud.google.com/grounding-api-redirect/XYZ"
        assert not _should_keep_gemini_url(url)

    def test_keeps_durable_vertex_ai_urls(self) -> None:
        """A non-redirect Vertex AI URL (if any shipped) is NOT stripped."""
        from truthbot.verify.adapters.gemini import _should_keep_gemini_url

        assert _should_keep_gemini_url("https://vertexaisearch.cloud.google.com/report")

    def test_keeps_normal_urls(self) -> None:
        from truthbot.verify.adapters.gemini import _should_keep_gemini_url

        for u in (
            "https://bls.gov/news.release/cpi.nr0.htm",
            "https://www.nytimes.com/article",
            "http://example.com/",
        ):
            assert _should_keep_gemini_url(u), u

    def test_rejects_empty_or_non_str(self) -> None:
        from truthbot.verify.adapters.gemini import _should_keep_gemini_url

        assert not _should_keep_gemini_url("")
        assert not _should_keep_gemini_url(None)  # type: ignore[arg-type]
        assert not _should_keep_gemini_url(42)  # type: ignore[arg-type]


# ── Import-time isolation ───────────────────────────────────────────────


def test_import_does_not_issue_http() -> None:
    """Importing the module must not touch the network.

    Guards against a refactor that accidentally moves httpx imports or
    probe calls to module scope.
    """
    import importlib

    import truthbot.verify.url_validation as _mod

    # Fresh reload; should not raise network errors even with no network.
    importlib.reload(_mod)
