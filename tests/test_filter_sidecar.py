"""Tests for ``filter_sidecar`` / ``filter_sidecar_row`` (Layer 3).

Covers:

1. All-ok pass-through (web_sources unchanged, broken empty).
2. Mixed ok + dead-4xx → broken URL stripped from web_sources.
3. Bot-blocked → kept (in web_sources via unverified_sources).
4. Malformed + DNS errors → both stripped to broken_sources.
5. End-to-end ``filter_sidecar`` round-trip via cache (no network).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from truthbot.verify.url_validation import (
    UrlCache,
    UrlCheckResult,
    filter_sidecar,
    filter_sidecar_row,
)


def _ok(url: str) -> UrlCheckResult:
    return UrlCheckResult(
        url=url,
        reachable=True,
        status=200,
        method_used="HEAD",
        checked_at=datetime.utcnow().isoformat(),
    )


def _dead_404(url: str) -> UrlCheckResult:
    return UrlCheckResult(
        url=url,
        reachable=False,
        status=404,
        error="http-404",
        method_used="HEAD",
        checked_at=datetime.utcnow().isoformat(),
    )


def _bot_blocked(url: str) -> UrlCheckResult:
    return UrlCheckResult(
        url=url,
        reachable=False,
        status=403,
        error="http-403",
        method_used="GET",
        checked_at=datetime.utcnow().isoformat(),
    )


def _malformed(url: str) -> UrlCheckResult:
    return UrlCheckResult(
        url=url,
        reachable=False,
        error="invalid-scheme",
        checked_at=datetime.utcnow().isoformat(),
    )


def _dns(url: str) -> UrlCheckResult:
    return UrlCheckResult(
        url=url,
        reachable=False,
        error="head:ConnectError:nodename nor servname provided",
        checked_at=datetime.utcnow().isoformat(),
    )


def test_filter_row_all_ok_passthrough():
    row = {
        "claim_id": "c1",
        "adapter_name": "openai",
        "web_sources": [
            "https://www.bls.gov/cps/",
            "https://www.cbo.gov/publication/12345",
        ],
        "model_reported_sources": [
            "https://www.bls.gov/cps/",
            "https://www.cbo.gov/publication/12345",
        ],
    }
    results = {u: _ok(u) for u in row["web_sources"]}

    out = filter_sidecar_row(row, results=results)

    assert out["verified_sources"] == row["web_sources"]
    assert out["unverified_sources"] == []
    assert out["broken_sources"] == []
    assert out["web_sources"] == row["web_sources"]
    # audit trail untouched
    assert out["model_reported_sources"] == row["model_reported_sources"]


def test_filter_row_mixed_ok_and_dead_strips_dead():
    good = "https://www.bls.gov/cps/"
    dead = "https://www.example.gov/this-page-does-not-exist"
    row = {
        "claim_id": "c1",
        "web_sources": [good, dead],
    }
    results = {good: _ok(good), dead: _dead_404(dead)}

    out = filter_sidecar_row(row, results=results)

    assert out["verified_sources"] == [good]
    assert out["unverified_sources"] == []
    assert out["broken_sources"] == [dead]
    assert out["web_sources"] == [good], "dead URL must be stripped"
    assert out["url_filter_classification"][dead] == "dead-4xx"


def test_filter_row_bot_blocked_kept_as_unverified():
    """A bot-blocked URL on a trusted .gov / news domain is almost
    certainly real — keep it but tag for unverified rendering."""
    blocked = "https://www.cbp.gov/newsroom/stats/something"
    row = {"claim_id": "c1", "web_sources": [blocked]}
    results = {blocked: _bot_blocked(blocked)}

    out = filter_sidecar_row(row, results=results)

    assert out["verified_sources"] == []
    assert out["unverified_sources"] == [blocked]
    assert out["broken_sources"] == []
    assert out["web_sources"] == [blocked], (
        "bot-blocked URLs from trusted domains must remain in web_sources "
        "so the publish layer can render them as unverified rather than "
        "silently dropping legitimate citations."
    )


def test_filter_row_malformed_and_dns_both_stripped():
    bad_scheme = "ftp://nope.example.com/x"
    bad_dns = "https://this-domain-does-not-exist-truthbot-test.invalid/"
    row = {"claim_id": "c1", "web_sources": [bad_scheme, bad_dns]}
    results = {bad_scheme: _malformed(bad_scheme), bad_dns: _dns(bad_dns)}

    out = filter_sidecar_row(row, results=results)

    assert out["verified_sources"] == []
    assert out["unverified_sources"] == []
    assert sorted(out["broken_sources"]) == sorted([bad_scheme, bad_dns])
    assert out["web_sources"] == []


def test_filter_sidecar_end_to_end_via_cache(tmp_path: Path):
    """Pre-populate the cache with deterministic results so no network
    I/O is needed; verify the cleaned file is correctly written."""
    good = "https://www.bls.gov/cps/"
    dead = "https://www.fake.gov/dead-link"

    src = tmp_path / "sidecar.jsonl"
    rows_in = [
        {"claim_id": "c1", "adapter_name": "openai", "web_sources": [good, dead]},
        {"claim_id": "c2", "adapter_name": "anthropic", "web_sources": [good]},
        {"claim_id": "c3", "adapter_name": "grok", "web_sources": []},
    ]
    with src.open("w") as f:
        for r in rows_in:
            f.write(json.dumps(r) + "\n")

    cache = UrlCache()
    cache.put(_ok(good))
    cache.put(_dead_404(dead))

    out = tmp_path / "sidecar.cleaned.jsonl"
    stats = filter_sidecar(src, out, cache=cache)

    assert stats == {"rows": 3, "verified": 2, "unverified": 0, "broken": 1}

    written = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(written) == 3
    assert written[0]["web_sources"] == [good]
    assert written[0]["broken_sources"] == [dead]
    assert written[1]["web_sources"] == [good]
    assert written[2]["web_sources"] == []
    assert written[2]["broken_sources"] == []
