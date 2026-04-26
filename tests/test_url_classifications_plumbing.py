"""Tests for the Layer 4 'Full' plumbing.

Covers:

1. ``classify_verdicts_in_place`` populates ``mv.url_classifications``
   on every verdict using the URL cache.
2. ``load_sidecar(path, cleaned_path=...)`` prefers the cleaned variant
   and translates ``url_filter_classification`` → ``url_classifications``.
3. ``_worse_classification`` returns the higher-ranked failure class.
4. Bundle-level rendering combines per-verdict classifications and
   strips broken URLs while keeping unverified ones with a badge.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from truthbot.models import ModelVerdict
from truthbot.publish.site import (
    _evidence_list_html,
    _worse_classification,
)
from truthbot.verify.batch import load_sidecar
from truthbot.verify.url_validation import (
    UrlCache,
    UrlCheckResult,
    classify_verdicts_in_place,
)


def _ok(url: str) -> UrlCheckResult:
    return UrlCheckResult(
        url=url,
        reachable=True,
        status=200,
        checked_at=datetime.utcnow().isoformat(),
    )


def _dead_404(url: str) -> UrlCheckResult:
    return UrlCheckResult(
        url=url,
        reachable=False,
        status=404,
        error="http-404",
        checked_at=datetime.utcnow().isoformat(),
    )


def _bot_blocked(url: str) -> UrlCheckResult:
    return UrlCheckResult(
        url=url,
        reachable=False,
        status=403,
        error="http-403",
        checked_at=datetime.utcnow().isoformat(),
    )


def _mv(claim_id: str, adapter: str, urls: list[str]) -> ModelVerdict:
    return ModelVerdict(
        adapter_name=adapter,
        model_id=f"{adapter}-test",
        claim_id=claim_id,
        label="True",
        confidence="High",
        explanation="x",
        web_sources=urls,
    )


def test_classify_verdicts_in_place_populates_per_verdict_map():
    good = "https://www.bls.gov/cps/"
    blocked = "https://www.cbp.gov/x"
    dead = "https://fake.gov/dead"

    cache = UrlCache()
    cache.put(_ok(good))
    cache.put(_bot_blocked(blocked))
    cache.put(_dead_404(dead))

    v1 = _mv("c1", "openai", [good, blocked])
    v2 = _mv("c1", "anthropic", [blocked, dead])

    stats = classify_verdicts_in_place([v1, v2], cache=cache)

    assert v1.url_classifications == {good: "ok", blocked: "bot-blocked"}
    assert v2.url_classifications == {blocked: "bot-blocked", dead: "dead-4xx"}
    assert stats == {"verified": 1, "unverified": 2, "broken": 1}


def test_classify_verdicts_in_place_empty_input_is_safe():
    stats = classify_verdicts_in_place([])
    assert stats == {"verified": 0, "unverified": 0, "broken": 0}


def test_load_sidecar_prefers_cleaned_and_translates_audit_field(
    tmp_path: Path,
):
    raw = tmp_path / "sidecar.jsonl"
    cleaned = tmp_path / "sidecar.cleaned.jsonl"

    raw_row = {
        "adapter_name": "gemini",
        "model_id": "gemini-2.5-pro",
        "claim_id": "c1",
        "label": "True",
        "confidence": "High",
        "explanation": "x",
        "web_sources": ["https://a.gov/", "https://b.example/"],
    }
    raw.write_text(json.dumps(raw_row) + "\n")

    cleaned_row = {
        **raw_row,
        "web_sources": ["https://a.gov/"],
        "verified_sources": ["https://a.gov/"],
        "unverified_sources": [],
        "broken_sources": ["https://b.example/"],
        "url_filter_classification": {
            "https://a.gov/": "ok",
            "https://b.example/": "dead-4xx",
        },
    }
    cleaned.write_text(json.dumps(cleaned_row) + "\n")

    out = load_sidecar(raw, cleaned_path=cleaned)
    assert len(out) == 1
    mv = out[0]
    assert mv.web_sources == ["https://a.gov/"]
    assert mv.url_classifications == {
        "https://a.gov/": "ok",
        "https://b.example/": "dead-4xx",
    }


def test_load_sidecar_falls_back_to_raw_when_cleaned_missing(tmp_path: Path):
    raw = tmp_path / "sidecar.jsonl"
    cleaned = tmp_path / "sidecar.cleaned.jsonl"
    raw.write_text(
        json.dumps(
            {
                "adapter_name": "openai",
                "model_id": "gpt",
                "claim_id": "c1",
                "label": "True",
                "confidence": "High",
                "explanation": "x",
                "web_sources": ["https://x.gov/"],
            }
        )
        + "\n"
    )
    out = load_sidecar(raw, cleaned_path=cleaned)
    assert len(out) == 1
    assert out[0].url_classifications == {}


def test_worse_classification_picks_higher_rank():
    assert _worse_classification(None, "ok") == "ok"
    assert _worse_classification("ok", "bot-blocked") == "bot-blocked"
    assert _worse_classification("bot-blocked", "ok") == "bot-blocked"
    assert _worse_classification("bot-blocked", "dead-4xx") == "dead-4xx"
    assert _worse_classification("dead-4xx", "bot-blocked") == "dead-4xx"


def test_evidence_list_with_combined_classifications_renders_three_tiers():
    """End-to-end check: a combined map across verdicts, including a
    URL that one verdict says ok and another says dead, gets rendered
    as broken (stripped) per ``_worse_classification`` semantics."""
    good = "https://www.bls.gov/cps/"
    blocked = "https://www.cbp.gov/y"
    contested = "https://contested.gov/z"
    classifications = {
        good: "ok",
        blocked: "bot-blocked",
        contested: _worse_classification("ok", "dead-4xx"),
    }
    html = _evidence_list_html(
        [good, blocked, contested], classifications=classifications
    )

    assert good in html
    assert blocked in html
    assert contested not in html, (
        "contested URL must be stripped — one verdict found it broken"
    )
    assert "source-verified" in html
    assert "source-unverified" in html
    assert ">unverified<" in html
