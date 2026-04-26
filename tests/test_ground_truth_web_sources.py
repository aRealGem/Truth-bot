"""Unit tests for ``ground_truth_web_sources`` and ``_normalize_url_for_compare``.

Layer 1a of the anti-hallucination defense-in-depth plan.
"""
from __future__ import annotations

import pytest

from truthbot.verify.adapters.base import (
    _normalize_url_for_compare,
    ground_truth_web_sources,
)


class TestNormalizeUrlForCompare:
    def test_blank_input_returns_empty(self):
        assert _normalize_url_for_compare("") == ""
        assert _normalize_url_for_compare("   ") == ""

    def test_non_string_returns_empty(self):
        assert _normalize_url_for_compare(None) == ""  # type: ignore[arg-type]
        assert _normalize_url_for_compare(12345) == ""  # type: ignore[arg-type]

    def test_non_http_scheme_rejected(self):
        assert _normalize_url_for_compare("ftp://example.com") == ""
        assert _normalize_url_for_compare("javascript:alert(1)") == ""
        assert _normalize_url_for_compare("file:///etc/hosts") == ""

    def test_strips_fragment(self):
        a = _normalize_url_for_compare("https://www.bls.gov/cpi#table")
        b = _normalize_url_for_compare("https://www.bls.gov/cpi")
        assert a == b

    def test_lowercases_scheme_and_host(self):
        a = _normalize_url_for_compare("HTTPS://WWW.BLS.GOV/news")
        b = _normalize_url_for_compare("https://www.bls.gov/news")
        assert a == b

    def test_strips_default_ports(self):
        assert _normalize_url_for_compare("https://example.com:443/x") == \
               _normalize_url_for_compare("https://example.com/x")
        assert _normalize_url_for_compare("http://example.com:80/x") == \
               _normalize_url_for_compare("http://example.com/x")

    def test_collapses_www_prefix(self):
        a = _normalize_url_for_compare("https://www.cnn.com/article")
        b = _normalize_url_for_compare("https://cnn.com/article")
        assert a == b

    def test_root_trailing_slash_equivalent(self):
        a = _normalize_url_for_compare("https://example.com/")
        b = _normalize_url_for_compare("https://example.com")
        assert a == b

    def test_trailing_slash_path_equivalent(self):
        a = _normalize_url_for_compare("https://example.com/foo/")
        b = _normalize_url_for_compare("https://example.com/foo")
        assert a == b

    def test_preserves_query_string(self):
        a = _normalize_url_for_compare("https://example.com/search?q=trump")
        b = _normalize_url_for_compare("https://example.com/search?q=biden")
        assert a != b


class TestGroundTruthWebSources:
    def test_empty_model_reported(self):
        kept, stripped = ground_truth_web_sources([], ["https://x.com"])
        assert kept == []
        assert stripped == 0

    def test_none_model_reported(self):
        kept, stripped = ground_truth_web_sources(None, ["https://x.com"])
        assert kept == []
        assert stripped == 0

    def test_empty_tool_retrieved_strips_everything(self):
        kept, stripped = ground_truth_web_sources(
            ["https://www.bls.gov/cpi"], []
        )
        assert kept == []
        assert stripped == 1

    def test_none_tool_retrieved_strips_everything(self):
        kept, stripped = ground_truth_web_sources(
            ["https://a.com", "https://b.com"], None
        )
        assert kept == []
        assert stripped == 2

    def test_exact_match_preserved(self):
        kept, stripped = ground_truth_web_sources(
            ["https://www.bls.gov/cpi.htm"],
            ["https://www.bls.gov/cpi.htm"],
        )
        assert kept == ["https://www.bls.gov/cpi.htm"]
        assert stripped == 0

    def test_returns_original_form_when_normalization_matches(self):
        # Model emits with trailing slash; tool emits without. Match still
        # succeeds. The returned URL should be the model's original form.
        kept, _ = ground_truth_web_sources(
            ["https://www.cnn.com/article/"],
            ["https://cnn.com/article"],
        )
        assert kept == ["https://www.cnn.com/article/"]

    def test_preserves_model_order_with_partial_match(self):
        kept, stripped = ground_truth_web_sources(
            [
                "https://www.bls.gov/cpi",
                "https://fake.example.com/hallucinated",
                "https://www.cbp.gov/border",
            ],
            [
                "https://cbp.gov/border",
                "https://bls.gov/cpi",
            ],
        )
        assert kept == [
            "https://www.bls.gov/cpi",
            "https://www.cbp.gov/border",
        ]
        assert stripped == 1

    def test_deduplicates_keeping_first(self):
        kept, stripped = ground_truth_web_sources(
            [
                "https://example.com/x",
                "https://www.example.com/x",
                "https://EXAMPLE.com/x/",
            ],
            ["https://example.com/x"],
        )
        # All three normalize to the same key; only the first form is kept.
        assert kept == ["https://example.com/x"]
        assert stripped == 0

    def test_strips_malformed_url(self):
        kept, stripped = ground_truth_web_sources(
            ["not-a-url-at-all", "https://real.com/page"],
            ["https://real.com/page"],
        )
        assert kept == ["https://real.com/page"]
        assert stripped == 1

    def test_strips_non_http_scheme(self):
        kept, stripped = ground_truth_web_sources(
            ["ftp://files.example.com", "https://x.com/y"],
            ["https://x.com/y"],
        )
        assert kept == ["https://x.com/y"]
        assert stripped == 1

    def test_mixed_case_query_strict(self):
        # Different query strings → no match (queries can be content-mat'l).
        kept, stripped = ground_truth_web_sources(
            ["https://x.com/s?q=trump"],
            ["https://x.com/s?q=biden"],
        )
        assert kept == []
        assert stripped == 1

    def test_distinct_stripped_count_dedupes(self):
        # Two model-reported URLs that normalize to the same key but are
        # both not in truth → counted as 1 stripped (distinct keys).
        kept, stripped = ground_truth_web_sources(
            [
                "https://fake.com/page",
                "https://www.fake.com/page/",
            ],
            ["https://real.com/x"],
        )
        assert kept == []
        assert stripped == 1
