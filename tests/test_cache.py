"""TB-06t: Claim cache — hit/miss, similarity matching, invalidation, round-trip."""

from __future__ import annotations

import pytest

from truthbot.cache.claims import CacheEntry, ClaimCache


@pytest.fixture
def cache(tmp_dir):
    return ClaimCache(cache_dir=tmp_dir, similarity_threshold=85, ttl_days=30)


class TestClaimCache:
    def test_miss_returns_none(self, cache):
        result = cache.get("Unemployment is at a record low.")
        assert result is None

    def test_put_and_get_exact(self, cache):
        cache.put(
            claim_text="Unemployment is at a 50-year low.",
            verdict_label="True",
            confidence="High",
            explanation="Supported by BLS data.",
        )
        result = cache.get("Unemployment is at a 50-year low.")
        assert result is not None
        assert result.verdict_label == "True"

    def test_size_increments(self, cache):
        assert cache.size() == 0
        cache.put("Claim one.", "True", "High", "Explanation.")
        assert cache.size() == 1

    def test_invalidate(self, cache):
        cache.put("Some claim.", "False", "Medium", "Because reasons.")
        assert cache.size() == 1
        removed = cache.invalidate("Some claim.")
        assert removed
        assert cache.size() == 0
        assert cache.get("Some claim.") is None

    def test_invalidate_missing_returns_false(self, cache):
        assert not cache.invalidate("This was never cached.")

    def test_clear(self, cache):
        cache.put("Claim A.", "True", "High", "Exp.")
        cache.put("Claim B.", "False", "Low", "Exp.")
        cache.clear()
        assert cache.size() == 0

    def test_evidence_urls_stored(self, cache):
        urls = ["https://bls.gov/1", "https://reuters.com/2"]
        cache.put("A claim.", "True", "High", "Exp.", evidence_urls=urls)
        result = cache.get("A claim.")
        assert result.evidence_urls == urls

    def test_round_trip(self, cache):
        cache.put("Test claim.", "Mostly True", "Medium", "Nuance matters.")
        result = cache.round_trip("Test claim.")
        assert result is not None
        assert result.verdict_label == "Mostly True"

    def test_cache_entry_serialization(self):
        from datetime import datetime, timezone
        entry = CacheEntry(
            claim_text="A claim.",
            verdict_label="True",
            confidence="High",
            explanation="Exp.",
            evidence_urls=["https://example.com"],
            cached_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            expires_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
        )
        d = entry.to_dict()
        restored = CacheEntry.from_dict(d)
        assert restored.verdict_label == "True"
        assert restored.evidence_urls == ["https://example.com"]

    def test_expired_entry_not_returned(self, tmp_dir):
        from datetime import datetime, timedelta, timezone
        cache = ClaimCache(cache_dir=tmp_dir, ttl_days=None)
        # Manually inject expired entry
        from truthbot.cache.claims import CacheEntry
        key = cache._hash("Old claim.")
        expired = CacheEntry(
            claim_text="Old claim.",
            verdict_label="True",
            confidence="High",
            explanation="Old.",
            evidence_urls=[],
            cached_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
            expires_at=datetime(2020, 2, 1, tzinfo=timezone.utc),  # expired
        )
        cache._memory[key] = expired
        result = cache.get("Old claim.")
        assert result is None
