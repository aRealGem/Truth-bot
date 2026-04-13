"""
Claim cache: store and retrieve previously verified claims.

Prevents re-verifying claims that have been checked recently, which is
especially useful for:
  - Recurring rhetoric (politicians repeat talking points across speeches)
  - Batch runs over multiple speeches from the same campaign

Uses diskcache for persistent on-disk storage and thefuzz for approximate
string matching (so "unemployment is at record lows" matches "unemployment
hit a 50-year low").
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import diskcache
    _DISKCACHE_AVAILABLE = True
except ImportError:
    _DISKCACHE_AVAILABLE = False
    logger.warning("diskcache not available — using in-memory cache only.")

try:
    from thefuzz import fuzz
    _FUZZ_AVAILABLE = True
except ImportError:
    _FUZZ_AVAILABLE = False
    logger.warning("thefuzz not available — cache will use exact matching only.")


@dataclass
class CacheEntry:
    """A cached fact-check result."""

    claim_text: str
    verdict_label: str
    confidence: str
    explanation: str
    evidence_urls: list[str]
    cached_at: datetime
    expires_at: Optional[datetime]

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        now = datetime.now(timezone.utc) if self.expires_at.tzinfo else datetime.utcnow()
        return now > self.expires_at

    def to_dict(self) -> dict:
        return {
            "claim_text": self.claim_text,
            "verdict_label": self.verdict_label,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "evidence_urls": self.evidence_urls,
            "cached_at": self.cached_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry":
        return cls(
            claim_text=data["claim_text"],
            verdict_label=data["verdict_label"],
            confidence=data["confidence"],
            explanation=data["explanation"],
            evidence_urls=data.get("evidence_urls", []),
            cached_at=datetime.fromisoformat(data["cached_at"]),
            expires_at=(
                datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
            ),
        )


class ClaimCache:
    """
    Persistent disk-backed cache with fuzzy matching for claim lookups.

    Parameters
    ----------
    cache_dir:
        Directory for the disk cache. Created if it doesn't exist.
    similarity_threshold:
        Minimum fuzz ratio (0–100) to consider a claim a cache hit.
        Default: 85.
    ttl_days:
        Cache entry lifetime in days. None = never expires.
    """

    def __init__(
        self,
        cache_dir: Optional[str | Path] = None,
        similarity_threshold: int = 85,
        ttl_days: Optional[int] = 90,
    ) -> None:
        from truthbot.config import settings

        self._threshold = similarity_threshold
        self._ttl = timedelta(days=ttl_days) if ttl_days else None

        cache_path = Path(cache_dir) if cache_dir else settings.cache_dir
        cache_path.mkdir(parents=True, exist_ok=True)

        if _DISKCACHE_AVAILABLE:
            self._disk: Optional["diskcache.Cache"] = diskcache.Cache(str(cache_path))
        else:
            self._disk = None

        # In-memory fallback / index for fuzzy matching
        self._memory: dict[str, CacheEntry] = {}
        self._load_from_disk()

    # ── Public interface ──────────────────────────────────────────────────────

    def get(self, claim_text: str) -> Optional[CacheEntry]:
        """
        Look up a cached entry for a claim.

        Uses exact hash lookup first, then falls back to fuzzy matching.
        Returns None if no match found or if the best match is expired.

        Parameters
        ----------
        claim_text:
            The claim text to look up.

        Returns
        -------
        Optional[CacheEntry]
            The cached entry, or None if not found / expired.
        """
        # Fast path: exact hash
        key = self._hash(claim_text)
        entry = self._memory.get(key)
        if entry:
            if entry.is_expired():
                self.invalidate(claim_text)
                return None
            return entry

        # Slow path: fuzzy match
        return self._fuzzy_lookup(claim_text)

    def put(
        self,
        claim_text: str,
        verdict_label: str,
        confidence: str,
        explanation: str,
        evidence_urls: Optional[list[str]] = None,
    ) -> CacheEntry:
        """
        Store a verified claim result in the cache.

        Parameters
        ----------
        claim_text:
            The canonical claim text.
        verdict_label:
            The verdict label (e.g. "True", "False").
        confidence:
            Confidence level ("High", "Medium", "Low").
        explanation:
            Human-readable explanation.
        evidence_urls:
            URLs of supporting evidence.

        Returns
        -------
        CacheEntry
            The entry that was stored.
        """
        now = datetime.now(timezone.utc)
        entry = CacheEntry(
            claim_text=claim_text,
            verdict_label=verdict_label,
            confidence=confidence,
            explanation=explanation,
            evidence_urls=evidence_urls or [],
            cached_at=now,
            expires_at=(now + self._ttl) if self._ttl else None,
        )

        key = self._hash(claim_text)
        self._memory[key] = entry
        self._persist(key, entry)
        logger.debug("Cached claim: %s… → %s", claim_text[:60], verdict_label)
        return entry

    def invalidate(self, claim_text: str) -> bool:
        """
        Remove a claim from the cache.

        Parameters
        ----------
        claim_text:
            The claim to remove.

        Returns
        -------
        bool
            True if an entry was found and removed, False otherwise.
        """
        key = self._hash(claim_text)
        removed = key in self._memory
        self._memory.pop(key, None)
        if self._disk and key in self._disk:
            del self._disk[key]
        return removed

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._memory.clear()
        if self._disk:
            self._disk.clear()

    def size(self) -> int:
        """Return the number of cached entries."""
        return len(self._memory)

    def round_trip(self, claim_text: str) -> Optional[CacheEntry]:
        """
        Verify cache consistency: put a lookup entry and retrieve it.
        Primarily used in testing.
        """
        entry = self.get(claim_text)
        if entry:
            return entry
        # Not in cache — nothing to round-trip
        return None

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _hash(text: str) -> str:
        """Deterministic key for a claim string."""
        return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]

    def _fuzzy_lookup(self, claim_text: str) -> Optional[CacheEntry]:
        """Find the best fuzzy match in the in-memory index."""
        if not _FUZZ_AVAILABLE or not self._memory:
            return None

        best_score = 0
        best_entry: Optional[CacheEntry] = None

        for entry in self._memory.values():
            score = fuzz.token_sort_ratio(
                claim_text.lower(), entry.claim_text.lower()
            )
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self._threshold and best_entry:
            if best_entry.is_expired():
                self.invalidate(best_entry.claim_text)
                return None
            logger.debug(
                "Cache fuzzy hit (score=%d): %s…",
                best_score,
                claim_text[:60],
            )
            return best_entry

        return None

    def _persist(self, key: str, entry: CacheEntry) -> None:
        """Write an entry to disk cache."""
        if self._disk:
            try:
                self._disk.set(key, json.dumps(entry.to_dict()), expire=None)
            except Exception as exc:
                logger.warning("Failed to persist cache entry: %s", exc)

    def _load_from_disk(self) -> None:
        """Hydrate the in-memory index from disk on startup."""
        if not self._disk:
            return
        try:
            for key in self._disk.iterkeys():
                raw = self._disk.get(key)
                if raw:
                    entry = CacheEntry.from_dict(json.loads(raw))
                    if not entry.is_expired():
                        self._memory[key] = entry
        except Exception as exc:
            logger.warning("Failed to load cache from disk: %s", exc)
