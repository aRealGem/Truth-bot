"""
Evidence ingestion behind a single port.

TruthBot does not implement domain-specific fetchers (BLS, FRED, Census, etc.);
those live in DataHoover or other services. This module provides:

- ``NoOpEvidenceProvider`` — default (no prefetched snippets; models use web search).
- ``ConnectorEvidenceProvider`` — legacy Brave / FactCheck / Gov parallel search.
- ``DataHooverEvidenceProvider`` — stub that returns ``[]`` until Hoover is wired
  (optional ``TRUTHBOT_DATAHOOVER_URL`` / ``TRUTHBOT_DATAHOOVER_MANIFEST`` reserved).
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from truthbot.models import Claim, Evidence
from truthbot.verify.sources.base import SourceConnector, TimeWindow

logger = logging.getLogger(__name__)


class EvidenceProvider(ABC):
    """Pluggable evidence source for ``VerificationEngine``."""

    @abstractmethod
    def get_evidence(self, claim: Claim, *, window: TimeWindow = None) -> list[Evidence]:
        """Return zero or more evidence items for the claim.

        ``window`` (Layer C) optionally scopes retrieval to the claim's era; a
        provider that cannot honour it MUST ignore it rather than fail."""
        ...


class NoOpEvidenceProvider(EvidenceProvider):
    """No prefetched evidence (models rely on their own tools / search)."""

    def get_evidence(self, claim: Claim, *, window: TimeWindow = None) -> list[Evidence]:
        logger.debug("NoOpEvidenceProvider: no snippets for claim %s", claim.id)
        return []


class ConnectorEvidenceProvider(EvidenceProvider):
    """Parallel queries across configured ``SourceConnector`` instances."""

    def __init__(
        self,
        connectors: list[SourceConnector],
        *,
        max_workers: int = 4,
    ) -> None:
        self._connectors = connectors
        self._max_workers = max_workers

    def get_evidence(self, claim: Claim, *, window: TimeWindow = None) -> list[Evidence]:
        available = [c for c in self._connectors if c.is_available()]
        if not available:
            return []

        all_evidence: list[Evidence] = []
        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(available))) as pool:
            futures = {pool.submit(c.search_windowed, claim, window): c for c in available}
            for future in as_completed(futures):
                connector = futures[future]
                try:
                    all_evidence.extend(future.result())
                except Exception as exc:
                    logger.error("Connector %s failed: %s", connector.source_name, exc)
        return all_evidence


class DataHooverEvidenceProvider(EvidenceProvider):
    """
    Stub for evidence served by the external **DataHoover** project.

    Wire ``TRUTHBOT_DATAHOOVER_URL`` (HTTP) or ``TRUTHBOT_DATAHOOVER_MANIFEST`` (path to
    NDJSON manifest) in a future change; until then this provider returns an empty list
    so the verification fan-out is unchanged.
    """

    def __init__(self) -> None:
        self._url = os.environ.get("TRUTHBOT_DATAHOOVER_URL", "").strip()
        self._manifest = os.environ.get("TRUTHBOT_DATAHOOVER_MANIFEST", "").strip()

    def get_evidence(self, claim: Claim, *, window: TimeWindow = None) -> list[Evidence]:
        logger.debug(
            "DataHooverEvidenceProvider stub: claim %s (url=%r manifest=%r) — returning []. "
            "Implement HTTP/file ingestion in TruthBot or consume Hoover output here.",
            claim.id,
            bool(self._url),
            bool(self._manifest),
        )
        return []


def build_evidence_provider(
    *,
    source: str,
    connectors: list[SourceConnector],
    max_workers: int = 4,
) -> EvidenceProvider:
    """
    Factory: ``source`` is ``none`` | ``connectors`` | ``datahoover``.
    """
    s = source.strip().lower()
    if s == "connectors":
        return ConnectorEvidenceProvider(connectors, max_workers=max_workers)
    if s == "datahoover":
        return DataHooverEvidenceProvider()
    return NoOpEvidenceProvider()
