"""
Abstract base class for source connectors.

All source connectors must implement this interface.
The VerificationEngine works against SourceConnector — never against
concrete implementations directly — so connectors can be swapped or
mocked freely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

from truthbot.models import Claim, Evidence, SourceTier

#: A closed date window ``[start, end]`` an evidence fetch should be scoped to
#: (Layer C: the claim's utterance/reference era). ``None`` means "no scoping".
TimeWindow = Optional[tuple[date, date]]


class SourceConnector(ABC):
    """
    Protocol for retrieving evidence about a claim.

    Implementors fetch evidence from a specific data source and return
    it as a list of Evidence objects. The connector should NOT score
    or render verdicts — that's the rubric's job.

    Parameters
    ----------
    max_results:
        Maximum number of Evidence items to return per search. Subclasses
        should respect this limit.
    timeout:
        HTTP request timeout in seconds.
    """

    #: Override in subclasses to identify this connector in logs / UI
    source_name: str = "Unknown"
    #: The trust tier assigned to evidence from this connector
    tier: SourceTier = SourceTier.OTHER

    def __init__(self, max_results: int = 5, timeout: float = 10.0) -> None:
        self.max_results = max_results
        self.timeout = timeout

    @abstractmethod
    def search(self, claim: Claim) -> list[Evidence]:
        """
        Search for evidence relevant to the given claim.

        Parameters
        ----------
        claim:
            The factual claim to investigate.

        Returns
        -------
        list[Evidence]
            Zero or more evidence items. Never raises — if the source
            is unavailable, return an empty list and log the error.
        """
        ...

    def search_windowed(self, claim: Claim, window: TimeWindow = None) -> list[Evidence]:
        """
        Search for evidence, optionally TIME-SCOPED to ``window`` (Layer C).

        Backward-compatible seam: the base implementation ignores ``window`` and
        delegates to :meth:`search`, so existing connectors keep working unchanged.
        Connectors that can scope by date (e.g. Brave) override this to narrow the
        query to the claim's utterance/reference era; the charter requires evidence
        to cover both the utterance date and the claim's reference period, and a
        window keeps stale or anachronistic hits out of the pack.

        Callers that care about temporal grounding should prefer this over
        :meth:`search`; ``window=None`` is exactly the legacy behaviour.
        """
        return self.search(claim)

    def is_available(self) -> bool:
        """
        Return True if this connector is configured and reachable.

        Default implementation always returns True; override in connectors
        that require API keys or network checks.
        """
        return True

    def _make_evidence(
        self,
        claim: Claim,
        source_url: str,
        snippet: str,
        source_name: Optional[str] = None,
        supports_claim: Optional[bool] = None,
        relevance_score: float = 0.5,
    ) -> Evidence:
        """
        Convenience factory for building Evidence objects.

        Subclasses can use this to reduce boilerplate.
        """
        return Evidence(
            claim_id=claim.id,
            source_name=source_name or self.source_name,
            source_url=source_url,
            source_tier=self.tier,
            snippet=snippet,
            supports_claim=supports_claim,
            relevance_score=relevance_score,
        )
