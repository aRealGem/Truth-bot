"""
Brave Search connector.

Uses the Brave Search API to gather web evidence for a claim.
Brave's index is large and returns clean, ad-free results suitable for
automated fact-checking.

API docs: https://api.search.brave.com/
"""

from __future__ import annotations

import logging
from typing import Optional

from truthbot.models import Claim, Evidence, SourceTier
from truthbot.verify.sources.base import SourceConnector

logger = logging.getLogger(__name__)

_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


class BraveSearchConnector(SourceConnector):
    """
    Retrieve web evidence via the Brave Search API.

    Parameters
    ----------
    api_key:
        Brave Search API key. Defaults to BRAVE_API_KEY env var.
    max_results:
        Max search results per claim (default 5).
    timeout:
        HTTP timeout in seconds (default 10).
    """

    source_name = "Brave Search"
    tier = SourceTier.OTHER  # tier gets upgraded per result based on domain

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 5,
        timeout: float = 10.0,
    ) -> None:
        super().__init__(max_results=max_results, timeout=timeout)
        import os
        if api_key is None:
            self._api_key = os.environ.get("BRAVE_API_KEY", "")
        else:
            self._api_key = api_key

    def is_available(self) -> bool:
        """Returns True if an API key is configured."""
        return bool(self._api_key)

    def search(self, claim: Claim) -> list[Evidence]:
        """
        Search Brave for evidence related to the claim.

        Constructs a fact-check query, calls the Brave API, and converts
        results to Evidence objects. Returns empty list on error.

        Parameters
        ----------
        claim:
            The claim to search for.

        Returns
        -------
        list[Evidence]
            Up to self.max_results evidence items.
        """
        if not self.is_available():
            logger.debug("BraveSearchConnector: no API key configured, skipping.")
            return []

        query = self._build_query(claim)

        try:
            return self._fetch(claim, query)
        except Exception as exc:
            logger.error("Brave search failed for claim %s: %s", claim.id, exc)
            return []

    def _build_query(self, claim: Claim) -> str:
        """Build a search query optimized for fact-checking."""
        # Prepend fact-check keywords to surface relevant journalism
        prefix = "fact check "
        if claim.category in ("economy", "jobs", "unemployment"):
            prefix = "data statistics "
        return f"{prefix}{claim.text}"[:200]

    def _fetch(self, claim: Claim, query: str) -> list[Evidence]:
        """Make the HTTP request and parse results."""
        import httpx

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._api_key,
        }
        params = {
            "q": query,
            "count": self.max_results,
            "search_lang": "en",
            "country": "us",
            "freshness": "py",  # past year
        }

        resp = httpx.get(
            _BRAVE_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("web", {}).get("results", [])
        evidence = []
        for r in results[: self.max_results]:
            tier = self._classify_tier(r.get("url", ""))
            ev = Evidence(
                claim_id=claim.id,
                source_name=r.get("profile", {}).get("name", r.get("meta_url", {}).get("hostname", "Unknown")),
                source_url=r.get("url", ""),
                source_tier=tier,
                snippet=r.get("description", "")[:500],
                retrieved_at=__import__("datetime").datetime.utcnow(),
            )
            evidence.append(ev)

        return evidence

    def _classify_tier(self, url: str) -> SourceTier:
        """Assign a trust tier based on the domain."""
        lower = url.lower()
        gov_domains = (".gov", ".mil", "bls.gov", "census.gov", "cbo.gov", "federalreserve.gov")
        wire_domains = ("apnews.com", "reuters.com")
        established_domains = (
            "nytimes.com", "washingtonpost.com", "bbc.com", "bbc.co.uk",
            "nbcnews.com", "cbsnews.com", "abcnews.go.com", "npr.org",
        )
        factcheck_domains = ("politifact.com", "factcheck.org", "snopes.com", "fullfact.org")

        if any(d in lower for d in gov_domains):
            return SourceTier.GOVERNMENT
        if any(d in lower for d in wire_domains):
            return SourceTier.WIRE
        if any(d in lower for d in established_domains):
            return SourceTier.ESTABLISHED
        if any(d in lower for d in factcheck_domains):
            return SourceTier.FACTCHECK
        return SourceTier.OTHER
