"""
Fact-check database connector.

Cross-references existing fact-check rulings from:
  - PolitiFact (politifact.com)
  - FactCheck.org
  - Snopes (limited to political claims)

These sources are assigned the FACTCHECK tier — high credibility but
recognized as having editorial positions. Used as supporting evidence,
not as definitive verdicts.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional
from urllib.parse import urlsplit

from truthbot.domains import host_matches, is_substantive_url, url_host
from truthbot.models import Claim, Evidence, SourceTier
from truthbot.verify.sources.base import SourceConnector, TimeWindow
from truthbot.verify.sources.brave import _clean_snippet, _freshness_for, _result_date

logger = logging.getLogger(__name__)


class FactCheckConnector(SourceConnector):
    """
    Cross-reference existing fact-checks via search APIs.

    Queries Brave (or fallback search) scoped to known fact-check domains
    to surface existing rulings on similar claims.

    Parameters
    ----------
    brave_api_key:
        Brave Search API key. Defaults to BRAVE_API_KEY env var.
    """

    source_name = "Fact-Check Databases"
    tier = SourceTier.FACTCHECK

    _FACTCHECK_DOMAINS = [
        "politifact.com",
        "factcheck.org",
        "snopes.com",
        "fullfact.org",
        "apnews.com/hub/ap-fact-check",
    ]

    def __init__(
        self,
        brave_api_key: Optional[str] = None,
        max_results: int = 3,
        timeout: float = 10.0,
    ) -> None:
        super().__init__(max_results=max_results, timeout=timeout)
        import os
        if brave_api_key is None:
            self._api_key = os.environ.get("BRAVE_API_KEY", "")
        else:
            self._api_key = brave_api_key

    def is_available(self) -> bool:
        return bool(self._api_key)

    def search(self, claim: Claim) -> list[Evidence]:
        """
        Search fact-check sites for prior rulings on this claim.

        Constructs a site-scoped query and retrieves matching articles.
        Returns empty list if the API key is missing or the search fails.

        Parameters
        ----------
        claim:
            The claim to look up in fact-check databases.

        Returns
        -------
        list[Evidence]
            Evidence from known fact-check organizations.
        """
        return self.search_windowed(claim, None)

    def search_windowed(self, claim: Claim, window: TimeWindow = None) -> list[Evidence]:
        """Time-scoped fact-check lookup (Layer C): ``window`` narrows Brave's
        ``freshness`` to the claim's era, so a ruling published years after the
        utterance can't land in the pack (how a 2026-02 PolitiFact piece ended up
        in a Biden-2022 pack). ``window=None`` is the legacy unscoped search."""
        if not self.is_available():
            logger.debug("FactCheckConnector: no API key, skipping.")
            return []

        query = self._build_query(claim)

        try:
            return self._fetch(claim, query, window)
        except Exception as exc:
            logger.error("FactCheckConnector search failed for claim %s: %s", claim.id, exc)
            return []

    def _build_query(self, claim: Claim) -> str:
        """Build a domain-scoped search query."""
        site_filter = " OR ".join(f"site:{d}" for d in self._FACTCHECK_DOMAINS[:3])
        return f"({site_filter}) {claim.text}"[:250]

    def _fetch(self, claim: Claim, query: str, window: TimeWindow = None) -> list[Evidence]:
        """Call Brave Search scoped to fact-check domains. A ``window`` becomes
        Brave's date-range ``freshness`` filter; None sends no recency filter
        (legacy behavior)."""
        import httpx

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._api_key,
        }
        params = {
            "q": query,
            "count": self.max_results,
            "search_lang": "en",
        }
        if window is not None:
            params["freshness"] = _freshness_for(window)

        resp = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        evidence = []
        for r in data.get("web", {}).get("results", []):
            url = r.get("url", "")
            if not self._is_factcheck_url(url):
                continue  # filter to only actual fact-check domains
            if not is_substantive_url(url):
                # site: queries return the domain homepage / listing indexes
                # when no article matches — those are not evidence.
                continue
            snippet = _clean_snippet(r.get("description", ""))[:400]
            published = _result_date(r)
            if published:
                snippet = f"[{published}] {snippet}"
            ev = self._make_evidence(
                claim,
                source_url=url,
                snippet=snippet,
                source_name=self._domain_name(url),
                relevance_score=0.8,
            )
            if published:
                ev.published_at = datetime.fromisoformat(published)
            evidence.append(ev)

        return evidence[: self.max_results]

    def _is_factcheck_url(self, url: str) -> bool:
        """Registered-domain filter (not substring — a lookalike host or a URL
        merely *mentioning* politifact.com in its path must not pass). Entries
        with a path component (the AP fact-check hub) also require the path
        prefix to match."""
        host = url_host(url)
        for entry in self._FACTCHECK_DOMAINS:
            if "/" in entry:
                domain, _, path_prefix = entry.partition("/")
                if host_matches(host, domain) and urlsplit(url).path.lstrip("/").startswith(path_prefix):
                    return True
            elif host_matches(host, entry):
                return True
        return False

    def _domain_name(self, url: str) -> str:
        """Map URL to a human-readable source name."""
        mapping = {
            "politifact.com": "PolitiFact",
            "factcheck.org": "FactCheck.org",
            "snopes.com": "Snopes",
            "fullfact.org": "Full Fact",
            "apnews.com": "AP Fact Check",
        }
        host = url_host(url)
        for domain, name in mapping.items():
            if host_matches(host, domain):
                return name
        return "Fact-Check Outlet"
