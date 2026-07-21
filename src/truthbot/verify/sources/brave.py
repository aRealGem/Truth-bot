"""
Brave Search connector.

Uses the Brave Search API to gather web evidence for a claim.
Brave's index is large and returns clean, ad-free results suitable for
automated fact-checking.

API docs: https://api.search.brave.com/
"""

from __future__ import annotations

import html
import logging
import re
from datetime import date, datetime
from typing import Optional

from truthbot.domains import is_substantive_url, url_matches_any
from truthbot.models import Claim, Evidence, SourceTier
from truthbot.verify.sources.base import SourceConnector, TimeWindow

logger = logging.getLogger(__name__)

_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


def _freshness_for(window: TimeWindow) -> str:
    """Brave ``freshness`` param for a Layer C window.

    A window ``(start, end)`` becomes Brave's date-range form
    ``YYYY-MM-DDtoYYYY-MM-DD`` so results are scoped to the claim's era; absent a
    window we keep the legacy ``py`` (past year) default."""
    if not window:
        return "py"
    start, end = window
    return f"{start.isoformat()}to{end.isoformat()}"


_TAG_RX = re.compile(r"<[^>]+>")


def _clean_snippet(text: str) -> str:
    """Strip HTML tags and unescape entities from a Brave description — the panel
    reads the snippet verbatim, so ``<strong>``/``&quot;`` noise shouldn't leak in."""
    return html.unescape(_TAG_RX.sub("", text or "")).strip()


def _result_date(result: dict) -> str:
    """Best-effort publication date (``YYYY-MM-DD``) from a Brave result, or ''.

    Brave returns ``page_age`` (ISO timestamp) and/or a human ``age``; we keep only
    the date part of ``page_age`` when present — it is the reliable machine field."""
    page_age = result.get("page_age")
    if isinstance(page_age, str) and len(page_age) >= 10 and page_age[4] == "-":
        return page_age[:10]
    return ""


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

        return self.search_windowed(claim, None)

    def search_windowed(self, claim: Claim, window: TimeWindow = None) -> list[Evidence]:
        """Time-scoped Brave search: ``window`` narrows ``freshness`` to the claim's
        era (Layer C). ``window=None`` reproduces the legacy past-year search."""
        if not self.is_available():
            logger.debug("BraveSearchConnector: no API key configured, skipping.")
            return []

        query = self._build_query(claim)
        try:
            return self._fetch(claim, query, _freshness_for(window))
        except Exception as exc:
            logger.error("Brave search failed for claim %s: %s", claim.id, exc)
            return []

    def search_query(self, claim: Claim, query: str, window: TimeWindow = None) -> list[Evidence]:
        """Fetch with an EXPLICIT query (relevance middle step: cheap-model
        query generation), still era-windowed. Returns [] on error/no key."""
        if not self.is_available():
            logger.debug("BraveSearchConnector: no API key configured, skipping.")
            return []
        try:
            return self._fetch(claim, query[:200], _freshness_for(window))
        except Exception as exc:
            logger.error("Brave query search failed for claim %s: %s", claim.id, exc)
            return []

    def _build_query(self, claim: Claim) -> str:
        """Build a search query optimized for fact-checking."""
        # Prepend fact-check keywords to surface relevant journalism
        prefix = "fact check "
        if claim.category in ("economy", "jobs", "unemployment"):
            prefix = "data statistics "
        return f"{prefix}{claim.text}"[:200]

    def _fetch(self, claim: Claim, query: str, freshness: str = "py") -> list[Evidence]:
        """Make the HTTP request and parse results. ``freshness`` is Brave's recency
        filter (``py`` or a ``YYYY-MM-DDtoYYYY-MM-DD`` date range from the window)."""
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
            "freshness": freshness,
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
            if not is_substantive_url(r.get("url", "")):
                continue    # homepages / listing indexes are not evidence
            tier = self._classify_tier(r.get("url", ""))
            snippet = _clean_snippet(r.get("description", ""))[:500]
            published = _result_date(r)
            if published:
                # Fold the publication date into the snippet so it survives into the
                # payload the panel sees — recency is signal for as-of-utterance judging.
                snippet = f"[{published}] {snippet}"
            ev = Evidence(
                claim_id=claim.id,
                source_name=r.get("profile", {}).get("name", r.get("meta_url", {}).get("hostname", "Unknown")),
                source_url=r.get("url", ""),
                source_tier=tier,
                snippet=snippet,
                retrieved_at=datetime.utcnow(),
                published_at=datetime.fromisoformat(published) if published else None,
            )
            evidence.append(ev)

        return evidence

    def _classify_tier(self, url: str) -> SourceTier:
        return classify_tier(url)


def classify_tier(url: str) -> SourceTier:
    """Assign a trust tier based on the URL's registered domain.

    Host-suffix matching (truthbot.domains), not substring — a substring
    rule made ``www.govtech.com`` rank Government because it contains
    ``.gov``, letting a trade magazine win pack slots. Module-level so the
    evidence-v2 retrievers (P67.8) classify identically to the Brave
    connector."""
    gov_domains = (".gov", ".mil", "federalreserve.gov")
    wire_domains = ("apnews.com", "reuters.com")
    established_domains = (
        "nytimes.com", "washingtonpost.com", "bbc.com", "bbc.co.uk",
        "nbcnews.com", "cbsnews.com", "abcnews.go.com", "npr.org",
    )
    factcheck_domains = ("politifact.com", "factcheck.org", "snopes.com", "fullfact.org")

    if url_matches_any(url, gov_domains):
        return SourceTier.GOVERNMENT
    if url_matches_any(url, wire_domains):
        return SourceTier.WIRE
    if url_matches_any(url, established_domains):
        return SourceTier.ESTABLISHED
    if url_matches_any(url, factcheck_domains):
        return SourceTier.FACTCHECK
    return SourceTier.OTHER
