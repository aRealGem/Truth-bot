"""
Government data source connector.

Queries authoritative US government data APIs:
  - BLS (Bureau of Labor Statistics) — employment, inflation, wages
  - FRED (Federal Reserve Economic Data) — broad macroeconomic indicators
  - Census Bureau API — population, demographics
  - CBO (Congressional Budget Office) — budget, deficit, spending projections

These are the highest-trust tier sources in the truth-bot hierarchy.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from truthbot.models import Claim, Evidence, SourceTier
from truthbot.verify.sources.base import SourceConnector

logger = logging.getLogger(__name__)

# Public BLS API — basic queries work without a key; key needed for higher rate limits
_BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
# FRED REST API
_FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
# Census Bureau Data API
_CENSUS_API_URL = "https://api.census.gov/data"


class GovernmentDataConnector(SourceConnector):
    """
    Retrieve evidence from authoritative US government data APIs.

    Currently implements BLS and FRED lookups for economic claims.
    Census and CBO support are planned for future releases.

    Parameters
    ----------
    fred_api_key:
        FRED API key. Defaults to FRED_API_KEY env var. Optional — some
        FRED endpoints work without a key at lower rate limits.
    """

    source_name = "US Government Data"
    tier = SourceTier.GOVERNMENT

    # Economic keywords that suggest a government data lookup
    _ECONOMIC_KEYWORDS = {
        "unemployment", "jobs", "employed", "inflation", "cpi",
        "gdp", "deficit", "debt", "wages", "salary", "income",
        "poverty", "population", "census",
    }

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        max_results: int = 3,
        timeout: float = 15.0,
    ) -> None:
        super().__init__(max_results=max_results, timeout=timeout)
        self._fred_api_key = fred_api_key or os.environ.get("FRED_API_KEY", "")

    def is_available(self) -> bool:
        """Government APIs are generally available; return True always."""
        return True

    def search(self, claim: Claim) -> list[Evidence]:
        """
        Query relevant government APIs for data related to the claim.

        Routes to the appropriate API based on keywords in the claim text.
        Returns empty list on errors or when no relevant APIs apply.

        Parameters
        ----------
        claim:
            The claim to gather government data for.

        Returns
        -------
        list[Evidence]
            Government data evidence items.
        """
        text_lower = claim.text.lower()
        evidence: list[Evidence] = []

        if any(kw in text_lower for kw in self._ECONOMIC_KEYWORDS):
            evidence.extend(self._search_fred(claim))

        return evidence[: self.max_results]

    def _search_fred(self, claim: Claim) -> list[Evidence]:
        """
        Stub: search FRED for macroeconomic data related to the claim.

        TODO: Implement FRED series lookup based on claim keywords.
        Map claim topics → FRED series IDs (e.g. "unemployment" → "UNRATE").
        """
        # Series mapping — expand this as needed
        _SERIES_MAP = {
            "unemployment": "UNRATE",
            "inflation": "CPIAUCSL",
            "gdp": "GDP",
            "deficit": "FYFSD",
            "jobs": "PAYEMS",
        }
        text_lower = claim.text.lower()
        series_id = next(
            (sid for kw, sid in _SERIES_MAP.items() if kw in text_lower),
            None,
        )

        if not series_id:
            return []

        if not self._fred_api_key:
            logger.debug("No FRED_API_KEY; returning stub government evidence.")
            return [
                self._make_evidence(
                    claim,
                    source_url=f"https://fred.stlouisfed.org/series/{series_id}",
                    snippet=f"[Stub] FRED series {series_id} — configure FRED_API_KEY for live data.",
                    source_name="FRED (Federal Reserve)",
                    relevance_score=0.7,
                )
            ]

        try:
            import httpx

            resp = httpx.get(
                _FRED_API_URL,
                params={
                    "series_id": series_id,
                    "api_key": self._fred_api_key,
                    "file_type": "json",
                    "limit": 5,
                    "sort_order": "desc",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            observations = data.get("observations", [])

            if not observations:
                return []

            latest = observations[0]
            snippet = (
                f"FRED {series_id}: latest value = {latest['value']} "
                f"(date: {latest['date']})"
            )
            return [
                self._make_evidence(
                    claim,
                    source_url=f"https://fred.stlouisfed.org/series/{series_id}",
                    snippet=snippet,
                    source_name="FRED (Federal Reserve)",
                    relevance_score=0.9,
                )
            ]
        except Exception as exc:
            logger.error("FRED lookup failed: %s", exc)
            return []
