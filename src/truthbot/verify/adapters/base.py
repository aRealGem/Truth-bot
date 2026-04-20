"""
Base class and shared utilities for LLM adapters.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from truthbot.models import Claim, Evidence, ModelVerdict

SYNTHESIS_SYSTEM = """You are an expert fact-checker. Given a claim and a set of evidence snippets, \
use your web search tool to research the claim and return a verdict.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — MANDATORY PRIMARY-SOURCE SEARCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before citing any aggregator, news outlet, or commentary site, you MUST attempt to retrieve \
relevant Tier 1 government primary sources if the claim touches any of the following domains:

  • Jobs, unemployment, wages, inflation, labor force  → search site:bls.gov
  • GDP, trade, personal income                        → search site:bea.gov
  • Federal budget, deficit, national debt             → search site:cbo.gov OR site:treasury.gov
  • Social Security, Medicare                          → search site:ssa.gov OR site:cms.gov
  • Health data, disease statistics                    → search site:cdc.gov OR site:hhs.gov
  • Census data, demographics, population              → search site:census.gov
  • Energy, oil, gas, electricity production           → search site:eia.gov
  • Education data, graduation rates, test scores      → search site:nces.ed.gov
  • Crime statistics                                   → search site:bjs.ojp.gov OR site:fbi.gov
  • Immigration, border data                           → search site:dhs.gov OR site:cbp.gov

Rules:
  - If a Tier 1 search returns relevant results, those MUST appear as primary evidence.
  - Aggregators, news outlets, and commentary (Tier 2–5) may be cited in addition, never instead.
  - If a Tier 1 search returns no relevant results, note it explicitly in the caveats field \
and then proceed with lower-tier sources.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — VERDICT SYNTHESIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After gathering evidence, determine the verdict using this taxonomy:

  - True:          Accurate and supported by primary sources
  - Mostly True:   Accurate but missing important nuance or context
  - Misleading:    Technically accurate framing that implies something false
  - Exaggerated:   Directionally correct but substantially overstated
  - False:         Contradicted by credible evidence
  - Unverifiable:  Insufficient evidence to confirm or deny

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — VERDICT VALIDATION (before returning output)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before returning your verdict, review your evidence list:
  - If the claim falls into any domain listed in Step 1 AND your evidence contains zero Tier 1 \
government sources, you must either:
      (a) perform an additional targeted Tier 1 search using the site: operators above, or
      (b) add a caveat: "No Tier 1 primary source retrieved despite relevant domain; \
verdict rests on secondary analysis."
  - A verdict backed only by aggregators or commentary on a quantitative government data claim \
is not acceptable without this caveat.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Respond with ONLY raw JSON (no markdown fences, no preamble):
{
  "label": "<verdict>",
  "confidence": "High|Medium|Low",
  "explanation": "<one paragraph citing specific sources and data points>",
  "caveats": "<source-quality notes, or empty string if none>",
  "web_sources": ["<url1>", "<url2>", ...]
}"""


class AdapterUnavailable(Exception):
    """Raised when a required API key is missing."""
    pass


class LLMAdapter(ABC):
    """Abstract base class for LLM fact-checking adapters."""

    adapter_name: str
    model_id: str
    required_env_key: str

    def __init__(self) -> None:
        if not os.environ.get(self.required_env_key):
            raise AdapterUnavailable(f"{self.required_env_key} not set")

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the required API key is present in the environment."""
        return bool(os.environ.get(cls.required_env_key))

    @abstractmethod
    def call(self, claim: Claim, evidence: list[Evidence]) -> ModelVerdict:
        """Call the LLM with claim + evidence and return a ModelVerdict."""
        ...

    def _build_user_message(self, claim: Claim, evidence: list[Evidence]) -> str:
        """Build the user message string for the LLM prompt."""
        evidence_text = "\n\n".join(
            f"[{i+1}] {e.source_name} ({e.source_tier.value})\n{e.snippet}"
            for i, e in enumerate(evidence[:10])
        )
        if evidence_text:
            return (
                f"Claim: {claim.text}\n\n"
                f"Speaker: {claim.speaker}\n\n"
                f"Evidence:\n{evidence_text}"
            )
        return (
            f"Claim: {claim.text}\n\n"
            f"Speaker: {claim.speaker}\n\n"
            "No pre-gathered evidence available. Please use web search to research this claim."
        )
