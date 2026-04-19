"""
Base class and shared utilities for LLM adapters.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from truthbot.models import Claim, Evidence, ModelVerdict

SYNTHESIS_SYSTEM = """You are an expert fact-checker. Given a claim and a set of evidence snippets, \
use your web search tool to research the claim further and determine the verdict according to this taxonomy:

  - True: Accurate and supported by primary sources
  - Mostly True: Accurate but missing nuance
  - Misleading: Technically accurate framing that implies something false
  - Exaggerated: Directionally correct but overstated
  - False: Contradicted by credible evidence
  - Unverifiable: Insufficient evidence

Respond with a JSON object (no markdown, just raw JSON):
{
  "label": "<verdict>",
  "confidence": "High|Medium|Low",
  "explanation": "<one paragraph explanation>",
  "web_sources": ["<url1>", "<url2>"]
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
