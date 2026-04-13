"""
LLM-powered claim extraction.

Decomposes a transcript into atomic, checkable factual claims.
Each claim should be:
  - A single specific assertion
  - Independently verifiable (not an opinion)
  - Self-contained (makes sense without the surrounding context)

This module is a stub. The LLM call is defined but returns placeholder data
until an Anthropic API key is configured.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

from truthbot.models import Claim, Transcript

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# System prompt sent to Claude for claim extraction
_SYSTEM_PROMPT = """You are a professional fact-checker. Your job is to extract verifiable factual claims from political speech transcripts.

For each claim, output a JSON object with these fields:
  - "text": The claim restated as a clear, standalone declarative sentence
  - "context": The surrounding quote from the original transcript (max 200 chars)
  - "category": Subject category (economy, immigration, healthcare, crime, foreign_policy, environment, education, other)
  - "is_checkable": true if this is a factual assertion, false if it's an opinion or value judgment

Rules:
1. Extract only atomic claims — one specific assertion per item
2. Exclude opinions, predictions, and value judgments
3. Normalize claims to third-person ("The speaker claimed X" → just "X")
4. Include statistical claims, policy claims, historical claims, comparative claims
5. Exclude rhetorical questions and vague platitudes

Return a JSON array of claim objects. Nothing else."""

_USER_PROMPT_TEMPLATE = """Extract all verifiable factual claims from the following transcript.
Speaker: {speaker}
Date: {date}

Transcript:
{text}

Return a JSON array of claim objects."""


class ClaimExtractor:
    """
    Extracts atomic factual claims from a Transcript using an LLM.

    Parameters
    ----------
    api_key:
        Anthropic API key. Defaults to settings.anthropic_api_key.
    model:
        Claude model to use. Defaults to settings.llm_model.
    max_claims:
        Maximum number of claims to extract. Defaults to settings.max_claims.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_claims: Optional[int] = None,
    ) -> None:
        from truthbot.config import settings

        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model or settings.llm_model
        self._max_claims = max_claims or settings.max_claims

    def extract(self, transcript: Transcript) -> list[Claim]:
        """
        Extract verifiable claims from a transcript.

        Calls the Anthropic API with a structured prompt. Returns a list of
        Claim objects, each representing one atomic factual assertion.

        Parameters
        ----------
        transcript:
            The normalized Transcript to extract claims from.

        Returns
        -------
        list[Claim]
            Extracted claims, at most self._max_claims items.
            Returns an empty list if the API call fails.
        """
        if not self._api_key:
            logger.warning("No ANTHROPIC_API_KEY set — returning stub claims.")
            return self._stub_claims(transcript)

        try:
            return self._call_llm(transcript)
        except Exception as exc:
            logger.error("Claim extraction failed: %s", exc)
            return []

    def _call_llm(self, transcript: Transcript) -> list[Claim]:
        """Make the actual Anthropic API call and parse the response."""
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)

        date_str = transcript.date.strftime("%Y-%m-%d") if transcript.date else "Unknown"
        user_msg = _USER_PROMPT_TEMPLATE.format(
            speaker=transcript.speaker,
            date=date_str,
            text=transcript.text[:12000],  # token budget guard
        )

        message = client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = message.content[0].text
        data = json.loads(raw)

        claims = []
        for item in data[: self._max_claims]:
            claim = Claim(
                transcript_id=transcript.id,
                text=item["text"],
                speaker=transcript.speaker,
                context=item.get("context"),
                category=item.get("category"),
                is_checkable=item.get("is_checkable", True),
            )
            claims.append(claim)

        logger.info("Extracted %d claims from transcript %s", len(claims), transcript.id)
        return claims

    def _stub_claims(self, transcript: Transcript) -> list[Claim]:
        """
        Return placeholder claims when no API key is available.

        Used in tests and dry-run mode.
        """
        sentences = [s.strip() for s in transcript.text.split(".") if len(s.strip()) > 20]
        return [
            Claim(
                transcript_id=transcript.id,
                text=sentence + ".",
                speaker=transcript.speaker,
                context=sentence[:100],
                category="other",
                is_checkable=True,
            )
            for sentence in sentences[:3]
        ]


# Fix missing import
import os  # noqa: E402
