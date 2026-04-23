"""
LLM-powered claim extraction using Claude Sonnet 4.6.

Decomposes a transcript into atomic, individually verifiable claims.
Each claim must stand alone — self-contained with context inlined,
ready to be fact-checked in isolation.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from truthbot.models import Claim, Transcript

logger = logging.getLogger(__name__)

# ── Extractor prompt ──────────────────────────────────────────────────────────

_EXTRACTOR_SYSTEM = """You are decomposing a political transcript into atomic, individually verifiable claims. \
Your output will be fact-checked one claim at a time, so each claim must stand alone."""

_EXTRACTOR_USER_TEMPLATE = """TRANSCRIPT METADATA
Speaker: {speaker}
Role: {role}
Date: {date}
Venue: {venue}

TRANSCRIPT
{transcript_text}

INSTRUCTIONS
1. Extract every factual assertion — statistics, historical events, claims about what someone did or said, \
quantitative comparisons, causal attributions.
2. Skip pure opinion, rhetorical framing, value judgments, and predictions about the future.
3. Each claim should be a single assertion, self-contained, with any necessary context inlined. \
"We created more jobs than any president" needs the speaker and timeframe inlined: \
"Speaker claims his administration created more jobs than any previous president."
4. Preserve the original phrasing where possible but resolve pronouns and deictic references.
5. For each claim, capture a context_window of ±2 sentences from the transcript so the \
fact-checker can see the surrounding framing.

OUTPUT FORMAT
Return valid JSON:
{{
  "claims": [
    {{
      "id": "c001",
      "text": "atomic claim statement",
      "context_window": "2-3 sentences of surrounding text",
      "is_checkable": true,
      "claim_type": "statistical" | "historical" | "attribution" | "comparison" | "other"
    }}
  ]
}}

Return JSON only. No preamble, no commentary."""

# ── Extractor model ───────────────────────────────────────────────────────────

# Hard safety cap on extracted claims so a runaway model can't drain the budget.
# User-facing "max claims to verify" lives at the pipeline layer; this is only a
# last-line guard at extract time.
_EXTRACT_HARD_CAP = 500

_EXTRACTOR_MODEL = "claude-sonnet-4-6"
# Characters of transcript fed to the extractor. Claude Sonnet 4.6 supports
# 200K tokens (~800K chars); 200K chars comfortably covers a 60K-char SOTU
# plus slack without runaway cost.
_TRANSCRIPT_CHAR_BUDGET = 200_000
_MAX_OUTPUT_TOKENS = 8_192


class ClaimExtractor:
    """
    Extract atomic factual claims from a Transcript using Claude Sonnet 4.6.

    Parameters
    ----------
    api_key:
        Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
    model:
        Claude model identifier. Defaults to claude-sonnet-4-6.
    max_claims:
        Hard cap on returned claims. Defaults to TRUTHBOT_MAX_CLAIMS env var or 30.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_claims: Optional[int] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model or _EXTRACTOR_MODEL
        # Extractor safety cap only; pipeline-level --max-claims controls
        # how many of these extracted claims actually get verified.
        self._max_claims = max_claims or _EXTRACT_HARD_CAP

    # ── Public interface ──────────────────────────────────────────────────────

    def extract(self, transcript: Transcript) -> list[Claim]:
        """
        Extract verifiable claims from a transcript.

        Returns an empty list (with a logged warning) if no API key is set.
        Returns an empty list (with a logged error) if the API call or parse fails.
        """
        if not self._api_key:
            logger.warning("No ANTHROPIC_API_KEY set — returning stub claims.")
            return self._stub_claims(transcript)

        try:
            return self._call_llm(transcript)
        except Exception as exc:
            logger.error("Claim extraction failed: %s", exc)
            return []

    # ── Private helpers ───────────────────────────────────────────────────────

    def _call_llm(self, transcript: Transcript) -> list[Claim]:
        """Call Claude Sonnet and parse the structured claim list."""
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)

        user_msg = _EXTRACTOR_USER_TEMPLATE.format(
            speaker=transcript.speaker or "Unknown",
            role=transcript.metadata.get("role", "Unknown"),
            date=transcript.date.strftime("%B %d, %Y") if transcript.date else "Unknown",
            venue=transcript.venue or "Unknown",
            transcript_text=transcript.text[:_TRANSCRIPT_CHAR_BUDGET],
        )

        if len(transcript.text) > _TRANSCRIPT_CHAR_BUDGET:
            logger.warning(
                "ClaimExtractor: transcript truncated from %d to %d chars before extraction",
                len(transcript.text),
                _TRANSCRIPT_CHAR_BUDGET,
            )

        response = client.messages.create(
            model=self._model,
            max_tokens=_MAX_OUTPUT_TOKENS,
            system=_EXTRACTOR_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw_text = response.content[0].text.strip()
        data = self._parse_response(raw_text)

        raw_claims = data.get("claims", [])
        capped = raw_claims[: self._max_claims]
        if len(raw_claims) > self._max_claims:
            logger.warning(
                "ClaimExtractor: model returned %d claims; capping at %d for safety",
                len(raw_claims),
                self._max_claims,
            )

        claims: list[Claim] = []
        for item in capped:
            claim = Claim(
                transcript_id=transcript.id,
                text=item["text"],
                speaker=transcript.speaker,
                context=item.get("context_window", ""),
                category=item.get("claim_type", "other"),
                is_checkable=bool(item.get("is_checkable", True)),
            )
            claims.append(claim)

        logger.info(
            "Extracted %d claims from transcript %s (model: %s)",
            len(claims),
            transcript.id,
            self._model,
        )
        return claims

    def _parse_response(self, text: str) -> dict:
        """Parse the JSON response, handling markdown fences if present."""
        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strip markdown fences
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Extract first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))

        raise ValueError(f"Could not parse JSON from extractor response: {text[:300]}")

    def _stub_claims(self, transcript: Transcript) -> list[Claim]:
        """
        Return minimal placeholder claims when no API key is available.
        Used in dry-run mode and tests.
        """
        sentences = [s.strip() for s in transcript.text.split(".") if len(s.strip()) > 20]
        return [
            Claim(
                transcript_id=transcript.id,
                text=sentence.strip() + ".",
                speaker=transcript.speaker,
                context=sentence[:120],
                category="other",
                is_checkable=True,
            )
            for sentence in sentences[:3]
        ]
