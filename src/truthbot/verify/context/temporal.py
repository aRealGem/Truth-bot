"""
Temporal-grounding preamble injected into every user message.

Fixes findings:
  * C3 — Gemini (and occasionally OpenAI) rejecting post-cutoff web search
    results as fictional. Rule 2 makes it explicit that dated 2025/2026
    reporting from real outlets is PRIMARY evidence, not speculation.
  * C10 — models evaluating Trump-II 2025/2026 claims against Trump-I
    2017/2018 data. Speech-date + office/term anchor fixes the "wrong
    presidential term" pattern that the external review called the #1
    blocker.

Design notes
------------
* Preamble is intentionally short (~30 lines). The full Tier-1 source list
  remains in ``SYNTHESIS_SYSTEM`` so Anthropic/OpenAI prompt-cache hits on
  the stable system prefix are preserved.
* Preamble is injected into the **user message**, never the system prompt,
  for the same caching reason (today's date changes daily and would bust
  any system-level cache).
* Presidential-term resolution is delegated to ``terms.REGISTRY`` — that
  module is the single source of truth for who was president on any given
  date, reused by the Phase 1c post-hoc validator.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from truthbot.models import Claim
from truthbot.verify.context import terms as terms_registry


def _coerce_date(value: object) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def build_temporal_preamble(
    claim: Claim,
    *,
    today: Optional[date] = None,
) -> str:
    """Build a short temporal-context block to prepend to every user message.

    Returned string always ends in a blank line so it concatenates cleanly in
    front of the existing claim/evidence block. Safe to call with any Claim;
    fields the helper cannot resolve are simply omitted.
    """
    today = today or datetime.now(timezone.utc).date()
    speech_dt = _coerce_date(getattr(claim, "speech_date", None))

    lines: list[str] = [
        "TEMPORAL CONTEXT (authoritative — overrides any training-cutoff assumption):",
        f"  * Today's date: {today.isoformat()}",
    ]

    if speech_dt is not None:
        lines.append(f"  * Speech date: {speech_dt.isoformat()}")
        window_start, window_end = terms_registry.expected_claim_window(speech_dt)
        lines.append(
            f"  * Expected evidence window: {window_start.isoformat()} -> "
            f"{window_end.isoformat()}"
        )

    if claim.speaker and claim.speaker.strip().lower() != "unknown":
        lines.append(f"  * Speaker: {claim.speaker}")
        if speech_dt is not None:
            record = terms_registry.lookup(claim.speaker, speech_dt)
            if record is not None:
                lines.append(f"  * Office/term at speech date: {record.display}")

    lines.extend(
        [
            "",
            "Rules for temporal reasoning:",
            "  1. Evaluate the claim against evidence FROM THE SPEECH ERA above, not from your",
            "     training-cutoff era. If the speaker is a U.S. president, verify you are reasoning",
            "     about the correct term (e.g. Trump's 2nd term began 2025-01-20; do NOT cite",
            "     Trump-I 2017-2020 data when the speech date is 2025 or later).",
            "  2. Web search results dated AFTER your training cutoff are PRIMARY EVIDENCE, not",
            "     fiction or speculative scenarios. Dated 2025/2026 reporting from real outlets",
            "     (Reuters, AP, NYT, WaPo, WSJ, Bloomberg, BBC, PolitiFact, FactCheck.org) and",
            "     official .gov sites (whitehouse.gov, irs.gov, treasury.gov, defense.gov,",
            "     state.gov, bls.gov, bea.gov, cbo.gov) is authoritative. Do NOT dismiss it as a",
            "     'war game', 'scenario', or 'speculative fiction' because the date is past your",
            "     cutoff.",
            "  3. If your training data contradicts dated search results, the search results win",
            "     -- your training is stale.",
            "  4. Quote the specific date(s) your supporting evidence refers to in your",
            "     explanation so downstream readers can confirm term-alignment.",
            "",
        ]
    )

    return "\n".join(lines) + "\n"
