"""
Post-hoc temporal-alignment validator.

After a model returns a verdict, scan its ``explanation`` + ``caveats`` for
year mentions. When the speech era is unambiguous (we know ``claim.speech_date``)
and the model references dates that fall well outside the expected claim
window, attach a ``TEMPORAL_MISMATCH`` flag to ``verdict.temporal_flags``.

Why a flag and not a verdict rewrite?
-------------------------------------
Catching wrong-term reasoning deterministically is a guardrail, not an
oracle — speakers legitimately reference "since 2021" or "in 2019" in
historical-comparison framing. The validator errs on the side of *surfacing*
the signal to the human-adjudication layer (Phase 3e) and to the
family-aware consensus weight (Phase 3c). Both layers can choose to
down-weight, escalate, or ignore.

Heuristic
---------
For a given speech date ``S``, define:

  * ``in_window_years = [S.year - 4, S.year + 1]`` — recent history +
    forward reporting window.
  * ``flagged_years  = any year BEFORE (S.year - 5)`` — deep-past citations.

Flag when at least one ``flagged_year`` appears in the verdict text.
Callers decide severity (number of flagged years, whether any in-window
years are also present, etc.).

This does NOT flag "2020" or "2021" references in a 2026-dated SOTU
speech, which are almost always legitimate Biden/COVID-era comparisons.
It DOES flag "2017" or "2018" references, which are the Pattern A
wrong-term bug the external review called the #1 blocker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from truthbot.models import Claim, ModelVerdict
from truthbot.verify.context import terms as terms_registry


_YEAR_RX = re.compile(r"\b(19\d{2}|20\d{2})\b")


@dataclass(frozen=True)
class TemporalFinding:
    """Structured result of the temporal-alignment scan on a single verdict."""

    flagged_years: tuple[int, ...]
    """Years cited that fall BEFORE the permissive lookback floor."""

    in_window_years: tuple[int, ...]
    """Years cited that fall within ``[speech_year - 4, speech_year + 1]``."""

    window_start_year: int
    """Speech-year minus 4 (permissive lookback floor)."""

    window_end_year: int
    """Speech-year plus 1 (permissive forward end)."""

    speech_year: int

    @property
    def is_flagged(self) -> bool:
        return bool(self.flagged_years)

    def format_flag(self) -> Optional[str]:
        """Render the flag string attached to ``ModelVerdict.temporal_flags``.

        Returns None when nothing is flagged (caller should not attach).
        """
        if not self.is_flagged:
            return None
        flagged_str = ", ".join(str(y) for y in self.flagged_years)
        in_str = (
            f" Also cited in-window: {', '.join(str(y) for y in self.in_window_years)}."
            if self.in_window_years
            else " No in-window year references detected in the reasoning."
        )
        return (
            f"TEMPORAL_MISMATCH: reasoning cites year(s) {flagged_str} which "
            f"fall before the permissive lookback floor "
            f"({self.window_start_year}) for a speech dated in "
            f"{self.speech_year}.{in_str}"
        )


def _coerce_date(value: object) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def _extract_years(text: str) -> set[int]:
    if not text:
        return set()
    out: set[int] = set()
    for match in _YEAR_RX.finditer(text):
        try:
            out.add(int(match.group(1)))
        except ValueError:
            continue
    return out


def scan_text(text: str, speech_date: date) -> TemporalFinding:
    """Pure scan: given free-text and a speech date, return a ``TemporalFinding``.

    Exposed separately from ``apply_temporal_flags`` so tests can exercise
    the heuristic on synthetic strings without constructing full
    ``ModelVerdict`` objects.
    """
    years = _extract_years(text)
    speech_year = speech_date.year
    window_start_year = speech_year - 4
    window_end_year = speech_year + 1
    lookback_floor = speech_year - 5  # flag strictly older than this

    flagged = sorted(y for y in years if y < lookback_floor)
    in_window = sorted(
        y for y in years if window_start_year <= y <= window_end_year
    )
    return TemporalFinding(
        flagged_years=tuple(flagged),
        in_window_years=tuple(in_window),
        window_start_year=window_start_year,
        window_end_year=window_end_year,
        speech_year=speech_year,
    )


def apply_temporal_flags(verdict: ModelVerdict, claim: Claim) -> ModelVerdict:
    """Mutate ``verdict.temporal_flags`` in place and return it.

    No-ops when ``claim.speech_date`` is unknown (we cannot build a window).
    Idempotent: re-applying on an already-flagged verdict does not duplicate
    the flag string.
    """
    speech_dt = _coerce_date(getattr(claim, "speech_date", None))
    if speech_dt is None:
        return verdict

    combined = f"{verdict.explanation or ''}\n{verdict.caveats or ''}"
    finding = scan_text(combined, speech_dt)
    flag = finding.format_flag()
    if flag is None:
        return verdict

    existing = list(verdict.temporal_flags or [])
    if flag not in existing:
        existing.append(flag)
        verdict.temporal_flags = existing
    return verdict


# Re-export ``expected_claim_window`` for callers that need the window
# without importing the terms module directly.
expected_claim_window = terms_registry.expected_claim_window
