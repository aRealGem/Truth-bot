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

Claim-text awareness (Phase 1c refinement)
------------------------------------------
Empirical evidence from the v-p1-p2 calibration run showed the raw
heuristic above false-positives on legitimate historical-baseline
reasoning. The refined contract:

1. Years that appear verbatim in the *claim text* are exempt from
   flagging (they are the comparison anchor the claim itself invokes —
   e.g. "lowest murder rate in 125 years, specifically referencing 1900"
   makes citing 1900 correct, not wrong-term).
2. When the claim contains a historical-comparison phrase
   ("lowest in N years", "first time in N years", "since YYYY",
   "in over N decades", etc.), the lookback floor is extended back
   to ``speech_year - N`` (decades multiplied ×10). Years inside that
   extended window are exempt from flagging.

Both adjustments target the C10 pattern (wrong-presidential-term
anchoring) while preserving the legitimate use of deep-history years as
comparison baselines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from truthbot.models import Claim, ModelVerdict
from truthbot.verify.context import terms as terms_registry


_YEAR_RX = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Historical-comparison phrases that expand the lookback floor back
# to ``speech_year - N`` (or ``N * 10`` for "decades"). Captures the
# numeric quantifier in group 1 and the unit in group 2.
#
# Examples (from real SOTU claims):
#   * "lowest in over 125 years"    -> floor back to speech_year - 125
#   * "highest since 1900"          -> handled separately via verbatim-year exemption
#   * "first time in four decades"  -> floor back to speech_year - 40
#
# Deliberately tolerant of small-word quantifiers ("five", "ten",
# "twenty") because claim extraction sometimes preserves spelled-out
# numerals. We only recognize low-risk word forms.
_N_YEARS_RX = re.compile(
    r"\b(?:in|for)\s+(?:over|more than|at least|about|nearly|almost)?\s*"
    r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|fifteen|twenty|twenty-five|thirty|forty|fifty|"
    r"sixty|seventy|eighty|ninety|one hundred)\s+"
    r"(years?|decades?)\b",
    re.IGNORECASE,
)

_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "twenty-five": 25, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "one hundred": 100,
}

# Open-ended historical-comparison phrases. When any of these appears
# in the claim, the claim invites *unbounded* historical comparison
# (e.g. "record levels", "all-time high", "highest ever"). In those
# cases, the validator should not flag any year citation in the model's
# reasoning — deep-past references are the evidentiary backbone of the
# comparison, not wrong-term anchoring.
_OPEN_ENDED_HISTORICAL_RX = re.compile(
    r"(?:"
    # Explicit historical-scope phrases
    r"\bin\s+(?:u\.?s\.?\s+)?(?:american\s+)?history\b"
    r"|\brecord\s+(?:level|high|low|setting|breaking)s?\b"
    r"|\ball[-\s]?time\s+(?:high|low|record|peak|best|worst)\b"
    # Superlative + ever (e.g. "lowest ever", "most arrests ever")
    r"|\b(?:highest|lowest|largest|smallest|worst|best|most|greatest|fewest|strongest|weakest)\s+ever\b"
    r"|\b(?:first|last)\s+time\s+ever\b"
    r"|\bnever\s+before\b"
    r"|\bunprecedented\b"
    r"|\bhistoric(?:al)?\s+(?:high|low|first|level|peak)s?\b"
    r")",
    re.IGNORECASE,
)


def _is_open_ended_historical(claim_text: str) -> bool:
    """True when the claim invokes unbounded historical comparison.

    These claims legitimize any deep-past year citation because the
    model is establishing a historical baseline, not anchoring its
    reasoning in the wrong era.
    """
    if not claim_text:
        return False
    return _OPEN_ENDED_HISTORICAL_RX.search(claim_text) is not None


def _historical_lookback_years(claim_text: str) -> int:
    """Return the largest N-year historical window implied by the claim.

    Returns 0 when no historical-comparison phrase is detected.
    """
    if not claim_text:
        return 0
    largest = 0
    for match in _N_YEARS_RX.finditer(claim_text):
        raw_num, unit = match.group(1).lower(), match.group(2).lower()
        if raw_num.isdigit():
            n = int(raw_num)
        else:
            n = _WORD_NUM.get(raw_num, 0)
        if unit.startswith("decade"):
            n *= 10
        if n > largest:
            largest = n
    return largest


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


def scan_text(
    text: str,
    speech_date: date,
    claim_text: Optional[str] = None,
) -> TemporalFinding:
    """Pure scan: given free-text and a speech date, return a ``TemporalFinding``.

    Exposed separately from ``apply_temporal_flags`` so tests can exercise
    the heuristic on synthetic strings without constructing full
    ``ModelVerdict`` objects.

    When ``claim_text`` is provided, years appearing verbatim in the claim
    are exempt from flagging, and any historical-comparison window implied
    by the claim ("in over 125 years") extends the lookback floor.
    """
    years = _extract_years(text)
    speech_year = speech_date.year
    window_start_year = speech_year - 4
    window_end_year = speech_year + 1
    lookback_floor = speech_year - 5  # flag strictly older than this

    claim_years = _extract_years(claim_text or "")

    # Open-ended historical framing: don't flag any deep-past citation.
    # Implemented as a blanket floor set to the earliest representable
    # year in our regex (1900). Leaves normal mixed-reference flagging
    # intact for non-historical claims.
    if _is_open_ended_historical(claim_text or ""):
        lookback_floor = 1900

    claim_lookback = _historical_lookback_years(claim_text or "")
    if claim_lookback:
        # Historical comparisons legitimize citations back to
        # speech_year - N. We include one extra year of buffer to cover
        # "more than N" / "over N" phrasing, which semantically means
        # ≥ N + 1 years ago. This is intentionally permissive; the goal
        # is to suppress false positives on legitimate historical-window
        # reasoning, not to catch subtle off-by-one anchoring bugs.
        lookback_floor = min(lookback_floor, speech_year - claim_lookback - 1)

    flagged = sorted(
        y for y in years
        if y < lookback_floor and y not in claim_years
    )
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
    finding = scan_text(combined, speech_dt, claim_text=getattr(claim, "text", None))
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
