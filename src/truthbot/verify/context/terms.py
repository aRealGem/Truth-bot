"""
U.S. presidential-term registry.

The temporal preamble (``temporal.py``) and the post-hoc validator
(``validator.py``, Phase 1c) both need a small, trustworthy table of who
was president on a given date and which term number / period that was. A
standalone registry keeps that table one edit away, makes it testable in
isolation, and lets us add vice-presidents, speakers of the house, or
foreign heads of state later without ratcheting the temporal module.

Table philosophy
----------------
* Terms are stored as half-open intervals ``[start_date, end_date)`` so
  inauguration day itself belongs to the incoming president.
* ``end_date=None`` marks the currently-serving term; lookup treats that as
  "open upper bound."
* Entries are ordered most-recent first so linear-scan lookup hits the hot
  path first (modern claims dominate SOTU-style workloads).
* Speaker matching is loose-containment (case-insensitive), so real-world
  strings like ``"President Donald J. Trump"`` or ``"Joseph R. Biden Jr."``
  match without hand-curated aliases. Exact aliases can be added here
  when a confusing collision appears (e.g. two presidents named Bush).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass(frozen=True)
class TermRecord:
    """A single U.S. presidential term."""

    speaker_name: str
    """Canonical display name, e.g. 'Donald Trump'."""

    presidency_number: int
    """Sequential U.S. presidency count (Biden=46, Trump=45/47, Obama=44, ...)."""

    term_index: int
    """1 for first term, 2 for second (non-consecutive counts as separate records)."""

    start_date: date
    """Inauguration date (inclusive)."""

    end_date: Optional[date]
    """Exclusive end date, or None for the currently-serving term."""

    aliases: tuple[str, ...] = ()
    """Optional extra strings that should match this term (lowercase compared)."""

    @property
    def label(self) -> str:
        """Human-readable term label used in the temporal preamble."""
        ord_suffix = _ordinal_suffix(self.presidency_number)
        return (
            f"{self.presidency_number}{ord_suffix} U.S. President, "
            f"{_ordinal_word(self.term_index)} term"
        )

    @property
    def display(self) -> str:
        """Full display string: '<name> — <label> (inaugurated YYYY-MM-DD)'."""
        return (
            f"{self.speaker_name} — {self.label} "
            f"(inaugurated {self.start_date.isoformat()})"
        )

    def contains(self, dt: date) -> bool:
        if dt < self.start_date:
            return False
        if self.end_date is not None and dt >= self.end_date:
            return False
        return True


def _ordinal_suffix(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


_ORDINAL_WORDS = {
    1: "1st",
    2: "2nd",
    3: "3rd",
    4: "4th",
}


def _ordinal_word(n: int) -> str:
    return _ORDINAL_WORDS.get(n, f"{n}th")


# Registry: most-recent first. Extend as future terms begin; adjust end_date on
# the current record when a successor is inaugurated.
REGISTRY: tuple[TermRecord, ...] = (
    TermRecord(
        speaker_name="Donald Trump",
        presidency_number=47,
        term_index=2,
        start_date=date(2025, 1, 20),
        end_date=None,
        aliases=("donald j. trump", "donald j trump", "trump"),
    ),
    TermRecord(
        speaker_name="Joe Biden",
        presidency_number=46,
        term_index=1,
        start_date=date(2021, 1, 20),
        end_date=date(2025, 1, 20),
        aliases=("joseph r. biden", "joseph r biden", "joseph biden", "biden"),
    ),
    TermRecord(
        speaker_name="Donald Trump",
        presidency_number=45,
        term_index=1,
        start_date=date(2017, 1, 20),
        end_date=date(2021, 1, 20),
        aliases=("donald j. trump", "donald j trump", "trump"),
    ),
    TermRecord(
        speaker_name="Barack Obama",
        presidency_number=44,
        term_index=2,
        start_date=date(2013, 1, 20),
        end_date=date(2017, 1, 20),
        aliases=("barack h. obama", "obama"),
    ),
    TermRecord(
        speaker_name="Barack Obama",
        presidency_number=44,
        term_index=1,
        start_date=date(2009, 1, 20),
        end_date=date(2013, 1, 20),
        aliases=("barack h. obama", "obama"),
    ),
    TermRecord(
        speaker_name="George W. Bush",
        presidency_number=43,
        term_index=2,
        start_date=date(2005, 1, 20),
        end_date=date(2009, 1, 20),
        aliases=("george w bush", "george walker bush", "bush 43"),
    ),
    TermRecord(
        speaker_name="George W. Bush",
        presidency_number=43,
        term_index=1,
        start_date=date(2001, 1, 20),
        end_date=date(2005, 1, 20),
        aliases=("george w bush", "george walker bush", "bush 43"),
    ),
)


def _coerce_date(value: object) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def _speaker_matches(record: TermRecord, raw_speaker: str) -> bool:
    raw = raw_speaker.strip().lower()
    if not raw:
        return False
    candidates = (record.speaker_name.lower(),) + tuple(a.lower() for a in record.aliases)
    for cand in candidates:
        if not cand:
            continue
        if cand in raw or raw in cand:
            return True
    return False


def lookup(speaker: Optional[str], at: object) -> Optional[TermRecord]:
    """Return the ``TermRecord`` that matches ``speaker`` active on ``at``.

    ``at`` may be a ``date`` or ``datetime``. Returns ``None`` when either
    input is missing/unparseable or no record fits — callers should then omit
    the office/term line from the preamble rather than fabricating one.
    """
    if not speaker:
        return None
    dt = _coerce_date(at)
    if dt is None:
        return None
    for record in REGISTRY:
        if not _speaker_matches(record, speaker):
            continue
        if record.contains(dt):
            return record
    return None


def expected_claim_window(speech_date: date) -> tuple[date, date]:
    """Default claim-evidence window: ~2y before speech, ~3mo after.

    Extracted here (rather than in ``temporal.py``) so the Phase 1c
    post-hoc validator can reuse the same window rule when flagging reasoning
    that references dates outside the expected band.
    """
    start = date(speech_date.year - 2, 1, 1)
    end_month = speech_date.month + 3
    end_year = speech_date.year + (end_month - 1) // 12
    end_month = ((end_month - 1) % 12) + 1
    end = date(end_year, end_month, 1)
    return start, end
