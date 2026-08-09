"""D16(α) — statistical-agency release in the post-speech band. FLAG, DEFAULT OFF.

THE PROBLEM
-----------
The era rule (remediation v2, item 1.3) marks every item published after the
utterance but inside the fair-game window ``post-speech · context-only``: kept,
displayed, and unable to credit ``MIN_BEARING_T13``. That was aimed at
same-speech fact-checks and reaction coverage — evidence that could not exist
before the speech and that judges the speech with hindsight.

It is ALSO silencing government statistical publications that report
PRE-UTTERANCE FACTS. The January 2006 Employment Situation is published on
3 February 2006 and measures January 2006; a 31 January 2006 speech about
January payrolls cannot be checked against it, because the era rule sees only
the publication date. Across the five rebuilt runs this band holds BLS payroll
releases, BEA GDP and personal-income releases, CBO outlooks, and EIA outlooks
— all measuring periods that had already ended when the speaker spoke.

WHAT WAS REJECTED, AND WHY THIS IS DIFFERENT
--------------------------------------------
The blanket form — "any Government-tier post-speech item may credit" — was
rejected: the two motivating examples turned out to be the principal's OWN
executive documents (``gwbush_2006:0217``'s ONDCP National Drug Control
Strategy, ``clinton_1998:0101``'s FY1999 President's Budget).
``principals.principal_relation`` keys on HOST, and those documents are served
from justice.gov, files.eric.ed.gov and gpo.gov, so they read "independent". A
document-class detector would catch them; that is real work and is deliberately
DEFERRED (logged D17-candidate).

D16(α) needs no detector, because it inverts the test. Instead of asking what
the document is NOT, it asks whether the PUBLISHER'S FUNCTION is statistical
measurement — and the President's Budget and the ONDCP Strategy are not
statistical-agency records no matter which host serves them.

THE RULE — three conditions, ALL required, each independently testable
---------------------------------------------------------------------
  1. **Function.** The host resolves through the versioned, fail-closed
     allowlist :mod:`truthbot.verify.statistical_agency` (BLS, BEA, Census,
     CBO, GAO, CRS, FRED/ALFRED, EIA, NCES, NCHS/CDC statistical products,
     USDA-NASS). Structurally excluded there: Executive Office of the
     President units (OMB, ONDCP, CEA), agency press offices, anything
     *whitehouse*, and document archives that reprint executive documents
     alongside statistical ones (fraser.stlouisfed.org).
  2. **Data period.** The item names a PARSEABLE data period at or before the
     utterance — a month, a quarter, a fiscal year, a year attached to a
     measurement noun, an anchor year ("since 2001"), or a year range. See
     :func:`data_periods`.

     This REPLACES an earlier heuristic — "any 4-digit year ≤ the utterance
     year anywhere in the snippet" — which must not survive, because
     ``gwbush_2006:0217`` passed it on the strength of its own PUBLICATION year
     ("2006"). For the same reason the ``[YYYY-MM-DD]`` prefix the connectors
     stamp into snippets, and every other bare calendar date, is MASKED OUT
     before parsing: a publication date is not a data period.

     FAIL CLOSED: no parseable period → no credit. Real statistical releases
     are missed by this (``biden_2022:0266``'s CDC Weekly Review names its
     reference week "Feb 23–Mar 1" without a year) and that is the intended
     failure mode — a false positive lets post-utterance world-state decide a
     verdict, which is the harm item 1.3 exists to prevent.
  3. **Cap.** The item is still inside the S-2 fair-game window
     (``utterance < published ≤ utterance + FAIR_GAME_DAYS``). S-2 is NOT
     touched: D16 releases the quota credit for items already inside the band,
     it never widens the band. An item dated after the cap matches nothing
     here, and an item dated BEFORE the utterance matches nothing either —
     it already credits, so it has no need of this rule.

THE EFFECT (when ratified)
--------------------------
A matching item may credit the quota exactly as a pre-utterance item of the
same tier and stance would; nothing else about it changes. Its pack payload
carries ``era_note: "post-speech · statistical release …"`` in place of
``post-speech · context-only``, because after this rule the old note would be
false. D15 still wins where both apply: a record of the utterance itself
credits nothing on any branch.

THE FLAG
--------
``TRUTHBOT_D16_STATISTICAL_RELEASE=1`` is the one switch. Default OFF: with the
flag unset nothing is classified, no item is released, and every gate outcome
is bit-for-bit what it is today. ``consolidate(..., statistical_release=True)``
is the same switch as an explicit argument, for tests and the $0 blast-radius
measurement (``scripts/measure_d16.py``). Ratification proposal:
``docs/decisions/D16-statistical-release.md``.

Structured exactly like D15 (:mod:`truthbot.verdict.utterance_record`) on
purpose — same flag shape, same rule-name vocabulary, same "returns the rule
that fired, or ''" contract — so the two proposals can be read, tested,
measured and ratified with one mental model.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

#: The one switch. Unset/empty = OFF. (The D16 analogue of D15's
#: ``TRUTHBOT_D15_UTTERANCE_RECORD``.)
FLAG_ENV = "TRUTHBOT_D16_STATISTICAL_RELEASE"

_TRUTHY = ("1", "true", "yes", "on")

#: The era note that REPLACES ``consolidator.POST_SPEECH_NOTE`` on a released
#: item: it is still post-speech, but it is no longer context-only.
RELEASE_NOTE = "post-speech · statistical release (pre-utterance data period)"

#: The fair-game cap this rule works inside (``era_lint.FAIR_GAME_DAYS``,
#: restated to keep this module importable on its own; pinned by a test).
FAIR_GAME_DAYS = 7

# ── period-rule names (stable identifiers; journaled and reported) ───────────
RULE_MONTH = "stat-period-month"
RULE_QUARTER = "stat-period-quarter"
RULE_FISCAL_YEAR = "stat-period-fiscal-year"
RULE_YEAR_DATA = "stat-period-year-data"
RULE_ANCHOR_YEAR = "stat-period-anchor-year"
RULE_YEAR_RANGE = "stat-period-year-range"

RULES: tuple[str, ...] = (RULE_MONTH, RULE_QUARTER, RULE_FISCAL_YEAR,
                          RULE_YEAR_DATA, RULE_ANCHOR_YEAR, RULE_YEAR_RANGE)

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

_QUARTER_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4,
                  "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
_QWORD_ALT = "|".join(_QUARTER_WORDS)

#: Nouns that make a year a MEASUREMENT period rather than a date. Kept to
#: data-product words on purpose: "the 2005 survey" is a period, "the 2005
#: strategy" is a document.
_DATA_NOUNS = ("surveys?", "census", "data", "datasets?", "estimates?",
               "statistics", "figures", "tabulations?", "series", "vintage",
               "benchmark", "readings?", "production", "output", "levels",
               "totals?", "cohort", "averages?")
_NOUN_ALT = "|".join(_DATA_NOUNS)

#: Prepositions that ANCHOR a series to a year. "in" is deliberately absent —
#: "published in 2006" is a publication date, and admitting it would rebuild
#: the bare-year heuristic this rule replaces.
_ANCHOR_ALT = "since|from|through|between|as of|during|beginning|starting"

# Calendar dates, masked out BEFORE parsing (see the module docstring).
_MASK_RX = (
    re.compile(r"\[\s*\d{4}-\d{2}-\d{2}\s*\]"),                 # snippet prefix
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                       # bare ISO date
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),                 # 3/4/2022
    re.compile(rf"\b(?:{_MONTH_ALT})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b",
               re.IGNORECASE),                                  # "Mar 4, 2022"
    re.compile(rf"\b\d{{1,2}}\s+(?:{_MONTH_ALT})\.?,?\s+\d{{4}}\b",
               re.IGNORECASE),                                  # "4 March 2022"
)

_MONTH_RX = re.compile(rf"\b({_MONTH_ALT})\.?\s+(\d{{4}})\b", re.IGNORECASE)
_QUARTER_RX = re.compile(r"\bQ([1-4])\s*[-/ ]?\s*(\d{4})\b", re.IGNORECASE)
_QUARTER_REV_RX = re.compile(r"\b(\d{4})\s*[-/ ]?\s*Q([1-4])\b", re.IGNORECASE)
_QUARTER_WORD_RX = re.compile(
    rf"\b({_QWORD_ALT})\s+quarter\s+(?:of\s+)?(\d{{4}})\b", re.IGNORECASE)
_FY_RX = re.compile(r"\b(?:FY|fiscal(?:\s+year)?)\s*[-–]?\s*(\d{4})\b",
                    re.IGNORECASE)
_YEAR_DATA_RX = re.compile(
    rf"\b(\d{{4}})\s+(?:[a-z]{{2,14}}\s+)?(?:{_NOUN_ALT})\b", re.IGNORECASE)
_DATA_YEAR_RX = re.compile(
    rf"\b(?:{_NOUN_ALT})\s+(?:for|in|of|from|through)\s+(\d{{4}})\b",
    re.IGNORECASE)
_ANCHOR_RX = re.compile(rf"\b(?:{_ANCHOR_ALT})\s+(\d{{4}})\b", re.IGNORECASE)
_YEAR_RANGE_RX = re.compile(r"\b(\d{4})\s*[-–—]\s*(\d{4})\b")

#: Sane calendar bounds — a 4-digit number is not automatically a year
#: ("$1,400 billion", "3200 series"). Anything outside is not a period.
_MIN_YEAR, _MAX_YEAR = 1900, 2100

#: Separators that turn a URL path into words ("personal-income-and-outlays-
#: january-2026" -> "... january 2026"). The URL is read as a SECOND prose
#: source, through the identical patterns — never with looser ones.
_URL_SPLIT_RX = re.compile(r"[/_\-.+%?=&#]+")


@dataclass(frozen=True)
class DataPeriod:
    """One parsed data-period reference."""
    rule: str            # one of RULES
    label: str           # the matched text, e.g. "January 2006"
    start: date          # the EARLIEST date the period covers
    position: int        # character offset of the match (for stable ordering)


def flag_enabled(env: Optional[dict] = None) -> bool:
    """Is D16 switched on? Read at call time, so a test can flip it."""
    src = os.environ if env is None else env
    return str(src.get(FLAG_ENV, "") or "").strip().lower() in _TRUTHY


def _year_ok(y: int) -> bool:
    return _MIN_YEAR <= y <= _MAX_YEAR


def mask_calendar_dates(text: str) -> str:
    """Blank out publication-date shapes so they cannot be read as periods.

    Replaces with spaces rather than deleting, so match offsets still line up
    with the original text and the ordering of hits stays meaningful."""
    out = text or ""
    for rx in _MASK_RX:
        out = rx.sub(lambda m: " " * len(m.group(0)), out)
    return out


def url_words(url: str) -> str:
    """A URL's own words, as prose. ``bea.gov/news/2026/personal-income-and-
    outlays-january-2026`` -> ``"... january 2026"``.

    Statistical agencies name the reference period in the slug far more
    reliably than the retriever's one-line snippet does, so the URL is parsed
    too — but through the SAME patterns, so nothing looser gets in. Separators
    become spaces, which is also why ``empsit_02032006`` stays unparseable: it
    has no separator between month and year."""
    if not url:
        return ""
    body = url.split("://", 1)[-1]
    body = body.split("/", 1)[1] if "/" in body else ""   # drop the host
    return _URL_SPLIT_RX.sub(" ", body)


def data_periods(text: str) -> list[DataPeriod]:
    """Every parseable data-period reference in ``text``, in document order.

    Pure and deterministic. Calendar dates are masked first (see
    :func:`mask_calendar_dates`), so a bare ``[2006-02-01]`` publication-date
    prefix yields NOTHING — which is the single most important negative in
    this module."""
    body = mask_calendar_dates(text or "")
    hits: list[DataPeriod] = []

    def add(rule: str, m: re.Match, start: date) -> None:
        hits.append(DataPeriod(rule=rule, label=m.group(0).strip(),
                               start=start, position=m.start()))

    for m in _MONTH_RX.finditer(body):
        y = int(m.group(2))
        if _year_ok(y):
            add(RULE_MONTH, m, date(y, _MONTHS[m.group(1).lower()], 1))
    for rx, yg, qg in ((_QUARTER_RX, 2, 1), (_QUARTER_REV_RX, 1, 2)):
        for m in rx.finditer(body):
            y, q = int(m.group(yg)), int(m.group(qg))
            if _year_ok(y):
                add(RULE_QUARTER, m, date(y, 3 * (q - 1) + 1, 1))
    for m in _QUARTER_WORD_RX.finditer(body):
        y = int(m.group(2))
        if _year_ok(y):
            q = _QUARTER_WORDS[m.group(1).lower()]
            add(RULE_QUARTER, m, date(y, 3 * (q - 1) + 1, 1))
    for m in _FY_RX.finditer(body):
        y = int(m.group(1))
        if _year_ok(y):
            # Federal fiscal year N runs 1 Oct (N-1) .. 30 Sep (N), so FY1998
            # had already begun when Clinton spoke in January 1998.
            add(RULE_FISCAL_YEAR, m, date(y - 1, 10, 1))
    for rx in (_YEAR_DATA_RX, _DATA_YEAR_RX):
        for m in rx.finditer(body):
            y = int(m.group(1))
            if _year_ok(y):
                add(RULE_YEAR_DATA, m, date(y, 1, 1))
    for m in _ANCHOR_RX.finditer(body):
        y = int(m.group(1))
        if _year_ok(y):
            add(RULE_ANCHOR_YEAR, m, date(y, 1, 1))
    for m in _YEAR_RANGE_RX.finditer(body):
        y = int(m.group(1))
        if _year_ok(y) and _year_ok(int(m.group(2))):
            add(RULE_YEAR_RANGE, m, date(y, 1, 1))

    hits.sort(key=lambda p: (p.position, p.rule))
    return hits


def pre_utterance_period(text: str, utterance: date) -> Optional[DataPeriod]:
    """The first data period in ``text`` that STARTS at or before ``utterance``.

    "At or before" is measured on the period's START: the January 2026
    Employment Situation reports a month that had already begun (indeed ended)
    when a 24 February 2026 speech was given, while a "February 2026" outlook
    published on 26 February names a period the speaker could not have
    observed in full — but which had begun, so it qualifies. What can never
    qualify is a period that had not started at all."""
    for p in data_periods(text):
        if p.start <= utterance:
            return p
    return None


def in_post_speech_band(item_date: Optional[date], utterance: Optional[date],
                        ) -> bool:
    """Condition 3: published after the utterance but inside the S-2 cap.

    Exactly ``consolidator``'s ``post_speech`` predicate, restated so it is
    testable on its own — D16 releases items ALREADY in that band and must
    never move its edges."""
    if item_date is None or utterance is None:
        return False
    return utterance < item_date <= utterance + timedelta(days=FAIR_GAME_DAYS)


def statistical_release_rule(url: str, snippet: str = "", *,
                             utterance: Optional[date],
                             item_date: Optional[date] = None) -> str:
    """Which D16 period rule (if any) releases this post-speech item.

    Returns the rule name (one of :data:`RULES`), or ``""`` for everything
    else. Pure, deterministic, no model call, no network. ALL THREE conditions
    must hold; the order below is cheapest-and-most-decisive first, so a
    denied host never even parses prose."""
    if not in_post_speech_band(item_date, utterance):     # 3. the S-2 cap
        return ""
    from truthbot.verify.statistical_agency import is_statistical_agency

    if not is_statistical_agency(url or ""):              # 1. the function
        return ""
    assert utterance is not None                          # narrowed by (3)
    period = pre_utterance_period(f"{snippet or ''} {url_words(url)}",
                                  utterance)              # 2. the data period
    return period.rule if period is not None else ""


def statistical_release_detail(url: str, snippet: str = "", *,
                               utterance: Optional[date],
                               item_date: Optional[date] = None,
                               ) -> Optional[dict]:
    """:func:`statistical_release_rule` plus the WHY, for telemetry.

    ``{"url", "rule", "agency", "reason", "period"}`` or ``None``. The agency
    and the registry reason are what let a reviewer ratify the allowlist one
    host at a time instead of as a bloc."""
    rule = statistical_release_rule(url, snippet, utterance=utterance,
                                    item_date=item_date)
    if not rule:
        return None
    from truthbot.verify.statistical_agency import agency_for, classify_ex

    assert utterance is not None
    period = pre_utterance_period(f"{snippet or ''} {url_words(url)}", utterance)
    return {"url": url, "rule": rule, "agency": agency_for(url),
            "reason": classify_ex(url)[1],
            "period": period.label if period else "",
            "period_start": period.start.isoformat() if period else ""}


def is_statistical_release(url: str, snippet: str = "", *,
                           utterance: Optional[date],
                           item_date: Optional[date] = None) -> bool:
    """Boolean form of :func:`statistical_release_rule`."""
    return bool(statistical_release_rule(url, snippet, utterance=utterance,
                                         item_date=item_date))
