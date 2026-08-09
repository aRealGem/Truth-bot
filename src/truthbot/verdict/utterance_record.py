"""D15 — utterance-derivative evidence (``utterance-record``). FLAG, DEFAULT OFF.

THE PROBLEM
-----------
Transcripts and official records OF THE SPEECH BEING EVALUATED carry
GOVERNMENT (govinfo / congress.gov) and WIRE tiers, so once the stance layer
gives them a bearing stance they credit ``MIN_BEARING_T13`` — and a claim is
let through on the strength of a document whose only content is the speaker
saying the thing. The claim witnesses itself.

Confirmed instances in the five rebuilt runs: the Daily Compilation of
Presidential Documents (``DCPD-2026xxxxx``), the Congressional Record for the
day of the address (``CREC-YYYY-MM-DD``, on both govinfo and congress.gov), the
Weekly Compilation (``WCPD-YYYY-MM-DD``), the American Presidency Project's
archive of the address itself (``presidency.ucsb.edu``), and same-speech "recap"
reporting whose snippet documents the speaker's WORDING rather than the
underlying fact (trump_2026:0469 E5 — an AP recap).

THE RULE
--------
A deterministic, model-free evidential role, computed from three things that
are all already on disk: the URL, the snippet, and the REGISTERED speech date
(``verdict.speech_context.speech_date_for``). Five independent rules, each
individually testable, each date-anchored so a Congressional Record from
another day — or a presidential document about another event — is NOT swept in:

  ``dcpd-daily-compilation``  govinfo DCPD package, item dated the speech date
                              or the day after (the Daily Compilation is filed
                              the morning after an evening address), same year
                              as the package id.
  ``crec-congressional-record``  a ``CREC-YYYY-MM-DD`` package id whose OWN date
                              is the speech date. The package id is read from
                              the URL, not from the item metadata, so the noisy
                              per-retriever ``published_at`` cannot move it.
  ``wcpd-weekly-compilation``  a ``WCPD-YYYY-MM-DD`` issue covering the speech
                              week (issue date within 7 days after the speech).
                              The LOOSEST rule here — a weekly issue also holds
                              that week's other presidential documents.
  ``presidency-ucsb-address``  an American Presidency Project document dated the
                              speech date whose slug or snippet names the
                              address itself (state of the union / joint
                              session / annual message). A same-day APP press
                              release about something else is deliberately NOT
                              matched.
  ``recap-language``          recap phrasing ("transcript of", "as delivered",
                              "recap of", "the President's wording") AND an
                              address token, on an item dated between the speech
                              and the fair-game end. This is the only rule that
                              reads prose, so it demands two independent cues.

Conservative by construction: every rule needs a date it can check, and an item
with no usable date matches nothing. A miss is the intended failure mode — a
false positive silently destroys real evidence.

THE EFFECT (RATIFIED 2026-08-09 — this is live)
------------------------------------------------
Quota credit 0: an ``utterance-record`` item can never be one of the
``MIN_BEARING_T13`` items that let a claim reach a decided verdict. It is still
KEPT and still DISPLAYED, carrying ``role: utterance-record`` in the pack
payload — provenance the reader can see, not evidence the gate can spend.

THE FLAG
--------
``TRUTHBOT_D15_UTTERANCE_RECORD`` is the one switch, and since the 2026-08-09
ratification its default is **ON**. Unset means enabled.

The env var survives as an OVERRIDE in both directions, which is the point:
``TRUTHBOT_D15_UTTERANCE_RECORD=0`` reproduces the pre-ratification gate
exactly, so a regression can be bisected against the old behaviour without
reverting code. ``consolidate(..., utterance_record=False)`` is the same
override as an explicit argument, and is what the $0 blast-radius measurements
use so they can never leave a flag set behind them.

Ratified decision: ``docs/decisions/D15-utterance-derivative.md``.

Stdlib only, on purpose: this module is a leaf so ``evidential_role`` and the
consolidator can both name the role without an import cycle.
"""
from __future__ import annotations

import os
import re
from datetime import date, timedelta
from typing import Optional
from urllib.parse import urlsplit

#: The role string. ``EvidentialRole.UTTERANCE_RECORD`` takes its value from
#: here so the two can never drift.
ROLE = "utterance-record"

#: The one switch. Unset/empty = the ratified DEFAULT, which is ON.
FLAG_ENV = "TRUTHBOT_D15_UTTERANCE_RECORD"

#: RATIFIED 2026-08-09 by the owner. Before that date this module shipped
#: default OFF pending ratification; the env var is now an OVERRIDE, kept so a
#: test or a $0 measurement can ask for the pre-ratification behaviour
#: explicitly (``TRUTHBOT_D15_UTTERANCE_RECORD=0``) rather than by unsetting
#: something and hoping.
DEFAULT_ENABLED = True

#: The date the owner ratified D15. Reported alongside outputs, so a run can be
#: tied to the decision that produced it.
RATIFIED = "2026-08-09"

_TRUTHY = ("1", "true", "yes", "on")

# ── rule names (stable identifiers; they are journaled and reported) ─────────
RULE_DCPD = "dcpd-daily-compilation"
RULE_CREC = "crec-congressional-record"
RULE_WCPD = "wcpd-weekly-compilation"
RULE_UCSB = "presidency-ucsb-address"
RULE_RECAP = "recap-language"

RULES: tuple[str, ...] = (RULE_DCPD, RULE_CREC, RULE_WCPD, RULE_UCSB, RULE_RECAP)

#: An evening address is compiled into the NEXT morning's Daily Compilation, and
#: retrievers disagree about which of the two days to report — biden_2022's
#: DCPD-202200127 appears with both 2022-03-01 and 2022-03-02. One day of slack,
#: no more.
DCPD_GRACE_DAYS = 1
#: The Weekly Compilation issue that covers the speech is dated at the end of
#: that week (WCPD-1998-02-02 for a 1998-01-27 address).
WCPD_WEEK_DAYS = 7
#: Recap coverage lands inside the speaker's fair-game window
#: (``era_lint.FAIR_GAME_DAYS``, restated here to keep this module a leaf; the
#: value is pinned by a test against era_lint).
RECAP_WINDOW_DAYS = 7

_DCPD_RX = re.compile(r"\bDCPD-(\d{4})(\d{4,6})\b", re.IGNORECASE)
_CREC_RX = re.compile(r"\bCREC-(\d{4})-(\d{2})-(\d{2})\b", re.IGNORECASE)
_WCPD_RX = re.compile(r"\bWCPD-(\d{4})-(\d{2})-(\d{2})\b", re.IGNORECASE)

_UCSB_HOST = "presidency.ucsb.edu"
#: APP document paths. ``/node/<id>`` is the legacy permalink for the same rows.
_UCSB_PATHS = ("/documents/", "/node/")

#: Slug/snippet tokens that name the ADDRESS ITSELF. Deliberately narrow: bare
#: "address"/"speech"/"remarks" are excluded because they match ordinary prose
#: ("address the deficit", "remarks on trade") and would turn a two-cue rule
#: into a one-cue rule.
_ADDRESS_TOKENS = (
    "state of the union",
    "state-the-union",          # APP slug form ("...-the-state-the-union-21")
    "state-of-the-union",
    "joint session of the congress",
    "joint session of congress",
    "address-before-joint-session",
    "annual message",
)

#: Phrases that mark a document as a RECORD OF THE SAYING rather than of the
#: fact. Each must co-occur with an address token.
_RECAP_TOKENS = (
    "transcript of",
    "full transcript",
    "as delivered",
    "as prepared for delivery",
    "recap of",
    "president's wording",
    "speaker's wording",
    "the president's words",
    "full text of the",
)


def flag_enabled(env: Optional[dict] = None) -> bool:
    """Is D15 switched on? Read at call time, so a test can flip it.

    Unset or empty means :data:`DEFAULT_ENABLED` — ON, ratified 2026-08-09.
    Anything else is an explicit override in either direction, so
    ``TRUTHBOT_D15_UTTERANCE_RECORD=0`` reproduces the pre-ratification gate."""
    src = os.environ if env is None else env
    raw = str(src.get(FLAG_ENV, "") or "").strip().lower()
    if not raw:
        return DEFAULT_ENABLED
    return raw in _TRUTHY


def _norm(text: str) -> str:
    """Lowercase and fold curly punctuation — stored snippets use U+2019
    ("President’s wording"), so a straight-apostrophe token list would
    silently never match."""
    return (text or "").replace("’", "'").replace("‘", "'").lower()


def _pkg_date(rx: re.Pattern, url: str) -> Optional[date]:
    """The YYYY-MM-DD baked into a govinfo/congress.gov package id, or None.

    Read from the URL rather than from item metadata on purpose: the package id
    is what GPO assigned, while ``published_at`` is whatever the retriever
    guessed (the same CREC PDF arrives with different dates from different
    retrievers)."""
    m = rx.search(url or "")
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:                       # e.g. CREC-2026-13-45
        return None


# ── the five rules, each independently testable ─────────────────────────────

def dcpd_package_year(url: str) -> Optional[int]:
    """Year from a Daily Compilation package id (``DCPD-202600136`` -> 2026)."""
    m = _DCPD_RX.search(url or "")
    return int(m.group(1)) if m else None


def crec_package_date(url: str) -> Optional[date]:
    """Date from a Congressional Record package id, on govinfo OR congress.gov.

    The BOUND Record (``GPO-CRECB-1998-pt1``) carries no date and therefore
    never matches — a deliberate miss."""
    return _pkg_date(_CREC_RX, url)


def wcpd_package_date(url: str) -> Optional[date]:
    """Issue date from a Weekly Compilation package id."""
    return _pkg_date(_WCPD_RX, url)


def is_presidency_ucsb_document(url: str) -> bool:
    """An American Presidency Project DOCUMENT url (not its homepage/search)."""
    try:
        parts = urlsplit(url or "")
    except ValueError:
        return False
    host = (parts.netloc or "").lower().split(":")[0]
    if not (host == _UCSB_HOST or host.endswith("." + _UCSB_HOST)):
        return False
    return any((parts.path or "").lower().startswith(p) for p in _UCSB_PATHS)


def names_the_address(*texts: str) -> bool:
    """Does any of ``texts`` name the address itself (slug or prose form)?"""
    blob = " ".join(_norm(t) for t in texts)
    return any(tok in blob for tok in _ADDRESS_TOKENS)


def has_recap_language(snippet: str) -> bool:
    """Recap phrasing AND an address token — two independent cues, because this
    is the only rule reading free prose."""
    text = _norm(snippet)
    if not any(tok in text for tok in _RECAP_TOKENS):
        return False
    return names_the_address(snippet)


def _within(d: Optional[date], start: date, days: int) -> bool:
    return d is not None and start <= d <= start + timedelta(days=days)


def utterance_record_rule(url: str, snippet: str = "", *,
                          speech_date: Optional[date],
                          item_date: Optional[date] = None) -> str:
    """Which D15 rule (if any) makes this item a record of the speech itself.

    Returns the rule name, or ``""`` for everything else. Pure, deterministic,
    no model call. ``speech_date`` is the REGISTERED utterance date; with no
    registered date nothing can be anchored, so nothing matches."""
    if speech_date is None:
        return ""
    url = url or ""

    # 1. Daily Compilation of Presidential Documents — the transcript.
    year = dcpd_package_year(url)
    if (year is not None and year == speech_date.year
            and _within(item_date, speech_date, DCPD_GRACE_DAYS)):
        return RULE_DCPD

    # 2. Congressional Record for the day the address was delivered.
    if crec_package_date(url) == speech_date:
        return RULE_CREC

    # 3. Weekly Compilation issue covering the speech week.
    if _within(wcpd_package_date(url), speech_date, WCPD_WEEK_DAYS):
        return RULE_WCPD

    # 4. American Presidency Project's archive copy of the address.
    if (is_presidency_ucsb_document(url) and item_date == speech_date
            and names_the_address(url, snippet)):
        return RULE_UCSB

    # 5. Same-speech recap coverage documenting the wording.
    if (_within(item_date, speech_date, RECAP_WINDOW_DAYS)
            and has_recap_language(snippet)):
        return RULE_RECAP

    return ""


def is_utterance_record(url: str, snippet: str = "", *,
                        speech_date: Optional[date],
                        item_date: Optional[date] = None) -> bool:
    """Boolean form of :func:`utterance_record_rule`."""
    return bool(utterance_record_rule(url, snippet, speech_date=speech_date,
                                      item_date=item_date))
