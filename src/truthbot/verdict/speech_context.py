"""Temporal grounding for Layer B/C verdicts (charter: judge veracity as-of the
utterance date, and treat contemporary reporting as authoritative).

Two axes the panel must respect:
  (a) the UTTERANCE date — when the claim was publicly made (derived from the sid's
      speech prefix here; a claim true when said is not false merely because reality
      moved later);
  (b) the REFERENCE period — the span the claim is about (passed in when known, e.g.
      from the verdict-gold fixture).

Speaker-BLIND by design (I3): the preamble anchors on DATES only, never the speaker —
the date already disambiguates the era (e.g. 2026 vs 2022) without naming who spoke.
Reuses ``verify.context.terms.expected_claim_window`` so Layer C evidence retrieval and
this preamble share one window rule.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from truthbot.verify.context.terms import expected_claim_window

# sid prefix -> utterance (speech) date. Extend as the corpus grows.
#
# All FIVE corpus speeches are pinned here statically (remediation v2 Phase A,
# A4). Only biden_2022 + trump_2026 used to be: the other three resolved to a
# date solely because a runner called register_speech_date() at run time, so
# any path that did NOT go through that runner — a re-render, a script, an
# ad-hoc consolidation — saw speech_date_for() -> None and silently ran with
# NO era gate at all. That is exactly how the Obama-2014 rescue leg shipped
# 2026-dated evidence into a 2014 speech. Static pinning + the publish-side
# check in publish.consistency.check_run_artifacts (every speech_id in a run
# artifact must resolve to a date, statically or by registration) closes it in
# both directions: nothing publishes on an unknown date, and the common case
# needs no registration call to be safe.
SPEECH_DATE: dict[str, date] = {
    "clinton_1998": date(1998, 1, 27),  # Clinton SOTU, 1998-01-27
    "gwbush_2006": date(2006, 1, 31),   # G.W. Bush SOTU, 2006-01-31
    "obama_2014": date(2014, 1, 28),    # Obama SOTU, 2014-01-28
    "biden_2022": date(2022, 3, 1),     # Biden SOTU, 2022-03-01
    "trump_2026": date(2026, 2, 24),    # Trump SOTU, 2026-02-24
    # Senate floor speeches. Pinned statically for the same reason the five
    # presidential speeches are: a re-render or a script does not go through
    # the CLI runner that would otherwise register them, and an unpinned
    # speech_id resolves to None -> no era gate.
    "budd_2025-04-02": date(2025, 4, 2),     # Budd, fentanyl, 2025-04-02
    "cruz_2026-06-24": date(2026, 6, 24),    # Cruz, Dobbs, 2026-06-24
    "tillis_2025-01-23": date(2025, 1, 23),  # Tillis, Trump admin, 2025-01-23
    "warren_2025-04-29": date(2025, 4, 29),  # Warren, first 100 days, 2025-04-29
}


def speech_date_for(sid: str) -> Optional[date]:
    """Utterance date for a claim sid ('<speech>:<offset>'), or None if unknown."""
    return SPEECH_DATE.get(sid.split(":", 1)[0]) if sid else None


class SpeechIdCollision(ValueError):
    """A speech_id was registered against a second, different utterance date."""


def register_speech_date(speech_id: str, utterance: date, *,
                         strict: bool = False) -> None:
    """Register a speech-prefix → utterance date so the temporal preamble and
    Layer C evidence window resolve for a transcript that isn't a pinned eval
    fixture. Used by the v2 publish path to thread the CLI ``--date`` into
    temporal grounding for an arbitrary speech_id. Idempotent; last write wins
    (a per-process CLI run adjudicates one speech).

    ``strict=True`` refuses a rebind to a DIFFERENT date, raising
    :class:`SpeechIdCollision`. That is the CLI publish path, where the id may
    have come from :func:`~truthbot.pipeline._default_speech_id` — which is
    only ``speaker_year``, so two speeches by the same senator in one year
    collide on it. Silently taking the last write there would file the second
    speech's claims under the first speech's date and merge two distinct
    speeches into one corpus entry. Default stays permissive: the in-process
    callers below (and the test suite) legitimately rebind a scratch id, and
    last-write-wins is the documented contract for them."""
    if strict:
        prior = SPEECH_DATE.get(speech_id)
        if prior is not None and prior != utterance:
            raise SpeechIdCollision(
                f"speech_id {speech_id!r} is already registered to "
                f"{prior.isoformat()}; refusing to rebind it to "
                f"{utterance.isoformat()}. Two different speeches are sharing "
                f"one id — pass an explicit --speech-id to distinguish them "
                f"(the default id is only speaker+year).")
    SPEECH_DATE[speech_id] = utterance


def build_temporal_preamble(sid: str, *, reference_period: Optional[str] = None,
                            today: Optional[date] = None) -> str:
    """Speaker-blind temporal block to prepend to a claim's context. Empty string if
    the utterance date is unknown (nothing to anchor on)."""
    utt = speech_date_for(sid)
    if utt is None:
        return ""
    today = today or datetime.now(timezone.utc).date()
    win_start, win_end = expected_claim_window(utt)
    lines = [
        "TEMPORAL CONTEXT (authoritative — overrides any training-cutoff assumption):",
        f"  * Utterance date (when the claim was publicly made): {utt.isoformat()}",
        f"  * Today's date: {today.isoformat()}",
        f"  * Expected evidence window: {win_start.isoformat()} -> {win_end.isoformat()}",
    ]
    if reference_period:
        lines.append(f"  * Period the claim is about: {reference_period}")
    lines += [
        "  * Judge veracity AS OF the utterance date: a claim accurate when made is not"
        " false merely because reality changed afterward (note if later revised).",
        "  * A referenced date being past your training cutoff is NOT grounds for"
        " UNVERIFIABLE. Any date <= today has either happened or not; contemporary dated"
        " reporting is primary evidence, not speculation. Reserve UNVERIFIABLE for what"
        " evidence genuinely cannot settle (private/undisclosed facts, or dates > today).",
    ]
    return "\n".join(lines) + "\n\n"
