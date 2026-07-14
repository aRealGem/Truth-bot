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
SPEECH_DATE: dict[str, date] = {
    "biden_2022": date(2022, 3, 1),    # Biden SOTU, 2022-03-01
    "trump_2026": date(2026, 2, 24),   # Trump SOTU, 2026-02-24
}


def speech_date_for(sid: str) -> Optional[date]:
    """Utterance date for a claim sid ('<speech>:<offset>'), or None if unknown."""
    return SPEECH_DATE.get(sid.split(":", 1)[0]) if sid else None


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
