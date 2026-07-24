"""Era gates for evidence packs and verdict rationales (P67.5 / remediation T1.1).

The 2026-07-21 audit showed the coded evidence window (``expected_claim_window``:
~2y before the speech to speech-month+3) admitting post-utterance world-state:
Trump SOTU packs carried items dated to 2026-05-01, and shipped rationales
falsified claims using events that happened *after* the speaker spoke (the
Iran-war gas-price surge, the later shutdown resolution). "Judge as of the
utterance date" does not survive packs containing later world-state.

Policy (jackie, 2026-07-21): evidence a verdict may rest on must be observable
by the audience shortly after the speech — the speaker's **fair-game window**,
utterance date + 7 days. The originally-coded window remains checked as the
outer retrieval bound; a violation of either is reported, and anything dated
after the fair-game window is cited in exactly those terms.

Three layers:

* ``build_evidence_pack`` filters dated items to the fair-game window at
  retrieval, so new packs never violate.
* ``lint_pack_items`` re-asserts the invariant at publish time — the build
  FAILS on violations (defense in depth behind the filter).
* ``lint_rationale`` flags reasoning text that cites dates after the
  fair-game window as candidate post-utterance world-state; flagged claims
  route to re-run (a reviewer pass decides — a rationale may legitimately
  mention a future *target* date the claim itself names).

Undated items still pass the pack filter (a date-less item cannot be
adjudicated by any window) but are counted in lint reports so reviewers see
how much of a pack is un-linted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Iterable, Optional

FAIR_GAME_DAYS = 7

# Historical-era policy (jackie, 2026-07-24; wiki projects:truthbot:
# historical-era-design). Speeches uttered before this date are "pre-web":
# the contemporaneous web record starts ~1994-1997 (major outlets + gov
# statistical sites), so from Clinton's 2nd term onward strict mode holds.
PRE_WEB_CUTOFF = date(1997, 1, 1)

# Future-tense markers → the claim is (heuristically) a PREDICTION and keeps
# strict era mode even pre-web: retrospective sources judge "There will be no
# recession" (Nixon 1974) with hindsight — exactly what fair-game prevents.
# A real Layer A `prediction` claim_type is future work; this is documented
# as a conservative text heuristic, not semantics.
_PREDICTIVE_RX = re.compile(r"\b(will|shall|won't|going to)\b", re.IGNORECASE)


def is_pre_web(utterance: Optional[date]) -> bool:
    """True when the speech predates the contemporaneous web record."""
    return utterance is not None and utterance < PRE_WEB_CUTOFF


def is_predictive_claim(text: str) -> bool:
    """Heuristic: does the claim assert future world-state? (See module note —
    predictions keep strict era mode so hindsight can't judge them.)"""
    return bool(_PREDICTIVE_RX.search(text or ""))


def era_mode_for(utterance: Optional[date], claim_text: str = "") -> str:
    """"lenient" for pre-web non-predictive claims, else "strict"."""
    if is_pre_web(utterance) and not is_predictive_claim(claim_text):
        return "lenient"
    return "strict"


def fair_game_end(utterance: date) -> date:
    """Last date the audience could observe and still call it the speech's
    era: utterance + 7 days (the speaker's fair-game window)."""
    return utterance + timedelta(days=FAIR_GAME_DAYS)


class EraLintError(RuntimeError):
    """A pack reached publish with items dated past the fair-game window."""


@dataclass(frozen=True)
class EraViolation:
    sid: str
    pack_id: str            # E<n> when known, else the source URL
    item_date: date
    message: str


@dataclass(frozen=True)
class RationaleFlag:
    sid: str
    cited_date: date
    excerpt: str            # the reasoning text around the date hit
    message: str


@dataclass
class ArtifactLintReport:
    """Per-artifact era-lint result: what must re-run before re-publish."""
    speech_id: str = ""
    utterance: Optional[date] = None
    pack_violations: list[EraViolation] = field(default_factory=list)
    rationale_flags: list[RationaleFlag] = field(default_factory=list)
    undated_items: int = 0
    dated_items: int = 0

    @property
    def rerun_sids(self) -> list[str]:
        """Claims routed to re-run: any pack violation or rationale flag."""
        sids = {v.sid for v in self.pack_violations}
        sids.update(f.sid for f in self.rationale_flags)
        return sorted(sids)


# ── item-date extraction ─────────────────────────────────────────────────────

_SNIPPET_DATE_RE = re.compile(r"^\[(\d{4})-(\d{2})-(\d{2})\]")


def item_date(published_at: Any, snippet: str = "") -> Optional[date]:
    """Best-known publication date for a pack item.

    Prefers a real ``published_at`` (date / datetime / ISO string); falls back
    to the ``[YYYY-MM-DD]`` prefix the connectors stamp into snippets —
    pre-fix artifacts carry the date ONLY there (published_at was dropped in
    serialization until P67.5)."""
    if published_at is not None:
        if isinstance(published_at, date) and not hasattr(published_at, "hour"):
            return published_at
        if hasattr(published_at, "date"):
            return published_at.date()
        if isinstance(published_at, str) and published_at:
            try:
                return date.fromisoformat(published_at[:10])
            except ValueError:
                pass
    m = _SNIPPET_DATE_RE.match(snippet or "")
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def _violation_message(d: date, utterance: date,
                       window: Optional[tuple[date, date]]) -> Optional[str]:
    """Compose the violation message, checking BOTH policies: the originally
    coded retrieval window stays asserted, and anything past the fair-game
    window is cited in those terms."""
    problems: list[str] = []
    if window is not None:
        start, end = window
        if not (start <= d <= end):
            problems.append(
                f"outside the coded evidence window {start} … {end}")
    fg = fair_game_end(utterance)
    if d > fg:
        problems.append(
            f"observed after the speaker's fair-game window "
            f"(utterance {utterance} + {FAIR_GAME_DAYS} days = {fg})")
    if not problems:
        return None
    return f"dated {d}: " + "; ".join(problems)


# ── pack lint ────────────────────────────────────────────────────────────────

def lint_pack_items(sid: str, items: Iterable[Any], utterance: date,
                    window: Optional[tuple[date, date]] = None,
                    ) -> tuple[list[EraViolation], int, int]:
    """Lint pack items (PackItem objects or artifact evidence dicts).

    Returns (violations, dated_count, undated_count)."""
    violations: list[EraViolation] = []
    dated = undated = 0
    for i, it in enumerate(items, start=1):
        if isinstance(it, dict):
            pub, snippet = it.get("published_at"), it.get("snippet") or ""
            pack_id = it.get("pack_id") or f"E{i}"
        else:
            pub = getattr(it, "published_at", None)
            snippet = getattr(it, "snippet", "") or ""
            pack_id = getattr(it, "pack_id", f"E{i}")
        d = item_date(pub, snippet)
        if d is None:
            undated += 1
            continue
        dated += 1
        msg = _violation_message(d, utterance, window)
        if msg:
            violations.append(EraViolation(
                sid=sid, pack_id=pack_id, item_date=d, message=msg))
    return violations, dated, undated


def assert_pack_within_era(pack, utterance: Optional[date],
                           era_mode: str = "strict") -> None:
    """Publish-time gate (T1.1: build fails on violations). No-op when the
    utterance date is unknown — there is no era to violate — or when the pack
    was built under the historical-era LENIENT policy, where retrospective
    items are admitted by design (the consolidator ranks them behind
    contemporaneous sources instead of dropping them)."""
    if utterance is None or era_mode == "lenient":
        return
    violations, _, _ = lint_pack_items(
        pack.sid, pack.items, utterance, window=pack.window)
    if violations:
        detail = "; ".join(f"{v.pack_id} {v.message}" for v in violations)
        raise EraLintError(
            f"evidence pack for {pack.sid} violates the era policy: {detail}")


# ── rationale lint ───────────────────────────────────────────────────────────

_MONTHS = ("January February March April May June July August September "
           "October November December").split()
_MONTH_NUM = {m: i + 1 for i, m in enumerate(_MONTHS)}

_ISO_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_PROSE_RE = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\s+(?:(\d{1,2}),\s*)?(20\d{2})\b")

_EXCERPT_RADIUS = 60


def _dates_in_text(text: str) -> list[tuple[date, str]]:
    """(date, excerpt) for every calendar-locatable date mention. A bare
    'Month YYYY' resolves to the 1st — conservative for the after-window
    test (flags only when even the month's first day is past the window)."""
    hits: list[tuple[date, str]] = []
    for m in _ISO_RE.finditer(text):
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
        hits.append((d, text[max(0, m.start() - _EXCERPT_RADIUS):
                             m.end() + _EXCERPT_RADIUS].strip()))
    for m in _PROSE_RE.finditer(text):
        month = _MONTH_NUM[m.group(1)]
        day = int(m.group(2)) if m.group(2) else 1
        try:
            d = date(int(m.group(3)), month, day)
        except ValueError:
            continue
        hits.append((d, text[max(0, m.start() - _EXCERPT_RADIUS):
                             m.end() + _EXCERPT_RADIUS].strip()))
    return hits


def lint_rationale(sid: str, reasoning: str, utterance: date) -> list[RationaleFlag]:
    """Flag reasoning that cites dates after the fair-game window as
    candidate post-utterance world-state. Flags are review inputs, not
    verdict changes: the claim routes to re-run / reviewer pass (T1.1)."""
    fg = fair_game_end(utterance)
    flags: list[RationaleFlag] = []
    for d, excerpt in _dates_in_text(reasoning or ""):
        if d > fg:
            flags.append(RationaleFlag(
                sid=sid, cited_date=d, excerpt=excerpt,
                message=(f"rationale cites {d} — observed after the "
                         f"speaker's fair-game window (utterance {utterance} "
                         f"+ {FAIR_GAME_DAYS} days = {fg})")))
    return flags


# ── artifact lint (historical runs → re-run routing) ─────────────────────────

def lint_artifact(artifact: dict) -> ArtifactLintReport:
    """Era-lint a persisted pca_runs artifact: every pack item + every row
    rationale. The report's ``rerun_sids`` is the Phase 1 re-run queue."""
    from truthbot.verdict import speech_context
    from truthbot.verify.context.terms import expected_claim_window

    meta = artifact.get("meta") or {}
    report = ArtifactLintReport(speech_id=meta.get("speech_id", ""))
    utt: Optional[date] = None
    if meta.get("date"):
        try:
            utt = date.fromisoformat(str(meta["date"])[:10])
        except ValueError:
            utt = None
    if utt is None and meta.get("speech_id"):
        utt = speech_context.speech_date_for(f"{meta['speech_id']}:0000")
    report.utterance = utt
    if utt is None:
        return report

    window = None
    try:
        window = expected_claim_window(utt)
    except Exception:  # pragma: no cover — defensive
        window = None

    for sid, evs in (artifact.get("evidence") or {}).items():
        violations, dated, undated = lint_pack_items(sid, evs, utt, window=window)
        report.pack_violations.extend(violations)
        report.dated_items += dated
        report.undated_items += undated

    for row in artifact.get("rows") or []:
        sid = row.get("item_id") or row.get("sid") or ""
        report.rationale_flags.extend(
            lint_rationale(sid, row.get("reasoning") or "", utt))
    return report
