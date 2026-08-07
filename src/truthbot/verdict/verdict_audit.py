"""Standing agreed-verdict audit — deterministic lints + model-pass selection
(remediation v2, 1.12).

The 2026-07-21 external audit's F8 class: when proposer and critic agree, a
claim is never escalated and never reviewed, so a SHARED misread (units,
baseline, tense, invented referent) ships as a confident verdict. The one-off
model harness (``scripts/audit_agreed_verdicts.py``) caught 17 such revisions;
this module makes the audit a STANDING pipeline stage in two tiers:

* **Deterministic lints** (this module, $0, every run): pure text checks over
  claim/rationale/evidence, each conservative by design (precision over
  recall — a lint that cries wolf gets ignored). Findings attach to row
  provenance as ``audit_flags``; a ``queue``-action finding additionally sets
  ``audit_queue`` and lands the row in the re-adjudication queue
  (``metrics/audits/readjudication_queue.jsonl``) for HUMAN approval. No
  finding ever triggers a model call — zero-unauthorized-spend is a hard
  project rule.

* **Model pass** (Phase 3, spend-gated, NOT implemented here): the selection
  contract is encoded as :func:`select_model_audit_rows` — every row with a
  non-empty CRM-114 override (``crm114.final``), every evidence-gate-forced
  row, plus a seeded random sample of the remaining decided rows. Phase 3
  consumes that selection; nothing in this module talks to a model.

Selection contract for the deterministic tier: ALL decided non-split rows
(verdict in TRUE/FALSE/MISLEADING, ``split`` false) are linted on every run.

Every lint is a pure function ``lint_<name>(claim_text, reasoning,
evidence_items, utterance) -> AuditFinding | None`` (``evidence_items``
accepts artifact evidence dicts or ``PackItem`` objects;
``lint_invented_referent`` additionally takes the claim's transcript context,
keyword-only, because its corpus is claim + context + evidence).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Optional

# A6: ONE superlative list for the whole codebase. The shape lint owns it;
# this module imports it. Do not fork a second copy here.
from truthbot.checkworthy.shape_lint import (  # noqa: F401  (re-exported)
    SUPERLATIVE_RX,
    SUPERLATIVE_TOKENS,
    has_superlative,
)

# Same "decided" vocabulary as the one-off model harness (prior art): the
# agreed-verdict problem class is a CONFIDENT published call. UNVERIFIABLE is
# an abstention — audited only via the gate-forced arm of the model-pass
# selection, never by the measure lints.
DECIDED = {"TRUE", "FALSE", "MISLEADING"}
ADVERSE = {"FALSE", "MISLEADING"}

QUEUE = "queue"
FLAG = "flag"


@dataclass(frozen=True)
class AuditFinding:
    """One deterministic-lint finding on one row."""

    lint: str
    detail: str
    action: str  # "flag" (disclose) | "queue" (re-adjudication queue)


# ── shared text helpers ──────────────────────────────────────────────────────

def _ev_texts(evidence_items: Optional[Iterable[Any]]) -> list[str]:
    """Snippet + source-name text from artifact evidence dicts or PackItems."""
    out: list[str] = []
    for it in evidence_items or []:
        if isinstance(it, dict):
            out.extend(str(it.get(k) or "") for k in ("snippet", "source_name"))
        else:
            out.extend(str(getattr(it, k, "") or "")
                       for k in ("snippet", "source_name"))
    return [t for t in out if t]


# ── citation → pack-item resolution (A6) ─────────────────────────────────────
#
# ``rows[].citations`` are E-refs ("E4") addressing the claim's evidence pack.
# In-process ``PackItem`` objects carry their own ``pack_id``; the published
# artifact stores a plain LIST whose ORDER is the pack order and whose ``id``
# is a content uuid, not an E-ref. Resolution therefore prefers a real
# ``pack_id`` and falls back to 1-based position — the addressing the pack
# builder assigns (``pack_id=f"E{i}"``).

_EREF_RX = re.compile(r"^E(\d+)$", re.IGNORECASE)


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def _pack_id(item: Any) -> str:
    """The item's own E-ref, or "" when it has none (artifact dicts)."""
    raw = _field(item, "pack_id") or _field(item, "id") or ""
    raw = str(raw).strip().upper()
    return raw if _EREF_RX.match(raw) else ""


def cited_items(evidence_items: Optional[Iterable[Any]],
                citations: Optional[Iterable[str]]) -> list[Any]:
    """The pack items a row actually cited, in citation order.

    Unresolvable refs are dropped silently — a citation outside the pack is
    an I4 violation caught at adjudication, not an audit-lint concern."""
    items = list(evidence_items or [])
    by_ref = {pid: it for it in items if (pid := _pack_id(it))}
    out: list[Any] = []
    for ref in citations or []:
        key = str(ref).strip().upper()
        if key in by_ref:
            out.append(by_ref[key])
            continue
        m = _EREF_RX.match(key)
        if m and 1 <= int(m.group(1)) <= len(items):
            out.append(items[int(m.group(1)) - 1])
    return out


def _strip_commas_in_numbers(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d)", "", text)


_NUMERAL_RX = re.compile(r"\d+(?:\.\d+)?")


def _numerals(text: str) -> set[str]:
    """Normalized numerals ('3.9', '357', ...) present in the text."""
    return set(_NUMERAL_RX.findall(_strip_commas_in_numbers(text)))


# Measure-token categories. Extraction from the CLAIM and engagement checks on
# the RATIONALE both go through these patterns, so a category read one way is
# recognized the same way on the other side.
_PP_RX = re.compile(r"\bpercentage[- ]points?\b", re.IGNORECASE)
_PCT_RX = re.compile(r"%|\bpercent(?:age)?\b|\bpct\b", re.IGNORECASE)
_ANNUAL_RX = re.compile(
    r"\bannual(?:ized|ly)?\b|\byearly\b|\bper year\b|\byear[- ]over[- ]year\b",
    re.IGNORECASE)
# "quarter" alone is colloquial ("three quarters of Americans"); only the
# calendar-quarter readings count.
_QUARTER_RX = re.compile(
    r"\bquarterly\b|\bQ[1-4]\b"
    r"|\b(?:first|second|third|fourth)[- ]quarter\b"
    r"|\bquarter of (?:19|20)\d{2}\b"
    r"|\blast three months\b|\bpast three months\b",
    re.IGNORECASE)
_RATE_RX = re.compile(r"\brates?\b", re.IGNORECASE)
# "level" is polysemous; institutional collocations don't count as a measure.
_LEVEL_RX = re.compile(
    r"(?<!federal )(?<!state )(?<!local )(?<!national )(?<!cabinet )"
    r"(?<!sea )(?<!every )\blevels?\b",
    re.IGNORECASE)
# "real" only as the economic modifier; bare "real" ("a real problem") is not
# a measure token.
_REAL_RX = re.compile(
    r"\breal\b(?=[- ](?:wages?|incomes?|earnings|GDP|gross|dollars?|terms|"
    r"growth|spending|value))|\binflation[- ]adjusted\b",
    re.IGNORECASE)
_NOMINAL_RX = re.compile(r"\bnominal\b", re.IGNORECASE)
_MEDIAN_RX = re.compile(r"\bmedian\b", re.IGNORECASE)
# "mean" is deliberately absent — as a verb it dominates political prose
# ("used to mean that…"), and calibration showed it queueing sound rows.
_AVG_RX = re.compile(r"\baverage\b|\bavg\b", re.IGNORECASE)
_PER_CAPITA_RX = re.compile(r"\bper[- ]capita\b|\bper person\b", re.IGNORECASE)

_MEASURE_CATEGORIES: dict[str, re.Pattern] = {
    "percentage_point": _PP_RX,
    "percent": _PCT_RX,
    "annual": _ANNUAL_RX,
    "quarterly": _QUARTER_RX,
    "rate": _RATE_RX,
    "level": _LEVEL_RX,
    "real": _REAL_RX,
    "nominal": _NOMINAL_RX,
    "median": _MEDIAN_RX,
    "average": _AVG_RX,
    "per_capita": _PER_CAPITA_RX,
}

# number+unit bigrams ("1.7 percent", "$357 billion", "400,000 veterans" is
# NOT one — the unit list is closed to magnitude/percent words).
_NUM_UNIT_RX = re.compile(
    r"\$?\d[\d,]*(?:\.\d+)?\s*(?:%|percent(?:age[- ]points?)?\b"
    r"|percentage[- ]points?\b|million\b|billion\b|trillion\b|thousand\b)",
    re.IGNORECASE)


# Categories too polysemous to carry a queue action on their own ("high
# levels", "at any rate", "real change", "on average"): they count as a
# stated measure only when the claim also quantifies (any numeral). The
# unambiguous categories count alone.
_WEAK_CATEGORIES = {"rate", "level", "average", "real"}


def claim_measure_tokens(claim_text: str) -> tuple[set[str], set[str]]:
    """(measure categories, numerals from number+unit bigrams) in a claim.
    Weak (polysemous) categories are only extracted from claims that also
    carry a numeral — see ``_WEAK_CATEGORIES``."""
    text = claim_text or ""
    cats = {name for name, rx in _MEASURE_CATEGORIES.items() if rx.search(text)}
    if not _numerals(text):
        cats -= _WEAK_CATEGORIES
    nums: set[str] = set()
    for m in _NUM_UNIT_RX.finditer(_strip_commas_in_numbers(text)):
        nums.update(_NUMERAL_RX.findall(m.group(0)))
    return cats, nums


# ── lints ────────────────────────────────────────────────────────────────────

def lint_measure_alignment(claim_text: str, reasoning: str,
                           evidence_items=None,
                           utterance: Optional[date] = None,
                           ) -> Optional[AuditFinding]:
    """QUEUE (the hard rule): the rationale must engage the claim's stated
    measure. Fires only when the claim carries >=1 measure token and the
    rationale contains NONE of them, no recognized equivalent (any measure
    category at all, or any number+unit bigram — the specific-mismatch lints
    below handle wrong-measure reasoning), and none of the claim's numerals.
    An empty rationale is skipped (nothing to lint)."""
    reasoning = (reasoning or "").strip()
    if not reasoning:
        return None
    cats, unit_nums = claim_measure_tokens(claim_text or "")
    if not cats and not unit_nums:
        return None
    # Engagement, most-specific first: a shared category …
    for name in cats:
        if _MEASURE_CATEGORIES[name].search(reasoning):
            return None
    # … a recognized equivalent: the rationale reasons in SOME measure
    # vocabulary or quantity (wrong-measure cases belong to the specific
    # flag lints, not the queue) …
    for rx in _MEASURE_CATEGORIES.values():
        if rx.search(reasoning):
            return None
    if _NUM_UNIT_RX.search(_strip_commas_in_numbers(reasoning)):
        return None
    # … or any numeral the claim itself states.
    r_nums = _numerals(reasoning)
    if _numerals(claim_text or "") & r_nums:
        return None
    return AuditFinding(
        "measure_alignment",
        f"claim states a measure ({', '.join(sorted(cats)) or 'number+unit'}"
        f"{', ' + '/'.join(sorted(unit_nums)) if unit_nums else ''}) but the "
        "rationale engages no measure token, equivalent, or claim numeral",
        QUEUE)


def lint_pct_vs_pp(claim_text: str, reasoning: str, evidence_items=None,
                   utterance: Optional[date] = None) -> Optional[AuditFinding]:
    """FLAG: claim speaks in percent while the rationale reasons in percentage
    points around a numeral the claim states (or vice versa)."""
    c, r = claim_text or "", reasoning or ""
    shared = _numerals(c) & _numerals(r)
    if not shared:
        return None
    c_no_pp, r_no_pp = _PP_RX.sub(" ", c), _PP_RX.sub(" ", r)
    c_pct, c_pp = bool(_PCT_RX.search(c_no_pp)), bool(_PP_RX.search(c))
    r_pct, r_pp = bool(_PCT_RX.search(r_no_pp)), bool(_PP_RX.search(r))
    if c_pct and not c_pp and r_pp and not r_pct:
        way = "claim states percent; rationale reasons in percentage points"
    elif c_pp and not c_pct and r_pct and not r_pp:
        way = "claim states percentage points; rationale reasons in percent"
    else:
        return None
    return AuditFinding(
        "pct_vs_pp", f"{way} around shared numeral(s) {sorted(shared)}", FLAG)


def _exclusive_cooccurrence(name: str, claim_text: str, reasoning: str,
                            rx_a: re.Pattern, label_a: str,
                            rx_b: re.Pattern, label_b: str,
                            ) -> Optional[AuditFinding]:
    """FLAG when claim carries exactly one of two measure classes and the
    rationale carries EXCLUSIVELY the other (a rationale that mentions both
    classes is engaging the distinction, not confusing it)."""
    c_a, c_b = bool(rx_a.search(claim_text)), bool(rx_b.search(claim_text))
    r_a, r_b = bool(rx_a.search(reasoning)), bool(rx_b.search(reasoning))
    if c_a and not c_b and r_b and not r_a:
        detail = f"claim speaks in {label_a}; rationale reasons only in {label_b}"
    elif c_b and not c_a and r_a and not r_b:
        detail = f"claim speaks in {label_b}; rationale reasons only in {label_a}"
    else:
        return None
    return AuditFinding(name, detail, FLAG)


def lint_quarterly_vs_annual(claim_text: str, reasoning: str,
                             evidence_items=None,
                             utterance: Optional[date] = None,
                             ) -> Optional[AuditFinding]:
    return _exclusive_cooccurrence(
        "quarterly_vs_annual", claim_text or "", reasoning or "",
        _QUARTER_RX, "quarterly terms", _ANNUAL_RX, "annual terms")


def lint_rate_vs_level(claim_text: str, reasoning: str, evidence_items=None,
                       utterance: Optional[date] = None,
                       ) -> Optional[AuditFinding]:
    return _exclusive_cooccurrence(
        "rate_vs_level", claim_text or "", reasoning or "",
        _RATE_RX, "a rate", _LEVEL_RX, "a level")


def lint_nominal_vs_real(claim_text: str, reasoning: str, evidence_items=None,
                         utterance: Optional[date] = None,
                         ) -> Optional[AuditFinding]:
    return _exclusive_cooccurrence(
        "nominal_vs_real", claim_text or "", reasoning or "",
        _REAL_RX, "real (inflation-adjusted) terms",
        _NOMINAL_RX, "nominal terms")


# Baseline anchors ("-class": took/take/came into/entered/assumed office, and
# the N-years-ago family).
_OFFICE_ANCHOR_RX = re.compile(
    r"\b(?:since|when|before|after|as)\s+(?:I|we)\s+"
    r"(?:took|take|came\s+into|come\s+into|entered|enter|assumed|assume)\s+"
    r"office\b",
    re.IGNORECASE)
_YEARS_AGO_RX = re.compile(r"\b(two|three|four|five|six|seven|eight|\d)\s+"
                           r"years\s+ago\b", re.IGNORECASE)
_YEARS_AGO_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                    "seven": 7, "eight": 8}
_YEAR_RX = re.compile(r"\b(?:19|20)\d{2}\b")
# A rationale that VERBALLY engages the office anchor ("when Clinton took
# office", "pre-Biden baseline") is reasoning about the right baseline even
# if it names no year — calibration showed flagging those is noise.
_OFFICE_ENGAGED_RX = re.compile(
    r"\b(?:took|taking|taken|came?\s+into|entered|assumed)\s+office\b"
    r"|\bpre-\w+\s+baseline\b|\binaugurat", re.IGNORECASE)


def lint_baseline_selection(claim_text: str, reasoning: str,
                            evidence_items=None,
                            utterance: Optional[date] = None,
                            ) -> Optional[AuditFinding]:
    """FLAG: the claim anchors its comparison on taking office (or "N years
    ago") but the rationale's named years include neither a plausible
    term-start year nor the year before it (the fiscal/calendar baseline as
    the term begins). Term starts derive from the utterance year: US
    inaugurations fall on years ≡ 1 (mod 4); both the current and the prior
    inauguration within 8 years are accepted (first- vs second-term "took
    office" is not text-derivable). A rationale naming NO years is skipped —
    there is no comparison year to check."""
    if utterance is None:
        return None
    c = claim_text or ""
    office = _OFFICE_ANCHOR_RX.search(c)
    ago = _YEARS_AGO_RX.search(c)
    if not office and not ago:
        return None
    if _OFFICE_ENGAGED_RX.search(reasoning or ""):
        return None
    r_years = {int(y) for y in _YEAR_RX.findall(reasoning or "")}
    if not r_years:
        return None
    anchors: set[int] = set()
    if office:
        inaugs = [y for y in range(utterance.year - 7, utterance.year + 1)
                  if y % 4 == 1]
        for y in inaugs:
            anchors.update({y, y - 1})
    if ago:
        raw = ago.group(1).lower()
        n = _YEARS_AGO_WORDS.get(raw, None)
        if n is None:
            try:
                n = int(raw)
            except ValueError:
                n = None
        if n:
            # ±1 year of slack: "four years ago" in a Feb speech legitimately
            # reads as either the calendar or the fiscal year either side.
            anchors.update({utterance.year - n - 1, utterance.year - n,
                            utterance.year - n + 1})
    if not anchors or r_years & anchors:
        return None
    return AuditFinding(
        "baseline_selection",
        f"claim anchors on {(office or ago).group(0)!r} but rationale years "
        f"{sorted(r_years)} include no anchor year {sorted(anchors)}",
        FLAG)


# Colloquial-recency literalism (audit error class d). The orchestrator runs
# this lint on ADVERSE rows only — the class is "hangs an ADVERSE verdict on
# recency words"; the lint itself is pure text.
_RECENCY_CLAIM_RX = re.compile(
    r"\b(?:recently|lately|these days|just now|just)\b", re.IGNORECASE)
_LITERALIST_RX = re.compile(
    r"\bnot\s+recent\b|\bhardly\s+recent\b|\bno\s+longer\s+recent\b"
    r"|\bmonths\s+(?:earlier|before|ago|prior)\b"
    r"|\bover\s+a\s+year\s+(?:ago|earlier|before)\b"
    r"|\byears?\s+(?:earlier|ago|before)\b",
    re.IGNORECASE)
_RECENT_WORD_RX = re.compile(r"\brecen(?:t|tly|cy)\b|\bjust\b", re.IGNORECASE)


def lint_colloquial_recency(claim_text: str, reasoning: str,
                            evidence_items=None,
                            utterance: Optional[date] = None,
                            ) -> Optional[AuditFinding]:
    """FLAG: the claim uses a colloquial recency word and the rationale both
    engages the recency reading AND measures the gap literally ("months
    earlier", "not recent", …) — the shape of falsifying "recently" over a
    span the word colloquially covers. Reuses era_lint's date extraction to
    name the earliest date the rationale cites (detail only)."""
    c, r = claim_text or "", reasoning or ""
    if not _RECENCY_CLAIM_RX.search(c):
        return None
    if not (_LITERALIST_RX.search(r) and _RECENT_WORD_RX.search(r)):
        return None
    cited = ""
    try:  # era_lint helper (usable prior art): dates the rationale cites.
        from truthbot.verdict.era_lint import _dates_in_text
        hits = sorted(d for d, _ in _dates_in_text(r))
        if hits:
            cited = f"; rationale cites {hits[0].isoformat()}"
    except Exception:  # pragma: no cover — detail enrichment only
        pass
    return AuditFinding(
        "colloquial_recency",
        "claim uses a colloquial recency word; rationale reads it literally "
        f"({_LITERALIST_RX.search(r).group(0)!r}){cited}",
        FLAG)


# Invented referents (audit error class e). Conservative: capitalized runs of
# >=2 words in the RATIONALE, absent as a phrase from claim+context+evidence,
# with at least one distinctive component token absent too. A hyphen may only
# join two CAPITALIZED parts ("Wal-Mart"); compound modifiers like
# "Obama-launched" / "Dec-2013" / "NSF-funded" break the run — calibration
# showed they are rationale shorthand, not referents.
_CAP_WORD = r"[A-Z][A-Za-z’']*(?:-[A-Z][A-Za-z’']*)*"
_CAP_RUN_RX = re.compile(rf"\b{_CAP_WORD}(?:\s+{_CAP_WORD})+\b")
_SENTENCE_START_RX = re.compile(r"(?:^|[.!?]\s+|[:;]\s+|[\"“]\s*)$")
_REFERENT_STOPWORDS = {
    # months / weekdays / common sentence furniture
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday",
    "the", "a", "an", "in", "on", "of", "and", "for", "but", "this", "that",
    # ubiquitous civic vocabulary — never distinctive enough to flag on
    "united", "states", "america", "american", "americans", "congress",
    "senate", "house", "white", "president", "vice", "state", "union",
    "federal", "government", "national", "washington", "u.s.", "us",
    "democrats", "republicans", "administration", "act", "party",
    "department", "dept", "agency", "office", "committee", "commission",
}


def lint_invented_referent(claim_text: str, reasoning: str,
                           evidence_items=None,
                           utterance: Optional[date] = None,
                           *, claim_context: str = "",
                           ) -> Optional[AuditFinding]:
    """FLAG: the rationale invokes a proper-noun phrase found nowhere in the
    claim text, the claim's transcript context, or any evidence snippet /
    source name. Skips sentence-initial capitalization, month names, and
    civic-vocabulary phrases; requires at least one distinctive token (>3
    chars, non-stopword) to be individually absent from the corpus."""
    r = reasoning or ""
    corpus = " ".join([claim_text or "", claim_context or "",
                       *_ev_texts(evidence_items)]).lower()
    missing: list[str] = []
    for m in _CAP_RUN_RX.finditer(r):
        tokens = m.group(0).split()
        # Sentence-initial first word carries no proper-noun signal — drop it
        # and require the remainder to still be a multi-word phrase.
        if _SENTENCE_START_RX.search(r[:m.start()]):
            tokens = tokens[1:]
        if len(tokens) < 2:
            continue
        phrase = " ".join(tokens)
        low = [t.lower().rstrip("’s").rstrip("'s") for t in tokens]
        if all(t in _REFERENT_STOPWORDS for t in low):
            continue
        if phrase.lower() in corpus:
            continue
        # Head-token grounding: if the phrase's final token is itself in the
        # corpus, the referent is anchored even when the full phrase is not —
        # "Cory Remsburg" in a rationale is grounded by evidence that says
        # "Remsburg's"; only the phrase's head going missing signals invention.
        if low[-1] in corpus:
            continue
        distinctive_missing = [
            t for t in low
            if len(t) > 3 and t not in _REFERENT_STOPWORDS and t not in corpus]
        if not distinctive_missing:
            continue
        missing.append(phrase)
    if not missing:
        return None
    shown = ", ".join(dict.fromkeys(missing[:3]))
    return AuditFinding(
        "invented_referent",
        f"rationale invokes referent(s) absent from claim/context/evidence: "
        f"{shown}",
        FLAG)


# ── superlative anti-gaming (A6 / D11 rev-B) ─────────────────────────────────
#
# The gaming vector the shape lint closes on the CLAIM side has a mirror on the
# VERDICT side: a superlative ("record numbers", "largest decline in recorded
# history") is a claim about a whole distribution, and the one class of source
# that can never establish it is the speaker's own press shop asserting it.
# When a superlative claim ships DECIDED on citations that are exclusively
# SELF/S5 — POLITICAL tier and/or an evidential role from the D11.2 SELF
# column — or exclusively preliminary/projected data, the verdict rests on the
# assertion it was supposed to test.
#
# Signed-off effect (D11 anti-gaming rule): force C-EVAL×SELF handling
# (attribution-only, weight 0) or escalate to the arbiter. In this
# DETERMINISTIC pass the effect is an ``action="queue"`` finding — the row
# lands in the human re-adjudication queue. Nothing here calls a model or
# re-adjudicates on its own: zero-unauthorized-spend is a standing project
# rule, and "escalate to arbiter" is a spend decision a human makes.

#: Evidential roles from the D11.2 SELF column (``verdict.evidential_role``).
#: CORROBORANT (PARTICIPANT) and NORMAL are deliberately absent — a
#: participant's record is independent enough to bear on a superlative.
SELF_ROLES = {"primary-record", "plain-s5", "attribution-only"}

#: The S5 tier value (``SourceTier.POLITICAL``), lowercased for comparison.
SELF_TIER = "political"

# Provisionality markers. Conservative on purpose: a superlative resting on a
# figure the source itself calls provisional is the trump_2026:0023 shape
# ("CCJ PROJECTS…", "FBI confirmation NOT YET AVAILABLE"). Hedges of degree
# ("potentially", "likely") are excluded — they qualify the superlative, not
# the vintage of the data, and calibration showed them queueing sound rows.
_PRELIMINARY_RX = re.compile(
    r"\bpreliminary\b|\bprovisional\b|\bunaudited\b"
    r"|\bproject(?:s|ed|ing|ion|ions)\b|\bon\s+pace\b|\bon\s+track\s+to\b"
    r"|\bpremature\b|\badvance\s+estimate\b|\binitial\s+estimate\b"
    r"|\bsubject\s+to\s+revision\b"
    r"|\bnot\s+yet\s+(?:available|confirmed|final|finalized|released|reported)\b"
    r"|\bif\s+(?:the\s+)?\w+(?:\s+\w+){0,2}\s+confirms?\b",
    re.IGNORECASE)


def _is_self_s5(item: Any) -> bool:
    """POLITICAL tier, or an evidential role from the D11.2 SELF column."""
    tier = _field(item, "source_tier")
    tier = str(getattr(tier, "value", tier) or "").strip().lower()
    role = _field(item, "role")
    role = str(getattr(role, "value", role) or "").strip().lower()
    return tier == SELF_TIER or role in SELF_ROLES


def _is_preliminary(item: Any) -> bool:
    return bool(_PRELIMINARY_RX.search(str(_field(item, "snippet") or "")))


def lint_superlative_self_citation(claim_text: str, reasoning: str,
                                   evidence_items=None,
                                   utterance: Optional[date] = None,
                                   *, citations: Optional[Iterable[str]] = None,
                                   ) -> Optional[AuditFinding]:
    """QUEUE: a superlative claim decided on citations that are EXCLUSIVELY
    the speaker's own record (SELF/S5) and/or preliminary-flagged data.

    Skipped when the claim carries no superlative token
    (:data:`SUPERLATIVE_TOKENS`, shared with the shape lint) or when the row
    cited nothing — an uncited row is an abstention or a gate-forced
    Unverifiable, both of which belong to other arms of the audit. One
    independent, final-vintage citation is enough to clear the lint: the rule
    targets verdicts with NO non-self, non-provisional support at all."""
    if not has_superlative(claim_text or ""):
        return None
    cited = cited_items(evidence_items, citations)
    if not cited:
        return None
    self_refs: list[str] = []
    prelim_refs: list[str] = []
    for n, item in enumerate(cited, 1):
        ref = _pack_id(item) or f"#{n}"
        if _is_self_s5(item):
            self_refs.append(ref)
        elif _is_preliminary(item):
            prelim_refs.append(ref)
        else:
            return None  # an independent, final-vintage citation clears it
    parts = []
    if self_refs:
        parts.append(f"{len(self_refs)} self/S5 ({', '.join(self_refs)})")
    if prelim_refs:
        parts.append(f"{len(prelim_refs)} preliminary ({', '.join(prelim_refs)})")
    return AuditFinding(
        "superlative_self_citation",
        f"superlative claim decided on {len(cited)} citation(s), all "
        f"non-probative for a superlative: {'; '.join(parts)} — D11 rev-B "
        "requires C-EVAL×SELF handling (weight 0) or arbiter escalation",
        QUEUE)


#: (name, fn, adverse_only) — the deterministic tier, in run order.
ALL_LINTS: list[tuple[str, Any, bool]] = [
    ("measure_alignment", lint_measure_alignment, False),
    ("pct_vs_pp", lint_pct_vs_pp, False),
    ("quarterly_vs_annual", lint_quarterly_vs_annual, False),
    ("rate_vs_level", lint_rate_vs_level, False),
    ("nominal_vs_real", lint_nominal_vs_real, False),
    ("baseline_selection", lint_baseline_selection, False),
    ("colloquial_recency", lint_colloquial_recency, True),
    ("invented_referent", lint_invented_referent, False),
    ("superlative_self_citation", lint_superlative_self_citation, False),
]

#: Lints that need more than the four positional arguments. Keyed by function
#: so :func:`run_lints` stays a loop instead of a chain of ``is`` checks.
_EXTRA_KWARGS: dict[Any, tuple[str, ...]] = {
    lint_invented_referent: ("claim_context",),
    lint_superlative_self_citation: ("citations",),
}


# ── orchestration ────────────────────────────────────────────────────────────

def run_lints(claim_text: str, reasoning: str, evidence_items=None,
              utterance: Optional[date] = None, *, claim_context: str = "",
              citations: Optional[Iterable[str]] = None,
              adverse: bool = True) -> list[AuditFinding]:
    """All deterministic lints over one row. ``adverse`` gates the
    adverse-only lints (colloquial_recency); ``citations`` are the row's
    E-refs (needed by superlative_self_citation to resolve which pack items
    the verdict actually leaned on)."""
    extra = {"claim_context": claim_context, "citations": citations}
    findings: list[AuditFinding] = []
    for name, fn, adverse_only in ALL_LINTS:
        if adverse_only and not adverse:
            continue
        kwargs = {k: extra[k] for k in _EXTRA_KWARGS.get(fn, ())}
        f = fn(claim_text, reasoning, evidence_items, utterance, **kwargs)
        if f is not None:
            findings.append(f)
    return findings


def _pack_items(evidence: Optional[dict], sid: str) -> list:
    """Evidence for one sid from either shape: ``sid -> [item dicts]``
    (artifact) or ``sid -> EvidencePack`` (in-process)."""
    if not evidence:
        return []
    ev = evidence.get(sid)
    if ev is None:
        return []
    items = getattr(ev, "items", None)
    if items is not None and not isinstance(ev, (list, tuple)):
        return list(items)
    return list(ev)


def _utterance_for(sid: str) -> Optional[date]:
    from truthbot.verdict import speech_context
    return speech_context.speech_date_for(sid)


def audit_rows(claims: list[dict], rows: list[dict],
               evidence: Optional[dict] = None,
               utterance: Optional[date] = None) -> dict[str, dict]:
    """The deterministic audit stage: lint ALL decided non-split rows.

    Returns ``sid -> {"audit_flags": [lint names], "audit_queue": bool}``
    for every audited row (non-decided / split rows get no entry).
    ``utterance`` may be omitted; it then resolves per-sid through the
    speech-date registry (empty registry → the date-dependent lints skip).
    Pure and offline — no model calls, ever."""
    claim_by_sid = {c["sid"]: c for c in claims or [] if "sid" in c}
    out: dict[str, dict] = {}
    for row in rows or []:
        sid = row.get("sid") or ""
        verdict = str(row.get("verdict") or "").strip().upper()
        if verdict not in DECIDED or row.get("split"):
            continue
        claim = claim_by_sid.get(sid) or {}
        utt = utterance if utterance is not None else _utterance_for(sid)
        findings = run_lints(
            claim.get("text", ""), row.get("reasoning") or "",
            _pack_items(evidence, sid), utt,
            claim_context=claim.get("context", "") or "",
            citations=row.get("citations") or (),
            adverse=verdict in ADVERSE)
        out[sid] = {
            "audit_flags": [f.lint for f in findings],
            "audit_queue": any(f.action == QUEUE for f in findings),
        }
    return out


def agreed_decided_rows(rows: list[dict]) -> list[dict]:
    """The one-off harness's selection, shared: non-escalated decided rows
    (the F8 agreed-verdict population)."""
    return [r for r in rows or []
            if not r.get("escalated")
            and str(r.get("verdict") or "").strip().upper() in DECIDED]


def select_model_audit_rows(rows: list[dict], k: int, seed: int) -> list[dict]:
    """Phase-3 model-pass selection contract (pure — NO model calls here).

    Mandatory coverage: every row with a non-empty CRM-114 override
    (``crm114.final`` — the "auto-FALSE"/auto-adjusted class) and every
    evidence-gate-forced row (``evidence_gate`` / legacy ``provenance_code``).
    Plus a seeded random sample of ``k`` rows from the remaining decided
    non-split rows. Deterministic given (rows order, k, seed); Phase 3 just
    consumes the returned rows — spend gating lives with the caller."""
    import random

    mandatory: list[dict] = []
    pool: list[dict] = []
    for row in rows or []:
        crm_final = str((row.get("crm114") or {}).get("final") or "")
        gate = str(row.get("evidence_gate") or row.get("provenance_code") or "")
        if crm_final or gate:
            mandatory.append(row)
        elif (str(row.get("verdict") or "").strip().upper() in DECIDED
                and not row.get("split")):
            pool.append(row)
    k = max(0, min(int(k), len(pool)))
    sample = random.Random(seed).sample(pool, k) if k else []
    return mandatory + sample
