"""Deterministic claim-shape lint (PR-A2.3 / D11.3) — anti-gaming guard.

The claim-shape axis relaxes evidence rules only for MINISTERIAL claims
(c-exist / c-count: "we convened a summit", "N attendees"). The obvious gaming
vector is a causal or superlative claim dressed as ministerial ("we launched
the initiative that ended veteran unemployment"). Layer A segmentation should
split such compounds; where it doesn't, this lint is the deterministic
belt-and-suspenders: any c-exist/c-count candidate whose text carries
superlative/comparative/causal tokens is FORCED to c-eval, regardless of what
the model emitted. Pure regex — no model, no speaker, no state.

Token list per the D11 sign-off draft, with two precision refinements (flagged
in the PR for line-item review):
  * the draft's bare ``\\w+est\\b`` matches interest/harvest/protest/West —
    the morphological form is matched only as "the <…>est" plus the explicit
    superlative word list;
  * bare "half"/"created"/"saved" over-fire on ministerial phrasing ("we
    created a program"); they count only in outcome form ("cut … in half",
    "created/saved <quantity>").

**The superlative list in this module is the single source of truth** (A6).
``truthbot.verdict.verdict_audit`` imports :data:`SUPERLATIVE_RX` for the
anti-gaming audit lint rather than keeping a second copy — two lists would
drift, and a superlative that is loud enough to force a shape is by
construction loud enough to be worth auditing on the verdict side.
"""
from __future__ import annotations

import re

#: Shapes the lint may force AWAY from (the ministerial class).
MINISTERIAL_SHAPES = {"c-exist", "c-count"}
FORCED_SHAPE = "c-eval"

#: Bare superlative words (D11 sign-off list, unchanged).
SUPERLATIVE_WORDS: tuple[str, ...] = (
    "largest", "biggest", "smallest", "first", "last", "most", "least",
    "best", "worst", "record", "historic", "unprecedented", "strongest",
    "fastest", "greatest", "highest", "lowest",
)

#: Rev-B additions (A6): multi-word superlative-scope markers the bare word
#: list cannot express. "first-ever" is nominally reachable through ``first``
#: — it is carried explicitly so the published token list and the compiled
#: regex are the same list, with no "well, ``first`` covers it" footnote.
#: "in recorded history" is genuinely new: ``\brecord\b`` does not match
#: "recorded". "in N years" is the superlative-SCOPE idiom ("the lowest level
#: in more than five years"); the qualifier list deliberately excludes
#: forward-looking framing ("in the next ten years"), which is a plan, not a
#: superlative.
SUPERLATIVE_PHRASES: tuple[str, ...] = (
    "first-ever", "never before", "in recorded history", "in N years",
)

#: The merged, published token list — words + rev-B phrases, in the order the
#: PR description quotes them.
SUPERLATIVE_TOKENS: tuple[str, ...] = SUPERLATIVE_WORDS + SUPERLATIVE_PHRASES

_NUMBER = (r"(?:\d[\d,]*|one|two|three|four|five|six|seven|eight|nine|ten|"
           r"eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|seventy|"
           r"eighty|ninety|hundred)")
#: "in N years" — bare, or behind a magnitude qualifier. NOT "in the next N
#: years" / "within N years": those are promises, and the ministerial-shape
#: lint must not drag a plan into c-eval.
_IN_N_YEARS = (rf"\bin\s+(?:more\s+than\s+|over\s+|nearly\s+|almost\s+|"
               rf"at\s+least\s+|under\s+|fewer\s+than\s+|less\s+than\s+)?"
               rf"{_NUMBER}\s+years\b")

SUPERLATIVE_RX = re.compile(
    r"\b(?:" + "|".join(SUPERLATIVE_WORDS) + r")\b"
    # rev-B phrases
    r"|\bfirst[- ]ever\b"
    r"|\bnever\s+before\b"
    r"|\b(?:in\s+)?recorded\s+history\b"
    rf"|{_IN_N_YEARS}"
    # morphological "-est" only in superlative position ("the …est") and not
    # for lexical -est words that aren't superlatives at all
    r"|\bthe\s+(?!(?:harvest|interest|honest|modest|earnest|protest|arrest|"
    r"request|conquest|tempest|contest|forest|west|east|rest|test|midwest|"
    r"northwest|southwest|northeast|southeast)\b)\w+est\b",
    re.IGNORECASE)

#: Historical private alias — the shape lint's own reference to the shared
#: pattern. Kept so this module reads the way it always did.
_SUPERLATIVE = SUPERLATIVE_RX


def has_superlative(text: str) -> bool:
    """True when ``text`` carries any token from :data:`SUPERLATIVE_TOKENS`.
    The one predicate both the shape lint and the verdict audit call."""
    return bool(SUPERLATIVE_RX.search(text or ""))
_COMPARATIVE = re.compile(
    r"\b(more|less|fewer|higher|lower|better|worse|faster|greater)\b"
    r"(?:\s+\S+){0,4}\s+than\b"
    r"|\b(doubled?|tripled?|twice|halved)\b"
    r"|\bin\s+half\b",
    re.IGNORECASE)
_CAUSAL = re.compile(
    r"\b(because( of)?|due\s+to|thanks\s+to|led\s+to|caused|resulted\s+in|"
    r"drove\s+(up|down)|boosted|spurred|helped)\b"
    r"|\b(created|saved|added)\s+(over|nearly|almost|about|more|millions?|"
    r"billions?|thousands?|hundreds?|jobs|\$?\d)",
    re.IGNORECASE)

_PATTERNS = (("superlative", _SUPERLATIVE), ("comparative", _COMPARATIVE),
             ("causal", _CAUSAL))


def shape_lint_hits(text: str) -> list[str]:
    """The lint categories whose tokens appear in ``text`` (order-stable)."""
    return [name for name, pat in _PATTERNS if pat.search(text or "")]


def enforce_shape(text: str, shape: str | None) -> str | None:
    """Force a ministerial shape to ``c-eval`` when the sentence carries
    superlative/comparative/causal tokens; all other shapes pass through.
    Deterministic and total — safe to run on every claim."""
    if shape in MINISTERIAL_SHAPES and shape_lint_hits(text):
        return FORCED_SHAPE
    return shape
