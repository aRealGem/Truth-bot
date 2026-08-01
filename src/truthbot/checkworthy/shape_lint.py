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
"""
from __future__ import annotations

import re

#: Shapes the lint may force AWAY from (the ministerial class).
MINISTERIAL_SHAPES = {"c-exist", "c-count"}
FORCED_SHAPE = "c-eval"

_SUPERLATIVE = re.compile(
    r"\b(largest|biggest|smallest|first|last|most|least|best|worst|record|"
    r"historic|unprecedented|strongest|fastest|greatest|highest|lowest)\b"
    # morphological "-est" only in superlative position ("the …est") and not
    # for lexical -est words that aren't superlatives at all
    r"|\bthe\s+(?!(?:harvest|interest|honest|modest|earnest|protest|arrest|"
    r"request|conquest|tempest|contest|forest|west|east|rest|test|midwest|"
    r"northwest|southwest|northeast|southeast)\b)\w+est\b",
    re.IGNORECASE)
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
