"""
Layer A / A1 — lexical check-worthiness prefilter (spec §4, Hassan et al. 2017).

Zero LLM tokens; runs on sandboxpi CPU. Produces a ClaimBuster-style
check-worthiness score in [0,1] from ClaimBuster-family features (numerics,
comparatives, assertion verbs, named entities, temporal anchors) minus
opinion/future markers. spaCy POS/NER is used when importable; otherwise a
dependency-free regex/wordlist featurizer runs (same feature semantics), so the
prefilter always runs CPU-only.

A1 is a *triage*, not the verdict: with two thresholds it routes each sentence to
DROP (clearly not check-worthy) / AMBIGUOUS (send to A2) / PASS (clearly
check-worthy). The gate is tuned on train to preserve check-worthy recall
(a missed claim is a silent failure; a leaked opinion is only wasted budget).

Speaker-blind by construction (Principle 1 / I3): features read only the
sentence text, never any speaker/source metadata.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

try:                                   # optional acceleration
    import spacy  # type: ignore
    _NLP = spacy.load("en_core_web_sm") if spacy.util.is_package("en_core_web_sm") else None
except Exception:
    _NLP = None

_NUM_RX = re.compile(r"\b\d[\d,]*\.?\d*\b|\b\d+\s?(?:percent|%)|\$\s?\d")
_PCT_RX = re.compile(r"%|\bpercent\b")
_YEAR_RX = re.compile(r"\b(?:18|19|20)\d{2}\b")
_TEMPORAL_RX = re.compile(r"(?i)\b(last year|this year|since|in \d{4}|today|yesterday|"
                          r"in the past \w+|over the past \w+)\b")
_COMPARATIVE_RX = re.compile(
    r"(?i)\b(most|more|less|fewer|greatest|biggest|largest|smallest|lowest|highest|"
    r"worst|best|strongest|weakest|record|first|only|than|double|triple|half|"
    r"[a-z]+est)\b")
_ASSERT_VERB_RX = re.compile(
    r"(?i)\b(is|are|was|were|has|have|had|did|created|cut|reduced|raised|passed|"
    r"signed|killed|rose|fell|increased|decreased|dropped|grew|added|ended|"
    r"secured|delivered|built|banned|deported|achieved|reached|hit|spent|saved)\b")
_ENTITY_RX = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_OPINION_RX = re.compile(
    r"(?i)\b(i think|i believe|we must|we should|should|must|great|greatest|"
    r"beautiful|incredible|amazing|wonderful|terrible|horrible|disaster|proud|"
    r"believe|honor|shame|love|hate|never before|like never|isn'?t it)\b")
_FUTURE_RX = re.compile(r"(?i)\b(will|going to|we'?ll|shall|promise|pledge|"
                        r"in the coming|next year|by 20\d{2})\b")
_CEREMONIAL_RX = re.compile(
    r"(?i)\b(thank you|thanks|good evening|my fellow|ladies and gentlemen|"
    r"speaker|vice president|first lady|god bless|please|welcome|tonight we)\b")


@dataclass
class A1Features:
    numeric: bool = False
    percent: bool = False
    temporal: bool = False
    comparative: bool = False
    assertion_verb: bool = False
    entity: bool = False
    opinion_marker: bool = False
    future_marker: bool = False
    ceremonial: bool = False
    n_tokens: int = 0
    detail: dict = field(default_factory=dict)


def extract_features(text: str) -> A1Features:
    f = A1Features()
    f.numeric = bool(_NUM_RX.search(text))
    f.percent = bool(_PCT_RX.search(text))
    f.temporal = bool(_TEMPORAL_RX.search(text) or _YEAR_RX.search(text))
    f.comparative = bool(_COMPARATIVE_RX.search(text))
    f.assertion_verb = bool(_ASSERT_VERB_RX.search(text))
    f.opinion_marker = bool(_OPINION_RX.search(text))
    f.future_marker = bool(_FUTURE_RX.search(text))
    f.ceremonial = bool(_CEREMONIAL_RX.search(text))
    if _NLP is not None:
        doc = _NLP(text)
        f.n_tokens = len(doc)
        f.entity = any(e.label_ in {"PERSON", "ORG", "GPE", "MONEY", "PERCENT",
                                    "CARDINAL", "DATE", "LAW", "EVENT", "NORP"}
                       for e in doc.ents)
    else:
        toks = text.split()
        f.n_tokens = len(toks)
        # dependency-free NER proxy: a capitalized token not at sentence start
        f.entity = bool(_ENTITY_RX.search(" " + " ".join(toks[1:])))
    return f


# Feature weights (ClaimBuster-style). Positive = pushes toward check-worthy;
# negative = pushes toward opinion/unimportant. Tuned lightly; the operating
# threshold is what gets calibrated on train.
_W = {
    "numeric": 0.35, "percent": 0.15, "temporal": 0.20, "comparative": 0.20,
    "assertion_verb": 0.25, "entity": 0.20,
    "opinion_marker": -0.35, "future_marker": -0.30, "ceremonial": -0.60,
}


def score(text: str) -> float:
    f = extract_features(text)
    s = 0.0
    for k, w in _W.items():
        if getattr(f, k):
            s += w
    # very short fragments rarely carry a material proposition
    if f.n_tokens <= 3:
        s -= 0.25
    return max(0.0, min(1.0, 0.5 + s))   # center at 0.5, clamp to [0,1]


@dataclass
class Route:
    sid: str
    score: float
    bucket: str          # "drop" | "ambiguous" | "pass"


def route(sid: str, text: str, tau_low: float, tau_high: float) -> Route:
    sc = score(text)
    if sc < tau_low:
        b = "drop"
    elif sc >= tau_high:
        b = "pass"
    else:
        b = "ambiguous"
    return Route(sid=sid, score=sc, bucket=b)
