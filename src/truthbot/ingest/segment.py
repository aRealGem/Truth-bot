"""Deterministic sentence segmenter for the v2 (HydraMind PCA) publish path.

The v2 pipeline's Layer A operates on *sentences* — ``{"sid","text","context"}``
records — not on ``ClaimExtractor`` output. This module turns raw transcript text
into exactly that inventory, with the same sid + context convention the eval
fixtures use (``eval/benchmarks/scorer/segment_sotu.py`` is the lineage; the
algorithm is duplicated here rather than imported so the segmenter ships with the
package and stays independent of the eval tree).

Rule-based and LLM-free so the sentence inventory is fully reproducible:
  * ``sid`` = ``"{speech_id}:{idx:04d}"`` — the ``speech_id`` prefix is what
    ``verdict.speech_context`` maps to an utterance date (temporal grounding) and
    what ``evidence_pack.window_for`` scopes retrieval by, so it MUST match a
    registered speech (or be registered from ``--date`` at run time).
  * ``context`` = ``"{prev} || {this} || {next}"`` — the neighbour window Layer A
    (A1/A2) and the PCA panel read for local disambiguation.
"""
from __future__ import annotations

import re

# Bracketed audience/stage cues to strip, e.g. "[Applause]", "[Laughter]".
_STAGE = re.compile(r"\[[^\]]*\]")

# Abbreviations whose '.' must not trigger a sentence split.
_ABBREV = [
    "U.S.A.", "U.S.", "U.N.", "D.C.", "Mr.", "Mrs.", "Ms.", "Dr.",
    "Sen.", "Rep.", "Gov.", "Gen.", "Lt.", "Sgt.", "St.", "Jr.", "Sr.",
    "vs.", "etc.", "No.", "a.m.", "p.m.",
]

# Split after ., !, or ? (optionally followed by a closing quote/paren) when the
# next non-space char starts a new sentence (capital, digit, or opening quote).
_SPLIT = re.compile(r'(?<=[.!?])(["\')\]]?)\s+(?=[A-Z0-9"“])')

# Control char that temporarily masks the '.' inside protected abbreviations;
# must not occur in transcript text.
_MASK = chr(1)


def _mask_abbrev(text: str) -> str:
    for a in _ABBREV:
        text = text.replace(a, a.replace(".", _MASK))
    return text


def _unmask(text: str) -> str:
    return text.replace(_MASK, ".")


def split_sentences(text: str) -> list[str]:
    """Segment raw transcript text into sentence strings (stage cues stripped,
    abbreviations protected, sub-8-char / alpha-free fragments dropped)."""
    body = _STAGE.sub(" ", text or "")
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return []
    masked = _mask_abbrev(body)
    parts = _SPLIT.split(masked)
    # _SPLIT keeps the optional closing-quote group; stitch it back on.
    sents: list[str] = []
    buf = ""
    for i, chunk in enumerate(parts):
        if i % 2 == 0:
            buf = chunk
        else:  # captured closing-quote group
            sents.append((buf + chunk).strip())
            buf = ""
    if buf.strip():
        sents.append(buf.strip())
    out: list[str] = []
    for s in sents:
        s = _unmask(s).strip()
        if len(s) >= 8 and re.search(r"[A-Za-z]", s):
            out.append(s)
    return out


def segment(text: str, speech_id: str) -> list[dict]:
    """Raw transcript text → Layer A sentence records.

    Returns ``[{"sid","speech","idx","text","context"}]`` — the input shape for
    ``checkworthy.pipeline.run_layer_a``. ``speech_id`` becomes the sid prefix
    (see module docstring on why it must resolve to an utterance date)."""
    sents = split_sentences(text)
    records: list[dict] = []
    for i, s in enumerate(sents):
        prev = sents[i - 1] if i > 0 else ""
        nxt = sents[i + 1] if i + 1 < len(sents) else ""
        records.append({
            "sid": f"{speech_id}:{i:04d}",
            "speech": speech_id,
            "idx": i,
            "text": s,
            "context": f"{prev} || {s} || {nxt}".strip(),
        })
    return records
