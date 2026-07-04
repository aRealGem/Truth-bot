#!/usr/bin/env python3
"""
Deterministic sentence segmenter for SOTU transcripts.

Input : a Historical-SOTU-Transcripts *.txt file (two '#'-prefixed header
        lines, blank line, then the speech body).
Output: JSONL, one object per sentence:
        {"sid": "trump_2026:0007", "speech": "trump_2026", "idx": 7,
         "text": "...", "context": "<prev> || <this> || <next>"}

The segmenter is intentionally simple and rule-based (no LLM, no external
deps) so that the sentence inventory is fully reproducible. Audience stage
directions ([Applause], [Laughter], [Audience chants "USA"]) are stripped
from sentence text. Common abbreviations are protected so "U.S." / "Mr."
do not trigger a false split.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Bracketed audience/stage cues to strip, e.g. "[Applause]", "[Laughter]".
_STAGE = re.compile(r"\[[^\]]*\]")

# Protect abbreviations from the sentence splitter by temporarily masking
# the period. Keep the list short and high-frequency for SOTU text.
_ABBREV = [
    "U.S.A.", "U.S.", "U.N.", "D.C.", "Mr.", "Mrs.", "Ms.", "Dr.",
    "Sen.", "Rep.", "Gov.", "Gen.", "Lt.", "Sgt.", "St.", "Jr.", "Sr.",
    "vs.", "etc.", "No.", "a.m.", "p.m.",
]
_MASK = ""  # private-use char unlikely to appear in transcripts

# Split after ., !, or ? (optionally followed by a closing quote/paren)
# when the next non-space char starts a new sentence (capital, digit, or quote).
_SPLIT = re.compile(r'(?<=[.!?])(["\')\]]?)\s+(?=[A-Z0-9"“])')

# Sentinel that temporarily replaces the '.' inside protected abbreviations.
# Defined here (after the block above) with an explicit control char so the
# value is unambiguous regardless of editor quirks. Must not occur in text.
_MASK = chr(1)


def _read_body(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    body = [ln for ln in lines if not ln.startswith("#")]
    return "\n".join(body).strip()


def _mask_abbrev(text: str) -> str:
    for a in _ABBREV:
        text = text.replace(a, a.replace(".", _MASK))
    return text


def _unmask(text: str) -> str:
    return text.replace(_MASK, ".")


def segment(path: Path) -> list[str]:
    body = _read_body(path)
    body = _STAGE.sub(" ", body)
    body = re.sub(r"\s+", " ", body).strip()
    masked = _mask_abbrev(body)
    parts = _SPLIT.split(masked)
    # _SPLIT keeps the optional closing-quote group; stitch it back on.
    sents: list[str] = []
    buf = ""
    for i, chunk in enumerate(parts):
        if i % 2 == 0:
            buf = chunk
        else:  # this is the captured closing-quote group
            sents.append((buf + chunk).strip())
            buf = ""
    if buf.strip():
        sents.append(buf.strip())
    out = []
    for s in sents:
        s = _unmask(s).strip()
        # Drop fragments that are pure whitespace/punctuation or too short
        # to carry a proposition (e.g. a stray "Thank you." is kept; a lone
        # "USA" left by a stripped chant is dropped).
        if len(s) >= 8 and re.search(r"[A-Za-z]", s):
            out.append(s)
    return out


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: segment_sotu.py <slug> <transcript.txt> [more slug txt ...]",
              file=sys.stderr)
        sys.exit(2)
    args = sys.argv[1:]
    pairs = list(zip(args[0::2], args[1::2]))
    for slug, txt in pairs:
        sents = segment(Path(txt))
        for i, s in enumerate(sents):
            prev = sents[i - 1] if i > 0 else ""
            nxt = sents[i + 1] if i + 1 < len(sents) else ""
            rec = {
                "sid": f"{slug}:{i:04d}",
                "speech": slug,
                "idx": i,
                "text": s,
                "context": f"{prev} || {s} || {nxt}".strip(),
            }
            print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
