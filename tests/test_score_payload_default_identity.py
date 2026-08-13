"""D17-c (a)/(a+) — the default scoring payload did not move.

``score_payload`` grew a ``max_snippet_chars`` parameter so D17-c can carry
series excerpts (~22,000 characters) without raising ``SCORE_SNIPPET_CHARS``
for every speech and every future run. The whole argument for that shape is
that existing callers are BYTE-unchanged, and an argument is not evidence.

So this file pins the default path against a committed fixture pack: the same
bytes, over real stored evidence, computed the way the pre-parameter code
computed them. If the default ever drifts, the D17-c flip census stops being a
measurement against the shipped baseline and becomes a comparison of two
different prompts.

The (a+) half is asserted too: a clip is always visible and always counted.
Offline — committed artifacts only, no network, no model.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from truthbot.models import Evidence
from truthbot.verify.relevance import (
    SCORE_SNIPPET_CHARS,
    TRUNCATION_MARKER,
    score_payload,
    score_payload_ex,
)

#: A shipped publishing head. Real stored packs, not synthesised ones.
HEAD = (Path(__file__).resolve().parents[1] / "metrics" / "pca_runs"
        / "91dd7a34-7a3c-4f40-bcdc-276b2cb15d26.json")


def _packs() -> list[tuple[str, list[Evidence]]]:
    """(claim text, evidence) for every claim in the head, as models."""
    doc = json.loads(HEAD.read_text())
    texts = {c["sid"]: c["text"] for c in doc["claims"]}
    out = []
    for sid, items in doc["evidence"].items():
        evs = [Evidence(claim_id=sid,
                        source_name=it["source_name"],
                        source_url=it["source_url"],
                        snippet=it.get("snippet") or "")
               for it in items]
        out.append((texts.get(sid, ""), evs))
    return out


def _payload_pre_parameter(claim_text: str, evidence: list[Evidence]) -> str:
    """Exactly what the function built before ``max_snippet_chars`` existed.

    Transcribed from the pre-change source rather than imported, so the test
    fails if the shipped implementation drifts toward it *or* away from it."""
    items = [{"i": i, "source": ev.source_name,
              "snippet": (ev.snippet or "")[:SCORE_SNIPPET_CHARS]}
             for i, ev in enumerate(evidence, start=1)]
    return json.dumps({"claim": claim_text, "items": items})


@pytest.mark.parametrize("claim_text,evidence", _packs())
def test_default_payload_is_byte_identical(claim_text, evidence) -> None:
    """The ruling's condition: every existing caller is byte-unchanged."""
    assert score_payload(claim_text, evidence) == _payload_pre_parameter(
        claim_text, evidence)


def test_default_matches_explicit_constant() -> None:
    """Passing the default explicitly is the same call, not a near-miss."""
    claim_text, evidence = _packs()[0]
    assert (score_payload(claim_text, evidence)
            == score_payload(claim_text, evidence, SCORE_SNIPPET_CHARS))


def test_nothing_in_the_head_is_clipped_at_the_default() -> None:
    """Why the default path is byte-identical at all: the cap never bites.

    Largest stored snippet across the five shipped heads is 207 characters."""
    for claim_text, evidence in _packs():
        _, meta = score_payload_ex(claim_text, evidence)
        assert all(m["chars_truncated"] == 0 for m in meta)


def test_uncapped_carries_the_whole_snippet() -> None:
    """``None`` means no limit — the D17-c excerpt path."""
    big = "x" * 25_000
    ev = [Evidence(claim_id="c", source_name="S", source_url="https://e/",
                   snippet=big)]
    payload, meta = score_payload_ex("claim", ev, None)
    assert json.loads(payload)["items"][0]["snippet"] == big
    assert meta[0]["chars_truncated"] == 0
    assert meta[0]["chars_sent"] == 25_000


def test_a_clip_is_visible_and_counted() -> None:
    """(a+): truncation is never silent, and says how much went missing."""
    ev = [Evidence(claim_id="c", source_name="S", source_url="https://e/",
                   snippet="y" * 500)]
    payload, meta = score_payload_ex("claim", ev, 400)
    sent = json.loads(payload)["items"][0]["snippet"]
    assert sent.startswith("y" * 400)
    assert sent.endswith(TRUNCATION_MARKER.format(n=100))
    assert meta[0]["chars_truncated"] == 100


def test_a_clip_exactly_at_the_cap_is_not_marked() -> None:
    """Off-by-one: a snippet the length of the cap loses nothing."""
    ev = [Evidence(claim_id="c", source_name="S", source_url="https://e/",
                   snippet="z" * 400)]
    payload, meta = score_payload_ex("claim", ev, 400)
    assert json.loads(payload)["items"][0]["snippet"] == "z" * 400
    assert meta[0]["chars_truncated"] == 0
