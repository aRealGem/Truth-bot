"""Sentinel tests for the CITATION DISCIPLINE block in SYNTHESIS_SYSTEM.

Layer 2 of the anti-hallucination defense-in-depth plan. Validates that:

1. The block exists in ``SYNTHESIS_SYSTEM`` with the expected critical
   phrases (so a future refactor can't silently delete it).
2. ``OPENAI_SYNTHESIS_SYSTEM`` still consists of ``SYNTHESIS_SYSTEM`` +
   the operational suffix unchanged, preserving the OpenAI prompt-cache
   prefix shape.
"""
from __future__ import annotations

from truthbot.verify.adapters.base import (
    OPENAI_SYNTHESIS_SYSTEM,
    SYNTHESIS_SYSTEM,
)


def test_synthesis_system_contains_citation_discipline_block():
    """Critical phrases that a future cleanup pass must not strip."""
    s = SYNTHESIS_SYSTEM
    assert "CITATION DISCIPLINE" in s
    assert "ONLY URLs that the web_search tool returned" in s
    assert "Do NOT fabricate" in s.replace("\n", " ")
    assert "fabrication-rate" in s
    assert 'web_sources": []' in s


def test_openai_prefix_is_stable_concatenation():
    """The OpenAI variant must be exactly SYNTHESIS_SYSTEM + suffix.

    OpenAI's prompt cache hits on a stable prefix; if some future change
    inserts an OpenAI-only block in the middle it would invalidate the
    cache for every claim. This test enforces the contract.
    """
    assert OPENAI_SYNTHESIS_SYSTEM.startswith(SYNTHESIS_SYSTEM)
    suffix = OPENAI_SYNTHESIS_SYSTEM[len(SYNTHESIS_SYSTEM):]
    # Suffix should be the OpenAI operational addendum (non-empty), not
    # an inserted block in the middle.
    assert suffix, "Expected non-empty OpenAI operational suffix"


def test_citation_discipline_appears_before_output_format():
    """Position matters — the model should see CITATION DISCIPLINE as the
    final guidance before the JSON spec, not buried earlier."""
    s = SYNTHESIS_SYSTEM
    cd_pos = s.find("CITATION DISCIPLINE")
    of_pos = s.find("OUTPUT FORMAT")
    assert cd_pos != -1
    assert of_pos != -1
    assert cd_pos < of_pos, (
        "CITATION DISCIPLINE block must appear before OUTPUT FORMAT so it "
        "is the last instruction before JSON emission."
    )
