"""Unit tests for ``normalize_verdict_label``.

Models in the wild emit a wide range of label strings that don't match the
canonical 6-value ``VerdictLabel`` enum exactly. The helper maps them to the
closest canonical bucket and logs a warning so the drift is visible in
telemetry.

This file is also the 'registry' of label variants we've seen. Each entry
doubles as documentation of the producer and the decision rationale.
"""

from __future__ import annotations

import logging

import pytest

from truthbot.models import VerdictLabel
from truthbot.verify.adapters.base import normalize_verdict_label


class TestCanonicalPassthrough:
    """Exact enum-value strings round-trip with no log noise."""

    @pytest.mark.parametrize(
        "raw",
        ["True", "Mostly True", "Misleading", "Exaggerated", "False", "Unverifiable"],
    )
    def test_exact_match_no_warning(self, raw, caplog):
        with caplog.at_level(logging.WARNING, logger="truthbot.verify.adapters.base"):
            label = normalize_verdict_label(raw)
        assert label == VerdictLabel(raw)
        assert caplog.records == [], f"exact match should not log for '{raw}'"


class TestCaseAndPunctuationNormalization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("true", VerdictLabel.TRUE),
            ("TRUE", VerdictLabel.TRUE),
            ("mostly true", VerdictLabel.MOSTLY_TRUE),
            ("Mostly-True", VerdictLabel.MOSTLY_TRUE),
            ("mostly_true", VerdictLabel.MOSTLY_TRUE),
            ("MISLEADING", VerdictLabel.MISLEADING),
            ("unverifiable", VerdictLabel.UNVERIFIABLE),
        ],
    )
    def test_canonicalizes_case_and_punct(self, raw, expected, caplog):
        with caplog.at_level(logging.WARNING, logger="truthbot.verify.adapters.base"):
            label = normalize_verdict_label(raw)
        assert label == expected
        assert any("non-canonical" in r.getMessage() for r in caplog.records), (
            "non-canonical case/punct normalizations must log a warning"
        )


class TestAliasMap:
    """Every alias variant we've seen in live traffic maps to its canonical bucket."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            # xAI grok-4 (seen 2026-04-23 SOTU sidecar)
            ("Mostly False", VerdictLabel.MISLEADING),
            ("mostly false", VerdictLabel.MISLEADING),
            ("mostly-false", VerdictLabel.MISLEADING),
            ("mostly_false", VerdictLabel.MISLEADING),
            # PolitiFact-style
            ("Pants on Fire", VerdictLabel.FALSE),
            ("pants-on-fire", VerdictLabel.FALSE),
            # Half-true family (partial)
            ("Half True", VerdictLabel.MISLEADING),
            ("half-true", VerdictLabel.MISLEADING),
            ("Half False", VerdictLabel.MISLEADING),
            # Not-enough-info family
            ("No Evidence", VerdictLabel.UNVERIFIABLE),
            ("no evidence", VerdictLabel.UNVERIFIABLE),
            ("N/A", VerdictLabel.UNVERIFIABLE),
            ("not applicable", VerdictLabel.UNVERIFIABLE),
            ("can't verify", VerdictLabel.UNVERIFIABLE),
            ("cannot verify", VerdictLabel.UNVERIFIABLE),
            # Partial true/false phrasing
            ("Partly True", VerdictLabel.MISLEADING),
            ("partially true", VerdictLabel.MISLEADING),
            ("Partly False", VerdictLabel.MISLEADING),
            ("Partially False", VerdictLabel.MISLEADING),
            # Exaggerated synonyms
            ("overstated", VerdictLabel.EXAGGERATED),
            ("Exaggeration", VerdictLabel.EXAGGERATED),
            # Context-needed (judgment call: lean Misleading)
            ("Needs context", VerdictLabel.MISLEADING),
        ],
    )
    def test_alias_mapping(self, raw, expected, caplog):
        with caplog.at_level(logging.WARNING, logger="truthbot.verify.adapters.base"):
            label = normalize_verdict_label(raw)
        assert label == expected
        assert any(
            "alias" in r.getMessage() or "non-canonical" in r.getMessage()
            for r in caplog.records
        ), "alias normalization must log a warning"


class TestInvalidLabels:
    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            normalize_verdict_label("")

    def test_garbage_raises(self):
        with pytest.raises(ValueError, match="not a recognized verdict label"):
            normalize_verdict_label("this is not a verdict")

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="label must be a string"):
            normalize_verdict_label(None)  # type: ignore[arg-type]

    def test_non_string_int_raises(self):
        with pytest.raises(ValueError, match="label must be a string"):
            normalize_verdict_label(42)  # type: ignore[arg-type]
