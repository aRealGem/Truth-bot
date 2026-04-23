"""
Unit tests for ``resolve_inject_evidence`` — the pipeline helper that decides
whether model prompts should include pre-gathered evidence snippets.

Goals:
- Default tracks ``evidence_source`` (none -> off; connectors/datahoover -> on)
  so telemetry ``evidence_injected`` reflects what actually happens in prompts.
- Explicit flags override the default.
- ``--no-inject-evidence`` wins over ``--inject-evidence`` when both are set,
  matching the guardrail that "force-off always wins" (users who pay attention
  to cost shouldn't be surprised).
"""

from __future__ import annotations

import pytest

from truthbot.pipeline import resolve_inject_evidence


class TestResolveInjectEvidenceDefaults:
    """Default behaviour tracks evidence_source when no flag is supplied."""

    def test_none_source_defaults_off(self):
        assert resolve_inject_evidence("none") is False

    def test_connectors_source_defaults_on(self):
        assert resolve_inject_evidence("connectors") is True

    def test_datahoover_source_defaults_on(self):
        assert resolve_inject_evidence("datahoover") is True

    @pytest.mark.parametrize("src", ["NONE", "None", " none ", "NONE  "])
    def test_none_source_is_case_and_whitespace_tolerant(self, src):
        assert resolve_inject_evidence(src) is False

    @pytest.mark.parametrize("src", ["CONNECTORS", "Connectors", " connectors "])
    def test_connectors_source_is_case_and_whitespace_tolerant(self, src):
        assert resolve_inject_evidence(src) is True

    def test_unknown_source_defaults_on(self):
        """
        Unknown source strings are treated as "something-is-fetching" for the
        purposes of this flag; build_evidence_provider still returns NoOp, but
        if someone adds a new source without updating this helper we prefer
        the safe-for-telemetry side of tracking the non-``none`` branch.
        """
        assert resolve_inject_evidence("future-provider") is True


class TestResolveInjectEvidenceFlags:
    """CLI flags override the default in both directions."""

    def test_no_inject_flag_forces_off_even_when_connectors_on(self):
        assert resolve_inject_evidence("connectors", no_inject_flag=True) is False

    def test_no_inject_flag_noop_when_already_off(self):
        assert resolve_inject_evidence("none", no_inject_flag=True) is False

    def test_inject_flag_forces_on_even_when_source_none(self):
        assert resolve_inject_evidence("none", inject_flag=True) is True

    def test_inject_flag_noop_when_already_on(self):
        assert resolve_inject_evidence("connectors", inject_flag=True) is True


class TestResolveInjectEvidenceConflict:
    """When both flags are set, --no-inject-evidence wins (safety / cost)."""

    @pytest.mark.parametrize("src", ["none", "connectors", "datahoover"])
    def test_no_inject_beats_inject(self, src):
        assert (
            resolve_inject_evidence(
                src,
                no_inject_flag=True,
                inject_flag=True,
            )
            is False
        )
