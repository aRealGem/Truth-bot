"""
Phase B of the 2-claim live smoke test: poll + reconcile.

Each test here reads the ``anthropic`` or ``openai`` entry from
``metrics/smoke/manifest.json`` (written by ``test_smoke_submit.py``),
polls until the batch completes or the provider's automated watch cap
is hit, reconciles to ``VerdictBundle`` s, and asserts verdicts match
ground truth.

The internal poll cadence is SLA-driven:
  - 60 s intervals for the first 5 min
  - 2 min intervals through 30 min
  - 5 min intervals afterwards

Automated watch cap per provider is ``AUTOMATED_WATCH_CAP_S`` (2.5 h).
If a batch is still pending at that point, the test FAILS rather than
silently hanging — but the descriptor + run_id stay on disk so operators
can resume manually with ``truthbot batch reconcile <run_id>`` any time
up to the 24 h vendor cutoff.

If the manifest has no entry for a provider (e.g. reconcile is run
without a prior submit in this working tree), the test is SKIPPED so
CI can legitimately run reconcile-only when the manifest was populated
in a prior invocation, but a missing manifest is not a hard failure.

Run with:

    pytest tests/smoke/test_smoke_reconcile.py -m live -v
"""

from __future__ import annotations

import pytest

from tests.smoke.conftest import (
    _run_reconcile_n,
    provider_timeout_s,
    require_key,
)


pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Per-provider reconcile tests
# ---------------------------------------------------------------------------
#
# The poll + reconcile + assert plumbing lives in
# ``tests.smoke.conftest._run_reconcile_n`` so the 2-claim smoke here and
# the 5-claim paginated smoke in ``test_smoke_reconcile_paginated.py``
# share the same battle-tested code path. The 2-claim truth pattern is
# ``[True, False]`` (Moon = TRUE, Eiffel = FALSE).


class TestReconcileAnthropicBatch:
    @classmethod
    def setup_class(cls):
        require_key("ANTHROPIC_API_KEY")

    def test_reconcile(self, two_claims, manifest_path, smoke_metrics_dir):
        from truthbot.verify.adapters.anthropic import AnthropicAdapter

        _run_reconcile_n(
            "anthropic",
            AnthropicAdapter(),
            manifest_path=manifest_path,
            metrics_dir=smoke_metrics_dir,
            claims=two_claims,
            truth_pattern=[True, False],
            manifest_key="anthropic",
            timeout_s=provider_timeout_s("anthropic_batch"),
            summary_mode="batch-2cpr",
        )


class TestReconcileOpenAIBatch:
    @classmethod
    def setup_class(cls):
        require_key("OPENAI_API_KEY")

    def test_reconcile(self, two_claims, manifest_path, smoke_metrics_dir):
        from truthbot.verify.adapters.openai import OpenAIAdapter

        _run_reconcile_n(
            "openai",
            OpenAIAdapter(),
            manifest_path=manifest_path,
            metrics_dir=smoke_metrics_dir,
            claims=two_claims,
            truth_pattern=[True, False],
            manifest_key="openai",
            timeout_s=provider_timeout_s("openai_batch"),
            summary_mode="batch-2cpr",
        )
