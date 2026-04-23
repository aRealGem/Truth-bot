"""
Phase B of the paginated smoke: poll + reconcile the 5-claim batches.

Reads the ``anthropic_pg`` / ``openai_pg`` entries from the manifest
(written by ``test_smoke_submit_paginated.py``), polls until each batch
completes or the automated watch cap fires, then runs ``reconcile_run``
and asserts all 5 claim verdicts match the truth pattern.

Shares the full poll/reconcile implementation with the 2-claim smoke via
``tests.smoke.conftest._run_reconcile_n``. The only per-variant thing
here is which manifest key to read + the 5-claim truth pattern.

Run with:

    pytest tests/smoke/test_smoke_reconcile_paginated.py -m live -v

Completion SLA + env overrides are identical to the 2-claim smoke; see
``tests/smoke/README.md``.
"""

from __future__ import annotations

import pytest

from tests.smoke.conftest import (
    _run_reconcile_n,
    provider_timeout_s,
    require_key,
)


pytestmark = pytest.mark.live


class TestReconcileAnthropicBatchPaginated:
    @classmethod
    def setup_class(cls):
        require_key("ANTHROPIC_API_KEY")

    def test_reconcile(
        self,
        five_claims,
        five_claims_truth_pattern,
        manifest_path,
        smoke_metrics_dir,
    ):
        from truthbot.verify.adapters.anthropic import AnthropicAdapter

        _run_reconcile_n(
            "anthropic",
            AnthropicAdapter(),
            manifest_path=manifest_path,
            metrics_dir=smoke_metrics_dir,
            claims=five_claims,
            truth_pattern=five_claims_truth_pattern,
            manifest_key="anthropic_pg",
            timeout_s=provider_timeout_s("anthropic_batch"),
            summary_mode="batch-2cpr-pg5",
        )


class TestReconcileOpenAIBatchPaginated:
    @classmethod
    def setup_class(cls):
        require_key("OPENAI_API_KEY")

    def test_reconcile(
        self,
        five_claims,
        five_claims_truth_pattern,
        manifest_path,
        smoke_metrics_dir,
    ):
        from truthbot.verify.adapters.openai import OpenAIAdapter

        _run_reconcile_n(
            "openai",
            OpenAIAdapter(),
            manifest_path=manifest_path,
            metrics_dir=smoke_metrics_dir,
            claims=five_claims,
            truth_pattern=five_claims_truth_pattern,
            manifest_key="openai_pg",
            timeout_s=provider_timeout_s("openai_batch"),
            summary_mode="batch-2cpr-pg5",
        )
