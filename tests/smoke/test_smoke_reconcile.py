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

import time
from pathlib import Path
from typing import Any

import pytest

from tests.smoke.conftest import (
    append_smoke_summary,
    is_false_label,
    is_true_label,
    load_manifest,
    poll_interval_for_elapsed,
    print_poll_line,
    provider_timeout_s,
    require_key,
    update_manifest,
)


pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Shared reconcile helper
# ---------------------------------------------------------------------------


def _run_reconcile(
    provider: str,
    adapter,
    *,
    manifest_path: Path,
    metrics_dir: Path,
    two_claims,
    timeout_s: int,
) -> None:
    """
    Poll → reconcile → assert for one batch provider.

    Pulls ``run_id`` / ``batch_id`` from the manifest. If the entry is
    missing (no submit ran), SKIPS the test rather than failing.
    """
    from truthbot.verify.batch import BatchDispatcher, reconcile_run
    from truthbot.verify.engine import VerificationEngine

    manifest = load_manifest(manifest_path)
    entry = manifest.get(provider)
    if not entry or not entry.get("run_id") or not entry.get("batch_id"):
        pytest.skip(
            f"{provider}: no submit entry in manifest {manifest_path} "
            f"(run test_smoke_submit.py first)"
        )

    run_id = entry["run_id"]
    batch_id = entry["batch_id"]
    submitted_at = entry.get("submitted_at") or time.time()

    dispatcher = BatchDispatcher(metrics_dir)
    t_start = time.monotonic()
    elapsed = 0.0
    last_status = "pending"

    # Poll loop. Prints one line per cycle; tail-friendly in the terminal file.
    print(
        f"[smoke] {provider}: polling run_id={run_id} batch_id={batch_id} "
        f"cap={timeout_s}s",
        flush=True,
    )

    while True:
        last_status = dispatcher.poll(run_id)
        elapsed = time.monotonic() - t_start
        print_poll_line(provider, last_status, elapsed)

        if last_status in ("complete", "failed", "missing"):
            break
        if elapsed >= timeout_s:
            # Persist the stall in the manifest so the monitoring loop
            # (and any subsequent manual reconcile) can see exactly
            # when we gave up.
            update_manifest(
                manifest_path,
                provider,
                {
                    "status": "pending_at_cap",
                    "last_status": last_status,
                    "automated_cap_hit_at": time.time(),
                    "elapsed_at_cap_s": round(elapsed, 1),
                },
            )
            pytest.fail(
                f"{provider}: still {last_status!r} after automated cap "
                f"of {timeout_s}s ({timeout_s / 3600:.2f}h). "
                f"run_id={run_id}, batch_id={batch_id}. "
                f"Resume manually with: truthbot batch reconcile {run_id}"
            )

        time.sleep(poll_interval_for_elapsed(elapsed))

    if last_status != "complete":
        update_manifest(
            manifest_path,
            provider,
            {
                "status": last_status,
                "terminal_at": time.time(),
                "elapsed_s": round(elapsed, 1),
            },
        )
        pytest.fail(
            f"{provider}: batch terminated as {last_status!r} after {elapsed:.0f}s. "
            f"run_id={run_id}, batch_id={batch_id}."
        )

    # Reconcile
    engine = VerificationEngine(run_id=run_id, inject_evidence=False)
    result = reconcile_run(
        metrics_dir,
        run_id,
        adapters_by_name={provider: adapter},
        engine=engine,
    )
    assert result["status"] == "complete", (
        f"{provider}: reconcile_run returned {result['status']}, expected complete"
    )
    bundles = result["bundles"]
    assert len(bundles) == 2, (
        f"{provider}: expected 2 bundles, got {len(bundles)}"
    )

    by_text = {b.claim.text: b for b in bundles}
    moon = by_text[two_claims[0].text]
    eiffel = by_text[two_claims[1].text]

    moon_labels = [mv.label for mv in moon.model_verdicts]
    eiffel_labels = [mv.label for mv in eiffel.model_verdicts]
    assert any(is_true_label(lbl) for lbl in moon_labels), (
        f"{provider}: Moon claim should be True-ish; got {moon_labels}"
    )
    assert any(is_false_label(lbl) for lbl in eiffel_labels), (
        f"{provider}: Eiffel claim should be False-ish; got {eiffel_labels}"
    )

    # Descriptor was just rewritten by reconcile_run; re-read the fresh
    # copy so the manifest matches what's on disk.
    descriptor = result["descriptor"] or {}
    provider_entry: dict[str, Any] = (descriptor.get("provider_jobs") or {}).get(
        provider, {}
    )

    total_elapsed = time.time() - submitted_at
    verdict_rows = [
        {
            "claim": b.claim.text,
            "consensus_label": b.consensus.consensus_label.value,
            "model_labels": [
                {"model": mv.adapter_name, "label": mv.label.value}
                for mv in b.model_verdicts
            ],
        }
        for b in bundles
    ]

    update_manifest(
        manifest_path,
        provider,
        {
            "status": "complete",
            "completed_at": time.time(),
            "elapsed_total_s": round(total_elapsed, 1),
            "elapsed_poll_s": round(elapsed, 1),
            "verdicts": verdict_rows,
            "chunk_size": provider_entry.get("chunk_size"),
            "request_count": provider_entry.get("request_count"),
        },
    )
    append_smoke_summary(
        metrics_dir,
        provider,
        "batch-2cpr",
        wall_clock_s=total_elapsed,
        claim_count=2,
        request_count=provider_entry.get("request_count", 1),
        verdicts=verdict_rows,
        notes=f"run_id={run_id}; batch_id={batch_id}",
    )


# ---------------------------------------------------------------------------
# Per-provider reconcile tests
# ---------------------------------------------------------------------------


class TestReconcileAnthropicBatch:
    @classmethod
    def setup_class(cls):
        require_key("ANTHROPIC_API_KEY")

    def test_reconcile(self, two_claims, manifest_path, smoke_metrics_dir):
        from truthbot.verify.adapters.anthropic import AnthropicAdapter

        _run_reconcile(
            "anthropic",
            AnthropicAdapter(),
            manifest_path=manifest_path,
            metrics_dir=smoke_metrics_dir,
            two_claims=two_claims,
            timeout_s=provider_timeout_s("anthropic_batch"),
        )


class TestReconcileOpenAIBatch:
    @classmethod
    def setup_class(cls):
        require_key("OPENAI_API_KEY")

    def test_reconcile(self, two_claims, manifest_path, smoke_metrics_dir):
        from truthbot.verify.adapters.openai import OpenAIAdapter

        _run_reconcile(
            "openai",
            OpenAIAdapter(),
            manifest_path=manifest_path,
            metrics_dir=smoke_metrics_dir,
            two_claims=two_claims,
            timeout_s=provider_timeout_s("openai_batch"),
        )
