"""Resilience gap in the PCA publish orchestrator — P67.3 (jackie, 2026-07-20).

``run_pca_verify`` adjudicates in sequential chunks but accumulates rows ONLY
in memory; artifacts and the site are written at the very end, so a mid-run
failure (proxy 429, OOM, ctrl-C) discards every completed chunk. Empirically:
the 2026-07-20 re-publish attempt died at ~chunk 25/30 on a proxy
budget-429 and ~$1.8 of completed adjudication spend was unrecoverable.

Two kinds of test here:
  * CHARACTERIZATION — pins today's lossy behavior so the eventual fix has a
    precise before/after (when checkpointing lands, these flip and must be
    updated deliberately, not accidentally).
  * DESIRED (xfail) — expresses the target contract for the P67.3 fix:
    completed chunks survive a mid-run failure and a resume path can skip
    already-adjudicated sids. Strict xfail: the day the fix lands, these
    start passing and pytest flags them so they get un-marked.

Fully offline; the adjudicate lane is an injected fake that detonates on a
chosen chunk.
"""
from __future__ import annotations

import pytest

from truthbot.verdict import publish_pipeline as pp

CHUNK = 2


def _sentences(n):
    return [
        {"sid": f"sp:{i:04d}", "text": f"Metric {i} rose by {i} percent in 2026.",
         "context": f"|| Metric {i} rose by {i} percent in 2026. ||"}
        for i in range(n)
    ]


def _fake_classify_all_checkworthy(sents):
    return [{"sid": s["sid"], "label": "check-worthy",
             "text": s["text"], "context": s["context"]} for s in sents]


class _DetonatingAdjudicator:
    """Resolves every claim TRUE until ``fail_on_chunk``, then raises the same
    shape the live lane does when the proxy 429s through its retries."""

    def __init__(self, fail_on_chunk: int):
        self.fail_on_chunk = fail_on_chunk
        self.calls = 0
        self.completed_rows: list[dict] = []

    def __call__(self, chunk):
        self.calls += 1
        if self.calls == self.fail_on_chunk:
            raise RuntimeError("HTTP Error 429: Too Many Requests (budget)")
        rows = [{"sid": c["sid"], "status": "resolved", "verdict": "TRUE",
                 "confidence": 0.9, "citations": [], "reasoning": "r",
                 "votes": {"TRUE": 2}, "split": False, "escalated": False}
                for c in chunk]
        self.completed_rows.extend(rows)
        return rows, {}


# ── characterization: today's lossy behavior ──────────────────────────────────

def test_mid_run_chunk_failure_discards_all_completed_chunks():
    """CURRENT behavior (the P67.3 gap): chunk 3 of 4 failing loses chunks 1-2's
    completed rows — the exception propagates and no partial result object,
    journal, or artifact carries them. When checkpointing lands this test must
    be REPLACED by the xfail contracts below, not silently deleted."""
    adj = _DetonatingAdjudicator(fail_on_chunk=3)
    with pytest.raises(RuntimeError, match="429"):
        pp.run_pca_verify(_sentences(8),
                          layer_a_fn=_fake_classify_all_checkworthy,
                          adjudicate_fn=adj, chunk_size=CHUNK)
    # Two chunks (4 rows) HAD completed inside the lane...
    assert len(adj.completed_rows) == 2 * CHUNK
    # ...but the caller has no way to receive them: run_pca_verify surfaces
    # nothing on failure (no partial-result channel exists today).


def test_failure_on_first_chunk_loses_nothing_but_layer_a_spend():
    """Boundary pin: an immediate chunk-1 failure at least wastes only the
    Layer A pass — documents that the blast radius scales with progress."""
    adj = _DetonatingAdjudicator(fail_on_chunk=1)
    with pytest.raises(RuntimeError, match="429"):
        pp.run_pca_verify(_sentences(4),
                          layer_a_fn=_fake_classify_all_checkworthy,
                          adjudicate_fn=adj, chunk_size=CHUNK)
    assert adj.completed_rows == []


# ── desired contract for the P67.3 fix (strict xfail until implemented) ───────

@pytest.mark.xfail(strict=True,
                   reason="P67.3 not implemented: completed chunks should survive a mid-run failure")
def test_completed_chunks_survive_a_mid_run_failure():
    """TARGET: after a chunk-3 failure, the completed rows from chunks 1-2 are
    recoverable from the run (partial result / journal), so their spend is not
    lost. Exact API is the P67.3 design decision; this asserts the weakest
    useful form — the orchestrator exposes the completed rows."""
    adj = _DetonatingAdjudicator(fail_on_chunk=3)
    partial = None
    try:
        pp.run_pca_verify(_sentences(8),
                          layer_a_fn=_fake_classify_all_checkworthy,
                          adjudicate_fn=adj, chunk_size=CHUNK)
    except RuntimeError as exc:
        partial = getattr(exc, "partial_result", None)
    assert partial is not None and len(partial.rows) == 2 * CHUNK


@pytest.mark.xfail(strict=True,
                   reason="P67.3 not implemented: resume should skip already-adjudicated sids")
def test_resume_skips_already_adjudicated_sids():
    """TARGET: a resumed run re-spends ONLY on sids that never completed.
    Weakest useful form: run_pca_verify accepts prior rows and does not call
    the lane for their sids again."""
    done_rows = [{"sid": "sp:0000", "status": "resolved", "verdict": "TRUE",
                  "confidence": 0.9, "citations": [], "reasoning": "r",
                  "votes": {"TRUE": 2}, "split": False, "escalated": False},
                 {"sid": "sp:0001", "status": "resolved", "verdict": "TRUE",
                  "confidence": 0.9, "citations": [], "reasoning": "r",
                  "votes": {"TRUE": 2}, "split": False, "escalated": False}]
    adj = _DetonatingAdjudicator(fail_on_chunk=99)   # never detonates
    result = pp.run_pca_verify(_sentences(4),
                               layer_a_fn=_fake_classify_all_checkworthy,
                               adjudicate_fn=adj, chunk_size=CHUNK,
                               resume_rows=done_rows)   # API TBD in P67.3
    adjudicated = {r["sid"] for rows in [adj.completed_rows] for r in rows}
    assert "sp:0000" not in adjudicated and "sp:0001" not in adjudicated
    assert len(result.rows) == 4
