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

def test_mid_run_chunk_failure_preserves_completed_chunks_on_exception():
    """P67.3 LANDED (2026-07-22, replacing the lossy characterization test per
    its own instruction): chunk 3 of 4 failing still raises, but chunks 1-2's
    completed rows ride on ``exc.partial_result`` — banked spend is never
    unreachable again."""
    adj = _DetonatingAdjudicator(fail_on_chunk=3)
    with pytest.raises(RuntimeError, match="429") as ei:
        pp.run_pca_verify(_sentences(8),
                          layer_a_fn=_fake_classify_all_checkworthy,
                          adjudicate_fn=adj, chunk_size=CHUNK)
    assert len(adj.completed_rows) == 2 * CHUNK
    partial = ei.value.partial_result
    assert [r["sid"] for r in partial.rows] == [r["sid"] for r in adj.completed_rows]


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


# ── P67.3 landed: journal + budget probe (options 1 + 3) ─────────────────────

def test_journal_appends_per_chunk_and_resumes(tmp_path):
    """Option 1: every completed chunk lands in the JSONL immediately; a
    resumed run loads it and re-spends only on the missing sids."""
    journal = tmp_path / "run.jsonl"
    adj = _DetonatingAdjudicator(fail_on_chunk=3)
    with pytest.raises(RuntimeError, match="429"):
        pp.run_pca_verify(_sentences(8),
                          layer_a_fn=_fake_classify_all_checkworthy,
                          adjudicate_fn=adj, chunk_size=CHUNK,
                          journal_path=journal)
    rows, packs, cost, roster = pp.load_chunk_journal(journal)
    assert len(rows) == 2 * CHUNK

    adj2 = _DetonatingAdjudicator(fail_on_chunk=99)
    result = pp.run_pca_verify(_sentences(8),
                               layer_a_fn=_fake_classify_all_checkworthy,
                               adjudicate_fn=adj2, chunk_size=CHUNK,
                               resume_rows=rows, resume_packs=packs,
                               journal_path=journal)
    resumed_sids = {r["sid"] for r in rows}
    assert not resumed_sids & {r["sid"] for r in adj2.completed_rows}
    assert len(result.rows) == 8
    assert adj2.calls == 2   # only the two missing chunks


def test_budget_probe_halts_early_with_work_journaled(tmp_path):
    """Option 3: when headroom drops below projected chunk cost, the run halts
    BEFORE spending, raising BudgetHalt with the partial result attached."""
    journal = tmp_path / "run.jsonl"

    base = _DetonatingAdjudicator(fail_on_chunk=99)   # reports $1/chunk via wrapper
    def adj(chunk):
        rows, _ = base(chunk)
        return rows, {"cost_usd": 1.0}

    headroom = iter([10.0, 10.0, 1.0, 1.0])   # probed per chunk
    with pytest.raises(pp.BudgetHalt, match="preflight") as ei:
        pp.run_pca_verify(_sentences(8),
                          layer_a_fn=_fake_classify_all_checkworthy,
                          adjudicate_fn=adj, chunk_size=CHUNK,
                          journal_path=journal,
                          budget_check=lambda: next(headroom))
    partial = ei.value.partial_result
    assert len(partial.rows) == 2 * CHUNK        # halted before chunk 3
    rows, _, cost, _ = pp.load_chunk_journal(journal)
    assert len(rows) == 2 * CHUNK and cost == 2.0
