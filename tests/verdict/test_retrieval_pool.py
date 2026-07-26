"""P120 PR-2: the parallel retrieval pool primitives (L1 shortlist fan-out + L2
claim pool). Offline: fake retrievers/builders, injected clock, tmp pressure file.
"""
from __future__ import annotations

import threading
import time

import pytest

from truthbot.verdict import retrieval_pool
from truthbot.verdict.pool_governor import PoolGovernor
from truthbot.verify.retrievers import ClaudeWorkerRetriever


class _R:
    def __init__(self, label):
        self.label = label


# ── L1: parallel_shortlists ───────────────────────────────────────────────────

def test_parallel_shortlists_preserves_pool_order():
    pool = [_R("A"), _R("B"), _R("C")]
    out = retrieval_pool.parallel_shortlists(pool, lambda r: [r.label])
    assert out == [["A"], ["B"], ["C"]]


def test_parallel_shortlists_runs_concurrently():
    # Three 50ms calls should overlap → wall-clock well under the 150ms serial sum.
    pool = [_R("A"), _R("B"), _R("C")]

    def slow(_r):
        time.sleep(0.05)
        return []

    start = time.monotonic()
    retrieval_pool.parallel_shortlists(pool, slow)
    assert time.monotonic() - start < 0.13


def test_r1_gate_caps_concurrent_cli_workers():
    # r1_cli_cap=1 → the two claude-CLI (R1) calls must NOT overlap.
    g = PoolGovernor(r1_cli_cap=1)
    pool = [ClaudeWorkerRetriever(label="R1a"), ClaudeWorkerRetriever(label="R1b")]
    active, peak, lock = [], [0], threading.Lock()

    def call(r):
        with lock:
            active.append(1)
            peak[0] = max(peak[0], len(active))
        time.sleep(0.05)
        with lock:
            active.pop()
        return [r.label]

    retrieval_pool.parallel_shortlists(pool, call, governor=g)
    assert peak[0] == 1                       # gate serialized the two R1 workers


def test_non_r1_not_gated():
    # Non-R1 retrievers ignore the r1_gate and overlap freely.
    g = PoolGovernor(r1_cli_cap=1)
    pool = [_R("R2"), _R("R3")]
    active, peak, lock = [], [0], threading.Lock()

    def call(r):
        with lock:
            active.append(1)
            peak[0] = max(peak[0], len(active))
        time.sleep(0.05)
        with lock:
            active.pop()
        return []

    retrieval_pool.parallel_shortlists(pool, call, governor=g)
    assert peak[0] == 2


# ── L2: build_packs_pooled ────────────────────────────────────────────────────

def _todo(n):
    return [{"sid": f"s{i}", "text": "t", "context": "c"} for i in range(n)]


def _ok_gov(tmp_path, **kw):
    p = tmp_path / "pressure.json"
    p.write_text('{"level":"ok","mem_avail_mb":8000,"ts":1000}')
    kw.setdefault("now_fn", lambda: 1000.0)
    kw.setdefault("sleep_fn", lambda s: None)
    return PoolGovernor(pressure_path=str(p), **kw)


def test_build_packs_pooled_builds_all_and_journals(tmp_path):
    g = _ok_gov(tmp_path, pool_max=3)
    journaled = []
    packs = retrieval_pool.build_packs_pooled(
        _todo(5), lambda sid, t, c: f"pack-{sid}", g,
        on_pack=lambda sid, p: journaled.append(sid))
    assert set(packs) == {f"s{i}" for i in range(5)}
    assert packs["s0"] == "pack-s0"
    assert sorted(journaled) == [f"s{i}" for i in range(5)]
    assert g.telemetry()["max_in_flight"] >= 1


def test_build_packs_pooled_empty_todo(tmp_path):
    g = _ok_gov(tmp_path)
    assert retrieval_pool.build_packs_pooled(
        [], lambda *a: None, g, on_pack=lambda *a: None) == {}


def test_build_packs_pooled_propagates_and_journals_completed(tmp_path):
    # Serial admission (target=1 via warn/missing pressure) → deterministic order:
    # s0, s1 complete+journal, then s2 raises and propagates.
    g = PoolGovernor(pressure_path=str(tmp_path / "none.json"),
                     now_fn=lambda: 1000.0, sleep_fn=lambda s: None, pool_max=3)
    journaled = []

    def pb(sid, t, c):
        if sid == "s2":
            raise RuntimeError("boom")
        return f"pack-{sid}"

    with pytest.raises(RuntimeError, match="boom"):
        retrieval_pool.build_packs_pooled(
            _todo(4), pb, g, on_pack=lambda sid, p: journaled.append(sid))
    assert journaled == ["s0", "s1"]          # completed work banked; s2 never journaled
    assert "s2" not in journaled


def test_build_packs_pooled_respects_low_target(tmp_path):
    # warn pressure → target 1 → never more than one build in flight at once.
    g = PoolGovernor(pressure_path=str(tmp_path / "none.json"),
                     now_fn=lambda: 1000.0, sleep_fn=lambda s: None, pool_max=4)
    active, peak, lock = [], [0], threading.Lock()

    def pb(sid, t, c):
        with lock:
            active.append(1)
            peak[0] = max(peak[0], len(active))
        time.sleep(0.02)
        with lock:
            active.pop()
        return sid

    retrieval_pool.build_packs_pooled(_todo(4), pb, g, on_pack=lambda *a: None)
    assert peak[0] == 1
