"""P120 PR-2: the adaptive pool governor — Pi-pressure sizing + API/subscription
backoff. Fully offline: pressure.json is a tmp file, the clock and sleep are injected.
"""
from __future__ import annotations

import json

import pytest

from truthbot.verdict.pool_governor import CriticalPressureTimeout, PoolGovernor
from truthbot.verify.retrievers import ClaudeWorkerRetriever, OpenAIBrowsingRetriever


def _write(tmp_path, level, mem=8000, ts=1000):
    p = tmp_path / "pressure.json"
    p.write_text(json.dumps({"level": level, "mem_avail_mb": mem, "ts": ts, "reason": ""}))
    return str(p)


def _gov(path, **kw):
    kw.setdefault("now_fn", lambda: 1000.0)
    kw.setdefault("sleep_fn", lambda s: None)
    return PoolGovernor(pressure_path=path, **kw)


# ── target sizing from Pi pressure ────────────────────────────────────────────

def test_ok_grows_to_ceiling(tmp_path):
    assert _gov(_write(tmp_path, "ok"), pool_max=3).target_claims_in_flight() == 3


def test_warn_pares_to_serial(tmp_path):
    assert _gov(_write(tmp_path, "warn"), pool_max=3).target_claims_in_flight() == 1


def test_critical_pares_to_serial(tmp_path):
    assert _gov(_write(tmp_path, "critical"), pool_max=3).target_claims_in_flight() == 1


def test_low_mem_pares_even_at_ok(tmp_path):
    g = _gov(_write(tmp_path, "ok", mem=1500), pool_max=3, mem_floor_mb=2000)
    assert g.target_claims_in_flight() == 1


def test_missing_pressure_is_failsafe_warn(tmp_path):
    g = _gov(str(tmp_path / "nope.json"), pool_max=3)
    assert g.read_pressure()["level"] == "warn"
    assert g.target_claims_in_flight() == 1


def test_stale_pressure_is_failsafe_warn(tmp_path):
    # ts=0 vs now=1e9 → far older than pressure_stale_s → warn regardless of level.
    g = PoolGovernor(pressure_path=_write(tmp_path, "ok", ts=0),
                     now_fn=lambda: 1e9, pool_max=3, pressure_stale_s=5400)
    assert g.read_pressure()["level"] == "warn"
    assert g.target_claims_in_flight() == 1


def test_no_adaptive_freezes_at_start(tmp_path):
    # Even with a healthy ok file, --no-adaptive holds the pool at pool_start.
    g = _gov(_write(tmp_path, "ok"), pool_start=2, pool_max=3, adaptive=False)
    assert g.target_claims_in_flight() == 2


# ── R1 Claude Max drop / re-admit ─────────────────────────────────────────────

def test_r1_drops_then_readmits_after_cooldown():
    t = [1000.0]
    g = PoolGovernor(now_fn=lambda: t[0], r1_cooldown_s=900)
    assert g.r1_available()
    g.note_max_signal()
    assert not g.r1_available()
    t[0] += 901
    assert g.r1_available()


def test_active_retrievers_drops_r1_when_backed_off():
    g = PoolGovernor(now_fn=lambda: 1000.0)
    pool = (ClaudeWorkerRetriever(), OpenAIBrowsingRetriever())
    assert len(g.active_retrievers(pool)) == 2      # healthy → all
    g.note_max_signal()
    kept = g.active_retrievers(pool)
    assert [type(r).__name__ for r in kept] == ["OpenAIBrowsingRetriever"]


def test_active_retrievers_never_empty_if_r1_only():
    g = PoolGovernor(now_fn=lambda: 1000.0)
    g.note_max_signal()
    pool = (ClaudeWorkerRetriever(),)
    assert len(g.active_retrievers(pool)) == 1       # R1 rides rather than build nothing


# ── R2/R3 429 lane backoff ────────────────────────────────────────────────────

def test_429_pares_target_then_recovers(tmp_path):
    t = [1000.0]
    g = PoolGovernor(pressure_path=_write(tmp_path, "ok"), now_fn=lambda: t[0], pool_max=3)
    assert g.target_claims_in_flight() == 3
    g.note_429("R2")
    assert g.target_claims_in_flight() == 2          # healthy but a lane is backing off
    t[0] += 61
    assert g.target_claims_in_flight() == 3          # backoff expired


# ── critical pause / timeout ──────────────────────────────────────────────────

def test_wait_if_critical_resumes_when_pressure_clears(tmp_path):
    p = tmp_path / "pressure.json"
    p.write_text(json.dumps({"level": "critical", "ts": 1000, "mem_avail_mb": 8000}))
    calls = {"n": 0}

    def sleeper(_s):
        calls["n"] += 1
        if calls["n"] >= 2:                           # flip to ok after 2 polls
            p.write_text(json.dumps({"level": "ok", "ts": 1000, "mem_avail_mb": 8000}))

    g = PoolGovernor(pressure_path=str(p), now_fn=lambda: 1000.0, sleep_fn=sleeper,
                     pressure_wait_s=600)
    g.wait_if_critical()                              # returns rather than raising
    assert calls["n"] >= 2
    assert g.telemetry()["events"].get("critical_pauses") == 1


def test_wait_if_critical_times_out(tmp_path):
    p = _write(tmp_path, "critical")
    g = PoolGovernor(pressure_path=p, now_fn=lambda: 1000.0, sleep_fn=lambda s: None,
                     pressure_wait_s=30)
    with pytest.raises(CriticalPressureTimeout):
        g.wait_if_critical()
