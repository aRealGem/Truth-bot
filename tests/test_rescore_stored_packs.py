"""B1a re-score script (scripts/rescore_stored_packs.py) — offline, $0.

The funded path is exercised end to end with a STUB llm and a fake spend probe:
no proxy key, no build_proxy_llm, no HTTP, no model call anywhere. What is
actually under test is the money machinery — the --go refusals, the per-claim
breaker firing BEFORE a call, sidecar resume so a halt never re-spends, and the
promise that the stored artifact is never touched.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "rescore_stored_packs", REPO / "scripts" / "rescore_stored_packs.py")
rs = importlib.util.module_from_spec(_SPEC)
sys.modules["rescore_stored_packs"] = rs
_SPEC.loader.exec_module(rs)        # must import clean with no key present


# ── helpers ──────────────────────────────────────────────────────────────────

def _ev(url, *, supports=None, relevance=0.5):
    """A stored evidence dump as the artifact really holds it: unscored."""
    return {"claim_id": "s:0001", "source_name": "AP", "source_url": url,
            "source_tier": "Wire", "snippet": "a snippet",
            "retrieved_at": "2026-08-03T04:20:27.824223",
            "published_at": "2026-02-20T00:00:00",
            "supports_claim": supports, "relevance_score": relevance}


def _artifact(n_sids=3, n_items=2):
    sids = [f"trump_2026:{i:04d}" for i in range(1, n_sids + 1)]
    return {
        "run_id": "test-run-0001",
        "claims": [{"sid": s, "text": f"claim text {s}"} for s in sids],
        "evidence": {s: [_ev(f"https://apnews.com/{s}/{j}")
                         for j in range(n_items)] for s in sids},
        "rows": [{"sid": s} for s in sids],
    }


def _write_artifact(tmp_path, art):
    p = tmp_path / "artifact.json"
    p.write_text(json.dumps(art), encoding="utf-8")
    return p


class _Args:
    """argparse.Namespace stand-in for run_rescore."""

    def __init__(self, **kw):
        self.speech = "trump_2026"
        self.artifact = None
        self.go = True
        self.budget = 5.0
        self.model = "claude-haiku"
        self.out = None
        self.limit = 0
        self.__dict__.update(kw)


def _stub_llm(calls, *, relevance=0.9, supports=True):
    """What a real Haiku reply parses to — built locally, spending nothing."""
    def llm(system, user):
        payload = json.loads(user)
        calls.append(payload)
        return {"scores": [{"i": item["i"], "relevance": relevance,
                            "supports": supports}
                           for item in payload["items"]]}
    return llm


@pytest.fixture
def funded(monkeypatch):
    """Wire the funded path to a stub lane + a scripted spend ledger."""
    from truthbot.verdict import proxy_lane
    from truthbot.verify import relevance

    state = {"spend": 0.0, "per_call": 0.0, "calls": []}
    monkeypatch.setattr(proxy_lane, "key_present", lambda: True)
    monkeypatch.setattr(proxy_lane, "proxy_key_spend", lambda: state["spend"])

    inner = _stub_llm(state["calls"])

    def llm(system, user):
        out = inner(system, user)
        state["spend"] += state["per_call"]      # the ledger moves as it would
        return out

    monkeypatch.setattr(relevance, "build_proxy_llm", lambda *a, **kw: llm)
    state["llm"] = llm          # so a test can restore the healthy lane
    return state


# ── $0 surfaces ─────────────────────────────────────────────────────────────

def test_estimate_prices_from_the_real_stored_payload():
    from truthbot.verify.relevance import _SCORE_SYSTEM, score_payload

    art = _artifact(n_sids=2, n_items=3)
    est = rs.estimate_speech(art)
    assert est["calls"] == 2 and est["items"] == 6

    # The measured prompt volume is the ACTUAL score_evidence prompt, not a guess.
    from truthbot.verdict.publish_pipeline import evidence_from_artifact_dict
    by_sid = evidence_from_artifact_dict(art["evidence"])
    expected = sum(len(_SCORE_SYSTEM) + len(score_payload(f"claim text {sid}", evs))
                   for sid, evs in by_sid.items())
    assert est["prompt_chars"] == expected
    assert est["cost_usd_est"] > 0


def test_estimate_skips_sids_with_no_claim_text():
    art = _artifact(n_sids=2)
    art["claims"][0]["text"] = ""          # never score against an empty claim
    est = rs.estimate_speech(art)
    assert est["calls"] == 1
    assert est["skipped_no_claim_text"] == ["trump_2026:0001"]


def test_estimate_is_zero_for_a_model_with_no_rate():
    assert rs.estimate_speech(_artifact(), model="no-such-model")["cost_usd_est"] == 0.0


# ── --go refusals ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("budget", [None, 0, -1.0])
def test_go_refused_without_a_positive_budget(budget):
    msg = rs.go_refusal(budget)
    assert msg and "--budget USD is REQUIRED" in msg and "No spend attempted" in msg


def test_go_allowed_with_a_budget():
    assert rs.go_refusal(1.50) is None


def test_run_refuses_and_spends_nothing_without_budget(tmp_path, funded):
    art = _artifact()
    rc = rs.run_rescore(_Args(artifact=str(_write_artifact(tmp_path, art)),
                              budget=None, out=str(tmp_path / "side.json")))
    assert rc == 1
    assert funded["calls"] == []                       # nothing was sent


# ── the funded path (stubbed) ───────────────────────────────────────────────

def test_rescore_writes_sidecar_and_never_touches_the_artifact(tmp_path, funded):
    art = _artifact(n_sids=3, n_items=2)
    art_path = _write_artifact(tmp_path, art)
    before = art_path.read_text(encoding="utf-8")
    out = tmp_path / "rescored.json"

    rc = rs.run_rescore(_Args(artifact=str(art_path), out=str(out)))
    assert rc == 0
    assert len(funded["calls"]) == 3                   # one call per sid

    # The record is untouched — archive-never-delete.
    assert art_path.read_text(encoding="utf-8") == before

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema"] == rs.SIDECAR_SCHEMA
    assert doc["source_run"] == "test-run-0001"
    assert set(doc["sids"]) == set(art["evidence"])
    row = doc["sids"]["trump_2026:0001"][0]
    assert row["relevance_score"] == 0.9 and row["supports_claim"] is True
    assert row["source_url"].startswith("https://apnews.com/")
    assert doc["soft_failures"] == []


def test_resume_never_re_spends_on_a_scored_sid(tmp_path, funded):
    art = _artifact(n_sids=3)
    art_path = _write_artifact(tmp_path, art)
    out = tmp_path / "rescored.json"

    rs.run_rescore(_Args(artifact=str(art_path), out=str(out), limit=1))
    assert len(funded["calls"]) == 1
    funded["calls"].clear()

    rs.run_rescore(_Args(artifact=str(art_path), out=str(out)))
    assert len(funded["calls"]) == 2                   # only the 2 unscored sids

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert len(doc["sids"]) == 3

    funded["calls"].clear()
    assert rs.run_rescore(_Args(artifact=str(art_path), out=str(out))) == 0
    assert funded["calls"] == []                       # idempotent, $0 re-run


def test_per_claim_breaker_fires_before_the_call(tmp_path, funded):
    """The cap must stop the NEXT call, not merely report the overrun after it."""
    art = _artifact(n_sids=5)
    art_path = _write_artifact(tmp_path, art)
    out = tmp_path / "rescored.json"
    funded["per_call"] = 0.40                          # each call costs $0.40

    rc = rs.run_rescore(_Args(artifact=str(art_path), out=str(out), budget=1.0))
    assert rc == 2                                     # halted, not completed
    # $0.40, $0.80 land; the third check sees $1.20 >= $1.00 and stops.
    assert len(funded["calls"]) == 3
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert len(doc["sids"]) == 3                       # everything paid for is banked
    assert doc["spend_usd"] == pytest.approx(1.2)


def test_halt_then_resume_completes_without_re_spending(tmp_path, funded, capsys):
    art = _artifact(n_sids=5)
    art_path = _write_artifact(tmp_path, art)
    out = tmp_path / "rescored.json"
    funded["per_call"] = 0.40

    assert rs.run_rescore(_Args(artifact=str(art_path), out=str(out), budget=1.0)) == 2
    assert "HALTED CLEANLY" in capsys.readouterr().out        # no traceback
    n_first = len(funded["calls"])

    funded["per_call"] = 0.0                                  # fresh budget
    assert rs.run_rescore(_Args(artifact=str(art_path), out=str(out), budget=9.0)) == 0
    assert len(funded["calls"]) == 5                          # 5 total, never 6+
    assert len(json.loads(out.read_text(encoding="utf-8"))["sids"]) == 5
    assert n_first == 3


def test_soft_failure_is_flagged_and_left_unbanked(tmp_path, funded, monkeypatch):
    """score_evidence fails SOFT — a hiccup leaves the 0.5 defaults in place and
    raises nothing. Banking that would record coverage never obtained AND make
    the resume skip it forever, so it must stay unbanked and visible."""
    from truthbot.verify import relevance

    monkeypatch.setattr(rs.time, "sleep", lambda s: None)
    monkeypatch.setattr(relevance, "build_proxy_llm",
                        lambda *a, **kw: (lambda s, u: {"scores": []}))
    art_path = _write_artifact(tmp_path, _artifact(n_sids=2))
    out = tmp_path / "rescored.json"

    assert rs.run_rescore(_Args(artifact=str(art_path), out=str(out))) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["sids"] == {}                       # nothing claimed as scored
    assert sorted(doc["soft_failures"]) == ["trump_2026:0001", "trump_2026:0002"]


def test_a_soft_failed_sid_is_retried_on_resume_and_can_recover(tmp_path, funded,
                                                                monkeypatch):
    from truthbot.verify import relevance

    monkeypatch.setattr(rs.time, "sleep", lambda s: None)
    monkeypatch.setattr(relevance, "build_proxy_llm",
                        lambda *a, **kw: (lambda s, u: {"scores": []}))
    art_path = _write_artifact(tmp_path, _artifact(n_sids=2))
    out = tmp_path / "rescored.json"
    rs.run_rescore(_Args(artifact=str(art_path), out=str(out)))
    assert json.loads(out.read_text(encoding="utf-8"))["sids"] == {}

    # Lane recovers; the resume picks the soft-failed sids back up.
    monkeypatch.setattr(relevance, "build_proxy_llm",
                        lambda *a, **kw: funded["llm"])
    assert rs.run_rescore(_Args(artifact=str(art_path), out=str(out))) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert len(doc["sids"]) == 2
    assert doc["soft_failures"] == []              # the flag is cleared on rescue


def test_a_lane_outage_retries_then_halts_cleanly(tmp_path, funded,
                                                  monkeypatch, capsys):
    """A dead lane must retry, then stop with resume instructions — never a
    traceback, never losing the sids already paid for, and never grinding
    through every remaining sid."""
    from truthbot.verify import relevance

    slept: list = []
    monkeypatch.setattr(rs.time, "sleep", slept.append)   # no real backoff wait
    n = {"i": 0}

    def flaky(system, user):
        n["i"] += 1
        if n["i"] == 1:                                   # first sid succeeds
            return {"scores": [{"i": 1, "relevance": 0.9, "supports": True},
                               {"i": 2, "relevance": 0.9, "supports": True}]}
        raise TimeoutError("proxy read timeout")          # swallowed → unscored

    monkeypatch.setattr(relevance, "build_proxy_llm", lambda *a, **kw: flaky)
    art_path = _write_artifact(tmp_path, _artifact(n_sids=8))
    out = tmp_path / "rescored.json"

    assert rs.run_rescore(_Args(artifact=str(art_path), out=str(out))) == 2
    text = capsys.readouterr().out
    assert "SOFT-FAILURE HALT" in text and "HALTED CLEANLY" in text
    assert "Resume" in text and "--go --budget" in text
    assert "Traceback" not in text
    # SOFT_FAIL_HALT sids failed, each retried CHUNK_RETRIES times.
    assert len(slept) == rs.SOFT_FAIL_HALT * (rs.CHUNK_RETRIES - 1)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert list(doc["sids"]) == ["trump_2026:0001"]       # the one that worked
    assert len(doc["soft_failures"]) == rs.SOFT_FAIL_HALT
    # It stopped at the outage instead of burning the other 4 sids.
    assert n["i"] == 1 + rs.SOFT_FAIL_HALT * rs.CHUNK_RETRIES


# ── sidecar integrity ───────────────────────────────────────────────────────

def test_sidecar_refuses_a_different_source_run(tmp_path):
    out = tmp_path / "rescored.json"
    out.write_text(json.dumps({"schema": rs.SIDECAR_SCHEMA,
                               "speech_id": "trump_2026",
                               "source_run": "OTHER-run", "sids": {}}),
                   encoding="utf-8")
    with pytest.raises(ValueError, match="different artifact revision"):
        rs.load_sidecar(out, "trump_2026", "test-run-0001")


def test_sidecar_refuses_another_speech(tmp_path):
    out = tmp_path / "rescored.json"
    out.write_text(json.dumps({"schema": rs.SIDECAR_SCHEMA,
                               "speech_id": "biden_2022",
                               "source_run": "r", "sids": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="speech_id"):
        rs.load_sidecar(out, "trump_2026", "r")


def test_missing_sidecar_starts_empty(tmp_path):
    doc = rs.load_sidecar(tmp_path / "nope.json", "trump_2026", "r")
    assert doc["sids"] == {} and doc["schema"] == rs.SIDECAR_SCHEMA


def test_rebuilt_run_ids_resolve_to_real_artifacts():
    """The registry points at the REBUILT runs the 4,344-item census covers —
    not phase3_rebuild.SPEECHES, which holds the superseded originals."""
    import importlib.util as iu

    spec = iu.spec_from_file_location("p3r", REPO / "scripts" / "phase3_rebuild.py")
    p3 = iu.module_from_spec(spec)
    sys.modules["p3r"] = p3
    spec.loader.exec_module(p3)

    assert set(rs.REBUILT_RUNS) == set(p3.SPEECHES)
    total = 0
    for speech, run_id in rs.REBUILT_RUNS.items():
        path = rs.artifact_path(speech)
        assert path.exists(), path
        assert run_id != p3.SPEECHES[speech]["run_id"]      # rebuild, not original
        art = rs.load_artifact(path)
        assert art["run_id"] == run_id
        total += sum(len(v) for v in art["evidence"].values())
    assert total == 4344            # the census the DC-B1 estimate is built on
