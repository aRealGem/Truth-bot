"""Phase-3 rebuild runner (scripts/phase3_rebuild.py) — offline, $0.

Everything here runs with no proxy key and no retriever: the guards are pure
functions, the budget breaker is exercised with a monkeypatched spend probe
and a fake pack builder, and the artifact writer round-trips through the same
bridge path scripts/rerender_pca_site.py uses. No model/API call anywhere.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "phase3_rebuild", REPO / "scripts" / "phase3_rebuild.py")
p3 = importlib.util.module_from_spec(_SPEC)
sys.modules["phase3_rebuild"] = p3
_SPEC.loader.exec_module(p3)       # must import clean with no key present


# ── helpers ──────────────────────────────────────────────────────────────────

def _row(sid, verdict, *, split=False, gate=""):
    """A verdict-contract row shaped like adjudicator output."""
    r = {"sid": sid, "status": "resolved" if verdict else "disagreement",
         "verdict": verdict, "confidence": 0.8 if verdict else None,
         "citations": [], "reasoning": "r", "votes": {}, "by_role": {},
         "split": split, "escalated": False}
    if gate:
        r["provenance_code"] = gate
    return r


def _ev(sid, url="https://www.bls.gov/x"):
    return {"claim_id": sid, "source_name": "BLS", "source_url": url,
            "source_tier": "Government", "snippet": "[2006-01-06] jobs data",
            "retrieved_at": "2026-08-02T00:00:00",
            "supports_claim": True, "relevance_score": 0.9}


def _packs(sids):
    from truthbot.verdict import publish_pipeline
    return publish_pipeline.packs_from_evidence_dict(
        {sid: [_ev(sid)] for sid in sids})


GATE = p3.GATE_INSUFFICIENT


def test_gate_constant_matches_consolidator():
    from truthbot.verdict.consolidator import GATE_INSUFFICIENT
    assert p3.GATE_INSUFFICIENT == GATE_INSUFFICIENT


# ── (a)/(b) --go refusals ────────────────────────────────────────────────────

def test_go_refused_without_economy_r2_model():
    msg = p3.go_refusal({}, 5.0)
    assert msg and "TRUTHBOT_R2_MODEL" in msg
    msg = p3.go_refusal({"TRUTHBOT_R2_MODEL": "gpt-5.5"}, 5.0)
    assert msg and "gpt-5-mini" in msg


def test_go_refused_without_budget():
    env = {"TRUTHBOT_R2_MODEL": "gpt-5-mini"}
    assert "--budget" in p3.go_refusal(env, None)
    assert "--budget" in p3.go_refusal(env, 0)
    # budget refusal fires even before the model check (never spend uncapped)
    assert "--budget" in p3.go_refusal({}, None)


def test_go_allowed_with_budget_and_economy_model():
    assert p3.go_refusal({"TRUTHBOT_R2_MODEL": "gpt-5-mini"}, 2.0) is None


# ── (c) per-claim breaker fires BEFORE retrieval ─────────────────────────────

def test_budget_halt_fires_before_retrieval(monkeypatch, tmp_path):
    from truthbot.verdict import proxy_lane
    monkeypatch.setattr(proxy_lane, "proxy_key_spend", lambda: 5.0)
    calls = []
    journal = tmp_path / "packs.jsonl"
    builder = p3.make_pack_builder(
        build_pack=lambda sid, text, ctx: calls.append(sid),
        cap=1.0, start_spend=0.0, packs_journal=journal)
    with pytest.raises(p3.BudgetHalt, match="before retrieving s:1"):
        builder("s:1", "text", "")
    assert calls == []                     # retrieval never ran
    assert not journal.exists()            # nothing journaled either


def test_pack_builder_runs_and_journals_under_cap(monkeypatch, tmp_path):
    from truthbot.verdict import proxy_lane
    monkeypatch.setattr(proxy_lane, "proxy_key_spend", lambda: 0.0)
    pack = _packs(["s:1"])["s:1"]
    journal = tmp_path / "packs.jsonl"
    builder = p3.make_pack_builder(
        build_pack=lambda sid, text, ctx: pack,
        cap=1.0, start_spend=0.0, packs_journal=journal)
    assert builder("s:1", "text", "") is pack
    rec = json.loads(journal.read_text().splitlines()[0])
    assert rec["sid"] == "s:1" and rec["evidence"]


def test_breaker_counts_offproxy_and_banked(monkeypatch):
    from truthbot.verdict import proxy_lane
    monkeypatch.setattr(proxy_lane, "proxy_key_spend", lambda: 0.0)
    builder = p3.make_pack_builder(
        build_pack=lambda *a: pytest.fail("retrieval must not run"),
        cap=1.0, start_spend=0.0, offproxy_est=lambda: 0.6, banked_cost=0.5)
    with pytest.raises(p3.BudgetHalt):
        builder("s:2", "t", "")


# ── (d) artifact writer round-trips through rerender's expectations ─────────

def _source_art():
    claims = [
        {"sid": "gwbush_2006:0001", "text": "Claim one.", "context": "ctx",
         "layer_a": {"label": "check-worthy", "source": "A2",
                     "claim_type": "statistic", "claim_shape": "c-self"}},
        {"sid": "gwbush_2006:0002", "text": "Claim two.", "context": "",
         "layer_a": {"label": "check-worthy", "source": "A2",
                     "claim_type": "historical", "claim_shape": "c-third"}},
    ]
    rows = [_row("gwbush_2006:0001", "TRUE"),
            _row("gwbush_2006:0002", "UNVERIFIABLE", gate=GATE)]
    return {"run_id": "old-run-id", "meta": {
                "speaker": "George W. Bush", "date": "2006-01-31",
                "speech_id": "gwbush_2006", "venue": "U.S. Capitol",
                "n_sentences": 10, "n_check_worthy": 2},
            "claims": claims, "rows": rows,
            "characterization": [{"sid": "gwbush_2006:0003", "text": "char"}],
            "evidence": {}}


def test_artifact_writer_roundtrip(tmp_path):
    src = _source_art()
    new_rows = [_row("gwbush_2006:0002", "FALSE"),   # order scrambled on purpose
                _row("gwbush_2006:0001", "TRUE")]
    packs = _packs([c["sid"] for c in src["claims"]])
    roster = {"name": "prod", "seats": {"proposer": ["opus-worker"]}}
    path, payload = p3.write_new_artifact(
        src, new_rows, packs, roster, speech_id="gwbush_2006",
        out_dir=tmp_path, cost_usd=1.23456)

    art = json.loads(path.read_text(encoding="utf-8"))
    for key in ("run_id", "meta", "claims", "rows", "evidence", "roster",
                "characterization"):
        assert key in art
    assert art["claims"] == src["claims"]              # verbatim — identity
    assert art["characterization"] == src["characterization"]
    assert art["meta"]["rebuild_of"] == "old-run-id"
    assert art["meta"]["pipeline_generation"] == "v2.3-role-axis-s5cap"
    assert art["meta"]["remediation"] == "phase-3 DC-5(b)"
    assert art["meta"]["speech_id"] == "gwbush_2006"
    assert art["meta"]["cost_usd"] == pytest.approx(1.2346)
    # rows re-ordered to the claims[] order
    assert [r["sid"] for r in art["rows"]] == [c["sid"] for c in src["claims"]]

    # The exact consumer path rerender_pca_site takes: evidence dumps →
    # packs_from_evidence_dict → bridge over rows+claims.
    from truthbot.verdict import bridge as bridge_mod
    from truthbot.verdict import publish_pipeline
    packs2 = publish_pipeline.packs_from_evidence_dict(art["evidence"])
    out = bridge_mod.bridge(art["rows"], art["claims"], packs2)
    assert len(out.bundles) == 2
    assert set(out.evidence) == {c["sid"] for c in src["claims"]}


def test_manifest_update_adds_unpublished_row_only(tmp_path):
    mpath = tmp_path / "methodology_manifest.json"
    mpath.write_text(json.dumps({
        "current_generation": "v2.3-role-axis-s5cap",
        "runs": {"old-run-id": {"speech_id": "gwbush_2006",
                                "generation": "v2.3-role-axis-s5cap",
                                "published": True}}}))
    p3.update_manifest("new-run-id", "gwbush_2006", manifest_path=mpath)
    m = json.loads(mpath.read_text())
    assert m["runs"]["new-run-id"] == {
        "speech_id": "gwbush_2006",
        "generation": "v2.3-role-axis-s5cap",
        "published": False}
    # the old run's row is untouched (un-publishing is human-gated)
    assert m["runs"]["old-run-id"]["published"] is True


# ── (e) verdict-diff classification ──────────────────────────────────────────

def test_outcome_label_separates_gated_from_panel_uv():
    assert p3.outcome_label(_row("s", "UNVERIFIABLE", gate=GATE)) \
        == "gated-UNVERIFIABLE"
    assert p3.outcome_label(_row("s", "UNVERIFIABLE")) == "UNVERIFIABLE"
    assert p3.outcome_label(_row("s", None, split=True)) == "Models split"
    assert p3.outcome_label(_row("s", None)) == "No verdict"
    # evidence_gate spelling is honored too
    assert p3.outcome_label({"sid": "s", "verdict": "UNVERIFIABLE",
                             "evidence_gate": GATE}) == "gated-UNVERIFIABLE"


def test_verdict_diff_classification():
    old = [_row("s:1", "TRUE"),
           _row("s:2", "FALSE"),
           _row("s:3", "TRUE"),
           _row("s:4", "UNVERIFIABLE", gate=GATE),
           _row("s:5", "TRUE"),
           _row("s:6", None, split=True)]
    new = [_row("s:1", "TRUE"),                         # unchanged
           _row("s:2", "MISLEADING"),                   # decided → decided
           _row("s:3", "UNVERIFIABLE", gate=GATE),      # newly gated
           _row("s:4", "FALSE"),                        # newly decided
           _row("s:5", None, split=True),               # split change
           _row("s:6", "UNVERIFIABLE")]                 # split → panel-decided
    diff = p3.build_verdict_diff(old, new)
    assert diff["n_compared"] == 6
    assert diff["counts"] == {"unchanged": 1,
                              "decided_to_decided_changed": 1,
                              "newly_gated": 1,
                              "newly_decided": 2,
                              "split_changes": 1,
                              "other": 0}
    # gate-forced UV in the new rows is COUNTED, not an error
    assert diff["gate_forced_new"] == 1
    by_sid = {e["sid"]: e for e in diff["per_sid"]}
    assert by_sid["s:3"]["category"] == "newly_gated"
    assert by_sid["s:4"]["old"] == "gated-UNVERIFIABLE"
    assert by_sid["s:4"]["category"] == "newly_decided"


def test_verdict_diff_is_partial_safe():
    old = [_row("s:1", "TRUE"), _row("s:2", "FALSE")]
    diff = p3.build_verdict_diff(old, [_row("s:1", "TRUE")])
    assert diff["n_compared"] == 1                 # only rebuilt sids compared


# ── (f) resume skips banked sids ─────────────────────────────────────────────

def test_resume_skips_banked_sids(tmp_path):
    from truthbot.verdict import publish_pipeline
    journal = tmp_path / "chunk.jsonl"
    publish_pipeline.append_chunk_journal(
        journal, 1, [_row("s:1", "TRUE")], {}, 0.25,
        roster={"name": "prod", "seats": {}})
    done_rows, _, banked, roster = publish_pipeline.load_chunk_journal(journal)
    assert banked == pytest.approx(0.25) and roster["name"] == "prod"
    claims = [{"sid": "s:1", "text": "a", "context": ""},
              {"sid": "s:2", "text": "b", "context": ""}]
    todo = p3.pending_claims(claims, done_rows)
    assert [c["sid"] for c in todo] == ["s:2"]


def test_estimate_is_offline_and_ranged():
    # Uses the real gwbush artifact when present (untracked file); skip on CI.
    if not p3.artifact_path("gwbush_2006").exists():
        pytest.skip("gwbush_2006 artifact not on disk")
    report = p3.estimate_report(["gwbush_2006"])
    n = len(p3.load_artifact("gwbush_2006")["claims"])
    lo, hi = n * p3.PER_CLAIM_EST[0], n * p3.PER_CLAIM_EST[1]
    assert f"${lo:.2f} - ${hi:.2f}" in report
    assert "$0" in report.splitlines()[0]          # explicitly a $0 projection


# ── transient-failure resilience (added 2026-08-05 after two long runs died
#    on single infrastructure blips: a proxy read timeout 60 claims into
#    biden, a worker-lane failure 80 claims into obama) ────────────────────

def test_transient_classifier_matches_infra_blips_only():
    import http.client
    import socket
    import urllib.error

    from phase3_rebuild import BudgetHalt, _is_transient

    for exc in (TimeoutError("timed out"),
                socket.timeout("timed out"),
                ConnectionResetError("reset by peer"),
                urllib.error.URLError("unreachable"),
                http.client.RemoteDisconnected("closed")):
        assert _is_transient(exc), exc

    class WorkerCallError(RuntimeError):
        pass

    assert _is_transient(WorkerCallError("lane died"))
    # Real defects and the budget breaker must NOT be retried.
    assert not _is_transient(ValueError("bad row"))
    assert not _is_transient(KeyError("sid"))
    assert not _is_transient(BudgetHalt("over cap"))


def test_chunk_retries_transient_then_succeeds(monkeypatch):
    from phase3_rebuild import _adjudicate_chunk

    monkeypatch.setattr("phase3_rebuild.time.sleep", lambda _s: None)
    calls = {"n": 0}

    class _Adj:
        def adjudicate(self, hm, chunk, **kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("timed out")
            return ["row"], {}, {"packs": {}}

    rows, _m, notes = _adjudicate_chunk(_Adj(), None, [], None, idx=1)
    assert rows == ["row"] and calls["n"] == 3 and notes == {"packs": {}}


def test_chunk_gives_up_cleanly_after_retries(monkeypatch):
    from phase3_rebuild import ChunkFailed, _adjudicate_chunk

    monkeypatch.setattr("phase3_rebuild.time.sleep", lambda _s: None)

    class _Adj:
        def adjudicate(self, hm, chunk, **kw):
            raise TimeoutError("timed out")

    with pytest.raises(ChunkFailed, match="attempts failed"):
        _adjudicate_chunk(_Adj(), None, [], None, idx=7)


def test_budget_halt_is_never_retried(monkeypatch):
    from phase3_rebuild import BudgetHalt, _adjudicate_chunk

    monkeypatch.setattr("phase3_rebuild.time.sleep", lambda _s: None)
    calls = {"n": 0}

    class _Adj:
        def adjudicate(self, hm, chunk, **kw):
            calls["n"] += 1
            raise BudgetHalt("over cap")

    with pytest.raises(BudgetHalt):
        _adjudicate_chunk(_Adj(), None, [], None, idx=1)
    assert calls["n"] == 1, "the cap must halt on the first raise"


def test_programming_errors_surface_immediately(monkeypatch):
    from phase3_rebuild import _adjudicate_chunk

    monkeypatch.setattr("phase3_rebuild.time.sleep", lambda _s: None)
    calls = {"n": 0}

    class _Adj:
        def adjudicate(self, hm, chunk, **kw):
            calls["n"] += 1
            raise ValueError("malformed row")

    with pytest.raises(ValueError):
        _adjudicate_chunk(_Adj(), None, [], None, idx=1)
    assert calls["n"] == 1, "real defects must not be masked by retries"
