"""Offline tests for the independent truth-bot run ledger. A scripted lane
produces a real RunManifest (project="truth-bot"); the ledger row is asserted to
carry the client identity, cost provenance, and run config. No network."""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "eval" / "benchmarks"))

from hydramind import HydraMind
from hydramind.types import Call, CallResult, Lane
from hydramind.registry import load_registry, SPECS_DIR
from hydramind.transport import Transport, call_key
from hydramind.manifest import NullSpendSink

import ledger as run_ledger


def _scripted(outs, cost=0.002, source="proxy"):
    def fn(call: Call) -> CallResult:
        return CallResult(call=call, output=outs[call_key(call)], lane=Lane.L_P,
                          cost_usd=cost, cost_source=source, tokens_in=10, tokens_out=5,
                          returned_model=call.binding.model)
    return fn


def _truthbot_pca_run():
    outs = {
        "proposer:c1": {"verdict": "TRUE", "confidence": 0.9, "citations": []},
        "critic:c1":   {"verdict": "FALSE", "confidence": 0.9, "citations": []},
        "arbiter:c1":  {"verdict": "TRUE", "confidence": 0.8, "citations": []},
        "proposer:c2": {"verdict": "TRUE", "confidence": 0.9, "citations": []},
        "critic:c2":   {"verdict": "TRUE", "confidence": 0.9, "citations": []},
    }
    hm = HydraMind(load_registry(SPECS_DIR), Transport(completion_fn=_scripted(outs)),
                   spend_sink=NullSpendSink(), project="truth-bot")
    items = [{"item_id": "c1", "payload": {"claim": "x"}},
             {"item_id": "c2", "payload": {"claim": "y"}}]
    return hm.run("verdict", items, "pca", roster="dev")


def test_build_record_carries_client_and_cost_provenance():
    result, manifest = _truthbot_pca_run()
    rec = run_ledger.build_record(
        manifest, notes=result.notes, ts="2026-07-09T00:00:00+00:00", run_id="testrun01",
        config={"key_label": "truth-bot", "base_url": "http://127.0.0.1:4141"})
    assert rec["client"] == "truth-bot"           # client identity, not the pca strategy
    assert rec["strategy"] == "pca" and rec["roster"] == "dev"
    assert rec["key_label"] == "truth-bot"
    assert rec["cost"]["cost_source_tally"].get("proxy")   # proxy-sourced, authoritative
    assert rec["cost"]["total_cost_usd"] > 0
    assert rec["budget_ceiling_usd"] == 2.00       # from pca.yaml resolved spec
    assert rec["result"]["escalation"]["escalated"] == 1   # c1 escalated to arbiter
    assert rec["run_id"] == "testrun01"


def test_append_run_writes_jsonl_roundtrip(tmp_path):
    result, manifest = _truthbot_pca_run()
    path = tmp_path / "sub" / "truthbot.jsonl"     # nested dir is created
    r1 = run_ledger.append_run(path, manifest, notes=result.notes, run_id="r1")
    r2 = run_ledger.append_run(path, manifest, notes=result.notes, run_id="r2")
    lines = path.read_text().splitlines()
    assert len(lines) == 2                         # append-only
    rows = [json.loads(l) for l in lines]
    assert [r["run_id"] for r in rows] == ["r1", "r2"]
    assert all(r["client"] == "truth-bot" for r in rows)
    assert rows[0] == r1 and rows[1] == r2         # returned == persisted
