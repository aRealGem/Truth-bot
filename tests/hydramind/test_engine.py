"""Engine + transport + strategy integration tests, using fake lane backends
(no network). Covers single/pca flow, I2 flagging, I4 fail, tune re-validation,
lane selection, batch reconciliation, and cost-ceiling halt."""
import pytest

from hydramind import HydraMind, TaskItem, StrategyResultKind, invariants as inv
from hydramind.types import Call, CallResult, Lane, Kind
from hydramind.registry import load_registry, SPECS_DIR
from hydramind.transport import Transport, call_key
from hydramind.manifest import NullSpendSink


def make_hm(completion_fn, batch_backend=None, sink=None):
    reg = load_registry(SPECS_DIR)
    tr = Transport(completion_fn=completion_fn, batch_backend=batch_backend)
    return HydraMind(reg, tr, spend_sink=sink or NullSpendSink())


# ── programmable fakes ────────────────────────────────────────────────────────

def scripted(outputs, cost=0.001):
    """outputs: dict keyed by call_key -> output dict."""
    def fn(call: Call) -> CallResult:
        out = outputs[call_key(call)]
        return CallResult(call=call, output=out, lane=Lane.L_P,
                          cost_usd=cost, tokens_in=10, tokens_out=5)
    return fn


class FakeBatch:
    def __init__(self, outputs, cost=0.0005):
        self.outputs = outputs
        self.cost = cost
        self.calls_seen = 0

    def run_batch(self, calls):
        self.calls_seen += len(calls)
        res = {}
        for c in calls:
            res[call_key(c)] = CallResult(call=c, output=self.outputs[call_key(c)],
                                          lane=Lane.L_B, cost_usd=self.cost,
                                          tokens_in=8, tokens_out=4)
        return res


# ── single ────────────────────────────────────────────────────────────────────

def test_single_flow():
    items = [TaskItem("s1", {"text": "a"}), TaskItem("s2", {"text": "b"})]
    outs = {"solo:s1": {"label": "check-worthy"}, "solo:s2": {"label": "opinion"}}
    hm = make_hm(scripted(outs))
    result, manifest = hm.run("classify", items, "single")
    labels = {r.item_id: r.value["label"] for r in result.items}
    assert labels == {"s1": "check-worthy", "s2": "opinion"}
    assert manifest.n_items == 2
    assert manifest.lane_tally.get("L-P") == 2
    assert manifest.dataset_hash


# ── pca: agreement, disagreement→arbiter, no-plurality flag, I4 ────────────────

def _pca_items(with_pack=True):
    pack = {"evidence_pack_ids": ["e1", "e2"]} if with_pack else {}
    return [TaskItem("c1", {"claim": "x", **pack}),
            TaskItem("c2", {"claim": "y", **pack})]


def test_pca_agreement_resolves_without_arbiter():
    outs = {
        "proposer:c1": {"verdict": "TRUE", "confidence": 0.9, "citations": ["e1"]},
        "critic:c1":   {"verdict": "TRUE", "confidence": 0.85, "citations": ["e1"]},
        "proposer:c2": {"verdict": "FALSE", "confidence": 0.8, "citations": ["e2"]},
        "critic:c2":   {"verdict": "FALSE", "confidence": 0.8, "citations": ["e2"]},
    }
    hm = make_hm(scripted(outs))
    result, manifest = hm.run("verdict", _pca_items(), "pca")
    kinds = {r.item_id: r.kind for r in result.items}
    assert kinds["c1"] == StrategyResultKind.RESOLVED
    assert manifest.lane_tally.get("L-B") is None  # below min_lot → all L-P
    # arbiter never fired (no disagreement)
    assert all(cr.role != "arbiter" for cr in manifest.cost_records)


def test_pca_disagreement_goes_to_arbiter_then_plurality():
    outs = {
        "proposer:c1": {"verdict": "TRUE", "confidence": 0.9, "citations": ["e1"]},
        "critic:c1":   {"verdict": "FALSE", "confidence": 0.9, "citations": ["e1"]},
        "arbiter:c1":  {"verdict": "TRUE", "confidence": 0.8, "citations": ["e1"]},
        "proposer:c2": {"verdict": "TRUE", "confidence": 0.9, "citations": ["e2"]},
        "critic:c2":   {"verdict": "TRUE", "confidence": 0.9, "citations": ["e2"]},
    }
    hm = make_hm(scripted(outs))
    result, _ = hm.run("verdict", _pca_items(), "pca")
    by = {r.item_id: r for r in result.items}
    # c1: TRUE/FALSE/arbiter TRUE → plurality TRUE
    assert by["c1"].kind == StrategyResultKind.RESOLVED
    assert by["c1"].value["verdict"] == "TRUE"


def test_pca_no_plurality_flags_disagreement():
    # proposer/critic disagree → arbiter adds a THIRD distinct label → no plurality
    outs = {
        "proposer:c1": {"verdict": "TRUE", "confidence": 0.9, "citations": ["e1"]},
        "critic:c1":   {"verdict": "FALSE", "confidence": 0.9, "citations": ["e1"]},
        "arbiter:c1":  {"verdict": "UNVERIFIABLE", "confidence": 0.5, "citations": ["e1"]},
        "proposer:c2": {"verdict": "TRUE", "confidence": 0.9, "citations": ["e2"]},
        "critic:c2":   {"verdict": "TRUE", "confidence": 0.9, "citations": ["e2"]},
    }
    hm = make_hm(scripted(outs))
    result, _ = hm.run("verdict", _pca_items(), "pca")
    by = {r.item_id: r for r in result.items}
    assert by["c1"].kind == StrategyResultKind.DISAGREEMENT_FLAGGED   # I2


def test_pca_i4_citation_outside_pack_hard_fails():
    outs = {
        "proposer:c1": {"verdict": "TRUE", "confidence": 0.9, "citations": ["eX"]},
        "critic:c1":   {"verdict": "TRUE", "confidence": 0.9, "citations": ["eX"]},
        "proposer:c2": {"verdict": "TRUE", "confidence": 0.9, "citations": ["e2"]},
        "critic:c2":   {"verdict": "TRUE", "confidence": 0.9, "citations": ["e2"]},
    }
    hm = make_hm(scripted(outs))
    with pytest.raises(inv.I4CitationError):
        hm.run("verdict", _pca_items(), "pca")


def test_pca_forced_arbitration_tune():
    outs = {
        "proposer:c1": {"verdict": "TRUE", "confidence": 0.9, "citations": ["e1"]},
        "critic:c1":   {"verdict": "TRUE", "confidence": 0.9, "citations": ["e1"]},
        "arbiter:c1":  {"verdict": "TRUE", "confidence": 0.9, "citations": ["e1"]},
        "proposer:c2": {"verdict": "TRUE", "confidence": 0.9, "citations": ["e2"]},
        "critic:c2":   {"verdict": "TRUE", "confidence": 0.9, "citations": ["e2"]},
        "arbiter:c2":  {"verdict": "TRUE", "confidence": 0.9, "citations": ["e2"]},
    }
    hm = make_hm(scripted(outs))
    _, manifest = hm.run("verdict", _pca_items(), "pca", tune={"flow.gate": "always"})
    arb = [c for c in manifest.cost_records if c.role == "arbiter"]
    assert len(arb) == 2   # both items forced to arbiter


# ── tune cannot defeat invariants ──────────────────────────────────────────────

def test_tune_injecting_grok_proposer_refails():
    hm = make_hm(scripted({}))
    with pytest.raises(inv.I1GrokPoolError):
        hm.run("verdict", _pca_items(), "pca",
               tune={"roles.proposer.providers": ["anthropic", "grok"]})


# ── transport lane selection + batch reconciliation ────────────────────────────

def test_lane_selects_batch_when_lot_and_eligible():
    # 100 items, single min_lot=100, wave1 eligible → L-B
    items = [TaskItem(f"s{i}", {"text": "t"}) for i in range(100)]
    outs = {f"solo:s{i}": {"label": "opinion"} for i in range(100)}
    fb = FakeBatch(outs)
    hm = make_hm(scripted(outs), batch_backend=fb)
    result, manifest = hm.run("classify", items, "single")
    assert fb.calls_seen == 100
    assert manifest.lane_tally.get("L-B") == 100
    assert manifest.lane_tally.get("L-P") is None
    assert len(result.items) == 100


def test_lane_stays_proxy_below_min_lot():
    items = [TaskItem(f"s{i}", {"text": "t"}) for i in range(10)]
    outs = {f"solo:s{i}": {"label": "opinion"} for i in range(10)}
    fb = FakeBatch(outs)
    hm = make_hm(scripted(outs), batch_backend=fb)
    _, manifest = hm.run("classify", items, "single")
    assert fb.calls_seen == 0
    assert manifest.lane_tally.get("L-P") == 10


# ── cost ceiling halt ───────────────────────────────────────────────────────────

def test_cost_ceiling_halts():
    items = [TaskItem(f"s{i}", {"text": "t"}) for i in range(10)]
    outs = {f"solo:s{i}": {"label": "opinion"} for i in range(10)}
    # single ceiling 2.00; make each call absurdly expensive → halt on wave1
    hm = make_hm(scripted(outs, cost=1.0))
    _, manifest = hm.run("classify", items, "single")
    assert manifest.halted
    assert "ceiling" in manifest.halt_reason
