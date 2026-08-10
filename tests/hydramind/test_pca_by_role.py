"""P67 Phase 3: pca.reduce captures per-seat vote attribution (agreement.by_role).

Kills the 2-1 tally ambiguity — the arbiter's own label is readable from the
artifact instead of provable only via the label_mismatch escalation theorem.
Offline — scripted lane, no proxy."""
from hydramind import HydraMind, StrategyResultKind
from hydramind.types import Call, CallResult, Lane
from hydramind.registry import load_registry, SPECS_DIR
from hydramind.transport import Transport, call_key


def _scripted(outputs, cost=0.001):
    def fn(call: Call) -> CallResult:
        return CallResult(call=call, output=outputs[call_key(call)], lane=Lane.L_P,
                          cost_usd=cost, tokens_in=10, tokens_out=5,
                          returned_model=call.binding.model)
    return fn


def _hm(outs):
    return HydraMind(load_registry(SPECS_DIR), Transport(completion_fn=_scripted(outs)))


def _items(*ids):
    return [{"item_id": i, "payload": {"claim": i}} for i in ids]


def test_reduce_captures_by_role_on_escalated_2_1():
    # P=MISLEADING, C=FALSE → escalate; A=MISLEADING → 2-1. Before by_role the
    # winner's seat was only inferable; now it's explicit in the agreement.
    outs = {
        "proposer:c1": {"verdict": "MISLEADING", "confidence": 0.8, "citations": []},
        "critic:c1":   {"verdict": "FALSE", "confidence": 0.9, "citations": []},
        "arbiter:c1":  {"verdict": "MISLEADING", "confidence": 0.7, "citations": []},
    }
    result, _ = _hm(outs).run("verdict", _items("c1"), "pca", roster="dev")
    it = result.items[0]
    assert it.kind is StrategyResultKind.RESOLVED
    assert it.value["verdict"] == "MISLEADING"
    assert it.agreement["by_role"] == {
        "proposer": ["MISLEADING"], "critic": ["FALSE"], "arbiter": ["MISLEADING"]}


def test_reduce_captures_by_role_on_unanimous_and_tie():
    outs = {
        # c1 unanimous, never escalates → no arbiter key
        "proposer:c1": {"verdict": "TRUE", "confidence": 0.9, "citations": []},
        "critic:c1":   {"verdict": "TRUE", "confidence": 0.9, "citations": []},
        # c2 three-way tie → DISAGREEMENT_FLAGGED, by_role still captured
        "proposer:c2": {"verdict": "MISLEADING", "confidence": 0.8, "citations": []},
        "critic:c2":   {"verdict": "FALSE", "confidence": 0.8, "citations": []},
        "arbiter:c2":  {"verdict": "UNVERIFIABLE", "confidence": 0.5, "citations": []},
    }
    result, _ = _hm(outs).run("verdict", _items("c1", "c2"), "pca", roster="dev")
    by_id = {r.item_id: r for r in result.items}
    assert by_id["c1"].agreement["by_role"] == {"proposer": ["TRUE"], "critic": ["TRUE"]}
    tie = by_id["c2"]
    assert tie.kind is StrategyResultKind.DISAGREEMENT_FLAGGED
    assert tie.agreement["by_role"] == {
        "proposer": ["MISLEADING"], "critic": ["FALSE"], "arbiter": ["UNVERIFIABLE"]}


# ── R-3 (2026-08-10): reduce also captures each seat's RATIONALE TEXT ─────────
# by_role says what each seat concluded; seat_rationales says why, verbatim.
# Without it a TIE carries no rationale anywhere on disk, which is how the
# stage-2 discriminator came to publish verdicts with an empty reasoning field.

def test_reduce_captures_seat_rationales_on_a_tie():
    outs = {
        "proposer:c2": {"verdict": "MISLEADING", "confidence": 0.8, "citations": ["E1"],
                        "reasoning": "real decline, overstated framing"},
        "critic:c2":   {"verdict": "FALSE", "confidence": 0.8, "citations": [],
                        "reasoning": "the series contradicts the figure"},
        "arbiter:c2":  {"verdict": "UNVERIFIABLE", "confidence": 0.5, "citations": [],
                        "reasoning": "nothing dated before the utterance settles it"},
    }
    result, _ = _hm(outs).run("verdict", _items("c2"), "pca", roster="dev")
    tie = result.items[0]
    assert tie.kind is StrategyResultKind.DISAGREEMENT_FLAGGED
    seats = {s["role"]: s for s in tie.agreement["seat_rationales"]}
    assert set(seats) == {"proposer", "critic", "arbiter"}
    assert seats["proposer"]["reasoning"] == "real decline, overstated framing"
    assert seats["proposer"]["verdict"] == "MISLEADING"
    assert seats["proposer"]["citations"] == ["E1"]
    assert seats["arbiter"]["reasoning"] == "nothing dated before the utterance settles it"


def test_reduce_captures_seat_rationales_on_a_resolved_row():
    outs = {
        "proposer:c1": {"verdict": "TRUE", "confidence": 0.9, "citations": [],
                        "reasoning": "BEA confirms the figure"},
        "critic:c1":   {"verdict": "TRUE", "confidence": 0.9, "citations": [],
                        "reasoning": "no contrary series in the pack"},
    }
    result, _ = _hm(outs).run("verdict", _items("c1"), "pca", roster="dev")
    seats = result.items[0].agreement["seat_rationales"]
    assert [s["reasoning"] for s in seats] == ["BEA confirms the figure",
                                               "no contrary series in the pack"]


def test_no_labels_item_carries_an_empty_seat_rationale_list():
    """A shape guarantee: every agreement dict has the key, so downstream code
    never has to distinguish 'no seats' from 'field not captured'."""
    outs = {
        "proposer:c3": {"confidence": 0.9, "citations": []},
        "critic:c3":   {"confidence": 0.9, "citations": []},
    }
    result, _ = _hm(outs).run("verdict", _items("c3"), "pca", roster="dev")
    assert result.items[0].agreement["seat_rationales"] == []
