"""Roster loading, roles_allowed hard guard, and roster-driven pca binding."""
import pytest

from hydramind import HydraMind, StrategyResultKind
from hydramind.types import Call, CallResult, Lane
from hydramind.registry import load_registry, SPECS_DIR
from hydramind.transport import Transport, call_key
from hydramind import rosters
from hydramind.rosters import load_rosters, get_roster, validate_roster, Roster, RosterRoleError


def test_dev_roster_complete_prod_incomplete():
    rs = load_rosters()
    assert rs["dev"].complete and rs["dev"].seats["proposer"] == ["mistral"]
    assert rs["dev"].seats["critic"] == ["dsv4-flash"]
    assert not rs["prod"].complete            # TBD seats
    assert rs["prod"].seats["critic"] == ["grok", "dsv4-flash"]


def test_get_prod_roster_refused_incomplete():
    with pytest.raises(RosterRoleError):
        get_roster("prod")                    # TBD seats ⇒ not runnable


def test_roles_allowed_guard():
    # grok may not propose; dsv4-flash may not arbitrate
    with pytest.raises(RosterRoleError):
        validate_roster(Roster("bad1", {"proposer": ["grok"], "critic": ["mistral"],
                                        "arbiter": ["claude-haiku"]}, True))
    with pytest.raises(RosterRoleError):
        validate_roster(Roster("bad2", {"proposer": ["mistral"], "critic": ["dsv4-flash"],
                                        "arbiter": ["dsv4-flash"]}, True))
    # legal: grok & dsv4-flash both as critics
    validate_roster(Roster("ok", {"proposer": ["mistral"], "critic": ["grok", "dsv4-flash"],
                                  "arbiter": ["claude-haiku"]}, True))


def _scripted(outputs):
    def fn(call: Call) -> CallResult:
        return CallResult(call=call, output=outputs[call_key(call)], lane=Lane.L_P,
                          cost_usd=0.001, tokens_in=10, tokens_out=5,
                          returned_model=call.binding.model)
    return fn


def test_pca_dev_roster_binds_seats_and_logs_split():
    reg = load_registry(SPECS_DIR)
    outs = {
        # c1: proposer(mistral) TRUE vs critic(dsv4-flash) FALSE → split → arbiter(claude-haiku)
        "proposer:c1": {"verdict": "TRUE", "confidence": 0.9, "citations": []},
        "critic:c1":   {"verdict": "FALSE", "confidence": 0.9, "citations": []},
        "arbiter:c1":  {"verdict": "TRUE", "confidence": 0.8, "citations": []},
        # c2: agreement → no split, no arbiter
        "proposer:c2": {"verdict": "TRUE", "confidence": 0.9, "citations": []},
        "critic:c2":   {"verdict": "TRUE", "confidence": 0.9, "citations": []},
    }
    hm = HydraMind(reg, Transport(completion_fn=_scripted(outs)))
    items = [{"item_id": "c1", "payload": {"claim": "x"}},
             {"item_id": "c2", "payload": {"claim": "y"}}]
    result, manifest = hm.run("verdict", items, "pca", roster="dev")
    # seat models bound from roster
    models = {(c.role): c.model for c in manifest.cost_records}
    assert models["proposer"] == "mistral"
    assert models["critic"] == "dsv4-flash"
    assert models["arbiter"] == "claude-haiku"
    # split + escalation logged
    assert result.notes["split_rate"] == 0.5
    assert result.notes["escalation_rate"] == 0.5
    by = {r.item_id: r for r in result.items}
    assert by["c1"].value["verdict"] == "TRUE"        # plurality after arbiter
    assert by["c1"].agreement["split"] and by["c1"].agreement["escalated"]
    assert not by["c2"].agreement["split"]
