"""P96.2.1 escalation criterion: the NAMED `label_mismatch` policy (confidence is
NOT part of the trigger) and the advisory escalation-rate monitor surfaced in the
manifest. Offline — scripted lane, no proxy."""
from hydramind import HydraMind
from hydramind.types import Call, CallResult, Lane
from hydramind.registry import load_registry, SPECS_DIR
from hydramind.transport import Transport, call_key
from hydramind import invariants as inv


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


# ── the named criterion resolver ──────────────────────────────────────────────

def test_is_escalation_split_label_mismatch_ignores_confidence():
    # labels agree, |Δconf| huge → NOT a split under label_mismatch
    assert not inv.is_escalation_split("label_mismatch", "TRUE", "TRUE", 0.9, 0.1, 0.25)
    # labels differ → split (confidence irrelevant)
    assert inv.is_escalation_split("label_mismatch", "TRUE", "FALSE", 0.9, 0.9, 0.25)


def test_is_escalation_split_material_disagreement_uses_confidence():
    # legacy rule still trips on a big confidence gap even with equal labels
    assert inv.is_escalation_split("material_disagreement", "TRUE", "TRUE", 0.9, 0.1, 0.25)


# ── pca honors label_mismatch (the pca.yaml decided policy) ────────────────────

def test_pca_confidence_only_gap_does_not_escalate():
    outs = {
        "proposer:c1": {"verdict": "TRUE", "confidence": 0.9, "citations": []},
        "critic:c1":   {"verdict": "TRUE", "confidence": 0.4, "citations": []},  # Δ0.5
        "proposer:c2": {"verdict": "TRUE", "confidence": 0.9, "citations": []},
        "critic:c2":   {"verdict": "TRUE", "confidence": 0.9, "citations": []},
    }
    result, manifest = _hm(outs).run("verdict", _items("c1", "c2"), "pca", roster="dev")
    assert result.notes["split_criterion"] == "label_mismatch"
    assert result.notes["escalation"]["criterion"] == "label_mismatch"
    assert result.notes["escalation_rate"] == 0.0
    assert all(c.role != "arbiter" for c in manifest.cost_records)   # never escalated


def test_pca_label_mismatch_escalates():
    outs = {
        "proposer:c1": {"verdict": "TRUE", "confidence": 0.9, "citations": []},
        "critic:c1":   {"verdict": "FALSE", "confidence": 0.9, "citations": []},
        "arbiter:c1":  {"verdict": "TRUE", "confidence": 0.8, "citations": []},
        "proposer:c2": {"verdict": "TRUE", "confidence": 0.9, "citations": []},
        "critic:c2":   {"verdict": "TRUE", "confidence": 0.9, "citations": []},
    }
    result, manifest = _hm(outs).run("verdict", _items("c1", "c2"), "pca", roster="dev")
    assert result.notes["escalation_rate"] == 0.5
    assert [c.item_id for c in manifest.cost_records if c.role == "arbiter"] == ["c1"]


def test_material_disagreement_tune_restores_confidence_trigger():
    outs = {
        "proposer:c1": {"verdict": "TRUE", "confidence": 0.9, "citations": []},
        "critic:c1":   {"verdict": "TRUE", "confidence": 0.4, "citations": []},  # Δ0.5
        "arbiter:c1":  {"verdict": "TRUE", "confidence": 0.8, "citations": []},
        "proposer:c2": {"verdict": "TRUE", "confidence": 0.9, "citations": []},
        "critic:c2":   {"verdict": "TRUE", "confidence": 0.9, "citations": []},
    }
    result, manifest = _hm(outs).run("verdict", _items("c1", "c2"), "pca", roster="dev",
                                     tune={"escalation.criterion": "material_disagreement"})
    assert result.notes["split_criterion"] == "material_disagreement"
    assert result.notes["escalation_rate"] == 0.5          # conf gap now escalates
    assert any(c.role == "arbiter" for c in manifest.cost_records)


# ── advisory monitor in the manifest (flags, never gates) ──────────────────────

def test_manifest_escalation_monitor_under_watermark():
    outs = {
        "proposer:c1": {"verdict": "TRUE", "confidence": 0.9, "citations": []},
        "critic:c1":   {"verdict": "FALSE", "confidence": 0.9, "citations": []},
        "arbiter:c1":  {"verdict": "TRUE", "confidence": 0.8, "citations": []},
        "proposer:c2": {"verdict": "TRUE", "confidence": 0.9, "citations": []},
        "critic:c2":   {"verdict": "TRUE", "confidence": 0.9, "citations": []},
    }
    _, manifest = _hm(outs).run("verdict", _items("c1", "c2"), "pca", roster="dev")
    mon = manifest.escalation
    assert mon["escalated"] == 1 and mon["total"] == 2 and mon["rate"] == 0.5
    assert mon["watermark"] == 0.50
    assert mon["over_watermark"] is False        # 0.5 is not > 0.5 → advisory clear


def test_manifest_escalation_monitor_trips_watermark():
    # both items escalate → rate 1.0; lower the watermark via tune → flag trips
    outs = {
        "proposer:c1": {"verdict": "TRUE", "confidence": 0.9, "citations": []},
        "critic:c1":   {"verdict": "FALSE", "confidence": 0.9, "citations": []},
        "arbiter:c1":  {"verdict": "TRUE", "confidence": 0.8, "citations": []},
        "proposer:c2": {"verdict": "FALSE", "confidence": 0.9, "citations": []},
        "critic:c2":   {"verdict": "TRUE", "confidence": 0.9, "citations": []},
        "arbiter:c2":  {"verdict": "FALSE", "confidence": 0.8, "citations": []},
    }
    _, manifest = _hm(outs).run("verdict", _items("c1", "c2"), "pca", roster="dev",
                                tune={"escalation.monitor.rate_watermark": 0.40})
    mon = manifest.escalation
    assert mon["rate"] == 1.0 and mon["watermark"] == 0.40
    assert mon["over_watermark"] is True         # flags only — run still completes
