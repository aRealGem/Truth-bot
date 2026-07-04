"""I1–I6 hard-guard unit tests. Guards must FAIL (raise), never warn."""
import copy
import pytest

from hydramind import invariants as inv
from hydramind.registry import build_spec, load_registry, SPECS_DIR

PCA_RAW = {
    "name": "pca", "caps": ["batch", "multi_round"],
    "roles": {
        "proposer": {"tier": "standard", "providers": ["anthropic", "openai"]},
        "critic": {"tier": "standard", "providers": ["mistral", "anthropic", "grok"]},
        "arbiter": {"tier": "frontier", "providers": ["anthropic", "openai"], "rotation": "round_robin"},
    },
    "flow": {"wave1": ["proposer", "critic"], "gate": "material_disagreement", "wave2": ["arbiter"]},
    "gate_threshold": 0.25, "tie_policy": "flag_disagreement",
    "batch": {"eligible_waves": ["wave1", "wave2"], "min_lot": 20},
}


def test_shipped_specs_load_clean():
    reg = load_registry(SPECS_DIR)
    assert {"single", "pca"} <= set(reg)


def test_i1_grok_ok_as_critic():
    build_spec(copy.deepcopy(PCA_RAW))  # grok is critic-only → fine


def test_i1_grok_in_proposer_fails():
    raw = copy.deepcopy(PCA_RAW)
    raw["roles"]["proposer"]["providers"].append("grok")
    with pytest.raises(inv.I1GrokPoolError):
        build_spec(raw)


def test_i1_grok_in_arbiter_fails():
    raw = copy.deepcopy(PCA_RAW)
    raw["roles"]["arbiter"]["providers"] = ["anthropic", "grok"]
    with pytest.raises(inv.I1GrokPoolError):
        build_spec(raw)


def test_i3_speaker_spec_key_fails():
    raw = copy.deepcopy(PCA_RAW)
    raw["roles"]["proposer"]["per_speaker"] = {"trump": {"tier": "frontier"}}
    with pytest.raises(inv.I3SpeakerConditionalError):
        build_spec(raw)


def test_i3_template_linter_rejects_speaker_conditional():
    with pytest.raises(inv.I3SpeakerConditionalError):
        inv.lint_template_for_speaker_conditionals(
            "t", "Judge this. {% if speaker == 'Trump' %}be harsh{% endif %}")


def test_i3_template_allows_model_provenance_conditional():
    # Conditioning on WHICH MODEL produced output is allowed (Principle 2).
    inv.lint_template_for_speaker_conditionals(
        "t", "If the model is grok, treat as critic-only input.")


def test_i2_material_disagreement_definition():
    assert inv.is_material_disagreement("TRUE", "FALSE", 0.9, 0.9, 0.25)   # label mismatch
    assert inv.is_material_disagreement("TRUE", "TRUE", 0.9, 0.6, 0.25)    # |Δconf| ≥ thr
    assert not inv.is_material_disagreement("TRUE", "TRUE", 0.9, 0.8, 0.25)


def test_i4_citations_subset():
    inv.check_i4_citations(["e1", "e2"], ["e1", "e2", "e3"])
    with pytest.raises(inv.I4CitationError):
        inv.check_i4_citations(["e1", "eX"], ["e1", "e2"])


def test_i5_provenance_required():
    inv.check_i5_provenance({"url": "u", "retrieved_at": "t", "sha256": "h", "tier": "1"})
    with pytest.raises(inv.I5ProvenanceError):
        inv.check_i5_provenance({"url": "u"})


def test_i6_heldout_read_once():
    g = inv.HeldoutGuard()
    g.read("claim_set.heldout", "rc1")
    g.read("claim_set.heldout", "rc2")      # different RC ok
    with pytest.raises(inv.I6HeldoutReuseError):
        g.read("claim_set.heldout", "rc1")   # second read same RC ⇒ fail
