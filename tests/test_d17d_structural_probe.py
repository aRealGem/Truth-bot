"""D17-d structural probe (scripts/d17d_structural_probe.py) — offline, $0.

Locks the properties that make the probe usable as evidence for an owner scope
decision:
  * structural-only — the rules never see claim text;
  * full coverage — all 128 gate-withheld claims are dispositioned;
  * determinism — same inputs -> byte-identical output;
  * fixture-lock — the per-rule confusion counts are pinned, so any change in
    the upstream signals surfaces as a test diff rather than a silent shift;
  * the SAFETY property — every committed error runs in the same direction
    (predicting 'undecidable' for a documentable claim). If that ever flips,
    the risk calculus for a "cannot be verified" render changes and this test
    must fail loudly.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "d17d_structural_probe", REPO / "scripts" / "d17d_structural_probe.py")
probe = importlib.util.module_from_spec(_SPEC)
sys.modules["d17d_structural_probe"] = probe
_SPEC.loader.exec_module(probe)      # must import clean with no key present


# ── structural-only guarantee ────────────────────────────────────────────────

def test_rules_read_structure_not_prose():
    """fire_rule takes no text argument by design; only structured fields move
    the outcome, and the strongest signal wins outright."""
    assert probe.fire_rule("statistical", None, False) == "R4-statistical-unattached"
    # an attached series beats every softer signal, even a substantive-looking one
    assert probe.fire_rule("attribution", "c-eval", True) == "R1-series-attached"
    # precedence: attribution outranks shape
    assert probe.fire_rule("attribution", "c-eval", False) == "R2-attribution-type"
    # nothing usable -> abstain loudly, never a silent default class
    assert probe.fire_rule(None, None, False) == "R6-no-signal"


def test_every_rule_id_is_declared():
    doc = probe.build()
    for c in doc["claims"]:
        assert c["rule_id"] in probe.RULES
        disp, predicted, residual, _, _ = probe.RULES[c["rule_id"]]
        assert c["disposition"] == disp
        assert c["predicted_class"] == predicted
        assert c["residual_class_range"] == residual


def test_abstentions_never_predict_a_class():
    """An abstention must carry a residual range and no prediction — the whole
    point is that it declines to commit."""
    doc = probe.build()
    for c in doc["claims"]:
        if c["disposition"] == "abstained":
            assert c["predicted_class"] is None
            assert c["agree"] is None
            assert c["residual_class_range"]
        else:
            assert c["predicted_class"] is not None
            assert isinstance(c["agree"], bool)


# ── full coverage ────────────────────────────────────────────────────────────

def test_covers_every_gate_withheld_claim():
    doc = probe.build()
    desk = json.loads(probe.DESK.read_text(encoding="utf-8"))
    t = doc["totals"]
    assert t["gate_withheld"] == desk["gate_withheld_total"] == 128
    assert len(doc["claims"]) == 128
    assert {c["sid"] for c in doc["claims"]} == {c["sid"] for c in desk["claims"]}
    assert t["committed"] + t["abstained"] == 128


# ── determinism ──────────────────────────────────────────────────────────────

def test_probe_is_deterministic():
    a = json.dumps(probe.build(), sort_keys=True)
    b = json.dumps(probe.build(), sort_keys=True)
    assert a == b


# ── fixture-lock: the measured shape of the disagreement ────────────────────

def test_totals_are_pinned():
    t = probe.build()["totals"]
    assert t == {
        "gate_withheld": 128,
        "committed": 37,
        "abstained": 91,
        "committed_agree": 7,
        "committed_error": 30,
        "abstained_residual_contains_desk": 87,
        "abstained_residual_misses_desk": 4,
    }


def test_per_rule_counts_are_pinned():
    pr = probe.build()["per_rule"]
    assert (pr["R1-series-attached"]["n_fired"],
            pr["R1-series-attached"]["n_agree"],
            pr["R1-series-attached"]["n_error"]) == (1, 1, 0)
    assert (pr["R2-attribution-type"]["n_fired"],
            pr["R2-attribution-type"]["n_agree"],
            pr["R2-attribution-type"]["n_error"]) == (17, 4, 13)
    assert (pr["R3-eval-shape"]["n_fired"],
            pr["R3-eval-shape"]["n_agree"],
            pr["R3-eval-shape"]["n_error"]) == (19, 2, 17)
    assert pr["R4-statistical-unattached"]["n_fired"] == 22
    assert pr["R4-statistical-unattached"]["n_residual_misses_desk"] == 3
    assert pr["R5-narrative-type"]["n_fired"] == 69
    assert pr["R5-narrative-type"]["n_residual_misses_desk"] == 1
    assert pr["R6-no-signal"]["n_fired"] == 0


# ── the safety property ──────────────────────────────────────────────────────

def test_every_committed_error_runs_one_direction():
    """All 30 committed errors predict 'substantive' for a claim the desk found
    documentable; none runs the other way. This is the polarity that argues
    against a "cannot be verified" render — if it ever flips, re-do the risk
    analysis before shipping a label."""
    doc = probe.build()
    errors = [c for c in doc["claims"]
              if c["disposition"] == "committed" and not c["agree"]]
    assert len(errors) == 30
    assert all(c["predicted_class"] == "substantive" for c in errors)
    assert all(c["desk_class"] != "substantive" for c in errors)


def test_commit_rules_have_the_measured_precision():
    """R2 and R3 are the rules a render would lean on, and both are worse than
    a coin flip against the fixture."""
    pr = probe.build()["per_rule"]
    r2 = pr["R2-attribution-type"]
    r3 = pr["R3-eval-shape"]
    assert r2["n_agree"] / r2["n_fired"] < 0.25
    assert r3["n_agree"] / r3["n_fired"] < 0.15


# ── what structure cannot express ────────────────────────────────────────────

def test_compound_split_is_structurally_inexpressible():
    inv = probe.build()["structurally_inexpressible"]["compound-split"]
    assert inv["desk_count"] == 5
    assert inv["n_recovered"] == 0


def test_series_attachment_reached_only_one_of_seven():
    inv = probe.build()["structurally_inexpressible"]["series-core"]
    assert inv["desk_count"] == 7
    assert inv["n_recovered"] == 1


def test_anecdote_precedence_overlap_is_pinned():
    """Both readings of 'anecdote-precedence' lose as a substantive signal."""
    o = probe.build()["anecdote_precedence_overlap"]
    assert o["desk_substantive_n"] == 35
    assert o["R5_narrative_all"]["n"] == 69
    assert o["R5_narrative_all"]["intersection_with_desk_substantive"] == 28
    assert o["R5_personal_anecdote_only"]["n"] == 47
    assert o["R5_personal_anecdote_only"]["intersection_with_desk_substantive"] == 22
    # precision below 0.5 either way — wrong more often than right
    assert o["R5_narrative_all"]["precision_if_treated_as_substantive"] < 0.5
    assert o["R5_personal_anecdote_only"]["precision_if_treated_as_substantive"] < 0.5
