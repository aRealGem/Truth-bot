"""D17-d R7 pack-anatomy probe — offline, $0.

Pins the two findings that decide whether pack anatomy can carry a decidability
signal: the field inventory (no per-item disqualification codes survive) and the
non-separation of web-tier1 from substantive.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "d17d_pack_anatomy_probe", REPO / "scripts" / "d17d_pack_anatomy_probe.py")
r7 = importlib.util.module_from_spec(_SPEC)
sys.modules["d17d_pack_anatomy_probe"] = r7
_SPEC.loader.exec_module(r7)


def test_probe_is_deterministic():
    assert json.dumps(r7.build(), sort_keys=True) == \
           json.dumps(r7.build(), sort_keys=True)


def test_covers_the_same_128_packs():
    doc = r7.build()
    assert doc["totals"]["claims"] == 128
    assert doc["totals"]["evidence_items"] == 969


def test_no_per_item_disqualification_codes_survive():
    """The gate's own reasoning is not recoverable from a stored item: role,
    era_note, utterance_rule, quota_credit, disqualification_code and gate_code
    are all absent. If any of these ever starts being persisted, this fails and
    R7 should be re-run — the analysis would change."""
    inv = r7.build()["field_inventory"]
    for f in ("role", "era_note", "utterance_rule", "quota_credit",
              "disqualification_code", "gate_code"):
        assert inv[f]["present"] is False, f"{f} now persisted — re-run R7"
    for f in ("source_tier", "supports_claim"):
        assert inv[f]["present"] is True


def test_stored_fields_cannot_reproduce_the_gate():
    """78 of 128 packs score >=2 credits under a reconstruction from stored
    fields, yet the real gate rejected all 128 — the artifact under-describes
    its own decision by 61%."""
    g = r7.build()["gate_reproduction_check"]
    assert g["packs"] == 128
    assert g["proxy_says_quota_met"] == 78
    assert g["proxy_disagreement_rate"] == 0.6094


def test_anatomy_does_not_separate_webtier1_from_substantive():
    """The two classes a render must never confuse are indistinguishable by
    pack anatomy — every mean differs by less than 0.4. If a future corpus
    separates them, that is a new signal and this test should be revisited
    deliberately, not silently."""
    s = r7.build()["separation"]
    web, sub = s["web_tier1"], s["substantive"]
    for k in ("n_tier13_mean", "n_bearing_mean", "proxy_quota_credits_mean"):
        assert abs(web[k] - sub[k]) < 0.4, (
            f"{k} now separates the classes ({web[k]} vs {sub[k]}) — "
            "re-read the R7 conclusion before relying on it")
    # the tier signal is faintly INVERTED: substantive packs carry marginally
    # more Tier-1..3 sources than documentable ones
    assert sub["n_tier13_mean"] > web["n_tier13_mean"]


def test_no_threshold_is_fitted_to_the_desk_pass():
    """R7 reports distributions only. A tuned cut would launder the fixture
    into the classifier, so the probe must expose no decision boundary."""
    doc = r7.build()
    assert "threshold" not in json.dumps(doc["separation"]).lower()
    for c in doc["claims"]:
        assert "predicted_class" not in c
