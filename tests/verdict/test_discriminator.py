"""CRM-114 discriminator — offline. Prompt lint + the pure override logic; the live
binary call is exercised by the --crm114 scoring run, not here."""
from __future__ import annotations

from types import SimpleNamespace

from truthbot.verdict import discriminator
from truthbot.verdict.discriminator import (
    CRM114_SYSTEM, apply_discrimination, apply_tie_routing, discriminate)


# ── prompt ────────────────────────────────────────────────────────────────────

def test_crm114_prompt_is_speaker_blind_and_binary():
    t = CRM114_SYSTEM
    assert "speaker" not in t.lower()
    assert "FALSE" in t and "MISLEADING" in t
    assert "CORE assertion" in t and "contradiction" in t.lower()
    # P67 Phase 3: the absolute-claim rule must be present — counterexamples
    # contradict an absolute core (zero/only/ended/...) ⇒ FALSE, not MISLEADING.
    assert "ABSOLUTE-CLAIM RULE" in t


# ── apply_discrimination: only re-labels resolved adverse rows ─────────────────

def _row(sid, status="resolved", verdict=None):
    return {"sid": sid, "status": status, "verdict": verdict, "citations": []}


def test_override_flips_adverse_labels_and_records():
    rows = [_row("a", verdict="MISLEADING"), _row("b", verdict="FALSE")]
    apply_discrimination(rows, {"a": "FALSE", "b": "FALSE"})
    assert rows[0]["verdict"] == "FALSE" and rows[0]["crm114"] == {"stage1": "MISLEADING", "final": "FALSE"}
    # b unchanged (disc == stage1) → no crm114 marker
    assert rows[1]["verdict"] == "FALSE" and "crm114" not in rows[1]


def test_override_ignores_true_abstain_and_missing():
    rows = [
        _row("t", verdict="TRUE"),                       # not adverse
        _row("u", verdict="UNVERIFIABLE"),               # not adverse
        _row("d", status="disagreement", verdict=None),  # not resolved
        _row("m", verdict="MISLEADING"),                 # adverse but no disc entry
    ]
    apply_discrimination(rows, {"t": "FALSE", "u": "FALSE"})  # disc for non-adverse ignored
    assert rows[0]["verdict"] == "TRUE"
    assert rows[1]["verdict"] == "UNVERIFIABLE"
    assert rows[2]["verdict"] is None
    assert rows[3]["verdict"] == "MISLEADING" and "crm114" not in rows[3]


def test_override_ignores_invalid_disc_label():
    rows = [_row("a", verdict="FALSE")]
    apply_discrimination(rows, {"a": "TRUE"})   # discriminator must stay binary
    assert rows[0]["verdict"] == "FALSE" and "crm114" not in rows[0]


# ── apply_tie_routing: resolves adverse-severity ties, explicitly (I2) ─────────

def test_tie_routing_resolves_flagged_row_and_records():
    rows = [dict(_row("d", status="disagreement"),
                 votes={"MISLEADING": 1, "FALSE": 1, "UNVERIFIABLE": 1})]
    apply_tie_routing(rows, {"d": "FALSE"})
    assert rows[0]["status"] == "resolved" and rows[0]["verdict"] == "FALSE"
    assert rows[0]["crm114"] == {"stage1": "DISAGREEMENT", "final": "FALSE"}
    assert rows[0]["votes"] == {"MISLEADING": 1, "FALSE": 1, "UNVERIFIABLE": 1}  # tie stays readable


def test_tie_routing_leaves_unrouted_and_resolved_rows_alone():
    rows = [
        dict(_row("d", status="disagreement"), votes={"TRUE": 1, "MISLEADING": 1}),  # not routed
        _row("r", verdict="MISLEADING"),                                             # resolved
    ]
    apply_tie_routing(rows, {"r": "FALSE"})    # disc entry for a resolved row is ignored here
    assert rows[0]["status"] == "disagreement" and rows[0]["verdict"] is None
    assert rows[1]["status"] == "resolved" and rows[1]["verdict"] == "MISLEADING"


def test_tie_routing_ignores_invalid_disc_label():
    rows = [dict(_row("d", status="disagreement"), votes={"MISLEADING": 1, "FALSE": 1})]
    apply_tie_routing(rows, {"d": "TRUE"})
    assert rows[0]["status"] == "disagreement" and "crm114" not in rows[0]


# ── discriminate: parses a valid binary label, drops the rest ─────────────────

class _FakeHM:
    def __init__(self, outputs):
        self._outputs = outputs   # item_id -> raw verdict string
        self.called_with = None

    def run(self, task, items, strat, *, roster=None, tune=None):
        self.called_with = (task, strat, roster, tune)
        it = [SimpleNamespace(item_id=i["item_id"],
                              value={"verdict": self._outputs.get(i["item_id"])})
              for i in items]
        return SimpleNamespace(items=it, notes={}), None


def test_discriminate_keeps_only_binary_labels():
    hm = _FakeHM({"a": "false", "b": "MISLEADING", "c": "UNVERIFIABLE", "d": None})
    items = [{"item_id": s, "payload": {}} for s in ("a", "b", "c", "d")]
    out = discriminate(hm, items)
    assert out == {"a": "FALSE", "b": "MISLEADING"}          # c/d dropped
    assert hm.called_with[1] == "single"                     # single strong seat
    assert hm.called_with[3]["prompt"] == CRM114_SYSTEM
    assert hm.called_with[3]["roles.solo.tier"] == "standard"   # sonnet by default


def test_discriminate_empty_is_noop():
    hm = _FakeHM({})
    assert discriminate(hm, []) == {}
    assert hm.called_with is None
