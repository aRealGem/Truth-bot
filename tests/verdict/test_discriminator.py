"""CRM-114 discriminator — offline. Prompt lint + the pure override logic; the live
binary call is exercised by the --crm114 scoring run, not here."""
from __future__ import annotations

from types import SimpleNamespace

from truthbot.verdict import discriminator
from truthbot.verdict.discriminator import CRM114_SYSTEM, apply_discrimination, discriminate


# ── prompt ────────────────────────────────────────────────────────────────────

def test_crm114_prompt_is_speaker_blind_and_binary():
    t = CRM114_SYSTEM
    assert "speaker" not in t.lower()
    assert "FALSE" in t and "MISLEADING" in t
    assert "CORE assertion" in t and "contradiction" in t.lower()


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
