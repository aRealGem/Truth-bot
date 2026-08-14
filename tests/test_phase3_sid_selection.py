"""--sids targeted retrieval (phase3_rebuild.select_claims) — offline, $0.

The retrieval lanes were speech-scoped, so the smallest purchasable unit was a
whole speech. The D17-d web-tier1 backlog is 81 claims across five speeches,
which made a 3-claim probe unbuyable: there was no runner for it. This is the
selection contract that makes a named set the unit.

Two properties matter enough to pin:
  * an unknown sid REFUSES — a typo that quietly retrieves nothing looks
    exactly like a lane with no work in it;
  * selection follows ARTIFACT order, not typed order, so two runs of the same
    set chunk identically and resume off the same journal.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "phase3_rebuild", REPO / "scripts" / "phase3_rebuild.py")
p3 = importlib.util.module_from_spec(_SPEC)
sys.modules["phase3_rebuild"] = p3
_SPEC.loader.exec_module(p3)


def _claims(*sids):
    return [{"sid": s, "text": f"text {s}", "context": ""} for s in sids]


CLAIMS = _claims("s:0001", "s:0002", "s:0003", "s:0004", "s:0005")


def test_no_selection_returns_everything():
    assert p3.select_claims(CLAIMS) == CLAIMS


def test_sids_narrows_to_the_named_set():
    got = p3.select_claims(CLAIMS, ["s:0002", "s:0004"])
    assert [c["sid"] for c in got] == ["s:0002", "s:0004"]


def test_unknown_sid_refuses_rather_than_silently_selecting_nothing():
    with pytest.raises(p3.UnknownSid, match="s:9999"):
        p3.select_claims(CLAIMS, ["s:0002", "s:9999"])


def test_refusal_names_every_missing_sid():
    with pytest.raises(p3.UnknownSid) as exc:
        p3.select_claims(CLAIMS, ["s:8888", "s:9999"])
    assert "s:8888" in str(exc.value) and "s:9999" in str(exc.value)


def test_selection_follows_artifact_order_not_typed_order():
    """Two runs of the same set must chunk identically, or resume re-spends."""
    a = p3.select_claims(CLAIMS, ["s:0004", "s:0001"])
    b = p3.select_claims(CLAIMS, ["s:0001", "s:0004"])
    assert [c["sid"] for c in a] == [c["sid"] for c in b] == ["s:0001", "s:0004"]


def test_duplicate_sids_are_not_retrieved_twice():
    got = p3.select_claims(CLAIMS, ["s:0002", "s:0002"])
    assert [c["sid"] for c in got] == ["s:0002"]


def test_limit_still_works_alone():
    assert [c["sid"] for c in p3.select_claims(CLAIMS, None, 2)] == \
        ["s:0001", "s:0002"]


def test_limit_acts_as_a_belt_on_a_named_set():
    got = p3.select_claims(CLAIMS, ["s:0002", "s:0003", "s:0005"], 2)
    assert [c["sid"] for c in got] == ["s:0002", "s:0003"]


def test_empty_sid_list_is_treated_as_no_selection():
    assert p3.select_claims(CLAIMS, [], 0) == CLAIMS


def test_selected_claims_keep_their_payload():
    got = p3.select_claims(CLAIMS, ["s:0003"])
    assert got[0] == {"sid": "s:0003", "text": "text s:0003", "context": ""}


# ── the safety property that makes a partial run safe ────────────────────────

def test_a_named_set_can_never_complete_a_speech():
    """A --sids run is inherently partial, and phase3_rebuild writes the
    artifact ONLY when every claim of the speech is banked. So a 3-claim probe
    cannot overwrite a 54-claim publishing head — results stay in the
    journals. This asserts the premise that guard relies on."""
    selected = p3.select_claims(CLAIMS, ["s:0002"])
    full_sids = {c["sid"] for c in CLAIMS}
    have_sids = {c["sid"] for c in selected}
    assert not (full_sids <= have_sids)
