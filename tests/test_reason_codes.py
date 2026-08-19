"""Reason codes (step 6) + the decidability v2 optional fields.

The invariants worth pinning are the ones that protect a READER: a code that
renders must have copy, the non-rendering state must never acquire any, and a
reason code must never appear on a row that is merely gate-withheld -- that
last one is the exact conflation the decidability axis exists to prevent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from truthbot.publish.decidability import (SCHEMA, DecidabilityError,
                                           load_decidability)
from truthbot.publish.reason_codes import (STATE_ONLY, ReasonCodeError,
                                           copy_for, known, load_reason_codes,
                                           renderable)

REPO = Path(__file__).resolve().parents[1]
CODES_PATH = REPO / "data" / "reason_codes.json"
REGISTRY_PATH = REPO / "data" / "decidability.json"


def _codes_doc(**over):
    doc = {"schema": "truthbot-reason-codes v1", "shared_footer": "Footer.",
           "codes": [{"code": "INTENT", "renders": True, "copy": "Body."},
                     {"code": STATE_ONLY, "renders": False, "copy": None}]}
    doc.update(over)
    return doc


def _write(tmp_path, doc, name="reason_codes.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _entry(**over):
    e = {"sid": "trump_2026:0153", "speech_id": "trump_2026",
         "decidability": "undecidable-from-public-record", "provenance": "desk",
         "date": "2026-08-17", "why": "A private conversation.",
         "review_trigger": "A published account by either party."}
    e.update(over)
    return e


def _registry_doc(entries):
    return {"schema": SCHEMA, "entries": entries}


# ── the shipped registry ─────────────────────────────────────────────────────

def test_shipped_registry_has_the_eleven_owner_ratified_codes() -> None:
    # Step-6 ratification (owner/Fable 2026-08-18) added CAUSAL-LINK,
    # GROUP-STATE, PROJECTION and NO-RECORD to the seven owner-approved codes.
    reg = load_reason_codes(CODES_PATH)
    assert known(reg) == {"INTENT", "PRIVATE-EVENT", "NO-INSTRUMENT",
                          "MASS-VOICE", "COUNTERFACTUAL", "NO-METRIC",
                          "CAUSAL-LINK", "GROUP-STATE", "PROJECTION",
                          "NO-RECORD", STATE_ONLY}
    # UNCODED is a pipeline state, not a label -- it can never reach a reader.
    assert STATE_ONLY not in renderable(reg)
    assert len(renderable(reg)) == 10


def test_shipped_copy_carries_the_shared_footer() -> None:
    reg = load_reason_codes(CODES_PATH)
    text = copy_for(reg, "INTENT")
    assert text.endswith(
        "This label is re-reviewed if a qualifying source or measure is "
        "identified.")
    assert "intends, believes, or aims to do" in text


def test_uncoded_has_no_copy_and_refuses_to_render() -> None:
    reg = load_reason_codes(CODES_PATH)
    with pytest.raises(ReasonCodeError, match="pipeline state"):
        copy_for(reg, STATE_ONLY)


def test_unknown_code_raises_rather_than_blanking(tmp_path) -> None:
    reg = load_reason_codes(_write(tmp_path, _codes_doc()))
    with pytest.raises(ReasonCodeError, match="unknown reason code"):
        copy_for(reg, "NOT-A-CODE")


def test_missing_registry_is_empty_not_an_error(tmp_path) -> None:
    assert load_reason_codes(tmp_path / "absent.json") == {
        "shared_footer": "", "codes": {}}


# ── registry validation, fail closed ─────────────────────────────────────────

def test_renderable_code_without_copy_is_rejected(tmp_path) -> None:
    doc = _codes_doc(codes=[{"code": "INTENT", "renders": True, "copy": "  "}])
    with pytest.raises(ReasonCodeError, match="carries no copy"):
        load_reason_codes(_write(tmp_path, doc))


def test_non_rendering_state_may_not_accumulate_copy(tmp_path) -> None:
    doc = _codes_doc(codes=[{"code": STATE_ONLY, "renders": False,
                             "copy": "sneaky prose"}])
    with pytest.raises(ReasonCodeError, match="must not accumulate"):
        load_reason_codes(_write(tmp_path, doc))


def test_duplicate_code_is_rejected(tmp_path) -> None:
    doc = _codes_doc(codes=[{"code": "INTENT", "renders": True, "copy": "a"},
                            {"code": "INTENT", "renders": True, "copy": "b"}])
    with pytest.raises(ReasonCodeError, match="duplicate code"):
        load_reason_codes(_write(tmp_path, doc))


def test_wrong_schema_is_rejected(tmp_path) -> None:
    with pytest.raises(ReasonCodeError, match="unknown schema"):
        load_reason_codes(_write(tmp_path, _codes_doc(schema="nope v9")))


# ── decidability v2 ──────────────────────────────────────────────────────────

def test_v1_still_loads_after_the_v2_bump(tmp_path) -> None:
    doc = {"schema": "truthbot-decidability v1", "entries": [_entry()]}
    p = _write(tmp_path, doc, "d.json")
    assert len(load_decidability(p)) == 1


def test_shipped_registry_loads_under_v2_and_stays_fail_closed() -> None:
    reg = load_reason_codes(CODES_PATH)
    entries = load_decidability(REGISTRY_PATH, reason_codes=reg)
    assert len(entries) == 128
    # Step-6 ratification (owner/Fable 2026-08-18) wrote the owner-ratified
    # codes in: 33 substantive rows carry a reason_code and 35 rows are
    # owner-ratified (the 33 coded plus the 2 reclassified-out rows). Every
    # coded row is still an undecidable-from-public-record row (fail closed).
    coded = [e for e in entries if e.get("reason_code")]
    assert len(coded) == 33
    assert all(e["decidability"] == "undecidable-from-public-record"
               for e in coded)
    assert len([e for e in entries
                if e["provenance"] == "owner-ratified"]) == 35


def test_reason_code_is_illegal_on_a_non_substantive_row(tmp_path) -> None:
    e = _entry(decidability="retrievable-pending-lane", reason_code="INTENT")
    e.pop("review_trigger")
    p = _write(tmp_path, _registry_doc([e]), "d.json")
    with pytest.raises(DecidabilityError, match="under-retrieved"):
        load_decidability(p)


def test_secondary_code_requires_a_primary(tmp_path) -> None:
    p = _write(tmp_path, _registry_doc([_entry(reason_code_2="COUNTERFACTUAL")]),
               "d.json")
    with pytest.raises(DecidabilityError, match="nothing to be secondary to"):
        load_decidability(p)


def test_dual_codes_may_not_repeat(tmp_path) -> None:
    p = _write(tmp_path, _registry_doc(
        [_entry(reason_code="MASS-VOICE", reason_code_2="MASS-VOICE")]), "d.json")
    with pytest.raises(DecidabilityError, match="repeats"):
        load_decidability(p)


def test_undefined_code_fails_when_the_registry_is_supplied(tmp_path) -> None:
    reg = load_reason_codes(CODES_PATH)
    p = _write(tmp_path, _registry_doc([_entry(reason_code="INVENTED")]), "d.json")
    with pytest.raises(DecidabilityError, match="not a defined"):
        load_decidability(p, reason_codes=reg)
    # Without a registry the shape is still checked, membership is not.
    assert len(load_decidability(p)) == 1


def test_genuine_dual_row_is_accepted(tmp_path) -> None:
    reg = load_reason_codes(CODES_PATH)
    p = _write(tmp_path, _registry_doc(
        [_entry(sid="trump_2026:0514", reason_code="MASS-VOICE",
                reason_code_2="COUNTERFACTUAL", review_after="2026-12-01")]),
        "d.json")
    entries = load_decidability(p, reason_codes=reg)
    assert entries[0]["reason_code_2"] == "COUNTERFACTUAL"
