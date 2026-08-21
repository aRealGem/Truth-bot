"""The decidability axis (D17-d Q2) — offline, $0.

The axis exists because the D17-d probes proved decidability cannot be derived
from the pipeline's structured fields. These tests defend the three properties
that make "recorded, not derived" actually hold:

  * FAIL CLOSED — only owner-ratified assignments can reach a page;
  * NEVER SAYS NEVER — undecidable-from-public-record needs a review trigger;
  * UNDROPPABLE — the lookup is keyed by sid, so no reconstruction path can
    silently lose it the way series_rows and bundle.speaker were lost.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from truthbot.publish.decidability import (
    PROVENANCE,
    PUBLISHABLE_PROVENANCE,
    SCHEMA,
    VALUES,
    DecidabilityError,
    by_sid,
    decidability_for,
    load_decidability,
    publishable_entries,
    summary,
)

REPO = Path(__file__).resolve().parent.parent
REGISTRY = REPO / "data" / "decidability.json"


def _write(tmp_path, entries, schema=SCHEMA):
    p = tmp_path / "decidability.json"
    p.write_text(json.dumps({"schema": schema, "entries": entries}))
    return p


def _entry(**kw):
    base = {"sid": "s:1", "speech_id": "s", "decidability": "retrievable-pending-lane",
            "provenance": "desk", "date": "2026-08-14", "why": "because"}
    base.update(kw)
    return base


# ── loader + validation ──────────────────────────────────────────────────────

def test_missing_file_is_empty_not_an_error(tmp_path):
    assert load_decidability(tmp_path / "nope.json") == []


def test_unknown_schema_raises(tmp_path):
    with pytest.raises(DecidabilityError, match="unknown schema"):
        load_decidability(_write(tmp_path, [], schema="something-else v9"))


def test_missing_required_field_raises(tmp_path):
    e = _entry()
    del e["why"]
    with pytest.raises(DecidabilityError, match="missing"):
        load_decidability(_write(tmp_path, [e]))


def test_unknown_value_raises(tmp_path):
    with pytest.raises(DecidabilityError, match="bad decidability"):
        load_decidability(_write(tmp_path, [_entry(decidability="probably-fine")]))


def test_unknown_provenance_raises(tmp_path):
    with pytest.raises(DecidabilityError, match="bad provenance"):
        load_decidability(_write(tmp_path, [_entry(provenance="vibes")]))


def test_duplicate_sid_raises(tmp_path):
    with pytest.raises(DecidabilityError, match="duplicate"):
        load_decidability(_write(tmp_path, [_entry(), _entry()]))


# ── never says never ─────────────────────────────────────────────────────────

def test_undecidable_without_a_review_trigger_is_rejected(tmp_path):
    """A fact-checker does not get to call a question permanently closed
    without naming what would reopen it. Enforced, not merely documented."""
    bad = _entry(decidability="undecidable-from-public-record")
    with pytest.raises(DecidabilityError, match="review_trigger"):
        load_decidability(_write(tmp_path, [bad]))


def test_undecidable_with_a_review_trigger_loads(tmp_path):
    ok = _entry(decidability="undecidable-from-public-record",
                review_trigger="A published first-hand account.")
    assert len(load_decidability(_write(tmp_path, [ok]))) == 1


# ── fail closed ──────────────────────────────────────────────────────────────

def test_only_owner_ratified_is_publishable(tmp_path):
    # All four rows are undecidable + reason-coded so provenance is the ONLY
    # variable; A1's render-set gate then admits exactly the owner-ratified one.
    def _pub(**kw):
        return _entry(decidability="undecidable-from-public-record",
                      review_trigger="A published first-hand account.",
                      reason_code="INTENT", **kw)
    entries = load_decidability(_write(tmp_path, [
        _pub(sid="s:1", provenance="desk"),
        _pub(sid="s:2", provenance="rule"),
        _pub(sid="s:3", provenance="model"),
        _pub(sid="s:4", provenance="owner-ratified"),
    ]))
    assert [e["sid"] for e in publishable_entries(entries)] == ["s:4"]


def test_owner_ratified_without_a_reason_code_does_not_publish(tmp_path):
    # A1 render-set gate: the step-6 reclassified-out rows are owner-ratified but
    # carry no reason_code (or are not undecidable), so they never reach a page.
    entries = load_decidability(_write(tmp_path, [
        _entry(sid="biden_2022:0194", provenance="owner-ratified",
               decidability="retrievable-pending-lane"),
        _entry(sid="trump_2026:0106", provenance="owner-ratified",
               decidability="needs-decomposition"),
    ]))
    assert publishable_entries(entries) == []


def test_desk_assignment_is_invisible_to_a_render(tmp_path):
    """ccagent's judgement must not be published as the system's."""
    entries = load_decidability(_write(tmp_path, [_entry(sid="s:1", provenance="desk")]))
    assert decidability_for(entries, "s:1") is None
    # still auditable when explicitly asked for
    assert decidability_for(entries, "s:1", publishable_only=False) == \
        "retrievable-pending-lane"


def test_absence_is_none_never_a_default_class(tmp_path):
    entries = load_decidability(_write(tmp_path, [_entry(sid="s:1")]))
    assert decidability_for(entries, "s:nonexistent") is None


def test_publishable_provenance_is_the_only_one_that_publishes():
    assert PUBLISHABLE_PROVENANCE == "owner-ratified"
    assert PUBLISHABLE_PROVENANCE in PROVENANCE


# ── undroppable: the adapted round-trip regression ───────────────────────────

def test_lookup_survives_the_offline_artifact_path(tmp_path):
    """series_rows vanished at render because THREE places rebuild a PackItem
    from an Evidence; bundles come back with speaker='Unknown' because the
    offline artifact path rebuilds claims as {sid, text, context, layer_a}.

    The axis is immune to that class of bug BY CONSTRUCTION: it is keyed by
    sid in a registry, so no reconstruction has to carry it. This test asserts
    the property holds against an offline-shaped claim that has been stripped
    of everything but its sid."""
    entries = load_decidability(_write(tmp_path, [
        _entry(sid="trump_2026:0153",
               decidability="undecidable-from-public-record",
               review_trigger="A published first-hand account.",
               reason_code="PRIVATE-EVENT",
               provenance="owner-ratified")]))

    # exactly what the offline artifact path reconstructs — no speaker, no
    # date, no decidability field on the object
    offline_claim = {"sid": "trump_2026:0153", "text": "…", "context": "",
                     "layer_a": {"claim_type": "attribution"}}
    assert "decidability" not in offline_claim

    assert decidability_for(entries, offline_claim["sid"]) == \
        "undecidable-from-public-record"


def test_by_sid_is_keyed_not_carried(tmp_path):
    entries = load_decidability(_write(tmp_path, [
        _entry(sid="a:1", provenance="owner-ratified",
               decidability="undecidable-from-public-record",
               review_trigger="A published first-hand account.",
               reason_code="INTENT"),
        _entry(sid="a:2", provenance="desk")]))
    assert set(by_sid(entries)) == {"a:1"}
    assert set(by_sid(entries, publishable_only=False)) == {"a:1", "a:2"}


# ── the shipped registry ─────────────────────────────────────────────────────

def test_shipped_registry_publishes_the_step6_ratified_set():
    """A1 (Wave A 2026-08-19): publishable_entries() is the RENDER SET -- the 33
    coded substantive rows. 35 rows are owner-ratified, but the 2 reclassified-out
    rows carry no reason_code and so are NOT publishable; 93 rows are still desk."""
    entries = load_decidability(REGISTRY)
    assert len(entries) == 128
    assert len(publishable_entries(entries)) == 33
    assert len([e for e in entries if e["provenance"] == "owner-ratified"]) == 35
    assert len([e for e in entries if e["provenance"] == "desk"]) == 93


def test_shipped_registry_value_distribution():
    s = summary(load_decidability(REGISTRY))
    assert s["publishable"] == 33  # A1: render set, not the 35 owner-ratified
    # Step-6 reclassed 2 rows: trump_2026:0106 -> needs-decomposition and
    # biden_2022:0194 -> retrievable-pending-lane, moving 88/35/5 to 89/33/6.
    assert s["by_value"] == {
        "retrievable-pending-lane": 89,
        "undecidable-from-public-record": 33,
        "needs-decomposition": 6,
    }


def test_publishable_render_set_contract():
    # A1 (Wave A 2026-08-19) -- the pinned render-set contract. Publishable iff
    # owner-ratified AND undecidable-from-public-record AND reason_code present.
    entries = load_decidability(REGISTRY)
    pub = publishable_entries(entries)
    sids = {e["sid"] for e in pub}
    assert len(pub) == 33
    # the 2 reclassified-out owner-ratified rows are EXCLUDED by name
    assert "biden_2022:0194" not in sids      # retrievable-pending-lane, no code
    assert "trump_2026:0106" not in sids       # needs-decomposition, no code
    # every published row satisfies all three legs of the gate
    assert all(e["provenance"] == "owner-ratified" for e in pub)
    assert all(e["decidability"] == "undecidable-from-public-record" for e in pub)
    assert all(e.get("reason_code") for e in pub)
    # both duals are IN, each carrying a non-rendering reason_code_2 (audit-only)
    by = {e["sid"]: e for e in pub}
    assert "trump_2026:0482" in sids and by["trump_2026:0482"]["reason_code_2"] == "INTENT"
    assert "trump_2026:0514" in sids and by["trump_2026:0514"]["reason_code_2"] == "COUNTERFACTUAL"
    # UNCODED is a pipeline state -- it can never enter the render set
    assert not any(e.get("reason_code") == "UNCODED" for e in pub)


def test_every_shipped_undecidable_names_a_review_trigger():
    for e in load_decidability(REGISTRY):
        if e["decidability"] == "undecidable-from-public-record":
            assert e.get("review_trigger"), e["sid"]


def test_retrieved_insufficient_is_reserved_not_seeded():
    """It describes a pack that was retrieved and fell short. The desk's point
    is that these lanes were never run, so seeding it would restate the gate's
    existing message and lose the distinction the axis exists to draw."""
    assert "retrieved-insufficient" in VALUES
    entries = load_decidability(REGISTRY)
    assert not [e for e in entries
                if e["decidability"] == "retrieved-insufficient"]


def test_registry_covers_exactly_the_desk_pass():
    desk = json.loads(
        (REPO / "metrics" / "remediation_v2" / "d17d_triage.json")
        .read_text(encoding="utf-8"))
    assert {e["sid"] for e in load_decidability(REGISTRY)} == \
        {c["sid"] for c in desk["claims"]}


# ── the seeder ───────────────────────────────────────────────────────────────

# Step-6 ratification writes the owner-ratified overlay (reason codes, ratified
# provenance, 2 reclassifications) directly into the shipped file, on top of the
# desk seed. The seeder still owns the desk skeleton, so it reproduces the
# shipped file with that overlay stripped back off.
_RECLASSED_IN_STEP6 = {"trump_2026:0106", "biden_2022:0194"}


def _desk_skeleton(entry):
    # Only strip the ratification overlay from rows that actually carry it
    # (Fable D2 tightening T1). A desk row carrying a reason_code is real seeder
    # drift and MUST fail the determinism test, not be normalized away.
    if entry.get("provenance") == "owner-ratified":
        e = {k: v for k, v in entry.items()
             if k not in ("reason_code", "reason_code_2", "note")}
        e["provenance"] = "desk"
    else:
        e = dict(entry)
    if entry["sid"] in _RECLASSED_IN_STEP6:
        e["decidability"] = "undecidable-from-public-record"
    return e


def test_step6_reclassification_set_is_pinned():
    # The helper that reconstructs reclassified rows must not grow silently
    # (Fable D2 tightening T2).
    assert _RECLASSED_IN_STEP6 == {"trump_2026:0106", "biden_2022:0194"}
    by = {e["sid"]: e for e in load_decidability(REGISTRY)}
    assert by["trump_2026:0106"]["decidability"] == "needs-decomposition"
    assert by["biden_2022:0194"]["decidability"] == "retrievable-pending-lane"


def test_render_set_invariant():
    # D1 tied to the suite (Fable D2 tightening T3): the only owner-ratified rows
    # WITHOUT a reason_code are the 2 reclassified-out rows, and every
    # reason_code-bearing row is undecidable-from-public-record.
    entries = load_decidability(REGISTRY)
    ratified_uncoded = {e["sid"] for e in entries
                        if e["provenance"] == "owner-ratified"
                        and not e.get("reason_code")}
    assert ratified_uncoded == _RECLASSED_IN_STEP6
    assert all(e["decidability"] == "undecidable-from-public-record"
               for e in entries if e.get("reason_code"))


def test_seeder_is_deterministic():
    spec = importlib.util.spec_from_file_location(
        "seed_decidability", REPO / "scripts" / "seed_decidability_from_desk.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["seed_decidability"] = mod
    spec.loader.exec_module(mod)
    a = json.dumps(mod.build(), sort_keys=True)
    b = json.dumps(mod.build(), sort_keys=True)
    assert a == b
    # and it reproduces the shipped file's desk skeleton (ratification overlay
    # stripped -- see _desk_skeleton).
    shipped = json.loads(REGISTRY.read_text(encoding="utf-8"))
    expected = dict(shipped)
    expected["entries"] = [_desk_skeleton(e) for e in shipped["entries"]]
    assert mod.build() == expected
