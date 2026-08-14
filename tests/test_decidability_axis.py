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
    entries = load_decidability(_write(tmp_path, [
        _entry(sid="s:1", provenance="desk"),
        _entry(sid="s:2", provenance="rule"),
        _entry(sid="s:3", provenance="model"),
        _entry(sid="s:4", provenance="owner-ratified"),
    ]))
    assert [e["sid"] for e in publishable_entries(entries)] == ["s:4"]


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
        _entry(sid="a:1", provenance="owner-ratified"),
        _entry(sid="a:2", provenance="desk")]))
    assert set(by_sid(entries)) == {"a:1"}
    assert set(by_sid(entries, publishable_only=False)) == {"a:1", "a:2"}


# ── the shipped registry ─────────────────────────────────────────────────────

def test_shipped_registry_loads_and_publishes_nothing_yet():
    """The seeded registry is the desk pass recorded, not ratified. Until step
    6 it must publish exactly nothing."""
    entries = load_decidability(REGISTRY)
    assert len(entries) == 128
    assert publishable_entries(entries) == []
    assert all(e["provenance"] == "desk" for e in entries)


def test_shipped_registry_value_distribution():
    s = summary(load_decidability(REGISTRY))
    assert s["publishable"] == 0
    assert s["by_value"] == {
        "retrievable-pending-lane": 88,     # 81 web-tier1 + 7 series-core
        "undecidable-from-public-record": 35,
        "needs-decomposition": 5,
    }


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

def test_seeder_is_deterministic():
    spec = importlib.util.spec_from_file_location(
        "seed_decidability", REPO / "scripts" / "seed_decidability_from_desk.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["seed_decidability"] = mod
    spec.loader.exec_module(mod)
    a = json.dumps(mod.build(), sort_keys=True)
    b = json.dumps(mod.build(), sort_keys=True)
    assert a == b
    # and it reproduces the shipped file exactly
    shipped = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert mod.build() == shipped
