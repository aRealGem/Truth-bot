"""Composition-bias telemetry (Claim Eval v3 fast-follow).

The number that matters is ``decided_rate_gap``: decided-rate among claims whose
pack carries no quarantined item MINUS the rate among claims that depend on one.
It is the visibility instrument for the S5 quarantine — a fail-closed rule is
only honest if the composition shift it causes is recorded per run.
"""
from __future__ import annotations

from truthbot.models import SourceTier
from truthbot.verdict.composition_telemetry import composition_report, format_report


def _ev(tier: SourceTier, supports=True):
    return {"source_tier": tier.value, "supports_claim": supports,
            "source_url": "https://example.test/x"}


def _row(sid, verdict):
    return {"sid": sid, "verdict": verdict, "status": "resolved"}


def test_gap_is_zero_when_quarantine_touches_nothing():
    rows = [_row("s:1", "TRUE"), _row("s:2", "FALSE")]
    ev = {"s:1": [_ev(SourceTier.GOVERNMENT)], "s:2": [_ev(SourceTier.WIRE)]}
    r = composition_report(rows, ev)
    assert r["overall"]["packs_exposed"] == 0
    assert r["overall"]["decided_rate_unexposed"] == 1.0
    # no exposed claims => gap undefined, not a spurious 0
    assert r["overall"]["decided_rate_gap"] is None


def test_gap_surfaces_quarantine_driven_abstention():
    """Exposed claims abstain, unexposed decide → a positive, visible gap."""
    rows = [_row("s:1", "TRUE"), _row("s:2", "TRUE"),
            _row("s:3", "UNVERIFIABLE"), _row("s:4", "UNVERIFIABLE")]
    ev = {
        "s:1": [_ev(SourceTier.GOVERNMENT)],
        "s:2": [_ev(SourceTier.GOVERNMENT)],
        "s:3": [_ev(SourceTier.POLITICAL)],
        "s:4": [_ev(SourceTier.POLITICAL)],
    }
    r = composition_report(rows, ev)["overall"]
    assert r["packs_exposed"] == 2 and r["pack_exposure_rate"] == 0.5
    assert r["decided_rate_unexposed"] == 1.0
    assert r["decided_rate_exposed"] == 0.0
    assert r["decided_rate_gap"] == 1.0
    # both exposed claims are sole-quarantined (every bearing item is S5)
    assert r["sole_quarantined"] == 2 and r["sole_quarantined_decided"] == 0


def test_sole_quarantined_requires_every_bearing_item():
    """A pack with one S5 item but other bearing evidence is exposed, NOT sole."""
    rows = [_row("s:1", "TRUE")]
    ev = {"s:1": [_ev(SourceTier.POLITICAL), _ev(SourceTier.GOVERNMENT)]}
    r = composition_report(rows, ev)["overall"]
    assert r["packs_exposed"] == 1
    assert r["sole_quarantined"] == 0


def test_non_bearing_items_do_not_make_a_claim_sole_quarantined():
    """supports_claim=None items don't bear on the verdict, so an S5-only pack of
    them is not the collapse shape."""
    rows = [_row("s:1", "TRUE")]
    ev = {"s:1": [_ev(SourceTier.POLITICAL, supports=None)]}
    assert composition_report(rows, ev)["overall"]["sole_quarantined"] == 0


def test_per_speaker_breakdown_exposes_asymmetry():
    """The I3-adjacent check: the same rule can land unevenly across speakers,
    and that must be visible rather than averaged away."""
    rows = [_row("a_2020:1", "TRUE"), _row("a_2020:2", "TRUE"),
            _row("b_2024:1", "UNVERIFIABLE"), _row("b_2024:2", "TRUE")]
    ev = {
        "a_2020:1": [_ev(SourceTier.GOVERNMENT)],
        "a_2020:2": [_ev(SourceTier.GOVERNMENT)],
        "b_2024:1": [_ev(SourceTier.POLITICAL)],
        "b_2024:2": [_ev(SourceTier.POLITICAL)],
    }
    r = composition_report(rows, ev)
    assert r["by_speaker"]["a_2020"]["pack_exposure_rate"] == 0.0
    assert r["by_speaker"]["b_2024"]["pack_exposure_rate"] == 1.0
    assert r["by_speaker"]["b_2024"]["decided_rate"] == 0.5


def test_accepts_evidence_objects_not_just_dicts():
    """Runs hand it Evidence models; stored artifacts hand it dicts."""
    from truthbot.models import Evidence

    ev_obj = Evidence(claim_id="s:1", source_name="n",
                      source_url="https://example.test/x",
                      source_tier=SourceTier.POLITICAL, snippet="...",
                      supports_claim=True, relevance_score=1.0)
    r = composition_report([_row("s:1", "TRUE")], {"s:1": [ev_obj]})
    assert r["overall"]["quarantined_items"] == 1


def test_empty_run_does_not_raise():
    r = composition_report([], {})
    assert r["overall"]["claims"] == 0 and r["by_speaker"] == {}
    assert "composition telemetry" in format_report(r)
