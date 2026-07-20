"""Layer C evidence-pack tests — offline, no network, no key.

Cover the structural evidence-entry seam: window derivation shares the preamble's
rule, packs dedup/rank/cap and carry I5 provenance, and the model-facing payload
exposes citable ids. A fake provider stands in for Brave/DataHoover."""
from __future__ import annotations

from datetime import date

import pytest

from truthbot.models import Claim, Evidence, SourceTier
from truthbot.verify.evidence_provider import EvidenceProvider
from truthbot.verify.sources.base import TimeWindow
from truthbot.verdict import evidence_pack
from truthbot.verdict.evidence_pack import build_evidence_pack, window_for


class FakeProvider(EvidenceProvider):
    """Returns canned evidence and records the window it was asked for."""

    def __init__(self, evidence: list[Evidence]):
        self._evidence = evidence
        self.seen_window: TimeWindow = None
        self.calls = 0

    def get_evidence(self, claim: Claim, *, window: TimeWindow = None) -> list[Evidence]:
        self.calls += 1
        self.seen_window = window
        return list(self._evidence)


def _ev(url, snippet="snip", tier=SourceTier.OTHER, name="src"):
    return Evidence(claim_id="c", source_name=name, source_url=url,
                    source_tier=tier, snippet=snippet)


# ── window_for: shares the preamble's expected_claim_window rule ───────────────

def test_window_for_known_sid_matches_expected_window():
    # trump_2026 utterance = 2026-02-24 → expected_claim_window
    assert window_for("trump_2026:3") == (date(2024, 1, 1), date(2026, 5, 1))


def test_window_for_unknown_sid_is_none():
    assert window_for("mystery_2099:1") is None


# ── build_evidence_pack: fetch → dedup → rank → cap → provenance ──────────────

def test_pack_passes_window_to_provider():
    p = FakeProvider([_ev("https://a/x")])
    build_evidence_pack("trump_2026:3", "claim text", p)
    assert p.calls == 1
    assert p.seen_window == (date(2024, 1, 1), date(2026, 5, 1))


def test_pack_assigns_stable_ids_and_provenance():
    p = FakeProvider([_ev("https://a/x", "alpha"), _ev("https://b/y", "beta")])
    pack = build_evidence_pack("trump_2026:3", "c", p)
    assert pack.ids == ["E1", "E2"]
    it = pack.items[0]
    assert it.source_url and it.retrieved_at and it.sha256 and it.tier
    # payload is model-facing and carries the citable id
    payload = pack.to_payload()
    assert payload[0]["id"] == "E1" and "snippet" in payload[0]


def test_pack_dedupes_urls_case_and_trailing_slash():
    p = FakeProvider([_ev("https://a/x/"), _ev("https://A/X"), _ev("https://b/y")])
    pack = build_evidence_pack("trump_2026:3", "c", p)
    assert [it.source_url for it in pack.items] == ["https://a/x/", "https://b/y"]


def test_pack_drops_urlless_evidence():
    p = FakeProvider([_ev("", "no url"), _ev("https://b/y", "ok")])
    pack = build_evidence_pack("trump_2026:3", "c", p)
    assert [it.source_url for it in pack.items] == ["https://b/y"]


def test_pack_ranks_government_above_other():
    p = FakeProvider([_ev("https://other/page", tier=SourceTier.OTHER),
                      _ev("https://gov.example.gov/report", tier=SourceTier.GOVERNMENT)])
    pack = build_evidence_pack("trump_2026:3", "c", p)
    assert pack.items[0].source_url == "https://gov.example.gov/report"   # trust rank wins over order


def test_pack_caps_to_max_items():
    p = FakeProvider([_ev(f"https://site{i}.com/article") for i in range(10)])
    pack = build_evidence_pack("trump_2026:3", "c", p, max_items=3)
    assert len(pack.items) == 3 and pack.ids == ["E1", "E2", "E3"]


def test_empty_pack_renders_empty_and_has_no_ids():
    pack = build_evidence_pack("trump_2026:3", "c", FakeProvider([]))
    assert pack.items == [] and pack.ids == [] and pack.render() == ""
    assert pack.to_payload() == []


def test_pack_render_contains_ids_and_snippets():
    p = FakeProvider([_ev("https://a/x", "alpha snippet", name="GovDept",
                          tier=SourceTier.GOVERNMENT)])
    r = build_evidence_pack("trump_2026:3", "c", p).render()
    assert "[E1]" in r and "alpha snippet" in r and "Government" in r


def test_pack_enforces_i5_on_malformed_evidence(monkeypatch):
    """A provider that yields evidence which can't be provenance-stamped fails
    closed at entry (I5) rather than reaching a verdict."""
    from hydramind.invariants import I5ProvenanceError

    # Force a blank sha256 so the I5 record is incomplete.
    monkeypatch.setattr(evidence_pack, "_sha256", lambda url, snip: "")
    with pytest.raises(I5ProvenanceError):
        build_evidence_pack("trump_2026:3", "c", FakeProvider([_ev("https://a/x")]))
