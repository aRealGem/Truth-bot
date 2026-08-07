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


# ── P67 Round B.5: stance surfaced into the panel payload ─────────────────────

def _ev_stance(url, supports, relevance=0.9, **kw):
    e = _ev(url, **kw)
    e.supports_claim = supports
    e.relevance_score = relevance
    return e


def test_pack_item_carries_relevance_layer_signals():
    p = FakeProvider([_ev_stance("https://a/x", True, relevance=0.8)])
    it = build_evidence_pack("trump_2026:3", "c", p).items[0]
    assert it.supports_claim is True and it.relevance_score == 0.8


def test_payload_surfaces_supports_and_refutes_stance():
    p = FakeProvider([_ev_stance("https://a/x", True),
                      _ev_stance("https://b/y", False)])
    payload = {d["id"]: d for d in build_evidence_pack("trump_2026:3", "c", p).to_payload()}
    assert payload["E1"]["stance"] == "supports"
    assert payload["E2"]["stance"] == "refutes"


def test_payload_omits_stance_when_unscored():
    # Default pack (no relevance layer) → supports_claim None → no stance key,
    # so the payload stays byte-identical to the pre-B.5 shape.
    p = FakeProvider([_ev("https://a/x")])
    item = build_evidence_pack("trump_2026:3", "c", p).to_payload()[0]
    assert "stance" not in item
    assert set(item) == {"id", "source", "tier", "url", "snippet"}


# ── A2: fact-check content is EXCLUDED, never reserved ────────────────────────
#
# These three tests used to pin the opposite contract (P67 Round B.5's reserved
# fact-check slot: a ruling that missed the cap displaced the last slot). T2.1
# reversed the policy — truth-bot reaches its own verdict from primary sources
# and must never launder another outlet's ruling into its evidence — and the v2
# consolidator has excluded fact-checkers since. The v1 builder went on FORCING
# one in. They are inverted here, not deleted, so the reversal is legible in
# the history rather than looking like coverage that quietly vanished.

def _relevant(url, tier, relevance):
    e = _ev(url, tier=tier)
    e.relevance_score = relevance
    return e


def test_factcheck_ruling_is_excluded_not_reserved_when_crowded_out():
    # WAS test_factcheck_ruling_reserved_when_crowded_out: 6 highly-relevant
    # explainers + one lower-relevance ruling used to end up in the last slot.
    explainers = [_relevant(f"https://news{i}.com/story", SourceTier.ESTABLISHED, 0.9)
                  for i in range(6)]
    ruling = _relevant("https://politifact.com/factchecks/x", SourceTier.FACTCHECK, 0.7)
    pack = build_evidence_pack("trump_2026:3", "c", FakeProvider(explainers + [ruling]))
    assert SourceTier.FACTCHECK not in [it.tier for it in pack.items]
    assert all("politifact" not in it.source_url for it in pack.items)
    assert len(pack.items) == 6          # the cap is filled by real evidence


def test_a_top_relevance_ruling_is_dropped_rather_than_ranked_first():
    # WAS test_factcheck_already_in_cap_is_noop: a 0.95-relevance ruling used to
    # rank E1 untouched. Relevance no longer buys a fact-checker a slot.
    ruling = _relevant("https://factcheck.org/x", SourceTier.FACTCHECK, 0.95)
    explainers = [_relevant(f"https://news{i}.com/story", SourceTier.ESTABLISHED, 0.9)
                  for i in range(6)]
    pack = build_evidence_pack("trump_2026:3", "c", FakeProvider([ruling] + explainers))
    assert pack.items[0].source_url == "https://news0.com/story"
    assert sum(it.tier == SourceTier.FACTCHECK for it in pack.items) == 0


def test_pack_without_any_ruling_is_byte_for_byte_what_it_always_was():
    # WAS test_no_factcheck_pack_unchanged, and still true — the overwhelmingly
    # common case is untouched by A2. Nothing about ordinary packs changed.
    explainers = [_relevant(f"https://news{i}.com/story", SourceTier.ESTABLISHED, 0.9)
                  for i in range(8)]
    pack = build_evidence_pack("trump_2026:3", "c", FakeProvider(explainers))
    assert len(pack.items) == 6
    assert all(it.tier == SourceTier.ESTABLISHED for it in pack.items)
    assert [it.source_url for it in pack.items] == [
        f"https://news{i}.com/story" for i in range(6)]


@pytest.mark.parametrize("position", ["first", "middle", "last", "only",
                                      "majority"])
def test_no_factcheck_item_survives_under_any_candidate_ordering(position):
    """The contract is unconditional: whatever order the provider yields, and
    whatever share of the candidates are rulings, no FACTCHECK-tier item
    reaches the pack. The old reserved slot was order-dependent (it fired only
    on a FULL cap), which is exactly why it was easy to miss."""
    def ruling(i):
        return _relevant(f"https://politifact.com/factchecks/{i}",
                         SourceTier.FACTCHECK, 0.99)

    def real(i):
        return _relevant(f"https://news{i}.com/story", SourceTier.ESTABLISHED, 0.5)

    orderings = {
        "first": [ruling(0)] + [real(i) for i in range(8)],
        "middle": [real(i) for i in range(4)] + [ruling(0)]
                  + [real(i) for i in range(4, 8)],
        "last": [real(i) for i in range(8)] + [ruling(0)],
        "only": [ruling(i) for i in range(3)],
        "majority": [ruling(i) for i in range(5)] + [real(0), real(1)],
    }
    pack = build_evidence_pack("trump_2026:3", "c",
                               FakeProvider(orderings[position]))
    assert all(it.tier != SourceTier.FACTCHECK for it in pack.items)
    assert all("politifact" not in it.source_url for it in pack.items)


def test_a_ruling_mis_tiered_as_established_is_still_excluded():
    """Belt and braces, matching consolidate(): the blocklist domain/path rules
    catch a fact-checker a retriever tiered as ordinary reporting."""
    mis_tiered = _relevant("https://www.snopes.com/fact-check/x",
                           SourceTier.ESTABLISHED, 0.99)
    pack = build_evidence_pack(
        "trump_2026:3", "c",
        FakeProvider([mis_tiered, _relevant("https://news0.com/story",
                                            SourceTier.ESTABLISHED, 0.5)]))
    assert [it.source_url for it in pack.items] == ["https://news0.com/story"]


def test_pack_enforces_i5_on_malformed_evidence(monkeypatch):
    """A provider that yields evidence which can't be provenance-stamped fails
    closed at entry (I5) rather than reaching a verdict."""
    from hydramind.invariants import I5ProvenanceError

    # Force a blank sha256 so the I5 record is incomplete.
    monkeypatch.setattr(evidence_pack, "_sha256", lambda url, snip: "")
    with pytest.raises(I5ProvenanceError):
        build_evidence_pack("trump_2026:3", "c", FakeProvider([_ev("https://a/x")]))
