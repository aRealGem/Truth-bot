"""Evidence-v2 core tests (P67.7 / PR-4, remediation T2.1-T2.4, T2.7).

Design note: wiki projects:truthbot:evidence-v2-design (wiki-first, published
2026-07-21 before this code). Pins: the config-maintained fact-checker
blocklist (domain + path rules), T2.2 query-constraint validation, the
deterministic consolidator (round-robin merge, dedup, era + fair-game
filters, tier quotas, cap, stable ordering — same input, same pack), the
T2.4 quality gate with its provenance code, and the evidence_mode enum.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from truthbot.models import Evidence, SourceTier
from truthbot.verdict.consolidator import (
    GATE_INSUFFICIENT,
    MAX_S5,
    MAX_T6,
    PACK_CAP_V2,
    consolidate,
)
from truthbot.verdict.evidence_mode import EvidenceMode
from truthbot.verify.factcheck_exclusion import (
    is_excluded_factchecker,
    query_violates_constraints,
)

UTT = date(2026, 2, 24)
WINDOW = (date(2024, 1, 1), date(2026, 5, 1))


def _ev(url: str, tier: SourceTier = SourceTier.ESTABLISHED,
        supports: bool | None = True, pub: str | None = "2026-02-20",
        snippet: str = "on-topic reporting") -> Evidence:
    return Evidence(
        claim_id="c", source_name="S", source_url=url, source_tier=tier,
        snippet=snippet, supports_claim=supports,
        published_at=(datetime.fromisoformat(pub).replace(tzinfo=timezone.utc)
                      if pub else None))


# ── T2.1 blocklist ───────────────────────────────────────────────────────────


def test_blocklist_domains_and_subdomains() -> None:
    for url in ("https://www.politifact.com/article/x/", "https://factcheck.org/2026/a/",
                "https://www.snopes.com/news/y/", "https://fullfact.org/economy/z/",
                "https://factcheck.afp.com/doc.123", "https://leadstories.com/hoax/",
                "https://checkyourfact.com/2026/02/x/"):
        assert is_excluded_factchecker(url), url


def test_blocklist_path_rules_spare_the_parent_domain() -> None:
    assert is_excluded_factchecker("https://www.reuters.com/fact-check/claim-x-2026/")
    assert not is_excluded_factchecker("https://www.reuters.com/world/us/story/")
    assert is_excluded_factchecker("https://apnews.com/hub/ap-fact-check")
    assert not is_excluded_factchecker("https://apnews.com/article/economy-123")
    assert is_excluded_factchecker(
        "https://www.washingtonpost.com/politics/fact-checker/claim/")
    assert not is_excluded_factchecker(
        "https://www.washingtonpost.com/politics/congress-story/")


def test_blocklist_no_lookalike_matches() -> None:
    assert not is_excluded_factchecker("https://mysnopes.example.com/x")
    assert not is_excluded_factchecker("https://notpolitifact.com.evil.net/x")


# ── T2.2 query constraints ───────────────────────────────────────────────────


def test_query_constraints_reject_factcheck_tokens_and_speaker_terms() -> None:
    assert "fact-check token" in query_violates_constraints("biden GDP fact check 2022")
    assert "fact-check token" in query_violates_constraints("FactCheck economy claims")
    assert "speaker term" in query_violates_constraints(
        "Trump tariff revenue 2026", ("Trump", "Donald"))
    assert query_violates_constraints("tariff revenue fiscal 2026", ("Trump",)) == ""


def test_generate_queries_drops_violating_queries() -> None:
    from truthbot.verify.relevance import generate_queries

    def llm(system: str, payload: str) -> dict:
        return {"queries": ["gdp growth 2021 annual rate",
                            "biden gdp fact check",
                            "Biden economy record 2021"]}

    out = generate_queries(llm, "Economy grew 5.7% last year.",
                           forbidden_terms=("Biden",))
    assert out == ["gdp growth 2021 annual rate"]


# ── T2.3 consolidator ────────────────────────────────────────────────────────


def test_round_robin_merge_and_dedup_is_deterministic() -> None:
    r1 = [_ev("https://a.gov/1", SourceTier.GOVERNMENT), _ev("https://a.com/2")]
    r2 = [_ev("https://b.com/1"), _ev("https://a.gov/1", SourceTier.GOVERNMENT)]
    res1 = consolidate("s", [("R1", r1), ("R2", r2)], utterance=UTT, window=WINDOW)
    res2 = consolidate("s", [("R1", r1), ("R2", r2)], utterance=UTT, window=WINDOW)
    urls = [it.evidence.source_url for it in res1.items]
    # round 0: a.gov/1 (R1, tier gov) then b.com/1 (R2); round 1: a.com/2;
    # R2's a.gov/1 deduped.
    assert urls == ["https://a.gov/1", "https://b.com/1", "https://a.com/2"]
    assert [it.evidence.source_url for it in res2.items] == urls  # deterministic
    assert res1.dropped.get("duplicate-url") == 1


def test_consolidator_enforces_exclusion_era_and_junk_filters() -> None:
    items = [
        _ev("https://www.politifact.com/factchecks/2026/x/"),        # excluded
        _ev("https://ok.com/a", pub="2026-04-30"),                   # past fair-game
        _ev("https://ok.com/b", pub="2023-01-01"),                   # before coded window
        _ev("https://snopes.com/"),                                  # excluded (+ homepage)
        _ev("https://ok.com/", pub="2026-02-20"),                    # homepage junk
        _ev("https://ok.com/good", pub="2026-02-20"),
        _ev("https://undated.com/story", pub=None, snippet="no date prefix"),
    ]
    res = consolidate("s", [("R1", items)], utterance=UTT, window=WINDOW)
    urls = [it.evidence.source_url for it in res.items]
    assert urls == ["https://ok.com/good", "https://undated.com/story"]
    assert res.dropped.get("factcheck-excluded") == 2
    assert res.dropped.get("after-fair-game-window") == 1
    assert res.dropped.get("outside-coded-window") == 1
    assert res.dropped.get("non-substantive-url") == 1


def test_consolidator_snippet_prefix_dates_are_honored() -> None:
    late = _ev("https://x.com/a", pub=None, snippet="[2026-05-01] later world-state")
    res = consolidate("s", [("R1", [late])], utterance=UTT, window=WINDOW)
    assert res.items == []
    assert res.dropped.get("after-fair-game-window") == 1


def test_t6_quota_and_pack_cap() -> None:
    others = [_ev(f"https://blog{i}.com/p", SourceTier.OTHER) for i in range(5)]
    goods = [_ev(f"https://n{i}.com/a") for i in range(12)]
    res = consolidate("s", [("R1", goods + others)], utterance=UTT, window=WINDOW)
    tiers = [it.evidence.source_tier for it in res.items]
    assert tiers.count(SourceTier.OTHER) <= MAX_T6
    assert len(res.items) <= PACK_CAP_V2
    assert res.dropped.get("t6-quota") == 3


def test_quality_gate_requires_two_bearing_t13_items() -> None:
    # Two T1-3 items but only one bears (supports/refutes) → gate fires.
    items = [_ev("https://a.com/1", supports=True),
             _ev("https://b.com/2", supports=None),
             _ev("https://c.com/3", SourceTier.OTHER, supports=True)]
    res = consolidate("s", [("R1", items)], utterance=UTT, window=WINDOW)
    assert not res.quota_met
    assert res.gate_code == GATE_INSUFFICIENT

    ok = consolidate("s", [("R1", [_ev("https://a.com/1", supports=True),
                                   _ev("https://b.com/2", supports=False)])],
                     utterance=UTT, window=WINDOW)
    assert ok.quota_met and ok.gate_code == ""


def test_v2_payload_schema() -> None:
    res = consolidate("s", [("R1", [_ev("https://a.com/1", supports=False)])],
                      utterance=UTT, window=WINDOW)
    payload = res.to_payload()[0]
    assert set(payload) == {"url", "date", "tier", "stance", "one_line_why"}
    assert payload["stance"] == "refutes"
    assert payload["date"] == "2026-02-20"
    assert res.schema_version == "evidence_pack v2.0"


# ── T2.4 gate code threads into provenance ───────────────────────────────────


def test_evidence_gate_threads_bridge_to_provenance() -> None:
    from truthbot.verdict.bridge import _build_provenance

    row = {"votes": {}, "split": False, "escalated": False,
           "evidence_gate": GATE_INSUFFICIENT}
    prov = _build_provenance(row, {})
    assert prov.evidence_gate == GATE_INSUFFICIENT


# ── T2.7 evidence mode ───────────────────────────────────────────────────────


def test_evidence_mode_enum_and_legacy_inference() -> None:
    assert EvidenceMode.SHARED_PACK_V2.value == "shared_pack_v2"
    assert EvidenceMode.infer_legacy(True) is EvidenceMode.SHARED_PACK_V1
    assert EvidenceMode.infer_legacy(False) is EvidenceMode.CLOSED_BOOK


# ── PR-A2.2: S5 saturation quota + pre-cap pool ──────────────────────────────


def test_s5_quota_caps_political_items() -> None:
    pols = [_ev(f"https://whitehouse.gov/briefing/{i}", SourceTier.POLITICAL)
            for i in range(6)]
    goods = [_ev(f"https://n{i}.com/a") for i in range(3)]
    res = consolidate("s", [("R1", goods + pols)], utterance=UTT, window=WINDOW)
    tiers = [it.evidence.source_tier for it in res.items]
    assert tiers.count(SourceTier.POLITICAL) == MAX_S5
    assert res.dropped.get("s5-quota") == 3
    # First-drawn S5 items are the kept ones.
    kept_pol = [it.evidence.source_url for it in res.items
                if it.evidence.source_tier == SourceTier.POLITICAL]
    assert kept_pol == [f"https://whitehouse.gov/briefing/{i}" for i in range(3)]


def test_s5_quota_frees_pack_cap_slots() -> None:
    # 8 S5 + 7 bearing T3 = 15 candidates. Without the S5 cap the round-robin
    # draw would carry all 8 S5 into the 10-slot pack and push T3 items past
    # the cap; with it, at most MAX_S5 political items survive and the freed
    # slots backfill with the T1-3 items the saturation crowded out.
    pols = [_ev(f"https://whitehouse.gov/briefing/{i}", SourceTier.POLITICAL)
            for i in range(8)]
    goods = [_ev(f"https://n{i}.com/a") for i in range(7)]
    res = consolidate("s", [("R1", pols + goods)], utterance=UTT, window=WINDOW)
    tiers = [it.evidence.source_tier for it in res.items]
    assert len(res.items) == PACK_CAP_V2
    assert tiers.count(SourceTier.POLITICAL) == MAX_S5
    assert tiers.count(SourceTier.ESTABLISHED) == 7
    assert res.quota_met


def test_pre_cap_pool_is_exposed_and_supersets_the_pack() -> None:
    goods = [_ev(f"https://n{i}.com/a") for i in range(14)]
    res = consolidate("s", [("R1", goods)], utterance=UTT, window=WINDOW)
    assert len(res.items) == PACK_CAP_V2
    assert len(res.pre_cap_items) == 14
    assert res.pre_cap_items[:PACK_CAP_V2] == res.items
    assert res.dropped.get("pack-cap") == 4


# ── remediation v2 (1.3): post-speech band + mutable endpoints ───────────────


def test_post_speech_items_admitted_but_never_credit_quota() -> None:
    # Dated after the utterance but within fair-game: admitted for context,
    # flagged, and excluded from the decided-verdict quota.
    post = [_ev(f"https://a.com/{i}", tier=SourceTier.GOVERNMENT,
                pub="2026-02-26") for i in range(2)]
    res = consolidate("s", [("R1", post)], utterance=UTT, window=WINDOW)
    assert [it.post_speech for it in res.items] == [True, True]
    assert all(it.to_payload_v2()["era_note"] == "post-speech · context-only"
               for it in res.items)
    assert not res.quota_met and res.gate_code == GATE_INSUFFICIENT
    # The same items dated on/before the utterance credit normally.
    pre = [_ev(f"https://a.com/{i}", tier=SourceTier.GOVERNMENT,
               pub="2026-02-20") for i in range(2)]
    res2 = consolidate("s", [("R1", pre)], utterance=UTT, window=WINDOW)
    assert res2.quota_met
    assert "era_note" not in res2.items[0].to_payload_v2()


def test_post_speech_flag_survives_role_annotation() -> None:
    from truthbot.verify.principals import PrincipalRelation

    items = [_ev("https://a.com/1", tier=SourceTier.GOVERNMENT, pub="2026-02-26"),
             _ev("https://b.com/2", tier=SourceTier.GOVERNMENT, pub="2026-02-20")]
    res = consolidate("s", [("R1", items)], utterance=UTT, window=WINDOW,
                      claim_shape="c-exist",
                      relation_of=lambda ev: PrincipalRelation.INDEPENDENT)
    flags = {it.evidence.source_url: it.post_speech for it in res.items}
    assert flags["https://a.com/1"] is True
    assert flags["https://b.com/2"] is False


def test_mutable_latest_endpoints_dropped_with_telemetry() -> None:
    live = _ev("https://www.bls.gov/news.release/empsit.htm",
               tier=SourceTier.GOVERNMENT, pub="2026-02-20")
    archived = _ev("https://www.bls.gov/news.release/archives/empsit_02072026.htm",
                   tier=SourceTier.GOVERNMENT, pub="2026-02-20")
    res = consolidate("s", [("R1", [live, archived])],
                      utterance=UTT, window=WINDOW)
    urls = [it.evidence.source_url for it in res.items]
    assert live.source_url not in urls
    assert archived.source_url in urls
    assert res.dropped["mutable-latest-endpoint"] == 1


def test_v2_builder_fails_closed_without_speech_date() -> None:
    import pytest

    from truthbot.verdict.era_lint import EraLintError
    from truthbot.verdict.evidence_pack_v2 import build_evidence_pack_v2

    class _R:
        label = "R1"

        def shortlist(self, claim_text, *, context="", utterance=None, window=None):
            return []

    with pytest.raises(EraLintError, match="no utterance date registered"):
        build_evidence_pack_v2("unregistered_speech_xyz:0001", "c", (_R(),))
    pack = build_evidence_pack_v2("unregistered_speech_xyz:0001", "c", (_R(),),
                                  era_exempt=True)
    assert pack.items == []
