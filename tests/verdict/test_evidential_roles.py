"""Evidential Role Axis tests (PR-A2.3, D11-approved 2026-08-01).

Pins the D11.2 role table cell-by-cell, the deterministic anti-gaming shape
lint (D11.3), the role-aware consolidator quota, and the four NAMED regression
fixtures from the directive — frozen around the real Obama-2014 claims that
motivated the axis. Design note: wiki projects:truthbot:evidential-role-design.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from functools import partial

from truthbot.checkworthy.shape_lint import enforce_shape, shape_lint_hits
from truthbot.models import Evidence, SourceTier
from truthbot.scoring.rubric import effective_tier_weight
from truthbot.verdict.consolidator import (
    GATE_INSUFFICIENT,
    consolidate,
)
from truthbot.verdict.evidential_role import EvidentialRole, evidential_role
from truthbot.verify.principals import PrincipalRelation, principal_relation

UTT = date(2014, 1, 28)
WINDOW = (date(2012, 1, 1), date(2014, 3, 1))
_WH = "https://obamawhitehouse.archives.gov/the-press-office/2014/01"

# The relation callable exactly as a live caller builds it: closed over
# speaker + utterance, computed identically for every speaker (I3-relational).
def _relation_of(ev, participants=()):
    return principal_relation(ev.source_url, "Barack Obama", UTT,
                              participants=participants)


def _ev(url: str, tier: SourceTier, supports: bool | None = True,
        snippet: str = "on-topic") -> Evidence:
    return Evidence(
        claim_id="c", source_name="R1", source_url=url, source_tier=tier,
        snippet=snippet, supports_claim=supports,
        published_at=datetime(2014, 1, 20, tzinfo=timezone.utc))


# ── D11.2 role table, cell by cell ───────────────────────────────────────────


def test_role_table_encodes_d11_exactly() -> None:
    S, P, I = (PrincipalRelation.SELF, PrincipalRelation.PARTICIPANT,
               PrincipalRelation.INDEPENDENT)
    assert evidential_role("c-exist", S) is EvidentialRole.PRIMARY_RECORD
    assert evidential_role("c-count", S) is EvidentialRole.PRIMARY_RECORD
    assert evidential_role("c-exist", P) is EvidentialRole.CORROBORANT
    assert evidential_role("c-count", P) is EvidentialRole.CORROBORANT
    assert evidential_role("c-eval", S) is EvidentialRole.ATTRIBUTION_ONLY
    assert evidential_role("c-third", S) is EvidentialRole.PLAIN_S5
    for shape in ("c-exist", "c-count", "c-eval", "c-third", "", None):
        assert evidential_role(shape, I) is EvidentialRole.NORMAL
    # Legacy/unshaped claims keep today's behavior for every relation.
    assert evidential_role("", S) is EvidentialRole.NORMAL
    assert evidential_role(None, P) is EvidentialRole.NORMAL


def test_attribution_only_weight_is_zero_and_tighter_than_s5() -> None:
    assert effective_tier_weight(SourceTier.POLITICAL) == 0.15
    assert effective_tier_weight(SourceTier.POLITICAL, "attribution-only") == 0.0
    # Even a top-tier item is worthless under attribution-only.
    assert effective_tier_weight(SourceTier.GOVERNMENT, "attribution-only") == 0.0
    assert effective_tier_weight(SourceTier.GOVERNMENT, "primary-record") == 1.0


# ── D11.3 anti-gaming shape lint ─────────────────────────────────────────────


def test_lint_forces_ministerial_shapes_with_loaded_tokens_to_c_eval() -> None:
    compound = ("We launched the initiative that ended veteran unemployment "
                "because our reforms worked.")
    assert "causal" in shape_lint_hits(compound)
    assert enforce_shape(compound, "c-exist") == "c-eval"
    assert enforce_shape("This was the largest jobs program in history.",
                         "c-count") == "c-eval"
    assert enforce_shape("We created more jobs than any prior administration.",
                         "c-exist") == "c-eval"


def test_lint_passes_clean_ministerial_claims_and_other_shapes() -> None:
    clean = "The White House convened a summit with 150 university presidents."
    assert shape_lint_hits(clean) == []
    assert enforce_shape(clean, "c-exist") == "c-exist"
    assert enforce_shape(clean, "c-count") == "c-count"
    # Non-ministerial shapes pass through untouched even with tokens.
    assert enforce_shape("It was the largest ever.", "c-eval") == "c-eval"
    assert enforce_shape("It was the largest ever.", None) is None


def test_lint_precision_refinements_do_not_overfire() -> None:
    # The D11 draft's bare "\\w+est" would have fired on all of these.
    for text in ("Interest rates on small-business loans fell.",
                 "We invested in the harvest season programs.",
                 "My administration made more loans to small business owners."):
        assert enforce_shape(text, "c-exist") == "c-exist", text
    # "created a program" is ministerial phrasing, not an outcome claim.
    assert enforce_shape("We created a manufacturing hub in Raleigh.",
                         "c-exist") == "c-exist"


# ── Named fixture 1: College Opportunity Summit (c-exist × SELF) ─────────────


def test_college_summit_primary_record_fills_at_most_one_slot() -> None:
    # The real failure shape: the only bearing sources are the speaker's own
    # press shop. PRIMARY-RECORD contributes exactly one credit — never two —
    # and a decided verdict still needs an independent or participant credit.
    self_items = [_ev(f"{_WH}/college-summit-{i}", SourceTier.POLITICAL)
                  for i in range(3)]
    res = consolidate("obama_2014:0046", [("R1", self_items)],
                      utterance=UTT, window=WINDOW,
                      claim_shape="c-exist", relation_of=_relation_of)
    assert [it.role for it in res.items] == ["primary-record"] * 3
    assert not res.quota_met
    assert res.gate_code == GATE_INSUFFICIENT

    # One independent on-core S1–S3 item + the primary record → decided-eligible.
    with_ap = self_items + [_ev("https://apnews.com/article/college-summit",
                                SourceTier.WIRE)]
    res2 = consolidate("obama_2014:0046", [("R1", with_ap)],
                       utterance=UTT, window=WINDOW,
                       claim_shape="c-exist", relation_of=_relation_of)
    assert res2.quota_met
    assert res2.gate_code == ""


# ── Named fixture 2: Joining Forces veterans-hiring (c-count × PARTICIPANT) ──


def test_veterans_hiring_participant_corroborant_fills_independent_slot() -> None:
    # A company NAMED as a program participant, publishing on its own domain,
    # corroborates the ministerial count regardless of its base tier (OTHER).
    rel = partial(_relation_of, participants=("cocacolacompany.com",))
    items = [
        _ev(f"{_WH}/joining-forces-report", SourceTier.POLITICAL),      # SELF
        _ev("https://www.cocacolacompany.com/press/veteran-hiring-commitment",
            SourceTier.OTHER),                                           # PARTICIPANT
    ]
    res = consolidate("obama_2014:0045", [("R1", items)],
                      utterance=UTT, window=WINDOW,
                      claim_shape="c-count", relation_of=rel)
    roles = [it.role for it in res.items]
    assert roles == ["primary-record", "corroborant"]
    # primary (1) + corroborant (1) = 2 credits, corroborant satisfies the
    # non-self requirement → decided-eligible.
    assert res.quota_met
    # Without the participant list the same company is INDEPENDENT tier-OTHER
    # → no credit, gate fires (behavior identical to pre-A2.3).
    res_no_part = consolidate("obama_2014:0045", [("R1", items)],
                              utterance=UTT, window=WINDOW,
                              claim_shape="c-count", relation_of=_relation_of)
    assert not res_no_part.quota_met


# ── Named fixtures 3+4: tightening — c-eval × SELF demoted to attribution ────


def test_c_eval_self_items_are_attribution_only_and_credit_nothing() -> None:
    # Fixture 3: a causal/effectiveness claim whose pack is stuffed with the
    # administration's own reports plus ONE bearing wire item. Under the old
    # quota the wire item + nothing-else already failed (1 < 2); the role axis
    # must not accidentally loosen that — self items credit ZERO here.
    items = [_ev(f"{_WH}/economy-worked-{i}", SourceTier.POLITICAL)
             for i in range(3)] + \
            [_ev("https://reuters.com/markets/analysis", SourceTier.WIRE)]
    res = consolidate("obama_2014:0003", [("R1", items)],
                      utterance=UTT, window=WINDOW,
                      claim_shape="c-eval", relation_of=_relation_of)
    assert [it.role for it in res.items] == ["attribution-only"] * 3 + ["normal"]
    assert not res.quota_met      # 1 independent credit < MIN_BEARING_T13
    assert res.role_tally == {"attribution-only": 3, "normal": 1}
    # Payload marks the demotion so the panel sees non-probative records.
    payload = res.to_payload()
    assert [p.get("role") for p in payload] == \
        ["attribution-only"] * 3 + [None]


def test_decided_c_eval_claim_with_self_items_keeps_its_verdict_path() -> None:
    # Fixture 4 (no epistemic regression): two independent bearing S1–S3 items
    # decide the claim exactly as before; the in-pack self items are labeled
    # attribution-only but the gate outcome is unchanged.
    items = [
        _ev("https://apnews.com/article/x", SourceTier.WIRE),
        _ev("https://www.bls.gov/data/report", SourceTier.GOVERNMENT),
        _ev(f"{_WH}/fact-sheet", SourceTier.POLITICAL),
    ]
    legacy = consolidate("s", [("R1", items)], utterance=UTT, window=WINDOW)
    role = consolidate("s", [("R1", items)], utterance=UTT, window=WINDOW,
                       claim_shape="c-eval", relation_of=_relation_of)
    assert legacy.quota_met and role.quota_met
    assert legacy.gate_code == role.gate_code == ""
    assert [it.evidence.source_url for it in role.items] == \
        [it.evidence.source_url for it in legacy.items]


# ── Opt-in discipline ────────────────────────────────────────────────────────


def test_without_relation_or_shape_behavior_is_bitwise_legacy() -> None:
    items = [_ev(f"{_WH}/college-summit", SourceTier.POLITICAL),
             _ev("https://apnews.com/article/y", SourceTier.WIRE)]
    plain = consolidate("s", [("R1", items)], utterance=UTT, window=WINDOW)
    shape_only = consolidate("s", [("R1", items)], utterance=UTT, window=WINDOW,
                             claim_shape="c-exist")
    rel_only = consolidate("s", [("R1", items)], utterance=UTT, window=WINDOW,
                           relation_of=_relation_of)
    for res in (plain, shape_only, rel_only):
        assert res.quota_met == plain.quota_met
        assert [it.role for it in res.items] == ["", ""]
        assert res.role_tally == {}
