"""Deterministic re-gate tool (scripts/regate_from_rescore.py) — offline, $0.

Nothing here touches a model, a proxy or the network: the subject is pure
arithmetic over stored data, and these fixtures are synthetic so the suite does
not depend on the real B1a sidecars existing. The two integration-flavoured
tests that DO read the real artifacts skip themselves when those are absent.

What is under test is the part a reviewer has to trust: that the sidecar join
reports both kinds of miss instead of dropping them, that the four flip classes
mean what they say, that the role-aware branch is selected exactly when the
original run would have selected it, and that the B1b costing does not
double-count a named extra that the gate already released.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
_SPEC = importlib.util.spec_from_file_location(
    "regate_from_rescore", REPO / "scripts" / "regate_from_rescore.py")
rg = importlib.util.module_from_spec(_SPEC)
sys.modules["regate_from_rescore"] = rg
_SPEC.loader.exec_module(rg)          # must import clean with no key present

from truthbot.verdict.consolidator import (GATE_INSUFFICIENT,  # noqa: E402
                                           MIN_BEARING_T13)

SPEECH = "trump_2026"
UTTERANCE = "2026-02-24"


# ── fixtures ─────────────────────────────────────────────────────────────────

def _ev(url, *, tier="Wire", supports=None, relevance=0.5, sid="s:0001",
        published="2026-02-20T00:00:00"):
    """A stored evidence dump exactly as an artifact holds it."""
    return {"claim_id": sid, "source_name": "AP", "source_url": url,
            "source_tier": tier, "snippet": "a snippet",
            "retrieved_at": "2026-02-25T04:20:27.824223",
            "published_at": published, "supports_claim": supports,
            "relevance_score": relevance}


def _artifact(evidence: dict, rows: dict, *, shapes: dict | None = None,
              speaker="Donald Trump", speech=SPEECH):
    """A minimal run artifact with the keys the re-gate actually reads."""
    shapes = shapes or {}
    claims = []
    for sid in evidence:
        layer_a = {"label": "check-worthy", "source": "A2"}
        if shapes.get(sid):
            layer_a["claim_shape"] = shapes[sid]
        claims.append({"sid": sid, "text": f"claim text for {sid}",
                       "context": "", "layer_a": layer_a})
    return {
        "run_id": "test-run", "meta": {"speaker": speaker, "date": UTTERANCE,
                                       "speech_id": speech},
        "claims": claims,
        "rows": [{"sid": sid, "status": "resolved", "verdict": v,
                  **({"provenance_code": GATE_INSUFFICIENT}
                     if v == "UNVERIFIABLE" else {})}
                 for sid, v in rows.items()],
        "evidence": evidence,
    }


def _sidecar(sids: dict, speech=SPEECH, source_run=""):
    return {"schema": rg.SIDECAR_SCHEMA, "speech_id": speech,
            "source_run": source_run, "model": "claude-haiku",
            "generated": "2026-08-08T00:00:00+00:00", "spend_usd": 0.01,
            "sids": sids, "soft_failures": []}


def _scored(url, *, supports, relevance=0.9):
    return {"source_url": url, "relevance_score": relevance,
            "supports_claim": supports}


# ── 1. the overlay join ──────────────────────────────────────────────────────

def test_overlay_applies_scores_and_reports_a_clean_join():
    from truthbot.verdict.publish_pipeline import evidence_from_artifact_dict

    evs = evidence_from_artifact_dict(
        {"s:0001": [_ev("https://apnews.com/a"), _ev("https://npr.org/b")]})["s:0001"]
    tel = rg.overlay_rescores(evs, [_scored("https://apnews.com/a", supports=True),
                                    _scored("https://npr.org/b", supports=False,
                                            relevance=0.7)])

    assert [e.supports_claim for e in evs] == [True, False]
    assert [e.relevance_score for e in evs] == [0.9, 0.7]
    assert tel == {"items": 2, "matched": 2, "sidecar_unmatched": [],
                   "artifact_unscored": []}


def test_overlay_join_key_tolerates_trailing_slash_and_case():
    from truthbot.verdict.publish_pipeline import evidence_from_artifact_dict

    evs = evidence_from_artifact_dict(
        {"s:0001": [_ev("https://APnews.com/a/")]})["s:0001"]
    tel = rg.overlay_rescores(evs, [_scored("https://apnews.com/a", supports=True)])

    assert evs[0].supports_claim is True
    assert tel["matched"] == 1 and not tel["sidecar_unmatched"]


def test_overlay_reports_misses_in_BOTH_directions_and_drops_nothing():
    """A sidecar row with no home, and a pack item the sidecar never scored,
    are both named by URL — never silently discarded."""
    from truthbot.verdict.publish_pipeline import evidence_from_artifact_dict

    evs = evidence_from_artifact_dict(
        {"s:0001": [_ev("https://apnews.com/a"),
                    _ev("https://govinfo.gov/never-scored")]})["s:0001"]
    tel = rg.overlay_rescores(evs, [
        _scored("https://apnews.com/a", supports=True),
        _scored("https://example.com/not-in-the-pack", supports=True)])

    assert tel["matched"] == 1
    assert tel["sidecar_unmatched"] == ["https://example.com/not-in-the-pack"]
    assert tel["artifact_unscored"] == ["https://govinfo.gov/never-scored"]
    # the unmatched artifact item keeps its stored (unscored) stance
    assert evs[1].supports_claim is None


# ── 2. the four classifications ──────────────────────────────────────────────

def test_classify_covers_the_four_cases():
    assert rg.classify(True, False) == "released"
    assert rg.classify(True, True) == "still_gated"
    assert rg.classify(False, True) == "newly_gated"
    assert rg.classify(False, False) == "unchanged_decided"


def _regate(evidence, rows, sidecar_sids, shapes=None):
    return rg.regate_speech(SPEECH, _artifact(evidence, rows, shapes=shapes),
                            _sidecar(sidecar_sids))


def test_released_a_gate_forced_claim_that_now_meets_quota():
    """The B1a repair in miniature: two Tier-1..3 items sat stanceless, so the
    pack could not credit MIN_BEARING_T13 and was gate-forced. With real stance
    it clears the quota."""
    sid = "trump_2026:0469"
    evidence = {sid: [_ev("https://npr.org/a", tier="Established", supports=True),
                      _ev("https://apnews.com/b", tier="Wire"),
                      _ev("https://govinfo.gov/c", tier="Government")]}
    res = _regate(evidence, {sid: "UNVERIFIABLE"},
                  {sid: [_scored("https://apnews.com/b", supports=True),
                         _scored("https://govinfo.gov/c", supports=True)]})

    assert res["counts"]["released"] == 1
    flip = res["flips"][0]
    assert flip["class"] == "released"
    assert flip["old_verdict"] == "UNVERIFIABLE"
    assert flip["old_gate_code"] == GATE_INSUFFICIENT
    assert flip["before"]["credits"] == 1 < MIN_BEARING_T13
    assert flip["after"]["credits"] == 3
    assert flip["before"]["quota_met"] is False
    assert flip["after"]["quota_met"] is True
    assert flip["baseline_reproduced"] is True


def test_still_gated_when_the_scores_do_not_rescue_it():
    sid = "trump_2026:0001"
    evidence = {sid: [_ev("https://apnews.com/b", tier="Wire"),
                      _ev("https://blog.example.com/c", tier="Other")]}
    res = _regate(evidence, {sid: "UNVERIFIABLE"},
                  {sid: [_scored("https://apnews.com/b", supports=True),
                         _scored("https://blog.example.com/c", supports=True)]})

    assert res["counts"]["still_gated"] == 1
    assert res["flips"] == []            # only released/newly_gated are listed
    assert res["counts"]["released"] == 0


def test_newly_gated_when_real_stance_withdraws_the_signal():
    """The repair withholding something: a decided claim whose stances were
    retriever-asserted turns out to be ambiguous context once actually scored."""
    sid = "trump_2026:0002"
    evidence = {sid: [_ev("https://apnews.com/b", tier="Wire", supports=True),
                      _ev("https://npr.org/c", tier="Established", supports=True)]}
    res = _regate(evidence, {sid: "TRUE"},
                  {sid: [_scored("https://apnews.com/b", supports=None,
                                 relevance=0.2),
                         _scored("https://npr.org/c", supports=None,
                                 relevance=0.1)]})

    assert res["counts"]["newly_gated"] == 1
    flip = res["flips"][0]
    assert flip["class"] == "newly_gated"
    assert flip["old_verdict"] == "TRUE"
    assert flip["before"]["credits"] == 2
    assert flip["after"]["credits"] == 0


def test_unchanged_decided_stays_out_of_the_flip_list():
    sid = "trump_2026:0003"
    evidence = {sid: [_ev("https://apnews.com/b", tier="Wire", supports=True),
                      _ev("https://npr.org/c", tier="Established", supports=True)]}
    res = _regate(evidence, {sid: "TRUE"},
                  {sid: [_scored("https://apnews.com/b", supports=True),
                         _scored("https://npr.org/c", supports=True)]})

    assert res["counts"] == {"released": 0, "still_gated": 0, "newly_gated": 0,
                             "unchanged_decided": 1, rg.NOT_RESCORED: 0}
    assert res["flips"] == []


def test_a_sid_the_sidecar_has_not_reached_is_not_rescored_not_unchanged():
    """B1a mid-flight: an unscored sid has an UNKNOWN after-state, so it must
    never be reported as if the repair had left it alone."""
    sid = "trump_2026:0004"
    evidence = {sid: [_ev("https://apnews.com/b", tier="Wire", supports=True),
                      _ev("https://npr.org/c", tier="Established", supports=True)]}
    res = _regate(evidence, {sid: "TRUE"}, {})

    assert res["counts"][rg.NOT_RESCORED] == 1
    assert res["counts"]["unchanged_decided"] == 0
    assert res["sidecar_complete"] is False
    assert res["join"]["sids_without_scores"] == [sid]


def test_the_before_arithmetic_is_not_polluted_by_the_overlay():
    """BEFORE and AFTER are built from two independent reconstructions."""
    sid = "trump_2026:0005"
    evidence = {sid: [_ev("https://apnews.com/b", tier="Wire"),
                      _ev("https://npr.org/c", tier="Established")]}
    res = _regate(evidence, {sid: "UNVERIFIABLE"},
                  {sid: [_scored("https://apnews.com/b", supports=True),
                         _scored("https://npr.org/c", supports=True)]})

    flip = res["flips"][0]
    assert flip["before"]["credits"] == 0      # stored stances were both null
    assert flip["after"]["credits"] == 2


# ── 3. role-aware vs legacy branch selection ─────────────────────────────────

def test_legacy_branch_when_the_claim_has_no_shape():
    sid = "trump_2026:0006"
    evidence = {sid: [_ev("https://apnews.com/b", tier="Wire", supports=True),
                      _ev("https://npr.org/c", tier="Established", supports=True)]}
    res = _regate(evidence, {sid: "TRUE"},
                  {sid: [_scored("https://apnews.com/b", supports=True),
                         _scored("https://npr.org/c", supports=True)]})
    # no flip row to read the breakdown off, so gate the sid directly
    from truthbot.verdict import speech_context
    from truthbot.verdict.publish_pipeline import evidence_from_artifact_dict

    speech_context.register_speech_date(SPEECH, date.fromisoformat(UTTERANCE))
    evs = evidence_from_artifact_dict(evidence)[sid]
    _, bd = rg.gate_once(sid, evs, utterance=date.fromisoformat(UTTERANCE),
                         claim_shape="", relation_of=lambda ev: None,
                         claim_text="a claim")
    assert bd["role_aware"] is False
    assert bd["corroborant"] == 0 and bd["primary"] == 0
    assert res["counts"]["unchanged_decided"] == 1


def test_role_aware_branch_when_shape_and_relation_are_both_present():
    from truthbot.verdict import speech_context
    from truthbot.verdict.publish_pipeline import evidence_from_artifact_dict
    from truthbot.verify.principals import principal_relation

    sid = "trump_2026:0007"
    utt = date.fromisoformat(UTTERANCE)
    speech_context.register_speech_date(SPEECH, utt)
    evidence = {sid: [_ev("https://apnews.com/b", tier="Wire", supports=True),
                      _ev("https://npr.org/c", tier="Established", supports=True)]}
    evs = evidence_from_artifact_dict(evidence)[sid]

    _, bd = rg.gate_once(sid, evs, utterance=utt, claim_shape="c-exist",
                         relation_of=lambda ev: principal_relation(
                             ev.source_url, "Donald Trump", utt),
                         claim_text="a claim")
    assert bd["role_aware"] is True
    # the branch it selected is the one whose arithmetic it reports, and that
    # arithmetic agrees with consolidate()'s own verdict
    assert bd["agrees"] is True


def test_relation_of_absent_forces_the_legacy_branch_even_with_a_shape():
    from truthbot.verdict import speech_context
    from truthbot.verdict.publish_pipeline import evidence_from_artifact_dict

    sid = "trump_2026:0008"
    utt = date.fromisoformat(UTTERANCE)
    speech_context.register_speech_date(SPEECH, utt)
    evs = evidence_from_artifact_dict(
        {sid: [_ev("https://apnews.com/b", tier="Wire", supports=True)]})[sid]
    _, bd = rg.gate_once(sid, evs, utterance=utt, claim_shape="c-exist",
                         relation_of=None, claim_text="a claim")
    assert bd["role_aware"] is False


def test_shape_map_prefers_the_artifact_and_fills_the_rest_from_the_sidecar(tmp_path):
    art = _artifact({"trump_2026:0001": [], "trump_2026:0002": []},
                    {"trump_2026:0001": "TRUE", "trump_2026:0002": "TRUE"},
                    shapes={"trump_2026:0001": "c-eval"})
    side = tmp_path / "shapes.json"
    side.write_text(json.dumps({
        "schema": "truthbot-shape-backfill v1", "speech_id": SPEECH,
        "source_run": rg.SPEECHES[SPEECH]["run_id"],
        "shapes": {"trump_2026:0001": "c-third",   # must NOT override
                   "trump_2026:0002": "c-exist"}}), encoding="utf-8")

    shapes, filled = rg.claim_shape_map(art, SPEECH, shapes_path=side)
    assert shapes == {"trump_2026:0001": "c-eval", "trump_2026:0002": "c-exist"}
    assert filled == 1


def test_shape_map_is_legacy_when_no_sidecar_exists(tmp_path):
    art = _artifact({"trump_2026:0001": []}, {"trump_2026:0001": "TRUE"})
    shapes, filled = rg.claim_shape_map(art, SPEECH,
                                        shapes_path=tmp_path / "absent.json")
    assert shapes == {"trump_2026:0001": ""} and filled == 0


def test_shape_sidecar_for_the_wrong_run_is_refused(tmp_path):
    art = _artifact({"trump_2026:0001": []}, {"trump_2026:0001": "TRUE"})
    side = tmp_path / "shapes.json"
    side.write_text(json.dumps({
        "schema": "truthbot-shape-backfill v1", "speech_id": SPEECH,
        "source_run": "some-other-artifact", "shapes": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="source_run"):
        rg.claim_shape_map(art, SPEECH, shapes_path=side)


# ── 4. sidecar guards ────────────────────────────────────────────────────────

def test_rescore_sidecar_from_another_run_is_refused(tmp_path):
    p = tmp_path / "rescored.json"
    p.write_text(json.dumps(_sidecar({}, source_run="run-A")), encoding="utf-8")
    with pytest.raises(ValueError, match="source_run"):
        rg.load_rescore_sidecar(p, SPEECH, "run-B")


def test_rescore_sidecar_for_another_speech_is_refused(tmp_path):
    p = tmp_path / "rescored.json"
    p.write_text(json.dumps(_sidecar({}, speech="biden_2022")), encoding="utf-8")
    with pytest.raises(ValueError, match="speech_id"):
        rg.load_rescore_sidecar(p, SPEECH, "")


# ── 5. the costed B1b summary ────────────────────────────────────────────────

def test_costed_summary_arithmetic():
    released = [f"trump_2026:{i:04d}" for i in range(100, 116)]   # 16
    k = rg.costed_summary(released)

    assert k["released"] == 16
    assert k["extras_named"] == len(rg.NAMED_EXTRAS) == 6
    assert k["extras_not_already_released"] == 6
    assert k["claims_to_adjudicate"] == 22
    lo, hi = rg.PER_CLAIM_USD
    assert k["cost_low_usd"] == round(22 * lo, 2)
    assert k["cost_high_usd"] == round(22 * hi, 2)
    assert k["remaining_usd"] == round(rg.BUDGET_CEILING_USD
                                       - rg.B1A_PLANNED_USD, 2)
    assert k["b1a_observed_usd"] is None and k["b1a_overran_plan"] is False
    assert k["fits_ceiling"] is True


def test_a_b1a_overrun_shrinks_the_headroom_instead_of_being_ignored():
    """The sidecars are ledger truth: spending more than planned must show up
    as less money left, never as unspent budget."""
    over = rg.B1A_PLANNED_USD + 1.00
    k = rg.costed_summary(["x:0001"], b1a_observed_usd=over)

    assert k["b1a_overran_plan"] is True
    assert k["b1a_observed_usd"] == round(over, 4)
    assert k["committed_b1a_usd"] == round(over, 2)
    assert k["remaining_usd"] == round(rg.BUDGET_CEILING_USD - over, 2)


def test_underspending_the_plan_does_not_free_up_extra_budget():
    k = rg.costed_summary(["x:0001"], b1a_observed_usd=0.01)
    assert k["b1a_overran_plan"] is False
    assert k["committed_b1a_usd"] == rg.B1A_PLANNED_USD
    assert k["remaining_usd"] == round(rg.BUDGET_CEILING_USD
                                       - rg.B1A_PLANNED_USD, 2)


def test_costed_summary_never_double_counts_a_released_extra():
    """A named extra the gate ALREADY released is one claim, not two."""
    k = rg.costed_summary(["trump_2026:0030", "trump_2026:0031", "x:0001"])
    assert k["released"] == 3
    assert k["extras_not_already_released"] == 4
    assert k["extras_overlapping_released"] == ["trump_2026:0030",
                                                "trump_2026:0031"]
    assert k["claims_to_adjudicate"] == 7


def test_costed_summary_dedups_a_repeated_released_sid():
    k = rg.costed_summary(["x:0001", "x:0001", "x:0002"])
    assert k["released"] == 2
    assert k["claims_to_adjudicate"] == 8


def test_newly_gated_claims_add_nothing_to_the_bill():
    """Withholding needs no panel call — the costing reads released only."""
    art = _artifact(
        {"a:1": [_ev("https://apnews.com/b", tier="Wire", supports=True),
                 _ev("https://npr.org/c", tier="Established", supports=True)]},
        {"a:1": "TRUE"})
    art["meta"]["speech_id"] = SPEECH
    res = rg.regate_speech(SPEECH, art, _sidecar(
        {"a:1": [_scored("https://apnews.com/b", supports=None),
                 _scored("https://npr.org/c", supports=None)]}))
    report = rg.build_report([res], missing=[])

    assert report["corpus_counts"]["newly_gated"] == 1
    assert report["costed_b1b"]["released"] == 0
    assert report["costed_b1b"]["claims_to_adjudicate"] == len(rg.NAMED_EXTRAS)


def test_a_named_extra_the_ratified_rules_gate_drops_off_the_bill():
    """T-1: a claim the gate now WITHHOLDS is answered already, for $0. Paying
    a panel to look at it again buys nothing, so it leaves the wave."""
    extra = rg.NAMED_EXTRAS[0]
    k = rg.costed_summary(["x:0001"], gated=[extra])

    assert k["extras_dropped_as_newly_gated"] == [extra]
    assert k["extras_not_already_released"] == len(rg.NAMED_EXTRAS) - 1
    assert k["claims_to_adjudicate"] == len(rg.NAMED_EXTRAS)   # 1 released + 5


def test_omitting_the_gated_set_reproduces_the_old_larger_bill():
    """The dedup is additive. Without it the arithmetic is what it always was,
    so an old artifact stays comparable to a new one."""
    assert (rg.costed_summary(["x:0001"])["claims_to_adjudicate"]
            == rg.costed_summary(["x:0001"], gated=[])["claims_to_adjudicate"]
            == 1 + len(rg.NAMED_EXTRAS))


def test_the_per_claim_band_is_imported_not_redeclared():
    """The same constant lived in two files once and drifted. It is ledger-
    derived — money spent over claims adjudicated — so the estimator
    recalibration must not touch it, and neither may this script."""
    from truthbot import costs

    assert rg.PER_CLAIM_USD is costs.PER_CLAIM_USD_MEASURED
    assert rg.PER_CLAIM_USD_PLANNING is costs.PER_CLAIM_USD_PLANNING
    # The planning band is never CHEAPER than the measurement at either end —
    # it is the measurement rounded up for headroom, so a budget set from it
    # cannot come in under what was actually observed per claim.
    assert rg.PER_CLAIM_USD_PLANNING[0] >= rg.PER_CLAIM_USD[0]
    assert rg.PER_CLAIM_USD_PLANNING[1] >= rg.PER_CLAIM_USD[1]


def test_headroom_is_not_rounded_before_the_subtraction():
    """$8.3962 and $8.40 are different statements, and the wave is close enough
    to its sub-cap that the difference is worth carrying."""
    k = rg.costed_summary(["x:0001"], b1a_observed_usd=1.6038)
    assert k["committed_b1a_usd"] == 1.6038
    assert k["remaining_usd"] == 8.3962


def test_the_wave_subcap_is_judged_on_the_planning_band():
    """"Fits" must not be true only on the optimistic number. The sub-cap is
    checked against the outward-rounded planning band, which is what a budget
    would actually be set from."""
    plo, phi = rg.PER_CLAIM_USD_PLANNING
    n = int(rg.WAVE_PLANNING_CEILING_USD / phi) + 5        # comfortably over
    k = rg.costed_summary([f"x:{i:04d}" for i in range(n)])
    assert k["cost_high_planning_usd"] > rg.WAVE_PLANNING_CEILING_USD
    assert k["fits_wave_planning_ceiling"] is False
    # ...and the measured band, being narrower, is not what decided it.
    assert k["cost_high_usd"] < k["cost_high_planning_usd"]


# ── 6. report assembly ───────────────────────────────────────────────────────

def test_the_report_records_which_rules_each_leg_ran():
    """A flip set is unreadable without knowing which gate produced it, and
    since the ratification the two legs no longer agree."""
    art = _artifact({"a:1": [_ev("https://apnews.com/b", tier="Wire")]},
                    {"a:1": "UNVERIFIABLE"})
    res = rg.regate_speech(SPEECH, art, _sidecar({}))

    default = rg.build_report([res], missing=[])
    assert default["rules"]["before"] == rg.PRE_RATIFICATION_RULES
    assert default["rules"]["after"] == {"utterance_record": True,
                                         "statistical_release": True}
    assert default["rules"]["ratified"] == "2026-08-09"
    assert "Rules." in rg.render_markdown(default)

    off = rg.build_report([res], missing=[],
                          rules={"utterance_record": False,
                                 "statistical_release": False})
    # The BEFORE leg is NOT configurable — it reproduces the artifacts.
    assert off["rules"]["before"] == rg.PRE_RATIFICATION_RULES


def test_the_before_leg_ignores_the_after_switches_entirely(monkeypatch):
    """The BEFORE leg must reproduce a PRE-ratification artifact. If it read the
    ambient default, gate reproduction would collapse the day the default moved
    — which is exactly what the ratification did."""
    monkeypatch.setenv("TRUTHBOT_D15_UTTERANCE_RECORD", "1")
    monkeypatch.setenv("TRUTHBOT_D16_STATISTICAL_RELEASE", "1")
    art = _artifact({"a:1": [_ev("https://apnews.com/b", tier="Wire")]},
                    {"a:1": "UNVERIFIABLE"})

    a_res = rg.regate_speech(SPEECH, art, _sidecar({}),
                             utterance_record=True, statistical_release=True)
    b_res = rg.regate_speech(SPEECH, art, _sidecar({}),
                             utterance_record=False, statistical_release=False)
    assert (a_res["gate_reproduction"]["mismatched"]
            == b_res["gate_reproduction"]["mismatched"] == 0)


def test_report_flags_a_missing_and_a_partial_sidecar():
    art = _artifact({"a:1": [_ev("https://apnews.com/b", tier="Wire")]},
                    {"a:1": "UNVERIFIABLE"})
    res = rg.regate_speech(SPEECH, art, _sidecar({}))
    report = rg.build_report([res], missing=["gwbush_2006"])

    assert report["speeches_missing_sidecar"] == ["gwbush_2006"]
    assert report["speeches_partial_sidecar"] == [SPEECH]
    assert report["complete"] is False
    md = rg.render_markdown(report)
    assert "Partial run" in md and "Partial sidecar" in md
    assert "not re-scored" in md


def test_markdown_renders_the_flip_tables_and_the_costed_block():
    sid = "trump_2026:0469"
    art = _artifact(
        {sid: [_ev("https://npr.org/a", tier="Established", supports=True),
               _ev("https://apnews.com/b", tier="Wire"),
               _ev("https://govinfo.gov/c", tier="Government")]},
        {sid: "UNVERIFIABLE"})
    res = rg.regate_speech(SPEECH, art, _sidecar(
        {sid: [_scored("https://apnews.com/b", supports=True),
               _scored("https://govinfo.gov/c", supports=True)]}))
    md = rg.render_markdown(rg.build_report([res], missing=[]))

    assert "## Flip set by speech" in md
    assert "## Released — 1 claim(s)" in md
    assert "## Newly gated — 0 claim(s)" in md
    assert "## Sidecar-join telemetry" in md
    assert "## Costed B1b summary" in md
    assert f"`{sid}`" in md
    assert "ind 1 / corr 0 / prim 0 → 1 of 2" in md      # BEFORE
    assert "ind 3 / corr 0 / prim 0 → 3 of 2" in md      # AFTER


# ── 7. integration — real artifacts, still $0 ────────────────────────────────

def _real(speech):
    return rg.artifact_path(speech), rg.sidecar_path(speech)


@pytest.mark.parametrize("speech", sorted(rg.REBUILT_RUNS))
def test_stored_gate_reproduces_exactly_on_the_real_artifacts(speech):
    """The BEFORE recomputation must match the gate code the artifact recorded,
    row for row — that equality is what makes the AFTER delta attributable to
    the re-score rather than to drift in the surrounding code."""
    art_p, side_p = _real(speech)
    if not art_p.exists():
        pytest.skip(f"{art_p} not present")
    artifact = rg.load_artifact(art_p)
    sidecar = (rg.load_rescore_sidecar(side_p, speech, rg.REBUILT_RUNS[speech])
               if side_p.exists() else _sidecar({}, speech, rg.REBUILT_RUNS[speech]))
    res = rg.regate_speech(speech, artifact, sidecar)

    assert res["gate_reproduction"]["mismatched"] == 0, \
        res["gate_reproduction"]["mismatches"]
    assert res["gate_reproduction"]["matched"] == res["claims"]
    assert res["breakdown_divergence"] == []


@pytest.mark.parametrize("speech", sorted(rg.REBUILT_RUNS))
def test_the_real_sidecar_joins_onto_the_real_artifact(speech):
    art_p, side_p = _real(speech)
    if not (art_p.exists() and side_p.exists()):
        pytest.skip("real artifact or B1a sidecar not present")
    artifact = rg.load_artifact(art_p)
    sidecar = rg.load_rescore_sidecar(side_p, speech, rg.REBUILT_RUNS[speech])
    res = rg.regate_speech(speech, artifact, sidecar)

    assert res["join"]["sidecar_unmatched"] == []
    assert res["join"]["artifact_unscored"] == []
    assert res["join"]["sidecar_sids_not_in_artifact"] == []
