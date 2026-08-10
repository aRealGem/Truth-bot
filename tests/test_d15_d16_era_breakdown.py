"""D15 + D16(α) era breakdown (scripts/d15_d16_era_breakdown.py) — offline, $0.

Nothing here touches a model, a proxy or the network. The fixtures are
synthetic so the suite never depends on the real artifacts, and what is under
test is what the M-6 check rests on: that the two rules are measured TOGETHER
as well as separately, that the decided-rate convention is applied in the
direction it claims, that the anecdote adjustment uses the same claim-type
convention the rest of the packet does, that the concentration finding is
COMPUTED rather than narrated — and that measuring never turns a flag on for
anybody else.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
_SPEC = importlib.util.spec_from_file_location(
    "d15_d16_era_breakdown", REPO / "scripts" / "d15_d16_era_breakdown.py")
eb = importlib.util.module_from_spec(_SPEC)
sys.modules["d15_d16_era_breakdown"] = eb
_SPEC.loader.exec_module(eb)          # must import clean with no key present

from truthbot.verdict import statistical_release as sr  # noqa: E402
from truthbot.verdict import utterance_record as ur  # noqa: E402
from truthbot.verdict.consolidator import GATE_INSUFFICIENT  # noqa: E402

SPEECH = "gwbush_2006"
UTTERANCE = "2006-01-31"

# A D15 utterance record: the Congressional Record for the day of the address.
CREC = ("https://www.congress.gov/content/pkg/CREC-2006-01-31/pdf/"
        "CREC-2006-01-31.pdf")
# A D16 statistical release: the January jobs report, published 3 days later.
BLS_A = "https://www.bls.gov/news.release/archives/empsit_02032006.pdf"
BLS_B = "https://www.bls.gov/news.release/history/empsit_02032006.txt"
BLS_SNIPPET = ("[2006-02-03] BLS Employment Situation (Jan 2006) — monthly "
               "nonfarm payroll series.")
NPR = "https://www.npr.org/2006/01/10/payrolls"


def _ev(url, *, tier="Government", supports=True, snippet="a snippet",
        published="2006-02-03T00:00:00"):
    return {"claim_id": "c", "source_name": "R1", "source_url": url,
            "source_tier": tier, "snippet": snippet,
            "retrieved_at": "2006-02-04T04:20:27.824223",
            "published_at": published, "supports_claim": supports,
            "relevance_score": 0.5}


def _artifact(evidence: dict, rows: dict, types: dict | None = None,
              speech: str = SPEECH, utterance: str = UTTERANCE,
              speaker: str = "George W. Bush"):
    types = types or {}
    return {
        "run_id": "test-run",
        "meta": {"speaker": speaker, "date": utterance, "speech_id": speech},
        "claims": [{"sid": sid, "text": f"claim text for {sid}", "context": "",
                    "layer_a": {"label": "check-worthy", "source": "A2",
                                "claim_type": types.get(sid, "statistical")}}
                   for sid in evidence],
        "rows": [{"sid": sid, "status": "resolved", "verdict": v,
                  **({"provenance_code": GATE_INSUFFICIENT}
                     if v == "UNVERIFIABLE" else {})}
                 for sid, v in rows.items()],
        "evidence": evidence,
    }


def _block(artifact, speech=SPEECH):
    gated = eb.gate_all_ways(speech, artifact, None)
    return eb.per_speech_block(speech, gated, "stored")


# ── the three views ─────────────────────────────────────────────────────────

def test_d15_and_d16_are_measured_separately_and_together():
    """One claim carried by the day's Congressional Record (D15 takes it away),
    one carried only by the post-speech jobs report (D16 gives it back)."""
    a, b = f"{SPEECH}:0001", f"{SPEECH}:0002"
    art = _artifact(
        {a: [_ev(NPR, tier="Established", published="2006-01-10T00:00:00"),
             _ev(CREC, published="2006-01-31T00:00:00"),
             _ev(CREC + "?x=1", published="2006-01-31T00:00:00")],
         b: [_ev(BLS_A, snippet=BLS_SNIPPET), _ev(BLS_B, snippet=BLS_SNIPPET)]},
        {a: "TRUE", b: "UNVERIFIABLE"})
    blk = _block(art)

    assert blk["d15"]["newly_gated_sids"] == [a]
    assert blk["d15"]["newly_gated_shipping_true"] == 1
    assert blk["d15"]["released_sids"] == []       # D15 never releases
    assert blk["d16"]["released_sids"] == [b]
    assert blk["d16"]["newly_gated_sids"] == []    # D16 never gates
    assert blk["combined"]["newly_gated"] == 1
    assert blk["combined"]["released"] == 1
    assert blk["combined"]["net"] == 0


def test_the_net_is_released_minus_gated_not_a_count_of_touched_claims():
    a, b, c = f"{SPEECH}:0001", f"{SPEECH}:0002", f"{SPEECH}:0003"
    art = _artifact(
        {sid: [_ev(NPR, tier="Established", published="2006-01-10T00:00:00"),
               _ev(CREC, published="2006-01-31T00:00:00"),
               _ev(CREC + f"?x={sid}", published="2006-01-31T00:00:00")]
         for sid in (a, b, c)},
        {a: "TRUE", b: "TRUE", c: "TRUE"})
    blk = _block(art)
    assert blk["combined"]["newly_gated"] == 3
    assert blk["combined"]["released"] == 0
    assert blk["combined"]["net"] == -3


# ── the decided-rate convention ─────────────────────────────────────────────

def test_decided_rate_applies_the_convention_it_documents():
    info = {"a": {"verdict": "TRUE"}, "b": {"verdict": "UNVERIFIABLE"},
            "c": {"verdict": "FALSE"}}
    sids = ["a", "b", "c"]
    before = eb.decided_rate(sids, info, gated=set(), released=set(),
                             releases_decide=False)
    assert (before["decided"], before["total"]) == (2, 3)

    # A newly-gated claim becomes undecided; a released one becomes decided
    # under the UPPER bound and stays put under the lower.
    upper = eb.decided_rate(sids, info, gated={"a"}, released={"b"},
                            releases_decide=True)
    lower = eb.decided_rate(sids, info, gated={"a"}, released={"b"},
                            releases_decide=False)
    assert upper["decided"] == 2       # b gained, a lost
    assert lower["decided"] == 1       # a lost, b unchanged


def test_a_row_with_no_verdict_counts_as_undecided():
    """Three of the five rebuilt runs carry rows with an empty verdict. Counting
    them as decided would inflate every 'before' figure in the packet."""
    info = {"a": {"verdict": ""}, "b": {"verdict": "TRUE"}}
    assert eb.decided_rate(["a", "b"], info, gated=set(), released=set(),
                           releases_decide=False)["decided"] == 1


def test_the_anecdote_adjustment_uses_the_shared_claim_type_convention():
    """Same string the renderer, the consistency checker and dc6_package key on
    — one convention, four consumers."""
    a, b = f"{SPEECH}:0001", f"{SPEECH}:0002"
    art = _artifact(
        {a: [_ev(NPR, tier="Established", published="2006-01-10T00:00:00")],
         b: [_ev(NPR + "?2", tier="Established",
                 published="2006-01-10T00:00:00")]},
        {a: "TRUE", b: "UNVERIFIABLE"},
        types={b: eb.ANECDOTE_CLAIM_TYPE})
    blk = _block(art)
    assert blk["anecdotes"] == 1
    # Raw: 1 of 2 decided. Adjusted drops the anecdote: 1 of 1.
    assert blk["decided_rate"]["raw_before"]["rate"] == 0.5
    assert blk["decided_rate"]["adjusted_before"]["rate"] == 1.0


# ── concentration, computed rather than narrated ────────────────────────────

def _row(speech, claims, gated, released=0):
    return {"speech": speech, "speaker": speech,
            "claims_measured": claims,
            "combined": {"newly_gated": gated, "released": released,
                         "net": released - gated}}


def test_concentration_flags_a_rate_ratio_of_two_or_more():
    conc = eb.concentration([_row("early", 100, 2), _row("late", 100, 20)])
    assert conc["rate_concentrated"] is True
    assert conc["withholding_rate_ratio"] == 10.0
    assert conc["verdict"].startswith("YES")
    assert "late" in conc["verdict"]


def test_concentration_reports_no_effect_when_the_rates_are_even():
    conc = eb.concentration([_row("early", 100, 10), _row("late", 200, 21)])
    assert conc["rate_concentrated"] is False
    assert conc["verdict"].startswith("No material era concentration")


def test_share_and_rate_can_disagree_and_both_are_reported():
    """The big speech carries the larger SHARE simply by being big, while the
    small one has the higher RATE. A packet that reported only the share would
    point at the wrong speaker."""
    conc = eb.concentration([_row("small", 50, 10), _row("big", 400, 40)])
    assert conc["withholding_top"]["speech"] == "big"        # 80% of the count
    assert conc["withholding_rate_spread"]["max_speech"] == "small"  # 20% vs 10%
    assert conc["rate_concentrated"] is True


def test_the_spread_reading_names_a_disagreement_between_the_two_bases():
    sp = {"raw_before": {"spread": 0.095}, "raw_after": {"spread": 0.126},
          "adjusted_before": {"spread": 0.039},
          "adjusted_after": {"spread": 0.036}}
    text = eb._spread_reading(sp)
    assert "disagreement is the finding" in text
    assert "widens by 3.1 pp" in text and "narrows by 0.3 pp" in text


# ── the flags stay off ──────────────────────────────────────────────────────

def test_measuring_never_switches_either_flag_on_for_anyone_else(monkeypatch):
    """Both switches are passed as arguments. If the measurement set the env
    instead, every later consolidation in the process would inherit them.

    Since the 2026-08-09 ratification the ambient default is ON for both, so
    the check pins them OFF, measures all four configurations, and confirms the
    measurement did not switch either back on behind everyone's back."""
    monkeypatch.setenv(ur.FLAG_ENV, "0")
    monkeypatch.setenv(sr.FLAG_ENV, "0")
    sid = f"{SPEECH}:0001"
    art = _artifact({sid: [_ev(CREC, published="2006-01-31T00:00:00"),
                           _ev(BLS_A, snippet=BLS_SNIPPET)]},
                    {sid: "TRUE"})
    eb.gate_all_ways(SPEECH, art, None)
    assert ur.flag_enabled() is False
    assert sr.flag_enabled() is False


def test_all_four_configurations_are_run_for_every_claim():
    sid = f"{SPEECH}:0001"
    art = _artifact({sid: [_ev(NPR, tier="Established",
                               published="2006-01-10T00:00:00")]},
                    {sid: "TRUE"})
    quota = eb.gate_all_ways(SPEECH, art, None)["quota"]["stored"]
    assert set(eb.CONFIGS) == {"base", "d15", "d16", "both"}
    # Four COMPUTED configurations, plus a fifth answer that is READ rather
    # than computed: ``shipped``, the gate outcome the artifact recorded. It is
    # the alternative baseline (--baseline shipped) and must never be mistaken
    # for a configuration of the gate — nothing is re-gated to produce it.
    assert set(quota[sid]) == set(eb.CONFIGS) | {"shipped"}
    assert quota[sid]["shipped"] is True     # no gate code on the row
