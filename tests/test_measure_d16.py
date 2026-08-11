"""D16(α) blast-radius measurement (scripts/measure_d16.py) — offline, $0.

Nothing here touches a model, a proxy or the network. The fixtures are
synthetic so the suite never depends on the real artifacts, and what is under
test is the part the ratification decision rests on: that the measurement runs
the gate BOTH ways over the same pack, that it attributes each released item to
an AGENCY (the unit a reviewer ratifies), that the two stance vintages are kept
apart, and — most important — that measuring never turns the flag on for
anybody else.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
_SPEC = importlib.util.spec_from_file_location(
    "measure_d16", REPO / "scripts" / "measure_d16.py")
md = importlib.util.module_from_spec(_SPEC)
sys.modules["measure_d16"] = md
_SPEC.loader.exec_module(md)          # must import clean with no key present

from truthbot.verdict import statistical_release as sr  # noqa: E402
from truthbot.verdict.consolidator import GATE_INSUFFICIENT  # noqa: E402

SPEECH = "gwbush_2006"
UTTERANCE = "2006-01-31"
SID = f"{SPEECH}:0134"

# Published 3 days AFTER the speech, reporting the month BEFORE it.
BLS_A = "https://www.bls.gov/news.release/archives/empsit_02032006.pdf"
BLS_B = "https://www.bls.gov/news.release/history/empsit_02032006.txt"
BLS_SNIPPET = ("[2006-02-03] BLS Employment Situation (Jan 2006) — monthly "
               "nonfarm payroll series.")
#: Same band, same tier, same stance — the principal's own executive document.
ONDCP = "https://www.justice.gov/archive/olp/pdf/ndcs06.pdf"
ONDCP_SNIPPET = ("[2006-02-01] ONDCP National Drug Control Strategy showing a "
                 "drop since 2001.")
NPR = "https://www.npr.org/2006/01/10/payrolls"


def _ev(url, *, tier="Government", supports=True, snippet=BLS_SNIPPET,
        published="2006-02-03T00:00:00"):
    return {"claim_id": SID, "source_name": "R1", "source_url": url,
            "source_tier": tier, "snippet": snippet,
            "retrieved_at": "2006-02-04T04:20:27.824223",
            "published_at": published, "supports_claim": supports,
            "relevance_score": 0.5}


def _artifact(evidence: dict, rows: dict):
    return {
        "run_id": "test-run",
        "meta": {"speaker": "George W. Bush", "date": UTTERANCE,
                 "speech_id": SPEECH},
        "claims": [{"sid": sid, "text": f"claim text for {sid}", "context": "",
                    "layer_a": {"label": "check-worthy", "source": "A2"}}
                   for sid in evidence],
        "rows": [{"sid": sid, "status": "resolved", "verdict": v,
                  **({"provenance_code": GATE_INSUFFICIENT}
                     if v == "UNVERIFIABLE" else {})}
                 for sid, v in rows.items()],
        "evidence": evidence,
    }


def _sidecar(sids: dict):
    return {"schema": "truthbot-rescore-sidecar v1", "speech_id": SPEECH,
            "source_run": "test-run", "model": "claude-haiku",
            "generated": "2026-08-08T00:00:00+00:00", "spend_usd": 0.01,
            "sids": sids, "soft_failures": []}


def test_two_post_speech_bls_releases_are_counted_as_a_gate_change():
    """A pack whose only bearing Tier-1..3 items are the jobs report published
    three days after the speech: gated today, decided under D16(α)."""
    art = _artifact({SID: [_ev(BLS_A), _ev(BLS_B)]}, {SID: "UNVERIFIABLE"})
    out = md.measure_speech(SPEECH, art, None)["vintages"]["stored"]

    assert out["released_items"] == 2
    assert out["claims_touched"] == 1
    assert out["by_agency"] == {"BLS": 2}
    assert out["by_rule"] == {sr.RULE_MONTH: 2}
    assert out["by_tier"] == {"Government": 2}
    assert out["released_bearing"] == 2 and out["released_bearing_t13"] == 2
    assert out["gate_changed"] == 1
    assert [f["sid"] for f in out["released_sids"]] == [SID]
    # D16 only ADDS credits, so this direction must always be empty.
    assert out["newly_gated_sids"] == []


def test_the_principals_own_executive_document_is_not_released():
    """The gwbush_2006:0217 shape: same band, same tier, same stance — and a
    valid pre-utterance period in the snippet. Excluded on the allowlist."""
    art = _artifact({SID: [_ev(ONDCP, snippet=ONDCP_SNIPPET,
                               published="2006-02-01T00:00:00"),
                           _ev("https://files.eric.ed.gov/fulltext/ED503096.pdf",
                               snippet=ONDCP_SNIPPET,
                               published="2006-02-01T00:00:00")]},
                    {SID: "UNVERIFIABLE"})
    out = md.measure_speech(SPEECH, art, None)["vintages"]["stored"]
    assert out["released_items"] == 0
    assert out["gate_changed"] == 0
    assert out["claims_touched"] == 0


def test_a_pre_utterance_pack_reports_no_change_at_all():
    """Nothing in the post-speech band means nothing for D16 to release."""
    art = _artifact({SID: [_ev(NPR, tier="Established",
                               published="2006-01-10T00:00:00"),
                           _ev("https://www.reuters.com/a", tier="Wire",
                               published="2006-01-11T00:00:00")]},
                    {SID: "TRUE"})
    out = md.measure_speech(SPEECH, art, None)["vintages"]["stored"]
    assert out["released_items"] == 0 and out["gate_changed"] == 0


def test_a_stanceless_release_is_counted_but_buys_no_credit():
    """Counting released items is not the same as counting credits gained — an
    item with a null stance was never able to credit the quota anyway."""
    art = _artifact({SID: [_ev(BLS_A, supports=None),
                           _ev(BLS_B, supports=None)]},
                    {SID: "UNVERIFIABLE"})
    out = md.measure_speech(SPEECH, art, None)["vintages"]["stored"]
    assert out["released_items"] == 2
    assert out["released_bearing"] == 0
    assert out["gate_changed"] == 0


def test_the_rescored_vintage_sees_what_b1a_bought_and_stored_does_not():
    other = f"{SPEECH}:0135"
    art = _artifact(
        {SID: [_ev(BLS_A, supports=None), _ev(BLS_B, supports=None)],
         other: [_ev(BLS_A, supports=None)]},
        {SID: "UNVERIFIABLE", other: "UNVERIFIABLE"})
    side = _sidecar({SID: [
        {"source_url": BLS_A, "relevance_score": 0.9, "supports_claim": True},
        {"source_url": BLS_B, "relevance_score": 0.9, "supports_claim": True},
    ]})
    v = md.measure_speech(SPEECH, art, side)["vintages"]

    assert v["stored"]["claims"] == 2
    assert v["stored"]["released_bearing"] == 0
    assert v["stored"]["gate_changed"] == 0
    # Only the scored sid appears in the rescored column.
    assert v["rescored"]["claims"] == 1
    assert v["rescored"]["released_bearing"] == 2
    assert v["rescored"]["gate_changed"] == 1


def test_measuring_never_switches_the_flag_on_for_anyone_else(monkeypatch):
    """The switch is passed as an argument. If the measurement set the env
    instead, every later consolidation in the process would inherit it.

    Since the 2026-08-09 ratification the ambient default is ON, so the check
    pins the env OFF, measures, and confirms the measurement did not switch it
    back on behind everyone's back."""
    monkeypatch.setenv(sr.FLAG_ENV, "0")
    art = _artifact({SID: [_ev(BLS_A), _ev(BLS_B)]}, {SID: "UNVERIFIABLE"})
    md.measure_speech(SPEECH, art, None)
    assert sr.flag_enabled() is False
