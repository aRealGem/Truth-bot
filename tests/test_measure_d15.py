"""D15 blast-radius measurement (scripts/measure_d15.py) — offline, $0.

Nothing here touches a model, a proxy or the network. The fixtures are
synthetic so the suite never depends on the real artifacts, and what is under
test is the part the ratification decision rests on: that the measurement runs
the gate BOTH ways over the same pack, that it counts a flagged item's
bearing-ness against the tiers that can actually credit the quota, that the two
stance vintages are kept apart, and — most important — that measuring never
turns the flag on for anybody else.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
_SPEC = importlib.util.spec_from_file_location(
    "measure_d15", REPO / "scripts" / "measure_d15.py")
md = importlib.util.module_from_spec(_SPEC)
sys.modules["measure_d15"] = md
_SPEC.loader.exec_module(md)          # must import clean with no key present

from truthbot.verdict import utterance_record as ur  # noqa: E402
from truthbot.verdict.consolidator import GATE_INSUFFICIENT  # noqa: E402

SPEECH = "trump_2026"
UTTERANCE = "2026-02-24"
SID = f"{SPEECH}:0469"

DCPD = ("https://www.govinfo.gov/content/pkg/DCPD-202600136/pdf/"
        "DCPD-202600136.pdf")
CREC = ("https://www.govinfo.gov/content/pkg/CREC-2026-02-24/pdf/"
        "CREC-2026-02-24.pdf")
NPR = "https://www.npr.org/2025/11/27/nx-s1-5622955/national-guard"


def _ev(url, *, tier="Government", supports=True, published="2026-02-24T00:00:00"):
    return {"claim_id": SID, "source_name": "R1", "source_url": url,
            "source_tier": tier, "snippet": "a snippet",
            "retrieved_at": "2026-02-25T04:20:27.824223",
            "published_at": published, "supports_claim": supports,
            "relevance_score": 0.5}


def _artifact(evidence: dict, rows: dict):
    return {
        "run_id": "test-run",
        "meta": {"speaker": "Donald Trump", "date": UTTERANCE,
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


def test_a_claim_carried_by_its_own_transcript_is_counted_as_a_gate_change():
    """Two GOVERNMENT records of the speech plus one real outside source: the
    quota is met today and unmet under D15."""
    art = _artifact(
        {SID: [_ev(NPR, tier="Established", published="2025-11-27T00:00:00"),
               _ev(DCPD), _ev(CREC)]},
        {SID: "TRUE"})
    out = md.measure_speech(SPEECH, art, None)["vintages"]["stored"]

    assert out["flagged_items"] == 2
    assert out["claims_touched"] == 1
    assert out["by_rule"] == {ur.RULE_DCPD: 1, ur.RULE_CREC: 1}
    assert out["by_tier"] == {"Government": 2}
    # Both are bearing AND Tier-1..3, so both are credits D15 would remove.
    assert out["flagged_bearing"] == 2
    assert out["flagged_bearing_t13"] == 2
    assert out["gate_changed"] == 1
    assert [f["sid"] for f in out["newly_gated_sids"]] == [SID]
    assert out["released_sids"] == []


def test_an_unflagged_pack_reports_no_change_at_all():
    art = _artifact({SID: [_ev(NPR, tier="Established",
                               published="2025-11-27T00:00:00"),
                           _ev("https://www.reuters.com/world/us/a", tier="Wire",
                               published="2025-11-28T00:00:00")]},
                    {SID: "TRUE"})
    out = md.measure_speech(SPEECH, art, None)["vintages"]["stored"]
    assert out["flagged_items"] == 0
    assert out["gate_changed"] == 0
    assert out["claims_touched"] == 0


def test_a_stanceless_transcript_is_flagged_but_costs_no_credit():
    """Counting flagged items is not the same as counting credits removed —
    an item with a null stance was never crediting the quota anyway."""
    art = _artifact({SID: [_ev(NPR, tier="Established", supports=True,
                               published="2025-11-27T00:00:00"),
                           _ev(DCPD, supports=None),
                           _ev(CREC, supports=None)]},
                    {SID: "UNVERIFIABLE"})
    out = md.measure_speech(SPEECH, art, None)["vintages"]["stored"]
    assert out["flagged_items"] == 2
    assert out["flagged_bearing"] == 0
    assert out["flagged_bearing_t13"] == 0
    assert out["gate_changed"] == 0        # gated before, gated after


def test_the_rescored_vintage_sees_what_b1a_bought_and_stored_does_not():
    """B1a is what gave the transcripts a bearing stance, so the two vintages
    must be measured separately — and a sid the sidecar never scored is left
    out of the rescored column rather than counted as unchanged."""
    other = f"{SPEECH}:0470"
    art = _artifact(
        {SID: [_ev(NPR, tier="Established", supports=True,
                   published="2025-11-27T00:00:00"),
               _ev(DCPD, supports=None), _ev(CREC, supports=None)],
         other: [_ev(DCPD, supports=None)]},
        {SID: "UNVERIFIABLE", other: "UNVERIFIABLE"})
    side = _sidecar({SID: [
        {"source_url": NPR, "relevance_score": 0.9, "supports_claim": True},
        {"source_url": DCPD, "relevance_score": 0.9, "supports_claim": True},
        {"source_url": CREC, "relevance_score": 0.9, "supports_claim": True},
    ]})
    v = md.measure_speech(SPEECH, art, side)["vintages"]

    assert v["stored"]["claims"] == 2
    assert v["stored"]["flagged_bearing"] == 0
    assert v["stored"]["gate_changed"] == 0
    # Only the scored sid appears in the rescored column.
    assert v["rescored"]["claims"] == 1
    assert v["rescored"]["flagged_bearing"] == 2
    assert v["rescored"]["gate_changed"] == 1


def test_measuring_never_switches_the_flag_on_for_anyone_else(monkeypatch):
    """The switch is passed as an argument. If the measurement set the env
    instead, every later consolidation in the process would inherit it.

    Since the 2026-08-09 ratification the ambient default is ON, so the check
    is the mirror of what it used to be: pin the env OFF, measure, and confirm
    the measurement did not switch it back on behind everyone's back."""
    monkeypatch.setenv(ur.FLAG_ENV, "0")
    art = _artifact({SID: [_ev(DCPD), _ev(CREC)]}, {SID: "TRUE"})
    md.measure_speech(SPEECH, art, None)
    assert ur.flag_enabled() is False
