"""B2 — the scoring-prompt fix, the arithmetic-hinge guard, and the two-sidecar
merge. Offline, $0: nothing here touches a model, a proxy or the network.

What has to be trustworthy before any money moves: that the new contract fields
survive the round trip from model reply to sidecar to re-gate, that a model
answering in the OLD shape still scores normally, that the targeted subset is
derived from rules a reviewer can read, and — the one that protects data we
already paid for — that merging the B2 sidecar over B1a's does not clobber
B1a's scores for sids B2 never touched.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)          # must import clean with no key
    return mod


rs = _load("rescore_stored_packs")
rg = _load("regate_from_rescore")
b2 = _load("b2_primary_series")

from truthbot.models import Evidence, SourceTier  # noqa: E402
from truthbot.verdict.consolidator import ConsolidatedItem  # noqa: E402
from truthbot.verify import relevance  # noqa: E402

BLS = "https://www.bls.gov/webapps/legacy/cpsatab1.htm"
FRED = "https://fred.stlouisfed.org/series/LNS12000000"
NPR = "https://www.npr.org/2026/01/05/jobs"


def _ev(url=BLS, **kw):
    kw.setdefault("snippet", "Employed persons, seasonally adjusted.")
    return Evidence(claim_id="c", source_name="BLS", source_url=url,
                    source_tier=SourceTier.GOVERNMENT, **kw)


# ── 1. the prompt ───────────────────────────────────────────────────────────

def test_the_prompt_tells_the_scorer_a_data_series_takes_a_side():
    p = relevance._SCORE_SYSTEM.lower()
    assert "primary data series" in p
    assert "never context" in p
    # The "context" option survives for genuine background — the fix must not
    # simply abolish the neutral answer.
    assert "reserve context for genuine background" in p
    assert "one_line_why" in p and "comparison you actually made" in p
    assert "arithmetic_hinge" in p


# ── 2. the contract round trip ──────────────────────────────────────────────

def test_score_evidence_records_the_comparison_and_the_hinge():
    ev = [_ev(), _ev(FRED)]

    def llm(system, user):
        return {"scores": [
            {"i": 1, "relevance": 0.95, "supports": True,
             "one_line_why": "claim says most ever working; table row Jan 2026 "
                             "shows 163.9M, the series maximum",
             "arithmetic_hinge": True},
            {"i": 2, "relevance": 0.9, "supports": False,
             "one_line_why": "claim says record; FRED series peaks in Nov 2025"},
        ]}

    relevance.score_evidence(llm, "More Americans are working than ever", ev)
    assert ev[0].supports_claim is True
    assert ev[0].arithmetic_hinge is True
    assert "series maximum" in ev[0].one_line_why
    assert ev[1].supports_claim is False
    # Absent field means "not asserted" — never silently true.
    assert ev[1].arithmetic_hinge is False


def test_a_reply_in_the_old_shape_still_scores_normally():
    """A model that ignores the two new keys must still buy us the stance we
    paid for. The fields are optional on the wire, by design."""
    ev = [_ev()]
    relevance.score_evidence(
        lambda s, u: {"scores": [{"i": 1, "relevance": 0.8, "supports": True}]},
        "a claim", ev)
    assert ev[0].supports_claim is True
    assert ev[0].relevance_score == 0.8
    assert ev[0].one_line_why is None
    assert ev[0].arithmetic_hinge is False


def test_a_malformed_hinge_never_switches_the_guard_on():
    ev = [_ev()]
    relevance.score_evidence(
        lambda s, u: {"scores": [{"i": 1, "relevance": 0.8, "supports": True,
                                  "arithmetic_hinge": "yes please"}]},
        "a claim", ev)
    assert ev[0].arithmetic_hinge is False


# ── 3. the pack payload ─────────────────────────────────────────────────────

def test_the_payload_prefers_the_comparison_and_flags_the_hinge():
    ev = _ev(supports_claim=True,
             published_at=datetime(2026, 1, 5, tzinfo=timezone.utc))
    ev.one_line_why = "claim says 3.2M; table row shows 2.9M"
    ev.arithmetic_hinge = True
    p = ConsolidatedItem(evidence=ev, draw_round=0, retriever="R1").to_payload_v2()
    assert p["one_line_why"] == "claim says 3.2M; table row shows 2.9M"
    assert p["arithmetic_hinge"] is True


def test_an_unscored_item_still_falls_back_to_its_snippet():
    ev = _ev(supports_claim=True,
             published_at=datetime(2026, 1, 5, tzinfo=timezone.utc))
    p = ConsolidatedItem(evidence=ev, draw_round=0, retriever="R1").to_payload_v2()
    assert p["one_line_why"] == "Employed persons, seasonally adjusted."
    assert "arithmetic_hinge" not in p


# ── 4. sidecar persistence ──────────────────────────────────────────────────

def test_scored_rows_write_the_new_fields_only_when_they_exist():
    plain = _ev(supports_claim=True)
    rich = _ev(FRED, supports_claim=False)
    rich.one_line_why = "series peaks earlier"
    rich.arithmetic_hinge = True
    rows = rs.scored_rows([plain, rich])
    assert "one_line_why" not in rows[0] and "arithmetic_hinge" not in rows[0]
    assert rows[1]["one_line_why"] == "series peaks earlier"
    assert rows[1]["arithmetic_hinge"] is True


def test_the_overlay_applies_the_new_fields_and_never_clears_a_hinge():
    ev = [_ev()]
    ev[0].arithmetic_hinge = True
    rg.overlay_rescores(ev, [{"source_url": BLS, "relevance_score": 0.7,
                              "supports_claim": True}])
    assert ev[0].supports_claim is True
    # A B1a-vintage row carries no hinge key; it must not blank one.
    assert ev[0].arithmetic_hinge is True


# ── 5. the targeting subset ─────────────────────────────────────────────────

def test_primary_record_detection_is_host_anchored():
    assert b2.is_primary_record(BLS) is True
    assert b2.is_primary_record(FRED) is True
    assert b2.is_primary_record("https://data.bls.gov/timeseries/LNS12000000")
    # A news story ABOUT the number is not the number.
    assert b2.is_primary_record(NPR) is False
    # govinfo carries everything, so the package id has to name a primary
    # collection.
    assert b2.is_primary_record(
        "https://www.govinfo.gov/content/pkg/BUDGET-1998-APP/pdf/x.pdf") is True
    assert b2.is_primary_record(
        "https://www.govinfo.gov/content/pkg/DCPD-202600136/pdf/x.pdf") is False


def test_the_subset_takes_only_stanceless_tier13_primary_items():
    art = {
        "run_id": "r", "meta": {"speaker": "X", "date": "2026-02-24"},
        "claims": [{"sid": "trump_2026:0054", "text": "t"},
                   {"sid": "trump_2026:0055", "text": "t"},
                   {"sid": "trump_2026:0056", "text": "t"}],
        "rows": [],
        "evidence": {
            # qualifies: stanceless, Tier-1, primary
            "trump_2026:0054": [{"source_url": BLS, "source_tier": "Government",
                                 "snippet": "", "supports_claim": None}],
            # already has a stance -> nothing to buy
            "trump_2026:0055": [{"source_url": BLS, "source_tier": "Government",
                                 "snippet": "", "supports_claim": True}],
            # stanceless but not a primary record
            "trump_2026:0056": [{"source_url": NPR, "source_tier": "Established",
                                 "snippet": "", "supports_claim": None}],
        },
    }
    out = b2.derive_subset("trump_2026", art, None)
    assert out["sids"] == ["trump_2026:0054"]
    assert out["trigger_items"] == 1


def test_the_subset_reflects_the_b1a_sidecar_not_the_stale_artifact():
    """B1a already bought a stance for some of these items; asking again would
    be paying twice for the same answer."""
    art = {"run_id": "r", "meta": {"speaker": "X", "date": "2026-02-24"},
           "claims": [{"sid": "trump_2026:0054", "text": "t"}], "rows": [],
           "evidence": {"trump_2026:0054": [
               {"source_url": BLS, "source_tier": "Government", "snippet": "",
                "supports_claim": None}]}}
    assert b2.derive_subset("trump_2026", art, None)["sids"] == ["trump_2026:0054"]
    side = {"sids": {"trump_2026:0054": [
        {"source_url": BLS, "relevance_score": 0.9, "supports_claim": True}]}}
    assert b2.derive_subset("trump_2026", art, side)["sids"] == []


def test_a_speech_record_is_excluded_from_the_subset():
    """D15 already says a transcript can never credit the quota, so buying it a
    stance buys nothing."""
    crec = ("https://www.govinfo.gov/content/pkg/CREC-2026-02-24/pdf/"
            "CREC-2026-02-24.pdf")
    art = {"run_id": "r", "meta": {"speaker": "X", "date": "2026-02-24"},
           "claims": [{"sid": "trump_2026:0054", "text": "t"}], "rows": [],
           "evidence": {"trump_2026:0054": [
               {"source_url": crec, "source_tier": "Government",
                "snippet": "Congressional Record.", "supports_claim": None,
                "published_at": "2026-02-24T00:00:00"}]}}
    out = b2.derive_subset("trump_2026", art, None)
    assert out["sids"] == []
    assert out["excluded_utterance_records"] == 1


def test_only_sids_narrows_the_run_without_defeating_the_resume_filter(tmp_path):
    art = {"evidence": {"s:1": [{}], "s:2": [{}], "s:3": [{}]}}
    texts = {"s:1": "a", "s:2": "b", "s:3": "c"}
    side = {"sids": {"s:2": []}}
    p = tmp_path / "sids.json"
    p.write_text(json.dumps(["s:1", "s:2"]))
    only = rs.load_only_sids(str(p))
    # s:2 is in the targeting list but already scored -> still skipped.
    assert rs.pending_sids(art, side, texts, only) == ["s:1"]
    assert rs.pending_sids(art, side, texts, None) == ["s:1", "s:3"]
    assert rs.load_only_sids(None) is None


# ── 6. the merge — the part that protects money already spent ───────────────

def _side(label, sids, spend):
    return {"schema": rs.SIDECAR_SCHEMA, "speech_id": "trump_2026",
            "source_run": "r", "model": "claude-haiku", "generated": "now",
            "spend_usd": spend, "sids": sids, "soft_failures": [],
            "pass_label": label}


def test_b2_wins_its_own_sids_and_leaves_b1a_untouched_elsewhere():
    b1a = _side("b1a", {"s:1": [{"source_url": BLS, "supports_claim": None}],
                        "s:2": [{"source_url": NPR, "supports_claim": True}]},
                1.0632)
    b2side = _side("b2", {"s:1": [{"source_url": BLS, "supports_claim": True,
                                   "one_line_why": "table row shows 2.9M"}]},
                   0.23)
    m = rg.merge_sidecars(b1a, b2side)
    assert m["sids"]["s:1"][0]["supports_claim"] is True
    assert m["sids"]["s:1"][0]["one_line_why"] == "table row shows 2.9M"
    # The sid B2 never targeted keeps B1a's row exactly.
    assert m["sids"]["s:2"] == [{"source_url": NPR, "supports_claim": True}]
    assert m["spend_usd"] == 1.2932
    assert m["spend_by_pass"] == {"b1a": 1.0632, "b2": 0.23}
    assert m["sids_by_pass"] == {"b1a": 2, "b2": 1}


def test_merging_with_no_b2_sidecar_is_the_b1a_result():
    b1a = _side("b1a", {"s:1": [{"source_url": BLS, "supports_claim": True}]}, 1.0)
    m = rg.merge_sidecars(b1a, None)
    assert m["sids"] == b1a["sids"]
    assert m["spend_usd"] == 1.0


# ── 7. the reported numbers ─────────────────────────────────────────────────

def test_stance_counts_read_the_overlay_when_there_is_one():
    art = {"evidence": {"s:1": [
        {"source_url": BLS, "supports_claim": None},
        {"source_url": NPR, "supports_claim": True}]}}
    before = rg.stance_counts(art)
    assert (before["null"], before["supports"]) == (1, 1)
    assert before["null_rate"] == 0.5
    after = rg.stance_counts(art, {"s:1": [
        {"source_url": BLS, "supports_claim": False, "arithmetic_hinge": True}]})
    assert (after["null"], after["refutes"], after["supports"]) == (0, 1, 1)
    assert after["arithmetic_hinge"] == 1
    assert after["null_rate"] == 0.0


def test_hinge_items_are_collected_with_their_claim_and_reasoning():
    art = {"claims": [{"sid": "s:1", "text": "More Americans are working"}],
           "evidence": {}}
    hinges = rg.hinge_items(art, {"s:1": [
        {"source_url": BLS, "supports_claim": True, "arithmetic_hinge": True,
         "one_line_why": "series maximum is Jan 2026"},
        {"source_url": NPR, "supports_claim": True}]})
    assert len(hinges) == 1
    assert hinges[0]["sid"] == "s:1"
    assert hinges[0]["claim"] == "More Americans are working"
    assert hinges[0]["one_line_why"] == "series maximum is Jan 2026"
