"""Historical-era evidence policy (wiki projects:truthbot:historical-era-design).

Pre-web speeches (< 1997-01-01) run lenient: retrospective sources admitted
behind contemporaneous ones, era-contemporaneous GOVERNMENT documents credit
the quota even stance-neutral, and predictions stay strict. Offline."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from truthbot.models import Evidence, SourceTier
from truthbot.verdict import era_lint
from truthbot.verdict.consolidator import GATE_INSUFFICIENT, consolidate
from truthbot.verdict.evidence_pack_v2 import build_evidence_pack_v2
from truthbot.verdict.speech_context import register_speech_date

NIXON = date(1974, 1, 30)
WINDOW = (date(1972, 2, 1), date(1974, 4, 30))
register_speech_date("pytest_nixon", NIXON)


def _ev(url, *, tier=SourceTier.GOVERNMENT, supports=None, year=1973, month=6):
    pub = datetime(year, month, 15, tzinfo=timezone.utc) if year else None
    return Evidence(claim_id="", source_name="S", source_url=url,
                    source_tier=tier, snippet="doc", supports_claim=supports,
                    published_at=pub)


# ── policy helpers ───────────────────────────────────────────────────────────

def test_pre_web_cutoff_is_clintons_second_term():
    assert era_lint.is_pre_web(date(1990, 1, 31))          # Bush Sr. — lenient
    assert era_lint.is_pre_web(date(1996, 12, 31))
    assert not era_lint.is_pre_web(date(1998, 1, 27))      # Clinton '98 — strict
    assert not era_lint.is_pre_web(None)


def test_predictions_keep_strict_mode_even_pre_web():
    assert era_lint.era_mode_for(NIXON, "Farm income is up 70 percent.") == "lenient"
    assert era_lint.era_mode_for(NIXON, "There will be no recession in the "
                                        "United States of America.") == "strict"
    assert era_lint.era_mode_for(date(2026, 2, 24), "anything") == "strict"


# ── consolidator lenient mode ────────────────────────────────────────────────

def test_lenient_admits_retrospective_ranked_last():
    retro = _ev("https://fed-history.example.gov/1974-review", year=2019,
                supports=True)
    contemp = _ev("https://fraser.stlouisfed.org/bls-dec-1973", year=1973,
                  supports=True)
    res = consolidate("s", [("R1", [retro, contemp])], utterance=NIXON,
                      window=WINDOW, era_mode="lenient")
    assert [it.evidence.source_url for it in res.items] == [
        contemp.source_url, retro.source_url]          # era class beats draw order
    assert res.retrospective == 1 and "outside-coded-window" not in res.dropped


def test_strict_mode_unchanged_drops_retrospective():
    retro = _ev("https://fed-history.example.gov/1974-review", year=2019)
    res = consolidate("s", [("R1", [retro])], utterance=NIXON, window=WINDOW)
    assert res.items == [] and res.dropped.get("outside-coded-window") == 1


def test_lenient_quota_credits_contemporaneous_gov_context_items():
    # Two era-dated GOVERNMENT documents, stance-neutral (the Nixon-probe
    # starvation shape) — lenient meets quota, strict does not.
    docs = [_ev("https://esmis.nal.usda.gov/FIS-1973", year=1973),
            _ev("https://apps.bea.gov/scb-1974-jan", year=1974, month=1)]
    lenient = consolidate("s", [("R1", docs)], utterance=NIXON,
                          window=WINDOW, era_mode="lenient")
    strict = consolidate("s", [("R1", docs)], utterance=NIXON, window=WINDOW)
    assert lenient.quota_met and lenient.gate_code == ""
    assert not strict.quota_met and strict.gate_code == GATE_INSUFFICIENT


def test_lenient_retrospective_does_not_credit_quota():
    docs = [_ev("https://gov.example/retro-a", year=2019, supports=True),
            _ev("https://gov.example/retro-b", year=2020, supports=False)]
    res = consolidate("s", [("R1", docs)], utterance=NIXON,
                      window=WINDOW, era_mode="lenient")
    # bearing Tier-1..3 counts regardless of era; these ARE bearing → quota met
    assert res.quota_met
    neutral = [_ev("https://gov.example/retro-c", year=2019),
               _ev("https://gov.example/retro-d", year=2020)]
    res2 = consolidate("s", [("R1", neutral)], utterance=NIXON,
                       window=WINDOW, era_mode="lenient")
    # stance-neutral RETROSPECTIVE gov docs get no credit — only era-dated ones
    assert not res2.quota_met


# ── pack builder end-to-end ──────────────────────────────────────────────────

class _R:
    label = "R1"

    def __init__(self):
        self.calls = []

    def shortlist(self, claim_text, *, context="", utterance=None, window=None):
        self.calls.append({"context": context, "utterance": utterance,
                           "window": window})
        return [_ev("https://esmis.nal.usda.gov/FIS-1973", year=1973),
                _ev("https://history.example.org/farm-income", year=2018,
                    tier=SourceTier.ESTABLISHED, supports=True)]


def test_lenient_pack_builds_with_retrospective_and_no_era_error():
    r = _R()
    pack = build_evidence_pack_v2("pytest_nixon:9000",
                                  "Farm income is up 70 percent.", (r,))
    # retriever briefed via context, not hard-scoped params
    assert "HISTORICAL CLAIM" in r.calls[0]["context"]
    assert r.calls[0]["utterance"] is None and r.calls[0]["window"] is None
    # era-dated gov doc first, retrospective admitted behind it, no EraLintError
    assert pack.items[0].source_url.startswith("https://esmis")
    assert len(pack.items) == 2 and pack.gate_code == ""


def test_predictive_claim_stays_strict_and_filters_retrospective():
    r = _R()
    pack = build_evidence_pack_v2("pytest_nixon:9001",
                                  "There will be no recession.", (r,))
    assert "HISTORICAL CLAIM" not in r.calls[0]["context"]
    assert r.calls[0]["utterance"] == NIXON
    # the 2018 retrospective item is filtered by strict consolidation
    assert [it.source_url for it in pack.items] == [
        "https://esmis.nal.usda.gov/FIS-1973"]


def test_fed_archive_domains_rank_government():
    """FRASER/FRED (St. Louis Fed) are Government-tier — the archival
    workhorse for pre-web claims must credit the T2.4 quota."""
    from truthbot.verify.sources.brave import classify_tier
    assert classify_tier("https://fraser.stlouisfed.org/title/employment-"
                         "situation-144/december-1973-56071").value == "Government"
    assert classify_tier("https://fred.stlouisfed.org/series/UNRATE").value == "Government"
    # substring-abuse guard: not-actually-fed domains stay OTHER
    assert classify_tier("https://notstlouisfed.org/x").value == "Other"
