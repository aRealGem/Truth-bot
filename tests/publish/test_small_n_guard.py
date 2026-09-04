"""Small-sample guard on the verdict-panel headline (step 3e).

Below 10 DECIDED claims a percent implies more precision than the sample
supports ("9 of 9" and "8 of 9" both round to headlines a reader will
over-trust), so the headline drops the percent and shows only the plain
"X of Y decided claims rated True" ratio, and the "Truthy or better" stat
frame swaps its own percent for a one-line caveat.

HARD CONSTRAINT: the five flagship presidential reports must never show
either the dropped-percent headline or the caveat line, regardless of how
many decided claims a given report happens to carry — the guard is exempted
by speech_id (belt-and-suspenders alongside their real decided counts, which
are all comfortably over the threshold).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from truthbot.models import (
    Claim,
    Confidence,
    ConsensusVerdict,
    ModelVerdict,
    VerdictBundle,
    VerdictLabel,
)
from truthbot.publish.aggregation import family_verdict
from truthbot.publish.site import SiteReport, _verdict_panel

PRESIDENTIAL_SPEECH_IDS = [
    "clinton_1998", "gwbush_2006", "obama_2014", "biden_2022", "trump_2026",
]

_SMALL_N_NOTE = "Small sample — read the claims, not the score."


def _bundle(i: int, label: VerdictLabel) -> VerdictBundle:
    claim = Claim(transcript_id="t", text=f"Claim {i}.", speaker="Speaker",
                  context="ctx", category="economy", is_checkable=True)
    mv = ModelVerdict(adapter_name="panel", model_id="m", claim_id=claim.id,
                      label=label, confidence=Confidence.HIGH, explanation="r")
    consensus = ConsensusVerdict(
        claim_id=claim.id, model_verdicts=[mv], consensus_label=label,
        consensus_verdict=label.value, confidence=Confidence.HIGH,
        agreement=True, consensus_strength="strong", explanation="x")
    return VerdictBundle(claim=claim, speaker="Speaker", date_str="2026-01-01",
                         model_verdicts=[mv], consensus=consensus,
                         sources_consulted=[])


def _report(n_decided: int, *, speech_id: str = "") -> SiteReport:
    """A report with exactly ``n_decided`` decided (True) claims and nothing
    else — the simplest fixture that lands on a chosen decided count.
    """
    bundles = [_bundle(i, VerdictLabel.TRUE) for i in range(n_decided)]
    return SiteReport(
        report_id="00000000-1111-2222-3333-444444444444",
        speaker="Speaker", role="Role",
        date=datetime(2026, 1, 1, tzinfo=timezone.utc), venue="Somewhere",
        transcript_source_url="", bundles=bundles, speech_id=speech_id)


def _headline_pct_text(n_decided: int) -> str:
    """The big-percent headline string (e.g. '100% True') that family_verdict
    would independently compute for an all-True distribution of this size —
    re-derived rather than hardcoded so this test can't drift from the
    production formula.
    """
    dist = {"True": n_decided, "Truthy": 0, "Unverifiable": 0,
             "Falsey": 0, "False": 0, "Models split": 0}
    return family_verdict(dist).label


def _ratio_text(n_decided: int) -> str:
    dist = {"True": n_decided, "Truthy": 0, "Unverifiable": 0,
             "Falsey": 0, "False": 0, "Models split": 0}
    return family_verdict(dist).ratio_text


# ── boundary: fires at 9, not at 10 ─────────────────────────────────────────

def test_fires_at_nine_decided():
    html = _verdict_panel(_report(9))
    assert _ratio_text(9) in html
    assert _headline_pct_text(9) not in html
    assert _SMALL_N_NOTE in html


def test_does_not_fire_at_ten_decided():
    html = _verdict_panel(_report(10))
    assert _headline_pct_text(10) in html
    assert _SMALL_N_NOTE not in html


# ── content when it fires ───────────────────────────────────────────────────

def test_ratio_wording_present_and_percent_absent_when_it_fires():
    html = _verdict_panel(_report(5))
    assert "5 of 5 decided claims rated True" in html
    assert _headline_pct_text(5) not in html


def test_truthy_line_is_exactly_the_small_sample_note():
    html = _verdict_panel(_report(5))
    assert _SMALL_N_NOTE in html
    # It stands alone as the stat-frame label, not appended to other text.
    assert f'<div class="vp-stat-lbl">{_SMALL_N_NOTE}</div>' in html


# ── hard constraint: never fires for the five presidential reports ─────────

@pytest.mark.parametrize("speech_id", PRESIDENTIAL_SPEECH_IDS)
def test_guard_never_fires_for_presidential_reports_even_with_few_decided(speech_id):
    """Pathological case: even if a presidential report somehow had only 3
    decided claims, the guard must not fire — the exemption is keyed on
    speech_id, not on whatever the count happens to be.
    """
    html = _verdict_panel(_report(3, speech_id=speech_id))
    assert _SMALL_N_NOTE not in html
    assert _headline_pct_text(3) in html
    assert "vp-verdict neutral" not in html


@pytest.mark.parametrize("speech_id", PRESIDENTIAL_SPEECH_IDS)
def test_guard_never_fires_for_presidential_reports_at_realistic_scale(speech_id):
    """The five flagship reports in production carry far more than 10 decided
    claims; unchanged headline rendering at a realistic scale too.
    """
    html = _verdict_panel(_report(120, speech_id=speech_id))
    assert _SMALL_N_NOTE not in html
    assert _headline_pct_text(120) in html
