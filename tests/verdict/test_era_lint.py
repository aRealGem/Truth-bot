"""Era-gate tests (P67.5 / PR-2, remediation T1.1 + audit F7).

Policy under test (jackie, 2026-07-21): evidence must fall inside BOTH the
originally-coded retrieval window (still asserted) and the speaker's
fair-game window — utterance + 7 days; violations cite the fair-game window
in those words. Named regression pins: the two shipped Trump rationales the
audit caught judging with post-utterance world-state (trump_2026:0035 gas
prices via the Iran-war surge, trump_2026:0407 the DHS shutdown resolved
later) must be flagged by the rationale lint on the real artifact.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from truthbot.models import Evidence, SourceTier
from truthbot.verdict import era_lint
from truthbot.verdict.era_lint import (
    EraLintError,
    assert_pack_within_era,
    fair_game_end,
    item_date,
    lint_artifact,
    lint_pack_items,
    lint_rationale,
)
from truthbot.verdict.evidence_pack import EvidencePack, PackItem, _sha256

UTT = date(2026, 2, 24)          # Trump SOTU 2026
FG_END = date(2026, 3, 3)        # utterance + 7 days
CODED_WINDOW = (date(2024, 1, 1), date(2026, 5, 1))


def _item(pack_id: str, pub: str | None, snippet: str = "x") -> PackItem:
    return PackItem(
        pack_id=pack_id, source_name="S", source_url=f"https://ex.com/{pack_id}",
        tier=SourceTier.OTHER, snippet=snippet, retrieved_at="2026-07-21T00:00:00",
        sha256=_sha256(f"https://ex.com/{pack_id}", snippet), published_at=pub)


# ── fair-game window math ────────────────────────────────────────────────────


def test_fair_game_end_is_utterance_plus_seven_days() -> None:
    assert fair_game_end(UTT) == FG_END
    assert fair_game_end(date(2022, 3, 1)) == date(2022, 3, 8)


# ── item_date extraction ─────────────────────────────────────────────────────


def test_item_date_prefers_published_at_over_snippet_prefix() -> None:
    assert item_date("2026-05-01", "[2026-04-30] text") == date(2026, 5, 1)
    assert item_date(datetime(2026, 5, 1, 12, tzinfo=timezone.utc)) == date(2026, 5, 1)
    assert item_date(date(2026, 5, 1)) == date(2026, 5, 1)


def test_item_date_falls_back_to_snippet_prefix_for_pre_fix_artifacts() -> None:
    assert item_date(None, "[2026-05-01] Prices have shot up") == date(2026, 5, 1)
    assert item_date(None, "no date here") is None


# ── pack lint: both policies checked, fair-game cited ────────────────────────


def test_lint_flags_post_fair_game_item_with_fair_game_wording() -> None:
    items = [_item("E1", "2026-02-20"), _item("E2", "2026-04-30")]
    violations, dated, undated = lint_pack_items(
        "trump_2026:0001", items, UTT, window=CODED_WINDOW)
    assert dated == 2 and undated == 0
    assert [v.pack_id for v in violations] == ["E2"]
    assert "fair-game window" in violations[0].message
    assert "+ 7 days" in violations[0].message


def test_lint_still_checks_originally_coded_window() -> None:
    # After the coded window's end too — both policies must be named.
    items = [_item("E1", "2026-06-15")]
    violations, _, _ = lint_pack_items(
        "trump_2026:0001", items, UTT, window=CODED_WINDOW)
    assert len(violations) == 1
    msg = violations[0].message
    assert "coded evidence window" in msg
    assert "fair-game window" in msg


def test_lint_passes_undated_items_but_counts_them() -> None:
    items = [_item("E1", None, snippet="no prefix")]
    violations, dated, undated = lint_pack_items("s", items, UTT)
    assert violations == [] and dated == 0 and undated == 1


def test_assert_pack_within_era_raises_on_violation() -> None:
    pack = EvidencePack(sid="trump_2026:0001", window=CODED_WINDOW,
                        items=[_item("E1", "2026-05-01")])
    with pytest.raises(EraLintError, match="fair-game window"):
        assert_pack_within_era(pack, UTT)
    # clean pack → no raise; unknown utterance → no-op
    assert_pack_within_era(
        EvidencePack(sid="s", window=CODED_WINDOW, items=[_item("E1", "2026-02-25")]),
        UTT)
    assert_pack_within_era(pack, None)


# ── build-time filter ────────────────────────────────────────────────────────


def test_build_filter_drops_post_fair_game_dated_evidence() -> None:
    from truthbot.verdict.evidence_pack import _within_fair_game

    def ev(pub: datetime | None) -> Evidence:
        return Evidence(claim_id="c", source_name="s", source_url="https://e.com/a",
                        snippet="t", published_at=pub)

    assert _within_fair_game(ev(datetime(2026, 2, 25, tzinfo=timezone.utc)), UTT)
    assert _within_fair_game(ev(datetime(2026, 3, 3, tzinfo=timezone.utc)), UTT)
    assert not _within_fair_game(ev(datetime(2026, 3, 4, tzinfo=timezone.utc)), UTT)
    assert not _within_fair_game(ev(datetime(2026, 5, 1, tzinfo=timezone.utc)), UTT)
    assert _within_fair_game(ev(None), UTT)          # undated passes
    assert _within_fair_game(ev(datetime(2026, 5, 1, tzinfo=timezone.utc)), None)


def test_pack_item_round_trips_published_at_through_bridge() -> None:
    from truthbot.verdict.bridge import _pack_to_evidence

    pack = EvidencePack(sid="s", window=None,
                        items=[_item("E1", "2026-02-25"), _item("E2", None)])
    evs = _pack_to_evidence("s", pack)
    assert evs[0].published_at is not None
    assert evs[0].published_at.date() == date(2026, 2, 25)
    assert evs[1].published_at is None


# ── rationale lint ───────────────────────────────────────────────────────────


def test_rationale_lint_flags_post_window_dates_iso_and_prose() -> None:
    text = ("CONTRADICTED by E6 (dated May 1, 2026, within the evidence "
            "window), which reports prices had surged")
    flags = lint_rationale("trump_2026:0035", text, UTT)
    assert len(flags) == 1
    assert flags[0].cited_date == date(2026, 5, 1)
    assert "fair-game window" in flags[0].message
    assert "prices had" in flags[0].excerpt

    flags_iso = lint_rationale("s", "resolved on 2026-04-30 by Congress", UTT)
    assert [f.cited_date for f in flags_iso] == [date(2026, 4, 30)]


def test_rationale_lint_ignores_pre_window_dates() -> None:
    text = ("E1, E2, E3 show inflation at 2.9% in December 2024, up from "
            "2.7% in November; stabilized by February 24, 2026")
    assert lint_rationale("s", text, UTT) == []


def test_rationale_lint_bare_month_resolves_to_first() -> None:
    # "March 2026" → 2026-03-01, inside the fair-game window (ends 03-03):
    # NOT flagged. "April 2026" → 04-01: flagged.
    assert lint_rationale("s", "as of March 2026", UTT) == []
    assert len(lint_rationale("s", "as of April 2026", UTT)) == 1


# ── artifact lint + named audit regressions (F7) ─────────────────────────────


def test_lint_artifact_on_synthetic_run() -> None:
    artifact = {
        "meta": {"speech_id": "trump_2026", "date": "2026-02-24"},
        "evidence": {
            "trump_2026:0001": [
                {"source_url": "https://a.com/x", "published_at": None,
                 "snippet": "[2026-05-01] later world-state"},
                {"source_url": "https://b.com/y", "published_at": None,
                 "snippet": "[2026-02-20] contemporaneous"},
            ],
        },
        "rows": [
            {"item_id": "trump_2026:0002",
             "reasoning": "falsified by the outcome on 2026-04-30"},
        ],
    }
    report = lint_artifact(artifact)
    assert report.utterance == UTT
    assert [v.sid for v in report.pack_violations] == ["trump_2026:0001"]
    assert [f.sid for f in report.rationale_flags] == ["trump_2026:0002"]
    assert report.rerun_sids == ["trump_2026:0001", "trump_2026:0002"]


_TRUMP_ARTIFACT = (Path(__file__).resolve().parents[2]
                   / "metrics" / "pca_runs"
                   / "5ebcabe3-05d9-4484-a8a5-e28e0bf883e1.json")


@pytest.mark.skipif(not _TRUMP_ARTIFACT.exists(),
                    reason="canonical Trump artifact not in this checkout")
def test_audit_f7_regressions_flagged_on_real_artifact() -> None:
    """The audit's shipped era-leakage cases must be caught: 0035 (gas price
    falsified via the post-speech surge) and 0407 (shutdown falsified via its
    later resolution) — pack violations and/or rationale flags."""
    artifact = json.loads(_TRUMP_ARTIFACT.read_text(encoding="utf-8"))
    report = lint_artifact(artifact)
    rerun = set(report.rerun_sids)
    assert "trump_2026:0035" in rerun
    assert "trump_2026:0407" in rerun
    # The audit counted 127 pack items past the fair-game window; the lint
    # must find at least that scale of violations (snippet-prefix dating).
    assert len(report.pack_violations) >= 100
