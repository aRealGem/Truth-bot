"""Consolidator gate behaviour at the REAL stance-null rate (Phase A, A1.3).

THE FINDING THIS MODULE IS: ``tests/verdict/test_consolidator_v2.py::_ev``
defaults ``supports=True``, so every pack the v2 suite has ever consolidated
had a 0% stance-null rate. The corpus does not look like that. Across the ten
stored runs (5 published + 5 rebuilt, 8,837 items) the stance-null rate runs
20.5%–34.2% — ``retrievers.py`` maps a retriever's stance "context" onto
``supports_claim=None``, and nothing downstream ever fills it in, because the
v2 path has no scoring step (``verify.relevance.score_evidence`` is reachable
only from the legacy v1 provider and the R4 archive retriever).

``consolidator._bearing`` requires stance in (True, False), so a null item can
never credit ``MIN_BEARING_T13``. At a ~25% null rate that quietly removes a
quarter of every pack from the quota — and the T2.4 gate then FORCES
Unverifiable. A suite that only ever sees supports=True cannot observe any of
this. These packs are built with a realistic, DETERMINISTIC null distribution
(no randomness — same input, same pack, like the consolidator itself) so the
gate behaviour that actually ships is what gets asserted.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from truthbot.models import Evidence, SourceTier
from truthbot.verdict.consolidator import (GATE_INSUFFICIENT, MIN_BEARING_T13,
                                           PACK_CAP_V2, consolidate,
                                           scoring_telemetry)

UTT = date(2026, 2, 24)
WINDOW = (date(2024, 1, 1), date(2026, 5, 1))

#: The observed stance-null band over the ten stored run artifacts — the
#: interval the fixture below is calibrated to sit inside.
CORPUS_NULL_RATE_MIN = 0.205
CORPUS_NULL_RATE_MAX = 0.342


def _ev(url: str, tier: SourceTier = SourceTier.ESTABLISHED,
        supports: bool | None = True, pub: str = "2026-02-20") -> Evidence:
    """Deliberately NO default-True shortcut in this module's callers: every
    call site states the stance it means."""
    return Evidence(
        claim_id="c", source_name="S", source_url=url, source_tier=tier,
        snippet="on-topic reporting", supports_claim=supports,
        published_at=datetime.fromisoformat(pub).replace(tzinfo=timezone.utc))


def realistic_stances(n: int) -> list[bool | None]:
    """A deterministic ~25% stance-null distribution, in the corpus band.

    Every 4th item is None; the rest alternate supports/refutes. Fixed
    sequence, no randomness — the consolidator is deterministic and its tests
    must be too, so a failure is always reproducible from the index alone.
    """
    out: list[bool | None] = []
    for i in range(n):
        out.append(None if i % 4 == 3 else (i % 2 == 0))
    return out


def _pack_with_realistic_nulls(n: int, tier: SourceTier = SourceTier.ESTABLISHED):
    stances = realistic_stances(n)
    return [_ev(f"https://news{i}.com/story-{i}", tier, stances[i])
            for i in range(n)]


# ── the fixture is itself the finding: pin its shape ─────────────────────────

def test_fixture_null_rate_sits_inside_the_observed_corpus_band():
    stances = realistic_stances(100)
    rate = stances.count(None) / len(stances)
    assert rate == 0.25
    assert CORPUS_NULL_RATE_MIN <= rate <= CORPUS_NULL_RATE_MAX, (
        "the fixture must reproduce the rate production actually produces")


def test_fixture_is_deterministic():
    assert realistic_stances(40) == realistic_stances(40)


def test_the_old_all_supports_default_hides_the_gate_entirely():
    """DIRECT contrast, the reason this module exists. The same eight Tier-1..3
    items pass the quota trivially when every stance is True — which is what
    the pre-existing v2 tests assert — and the realistic pack is what shows
    the null items dropping out of the credit count."""
    all_true = [_ev(f"https://news{i}.com/s{i}", SourceTier.ESTABLISHED, True)
                for i in range(8)]
    optimistic = consolidate("s:0001", [("R1", all_true)],
                             utterance=UTT, window=WINDOW)
    realistic = consolidate("s:0001", [("R1", _pack_with_realistic_nulls(8))],
                            utterance=UTT, window=WINDOW)
    assert optimistic.quota_met and not optimistic.gate_code
    # Same tier, same count, same order — only the stance distribution moved.
    assert len(optimistic.items) == len(realistic.items) == 8
    assert scoring_telemetry(
        [it.evidence for it in optimistic.items])["stance_null"] == 0
    assert scoring_telemetry(
        [it.evidence for it in realistic.items])["stance_null"] == 2


# ── gate behaviour under a realistic null distribution ───────────────────────

def test_realistic_nulls_still_pass_when_enough_bearing_items_survive():
    """25% null is survivable: 8 Tier-1..3 items minus 2 nulls still clears
    MIN_BEARING_T13. The gate is not supposed to be a hair trigger."""
    res = consolidate("s:0001", [("R1", _pack_with_realistic_nulls(8))],
                      utterance=UTT, window=WINDOW)
    bearing = sum(1 for it in res.items
                  if it.evidence.supports_claim in (True, False))
    assert bearing == 6 >= MIN_BEARING_T13
    assert res.quota_met and res.gate_code == ""


def test_a_pack_that_is_all_nulls_gates_however_good_the_tiers_are():
    """Ten GOVERNMENT-tier items, every one of them stance-null: zero credits.
    Tier quality is irrelevant to _bearing — this is the failure mode the v2
    path walks into whenever the retrievers return "context"."""
    items = [_ev(f"https://agency{i}.gov/report", SourceTier.GOVERNMENT, None)
             for i in range(10)]
    res = consolidate("s:0001", [("R1", items)], utterance=UTT, window=WINDOW)
    assert len(res.items) == 10
    assert not res.quota_met and res.gate_code == GATE_INSUFFICIENT


def test_credits_count_only_bearing_items_at_the_quota_boundary():
    """Exactly MIN_BEARING_T13 bearing items passes; one fewer gates. Pins the
    boundary so a future quota change is a deliberate edit, not a drift."""
    def _res(n_bearing: int):
        items = [_ev(f"https://news{i}.com/b{i}", SourceTier.ESTABLISHED, True)
                 for i in range(n_bearing)]
        items += [_ev(f"https://news{i}.com/n{i}", SourceTier.ESTABLISHED, None)
                  for i in range(n_bearing, 8)]
        return consolidate("s:0001", [("R1", items)],
                           utterance=UTT, window=WINDOW)

    assert _res(MIN_BEARING_T13).quota_met
    assert not _res(MIN_BEARING_T13 - 1).quota_met


# ── the Beckstrom regression ─────────────────────────────────────────────────

def test_beckstrom_shape_one_bearing_t13_item_gates_to_insufficient():
    """REGRESSION, reproducing trump_2026:0469 ("Sarah Beckstrom died in order
    to defend our capital") from the rebuilt trump artifact
    4ee5a251: a FULL pack of ten — 3 Political, 2 Other, 5 Tier-1..3 — in
    which exactly ONE Tier-1..3 item carries a stance and the other four are
    null. Credits = 1 < MIN_BEARING_T13, so a claim with ten sources on it,
    including wire and government records, is forced Unverifiable with
    provenance code ``insufficient-qualifying-evidence``.

    Political items cannot credit the quota by design (a press release never
    proves a claim), and the four null Tier-1..3 items cannot credit it
    because nothing ever gave them a stance. The gate is not wrong here —
    it is reporting, accurately, that the SCORING never happened.
    """
    items = [
        # Tier-1..3, exactly one of them bearing.
        _ev("https://www.npr.org/2025/11/27/guard-shooting",
            SourceTier.ESTABLISHED, True),
        _ev("https://apnews.com/article/fba1273c", SourceTier.WIRE, None),
        _ev("https://www.govinfo.gov/content/pkg/DCPD-202600136/pdf/x.pdf",
            SourceTier.GOVERNMENT, None),
        _ev("https://www.govinfo.gov/content/pkg/CREC-2026-02-24/pdf/y.pdf",
            SourceTier.GOVERNMENT, None),
        _ev("https://www.nbcnews.com/news/us-news/beckstrom",
            SourceTier.ESTABLISHED, None),
        # Political — at the MAX_S5 cap, and non-crediting by construction.
        _ev("https://dc.ng.mil/Public-Affairs/News-Release/Article/4344097",
            SourceTier.POLITICAL, True),
        _ev("https://www.justice.gov/usao-dc/pr/afghan-national-charged",
            SourceTier.POLITICAL, True),
        _ev("https://mast.house.gov/2025/12/honoring-sarah-beckstrom",
            SourceTier.POLITICAL, True),
        # Other — at the MAX_T6 cap.
        _ev("https://www.spokesman.com/stories/2025/nov/27/president-says",
            SourceTier.OTHER, True),
        _ev("https://www.axios.com/2025/11/28/guard-member-shot",
            SourceTier.OTHER, True),
    ]
    res = consolidate("trump_2026:0469", [("R1", items)],
                      utterance=UTT, window=WINDOW)

    assert len(res.items) == PACK_CAP_V2 == 10, "a FULL pack, not a thin one"
    tel = scoring_telemetry([it.evidence for it in res.items])
    assert tel["stance_null"] == 4 and tel["relevance_scored"] == 0
    bearing_t13 = sum(
        1 for it in res.items
        if it.evidence.source_tier in (SourceTier.GOVERNMENT, SourceTier.WIRE,
                                       SourceTier.ESTABLISHED)
        and it.evidence.supports_claim in (True, False))
    assert bearing_t13 == 1
    assert not res.quota_met
    assert res.gate_code == GATE_INSUFFICIENT


def test_beckstrom_shape_clears_the_gate_once_the_nulls_are_scored():
    """The counterfactual that makes the regression above a SCORING defect and
    not an evidence defect: give the same four Tier-1..3 items the stance a
    relevance layer would have assigned, and the identical pack decides."""
    items = [
        _ev("https://www.npr.org/2025/11/27/guard-shooting",
            SourceTier.ESTABLISHED, True),
        _ev("https://apnews.com/article/fba1273c", SourceTier.WIRE, True),
        _ev("https://www.govinfo.gov/content/pkg/DCPD-202600136/pdf/x.pdf",
            SourceTier.GOVERNMENT, True),
        _ev("https://www.govinfo.gov/content/pkg/CREC-2026-02-24/pdf/y.pdf",
            SourceTier.GOVERNMENT, None),
        _ev("https://www.nbcnews.com/news/us-news/beckstrom",
            SourceTier.ESTABLISHED, None),
    ]
    res = consolidate("trump_2026:0469", [("R1", items)],
                      utterance=UTT, window=WINDOW)
    assert res.quota_met and res.gate_code == ""
