"""DC-6' acceptance gate — the named regression suite that must pass before publish.

Every case here is a claim a human already adjudicated by hand during the
2026-07-21 external audit, the DC-5 worksheet, or the DC-6 review. They are the
things we said the rebuild had to get right; this file is where "we said so"
becomes "CI says so". The suite runs against the STAGED Phase-3 artifacts (the
five rebuilt runs listed in :data:`RUNS`) — not against the live ``site-pca/``
tree, which still renders the pre-remediation runs.

Run it as the publish gate::

    .venv/bin/python -m pytest -m acceptance -q

It is also part of the default suite, so a regression cannot reach main by
someone forgetting the flag. If the artifacts are absent the whole module
skips — a checkout without ``metrics/pca_runs/`` is not a failing gate.

**Several cases are EXPECTED TO FAIL right now.** The gate defect the DC-6
review found — the relevance layer never ran, so the pack-quality gate saw no
qualifying sources and force-gated claims with perfectly good packs — is not
repaired yet. Those cases carry ``xfail(strict=True)`` naming the B1a re-score
as the repair. Strict is the point: when B1a lands they flip to passing, and a
strict xfail turns that flip into a loud XPASS instead of a silent one. They
are NOT to be forced green by weakening the assertion.

Current state (A7):

===============================================  ========  ==================
case                                             status    repaired by
===============================================  ========  ==================
Beckstrom pair 0462 + 0469                       xfail     B1a re-score
inflation pair 0030 + 0031                       passing   —
DEI claim 0056                                   passing   —
Biden 5.7% GDP 0115                              passing   —
Biden deficit-half 0244                          xfail     B1a re-score
Obama College Opportunity Summit 0046            passing   —
Obama Joining Forces 0045                        passing   —
murder-rate pair 0023 + 0024 (COHERENCE)         xfail     B1a re-score
eggs framing disclosure 0219                     passing   —
===============================================  ========  ==================
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pytest

from truthbot.verdict import verdict_audit as va

REPO = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO / "metrics" / "pca_runs"

#: speech_id → run-id prefix of the STAGED Phase-3 rebuild.
RUNS = {
    "gwbush_2006": "74a89c5f",
    "clinton_1998": "d0010426",
    "obama_2014": "4de8a551",
    "biden_2022": "37744fc8",
    "trump_2026": "4ee5a251",
}

B1A_GATE = ("gate defect not yet repaired: the relevance layer never ran, so "
            "the pack-quality gate counted no qualifying sources and "
            "force-gated a well-sourced claim — repaired by the B1a re-score")

B1A_COHERENCE = ("adjacent claims on one statistic were adjudicated "
                 "independently and disagree, unannotated — repaired by the "
                 "B1a re-score, which scores the pair together")


def _artifact_path(speech_id: str) -> Path | None:
    hits = sorted(RUNS_DIR.glob(f"{RUNS[speech_id]}*.json"))
    return hits[0] if hits else None


_MISSING = [s for s in RUNS if _artifact_path(s) is None]
pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.skipif(
        bool(_MISSING),
        reason=f"staged rebuild artifacts absent: {', '.join(_MISSING)}"),
]


@lru_cache(maxsize=None)
def _run(speech_id: str) -> dict:
    return json.loads(_artifact_path(speech_id).read_text("utf-8"))


def _sid_speech(sid: str) -> str:
    return sid.split(":", 1)[0]


def claim(sid: str) -> dict:
    run = _run(_sid_speech(sid))
    return next(c for c in run["claims"] if c["sid"] == sid)


def row(sid: str) -> dict:
    run = _run(_sid_speech(sid))
    return next(r for r in run["rows"] if r["sid"] == sid)


def verdict(sid: str) -> str:
    return str(row(sid).get("verdict") or "").strip().upper()


def gate_code(sid: str) -> str:
    r = row(sid)
    return str(r.get("evidence_gate") or r.get("provenance_code") or "")


def is_decided(sid: str) -> bool:
    """A substantive published ruling: a decided label, not split, not
    gate-forced. This is the same "decided" the parity metric counts."""
    r = row(sid)
    return (verdict(sid) in va.DECIDED and not r.get("split")
            and not gate_code(sid))


# ── the cases ────────────────────────────────────────────────────────────────

@pytest.mark.xfail(strict=True, reason=B1A_GATE)
def test_beckstrom_pair_is_decided_on_its_abundant_pack():
    """trump_2026:0462 + :0469 — a soldier's deployment and death, covered by
    the Army, the National Guard, the DoJ, AP, NPR and NBC. Both packs hold
    ten items each. 0469 nonetheless ships gate-forced Unverifiable and 0462
    ships as a models-split with no verdict at all. If a claim this well
    sourced cannot be decided, the gate is broken, not the claim."""
    for sid in ("trump_2026:0462", "trump_2026:0469"):
        assert len(_run("trump_2026")["evidence"].get(sid, [])) >= 5
        assert is_decided(sid), (
            f"{sid}: verdict={verdict(sid)!r} split={row(sid).get('split')} "
            f"gate={gate_code(sid)!r}")
    assert verdict("trump_2026:0469") == "TRUE"


def test_inflation_pair_discriminates_the_two_measures():
    """trump_2026:0030 + :0031 — adjacent sentences about core inflation that
    are NOT the same number. 0030 claims a five-year low in the LEVEL (false:
    the supported figure is only a low since March 2021). 0031 claims 1.7% for
    the last three months, which is the three-month ANNUALIZED rate and is
    true. The pre-rebuild run failed 0031 by checking it against the 2.7%
    year-over-year figure — the exact measure-mismatch class A8's computed
    exhibit exists to make legible."""
    assert verdict("trump_2026:0030") == "FALSE"
    assert verdict("trump_2026:0031") == "TRUE"
    assert is_decided("trump_2026:0031")
    r31 = (row("trump_2026:0031").get("reasoning") or "").lower()
    assert "annualized" in r31 and "three-month" in r31


def test_dei_claim_is_adverse_not_true():
    """trump_2026:0056 "We ended DEI in America." — an absolute whose
    executive orders reached federal programs only. Must not ship TRUE."""
    assert verdict("trump_2026:0056") == "FALSE"
    assert is_decided("trump_2026:0056")


def test_biden_gdp_5_7_survives_as_decided_true():
    """biden_2022:0115 — 5.7% 2021 growth, "strongest in nearly 40 years".
    BEA confirms; fastest since 1984 is ~37 years, fairly framed. The model
    audit ruled it sound and it must stay decided through the rebuild."""
    assert verdict("biden_2022:0115") == "TRUE"
    assert is_decided("biden_2022:0115")


@pytest.mark.xfail(strict=True, reason=B1A_GATE)
def test_biden_deficit_half_stays_decided():
    """biden_2022:0244 — the deficit claim shipped DECIDED in the
    pre-remediation run and the model audit ruled it sound. The rebuild
    force-gated it to Unverifiable. A remediation that turns a sound decided
    verdict into an abstention is a regression, not caution."""
    assert is_decided("biden_2022:0244"), (
        f"gate={gate_code('biden_2022:0244')!r}")


def test_obama_college_opportunity_summit_is_decided():
    """obama_2014:0046 — the c-exist x SELF named fixture: the White House
    convening its own summit, where the speaker's own record IS the primary
    record of the act. It should be decidable, and is."""
    assert verdict("obama_2014:0046") == "TRUE"
    assert is_decided("obama_2014:0046")


def test_obama_joining_forces_veterans_hiring_is_repaired():
    """obama_2014:0045 — Joining Forces veterans hiring. Gate-forced
    Unverifiable in the pre-remediation run (the original T2.4 casualty); the
    rebuild decides it. This case is the control that says the rebuild really
    did fix some of the class, not none of it."""
    assert is_decided("obama_2014:0045")
    assert verdict("obama_2014:0045") == "TRUE"


@pytest.mark.xfail(strict=True, reason=B1A_COHERENCE)
def test_murder_rate_pair_is_coherent():
    """trump_2026:0023 + :0024 COHERENCE — adjacent claims rating the SAME
    statistic (the 2025 homicide decline) must not carry contradictory
    rationales without an annotation. Today 0023 ships MISLEADING ("only a
    projection") and 0024 ships TRUE ("the largest one-year drop on record"),
    side by side, unannotated: the page contradicts itself.

    The check is deterministic (``verdict_audit.adjacent_coherence_conflicts``)
    and, run over all five staged rebuilds, this pair is the ONLY conflict it
    finds — so a failure here is signal, not noise."""
    run = _run("trump_2026")
    conflicts = va.adjacent_coherence_conflicts(run["claims"], run["rows"])
    pair = [c for c in conflicts
            if c["sids"] == ["trump_2026:0023", "trump_2026:0024"]]
    assert pair == [], pair[0]["detail"] if pair else ""


# A framing disclosure: a TRUE that rests on one framing of a number has to say
# so, by naming a contrasting framing. Deterministic — a contrast connective
# plus a second, different framing/measure marker in the same rationale.
_CONTRAST_RX = re.compile(
    r"\b(?:though|although|while|whereas|but|however|versus|vs\.?)\b",
    re.IGNORECASE)
_FRAMING_RX = re.compile(
    r"\byear[- ]over[- ]year\b|\bYoY\b|\bannual(?:ized|ly)?\b|\bpeak\b"
    r"|\bpeak[- ]to[- ]\w+\b|\bsince\b|\bmonth[- ]over[- ]month\b",
    re.IGNORECASE)


def _discloses_alternative_framing(reasoning: str) -> bool:
    return bool(_CONTRAST_RX.search(reasoning or "")
                and _FRAMING_RX.search(reasoning or ""))


def test_eggs_price_claim_discloses_its_framing():
    """trump_2026:0219 "The price of eggs is down 60 percent." — true only on
    a peak-to-current reading; year-over-year it is ~34%. A TRUE that depends
    on the framing must disclose the framing, and this rationale does."""
    r = row("trump_2026:0219").get("reasoning") or ""
    assert verdict("trump_2026:0219") == "TRUE"
    assert _discloses_alternative_framing(r), r
    assert "peak" in r.lower()


# ── the gate itself ──────────────────────────────────────────────────────────

#: Every named case, so the suite cannot silently shrink.
NAMED_CASES = (
    "test_beckstrom_pair_is_decided_on_its_abundant_pack",
    "test_inflation_pair_discriminates_the_two_measures",
    "test_dei_claim_is_adverse_not_true",
    "test_biden_gdp_5_7_survives_as_decided_true",
    "test_biden_deficit_half_stays_decided",
    "test_obama_college_opportunity_summit_is_decided",
    "test_obama_joining_forces_veterans_hiring_is_repaired",
    "test_murder_rate_pair_is_coherent",
    "test_eggs_price_claim_discloses_its_framing",
)


def test_the_gate_still_covers_every_named_case():
    missing = [n for n in NAMED_CASES if n not in globals()]
    assert missing == [], f"acceptance case(s) deleted: {missing}"


def test_the_gate_reads_staged_artifacts_not_the_published_site():
    """The published tree still renders the OLD runs. If this suite ever
    starts reading site-pca/ it stops being an acceptance gate for the
    rebuild and becomes a tautology about what is already live."""
    for speech_id in RUNS:
        path = _artifact_path(speech_id)
        assert path is not None and RUNS_DIR in path.parents
        assert "site-pca" not in str(path)
        assert _run(speech_id)["meta"]["speech_id"] == speech_id
        # staged, not published
        assert _run(speech_id)["meta"].get("rebuild_of")
