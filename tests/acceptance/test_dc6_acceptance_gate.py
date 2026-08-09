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

**Some cases are EXPECTED TO FAIL right now**, and each one names the pending
event that will flip it. They carry ``xfail(strict=True)``. Strict is the
point: when the event lands they flip to passing, and a strict xfail turns that
flip into a loud XPASS instead of a silent one. They are NOT to be forced green
by weakening the assertion.

Current state (T-2, 2026-08-09):

===============================================  ========  ==================
case                                             status    resolved by
===============================================  ========  ==================
Beckstrom 0469 (purposive clause)                passing   ratified conversion
Beckstrom 0462 (models-split)                    xfail     adjudication wave
inflation pair 0030 + 0031                       passing   —
DEI claim 0056                                   passing   —
Biden 5.7% GDP 0115                              passing   —
Biden deficit-half 0244                          xfail     adjudication wave
Obama College Opportunity Summit 0046            passing   —
Obama Joining Forces 0045                        passing   —
murder-rate pair 0023 + 0024 (COHERENCE)         xfail     adjudication wave
eggs framing disclosure 0219                     passing   —
===============================================  ========  ==================

THE BECKSTROM PAIR SPLIT IN TWO (T-2)
--------------------------------------
Until 2026-08-09 this module carried ONE case asserting that both 0462 and 0469
should end up decided, xfailed as "gate defect, repaired by the B1a re-score".
The owner then ratified the opposite reading for 0469, and a header that still
called it a defect was contradicting a ratified decision. The pair is now two
cases with two different fates:

  * **0469 — "Sarah Beckstrom died in order to defend our capital."** The
    ratified conversion: the FACTUAL CORE (she died, shot on Guard duty near
    the White House) is confirmed by several independent non-Political sources,
    but the PURPOSIVE clause — *in order to defend* — is a statement about
    purpose, and the only item in the pack that speaks to purpose at all is a
    House member's tribute page, which is Political-tier and under the Claim
    Eval v3 ruling is attribution, never proof. Unverifiable is therefore the
    CORRECT verdict, not a gate failure. The case now PASSES, and it verifies
    all three limbs of that rationale against the artifact rather than merely
    quoting it.
  * **0462 — "she voluntarily extended her service…"** unaffected by the
    ratification. It ships as a models-split with no verdict at all, which no
    deterministic re-gate can settle. It stays xfail pending a panel call.

XFAIL INVENTORY — all four runtime xfails in the suite
-------------------------------------------------------
Kept here so the whole set is legible from one place; the same table lives in
``metrics/remediation_v2/t2_xfail_inventory.md``.

1. ``test_beckstrom_0462_split_is_awaiting_a_panel_call`` — **trump_2026:0462**.
   Asserts the claim reaches a substantive decided verdict (not a models-split,
   not gate-forced). Tied to: **the adjudication wave.** A split with no verdict
   is a disagreement between models, and only a panel call can resolve it.
   CAVEAT, stated because it matters for planning: 0462 is NOT in the wave's
   released set and NOT one of the six named extras, so on the current scope it
   would stay xfail after the wave runs. Either the scope grows by one or this
   marker outlives the wave.
2. ``test_biden_deficit_half_stays_decided`` — **biden_2022:0244**. Asserts the
   claim is decided (it shipped decided pre-remediation; the rebuild
   force-gated it). Tied to: **the adjudication wave.** B1a+B2 already did its
   part — the re-gate RELEASES 0244 from the quality gate — but release only
   makes a claim ELIGIBLE for a decided verdict. The panel call is what
   actually flips this marker.
3. ``test_murder_rate_pair_is_coherent`` — **trump_2026:0023 + :0024**. Asserts
   the deterministic adjacent-coherence check finds no conflict between the two
   claims rating the same homicide statistic. Tied to: **the adjudication
   wave**, which re-adjudicates the pair TOGETHER; both sids are among the six
   named extras, so this one is in scope.
4. ``tests/test_bluesky.py::…::test_post_report_returns_none_when_unconfigured``
   — not a claim at all, and not tied to any decision. Asserts
   ``post_report`` returns ``None`` when the publisher is unconfigured; today it
   raises ``NotImplementedError``. Tied to: **the Bluesky v2 publisher landing
   in Phase 7.** Listed here only so the inventory is complete.
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

WAVE_SPLIT = ("models-split with no verdict at all — no deterministic re-gate "
              "can settle a disagreement between models; flips when the "
              "adjudication wave makes a panel call on this claim")

WAVE_RELEASED = ("the B1a+B2 re-gate RELEASES this claim from the quality "
                 "gate, but release only makes it ELIGIBLE for a decided "
                 "verdict — flips when the adjudication wave re-adjudicates it")

WAVE_COHERENCE = ("adjacent claims on one statistic were adjudicated "
                  "independently and disagree, unannotated — flips when the "
                  "adjudication wave re-adjudicates the pair TOGETHER (both "
                  "sids are named extras, so this is in wave scope)")

#: The ratified conversion for trump_2026:0469, recorded verbatim so the reason
#: a claim is Unverifiable is reviewable next to the assertion that checks it.
#: Each clause is verified against the artifact by the test below — the string
#: is documentation, the assertions are the gate.
BECKSTROM_0469_RATIONALE = (
    "Unverifiable is CORRECT, not a gate failure: the purposive clause "
    "(\"in order to defend our capital\") is uncheckable; the factual core "
    "(Sarah Beckstrom died, shot on National Guard duty near the White House) "
    "is confirmed by independent non-Political sources; and the sole item in "
    "the pack that supports the PURPOSE is Political-tier, which under Claim "
    "Eval v3 is attribution, never proof."
)

#: Language that speaks to PURPOSE rather than to the fact of the death.
#: Deliberately narrow: "died of her wounds", "died following the shooting" and
#: "while serving with the National Guard" are reports of the FACT and must not
#: match, or the "sole purposive support" limb becomes untestable.
_PURPOSIVE_RX = re.compile(
    r"\bin order to\b|\bto (?:protect|defend)\b|\bdefend(?:ing|ed)\b"
    r"|\bsacrific\w*|\bgave (?:her|his) life\b|\bdied for\b",
    re.IGNORECASE)

#: Tiers that can carry PROOF. "Political" is excluded on purpose — that
#: exclusion is the whole content of the ratified 0469 reading.
_NON_POLITICAL_TIERS = ("Established", "Wire", "Other", "Government")


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

def test_beckstrom_0469_is_unverifiable_on_its_purposive_clause():
    """trump_2026:0469 "Sarah Beckstrom died in order to defend our capital."
    — the RATIFIED conversion (2026-08-09).

    This case used to be half of an xfail asserting the claim should ship TRUE.
    The owner ratified the opposite: Unverifiable is the right answer here, and
    for a reason that has nothing to do with the pack being thin. The pack is
    not thin — ten items, AP, NPR, NBC, Axios, DoJ, two Guard releases.

    The claim has two parts and they are not equally checkable. The FACTUAL
    CORE is confirmed several times over by sources that are not anybody's
    press shop. The PURPOSIVE clause — *in order to defend* — asserts why she
    was there, and the only item in the pack that speaks to purpose at all is a
    House member's tribute page. Political-tier, and under the Claim Eval v3
    ruling a partisan release is attribution, never proof.

    All three limbs are checked against the artifact rather than asserted from
    the prose, because a rationale nobody verifies is just a comment."""
    sid = "trump_2026:0469"
    pack = _run("trump_2026")["evidence"].get(sid, [])
    assert len(pack) >= 5, f"{sid}: pack is thin ({len(pack)}) — different bug"

    # Limb 1: the verdict really is Unverifiable, and really is undecided.
    assert verdict(sid) == "UNVERIFIABLE", BECKSTROM_0469_RATIONALE
    assert not is_decided(sid)

    # Limb 2: the factual core is confirmed — more than one bearing supporter
    # from a tier that can carry proof.
    core = [e for e in pack
            if e.get("supports_claim") is True
            and str(e.get("source_tier")) in _NON_POLITICAL_TIERS]
    assert len(core) >= 2, (
        f"{sid}: factual core is NOT confirmed ({len(core)} non-Political "
        f"supporters) — that would be a different finding than the ratified "
        f"one")

    # Limb 3: the SOLE purposive support is Political-tier.
    purposive = [e for e in pack
                 if e.get("supports_claim") is True
                 and _PURPOSIVE_RX.search(e.get("snippet") or "")]
    assert purposive, f"{sid}: no purposive support at all — rationale is stale"
    tiers = sorted({str(e.get("source_tier")) for e in purposive})
    assert tiers == ["Political"], (
        f"{sid}: purposive support is no longer Political-only ({tiers}) — "
        f"the ratified rationale needs revisiting, not this assertion "
        f"weakening. {BECKSTROM_0469_RATIONALE}")


@pytest.mark.xfail(strict=True, reason=WAVE_SPLIT)
def test_beckstrom_0462_split_is_awaiting_a_panel_call():
    """trump_2026:0462 "After a four-month deployment, she voluntarily extended
    her service, and her rank was going to be lifted." — ten items in the pack,
    six of them bearing, and it still ships as a models-split with NO verdict.

    Unlike its sibling 0469 this is not a gate outcome and not a ratification
    question: the models disagreed and nothing deterministic can break the tie.
    It flips when a panel adjudicates it. See the module docstring for why that
    may not happen in the current wave — 0462 is not in the released set and is
    not one of the six named extras."""
    sid = "trump_2026:0462"
    assert len(_run("trump_2026")["evidence"].get(sid, [])) >= 5
    assert is_decided(sid), (
        f"{sid}: verdict={verdict(sid)!r} split={row(sid).get('split')} "
        f"gate={gate_code(sid)!r}")


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


@pytest.mark.xfail(strict=True, reason=WAVE_RELEASED)
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


@pytest.mark.xfail(strict=True, reason=WAVE_COHERENCE)
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
    "test_beckstrom_0469_is_unverifiable_on_its_purposive_clause",
    "test_beckstrom_0462_split_is_awaiting_a_panel_call",
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
