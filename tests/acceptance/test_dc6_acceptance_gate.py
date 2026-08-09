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

Current state (after the adjudication wave, 2026-08-09):

===============================================  ========  ==================
case                                             status    resolved by
===============================================  ========  ==================
Beckstrom 0469 (purposive clause)                passing   ratified conversion
Beckstrom 0462 (models-split)                    xfail     NOT the wave — see
                                                           the inventory below
inflation pair 0030 + 0031                       passing   —
DEI claim 0056                                   passing   —
Biden 5.7% GDP 0115                              passing   —
Biden deficit-half 0244                          passing   adjudication wave
Obama College Opportunity Summit 0046            passing   —
Obama Joining Forces 0045                        passing   —
murder-rate pair 0023 + 0024 (COHERENCE)         passing   NOT resolved — the
                                                           case was renamed;
                                                           read it before you
                                                           trust the "passing"
eggs framing disclosure 0219                     passing   —
===============================================  ========  ==================

WHAT THE WAVE ACTUALLY SETTLED
-------------------------------
The wave re-adjudicated 29 claims on stored packs (no retrieval). Of the three
markers that named it:

  * **0244 flipped for real.** The panel decided it TRUE. The case is now a
    plain assertion of that outcome, which is the intended lifecycle for a
    strict xfail: the event landed, so the test states the result.
  * **0462 did not flip.** The wave made the panel call the marker was waiting
    for and the models split anyway. The xfail stays, but its REASON had to
    change — "awaiting a panel call" became false the moment the call happened,
    and a marker that misdescribes what it is waiting for is worse than no
    marker.
  * **the murder-rate pair flipped for the WRONG REASON**, which is why it is
    renamed rather than converted. See
    ``test_murder_rate_pair_conflict_is_hidden_by_an_empty_rationale``: the
    deterministic checker stopped reporting the conflict because 0023 now
    carries NO RATIONALE AT ALL, not because the two claims stopped
    contradicting each other. Asserting "coherent" here would have converted a
    detection blind spot into a green check, which is exactly the test-fitting
    the strict markers exist to prevent.

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

XFAIL INVENTORY — both remaining runtime xfails in the suite
-------------------------------------------------------------
Kept here so the whole set is legible from one place; the same table lives in
``metrics/remediation_v2/t2_xfail_inventory.md``.

1. ``test_beckstrom_0462_split_survives_its_panel_call`` — **trump_2026:0462**.
   Asserts the claim reaches a substantive decided verdict (not a models-split,
   not gate-forced). It was tied to the adjudication wave, and the wave was
   added to specifically so this marker could resolve. It did not: the panel
   ran on the re-scored pack and the seats still split, so the claim ships
   with no verdict. Now tied to: **a decision nobody has made yet** — either an
   escalation policy for a persistent split, or an owner ruling on the claim.
   Re-running the same panel would just buy the same disagreement again.
2. ``tests/test_bluesky.py::…::test_post_report_returns_none_when_unconfigured``
   — not a claim at all, and not tied to any decision. Asserts
   ``post_report`` returns ``None`` when the publisher is unconfigured; today it
   raises ``NotImplementedError``. Tied to: **the Bluesky v2 publisher landing
   in Phase 7.** Listed here only so the inventory is complete.

Retired by the wave: ``test_biden_deficit_half_stays_decided`` (now a plain
assertion — the panel decided it TRUE) and ``test_murder_rate_pair_is_coherent``
(renamed, NOT resolved — see the case itself).
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

#: speech_id → run-id prefix of the STAGED artifact this gate reads.
#:
#: These were the Phase-3 rebuilds until 2026-08-09. They are now the
#: ADJUDICATION WAVE's artifacts, which are those rebuilds with 29 claims
#: re-adjudicated and everything else carried over verbatim (``rebuild_of``
#: points back at the rebuild, which is still on disk — archive-never-delete).
#: The gate has to read the newest staged artifact or it stops measuring what
#: would be published: pointed at the rebuilds it would have gone on reporting
#: biden_2022:0244 as gate-forced after the panel had actually decided it.
RUNS = {
    "gwbush_2006": "0ae0f3b8",
    "clinton_1998": "fcbc8db2",
    "obama_2014": "91d400ba",
    "biden_2022": "8577979b",
    "trump_2026": "9c4262a7",
}

#: 0462's marker after the wave. The old reason said "flips when the wave makes
#: a panel call"; the wave made it, on the fully re-scored pack, and the seats
#: still split. Leaving the old wording would have pointed the next reader at an
#: event that has already happened.
PERSISTENT_SPLIT = (
    "the adjudication wave made the panel call this marker was waiting for "
    "(2026-08-09, on the B1a+B2 re-scored pack) and the seats split anyway, so "
    "the claim still ships with no verdict — this now needs an escalation "
    "policy for a persistent split or an owner ruling, NOT another identical "
    "panel call")

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


@pytest.mark.xfail(strict=True, reason=PERSISTENT_SPLIT)
def test_beckstrom_0462_split_survives_its_panel_call():
    """trump_2026:0462 "After a four-month deployment, she voluntarily extended
    her service, and her rank was going to be lifted." — ten items in the pack,
    six of them bearing, and it still ships as a models-split with NO verdict.

    Unlike its sibling 0469 this is not a gate outcome and not a ratification
    question: the models disagreed and nothing deterministic can break the tie.

    The claim was added to the adjudication wave FOR this marker — it is not in
    the released set and was not one of the named extras, so it was carried
    purely so the test could resolve. It did not resolve. The panel ran on the
    fully re-scored pack and split again, which is a real answer: this is a
    durable disagreement, not a missing panel call. Re-running the same roster
    would buy the same split, so what this now waits on is a POLICY (how a
    persistent split is escalated or published) rather than more spend."""
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


def test_biden_deficit_half_stays_decided():
    """biden_2022:0244 — the deficit claim shipped DECIDED in the
    pre-remediation run and the model audit ruled it sound. The rebuild
    force-gated it to Unverifiable. A remediation that turns a sound decided
    verdict into an abstention is a regression, not caution.

    RESOLVED by the adjudication wave (2026-08-09). This was a strict xfail
    until the B1a+B2 re-score released the claim from the quality gate and the
    panel decided it TRUE on the released pack. The marker is gone and the
    outcome is asserted directly — which is what a strict xfail is FOR: it
    announced the flip as an XPASS instead of letting it pass unnoticed."""
    assert is_decided("biden_2022:0244"), (
        f"gate={gate_code('biden_2022:0244')!r}")
    assert verdict("biden_2022:0244") == "TRUE"


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


def test_murder_rate_pair_conflict_is_hidden_by_an_empty_rationale():
    """trump_2026:0023 + :0024 COHERENCE — the case formerly known as
    ``test_murder_rate_pair_is_coherent``, RENAMED because after the
    adjudication wave that name would have been false.

    The pair rates the same statistic (the 2025 homicide decline). 0023 ships
    MISLEADING and 0024 ships TRUE, side by side, unannotated. The wave
    re-adjudicated both together, and both came back with the SAME verdicts
    they had before. Nothing about the contradiction changed.

    What changed is that ``adjacent_coherence_conflicts`` stopped REPORTING it.
    0023 came back as a three-way split (MISLEADING / FALSE / UNVERIFIABLE)
    resolved by the stage-2 discriminator, and that path emits a verdict with
    NO rationale text. The coherence check links two claims partly through
    their rationales (``same_statistic``), so with 0023's rationale empty it
    can no longer see that the two are about the same number.

    That is a blind spot, not a repair, and this test says so out loud: it
    asserts the contradiction is still on the page, that the checker is silent,
    and — by restoring the rationale the PRIOR artifact recorded for the same
    claim — that the silence is caused by the missing text. Converting the old
    xfail into a green "coherent" assertion would have laundered a detection
    gap into a passing gate."""
    run = _run("trump_2026")
    a, b = "trump_2026:0023", "trump_2026:0024"

    # 1. The contradiction is still published, unannotated.
    assert verdict(a) == "MISLEADING" and verdict(b) == "TRUE"

    # 2. And 0023 ships a DECIDED verdict with no rationale at all — a
    #    fact-check whose reason is blank, which is its own defect.
    assert (row(a).get("reasoning") or "").strip() == "", (
        "0023 has a rationale again — re-check whether the coherence blind "
        "spot below still exists before trusting this case")
    assert row(a).get("split") is True

    # 3. The deterministic checker reports nothing. This is what the gate sees.
    conflicts = va.adjacent_coherence_conflicts(run["claims"], run["rows"])
    assert [c for c in conflicts if c["sids"] == [a, b]] == []

    # 4. Restore the rationale the pre-wave artifact recorded for 0023 and the
    #    SAME checker finds the SAME conflict — so step 3's silence is about
    #    the missing text, not about the claims having been reconciled.
    prior = json.loads(
        (RUNS_DIR / f"{run['meta']['rebuild_of']}.json").read_text("utf-8"))
    prior_reasoning = next(r.get("reasoning") or "" for r in prior["rows"]
                           if r["sid"] == a)
    assert prior_reasoning.strip(), "prior artifact has no rationale either"
    patched = [dict(r, reasoning=prior_reasoning) if r["sid"] == a else r
               for r in run["rows"]]
    revealed = va.adjacent_coherence_conflicts(run["claims"], patched)
    assert [c for c in revealed if c["sids"] == [a, b]], (
        "with 0023's prior rationale restored the checker STILL finds no "
        "conflict — the pair may genuinely have been reconciled, in which "
        "case this case should be rewritten as a real coherence assertion")


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
#:
#: Two entries were RENAMED by the adjudication wave (2026-08-09), not removed:
#: ``…0462_split_is_awaiting_a_panel_call`` → ``…0462_split_survives_its_panel_call``
#: (the call happened) and ``…murder_rate_pair_is_coherent`` →
#: ``…murder_rate_pair_conflict_is_hidden_by_an_empty_rationale`` (it is not
#: coherent). Renaming a case that now asserts something different is the point
#: of this list, not a way around it — the count is unchanged.
NAMED_CASES = (
    "test_beckstrom_0469_is_unverifiable_on_its_purposive_clause",
    "test_beckstrom_0462_split_survives_its_panel_call",
    "test_inflation_pair_discriminates_the_two_measures",
    "test_dei_claim_is_adverse_not_true",
    "test_biden_gdp_5_7_survives_as_decided_true",
    "test_biden_deficit_half_stays_decided",
    "test_obama_college_opportunity_summit_is_decided",
    "test_obama_joining_forces_veterans_hiring_is_repaired",
    "test_murder_rate_pair_conflict_is_hidden_by_an_empty_rationale",
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
