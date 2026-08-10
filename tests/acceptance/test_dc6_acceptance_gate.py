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

Current state (after the wave rulings, 2026-08-10):

===============================================  ========  ==================
case                                             status    resolved by
===============================================  ========  ==================
Beckstrom 0469 (purposive clause)                passing   ratified conversion
Beckstrom 0462 (models-split)                    passing   escalation POLICY —
                                                           persistent split
                                                           publishes as split
inflation pair 0030 + 0031                       passing   —
DEI claim 0056                                   passing   —
Biden 5.7% GDP 0115                              passing   —
Biden deficit-half 0244                          passing   adjudication wave
Obama College Opportunity Summit 0046            passing   —
Obama Joining Forces 0045                        passing   ratified conversion
                                                           — now asserts GATED
murder-rate pair 0023 + 0024 (COHERENCE)         passing   R-3 + D14 ANNOTATE
eggs framing disclosure 0219                     passing   —
no blank rationales (corpus)                     xfail     one claim cannot be
                                                           repaired for $0
===============================================  ========  ==================

WHAT THE 2026-08-10 RULINGS SETTLED
------------------------------------
  * **0462 — persistent-split-after-2 PUBLISHES.** Two independent panels split
    the same way on the same evidence. The ruling is that this is a legitimate
    outcome in the verdict vocabulary, not a failure awaiting a third call, so
    the case inverts: it now asserts the split as STABLE. Owner-revisitable.
  * **0045 — the D15 flags were CORRECT.** The owner's readout found both
    flagged items (govinfo DCPD-201400050, CREC-2014-01-28) literally reprint
    the sentence under evaluation. Gating is right and the rule needs no
    change, so the case converts from "the rebuild repaired it" to "it is
    withheld, and here is why" — the same discipline as the 0469 conversion.
  * **the murder-rate pair is annotated, not reconciled.** 0023's rationale was
    re-emitted from stored panel output, which un-blinded the coherence
    checker; the D14 disposition for this publish is ANNOTATE, so both rows
    carry a ``coherence_note`` and the labels were NOT forced to agree.
  * **no blank rationales is now a publish-blocking gate.** It caught a second
    claim nobody had named: biden_2022:0432, tie-routed with no rationale in
    any run of its lineage. That is the remaining xfail.

WHAT THE WAVE SETTLED BEFORE THAT (2026-08-09)
-----------------------------------------------
The wave re-adjudicated 29 claims on stored packs (no retrieval). **0244
flipped for real** — the panel decided it TRUE, so the marker is gone and the
outcome is asserted directly. **0462 did not flip**, which is what made the
escalation policy necessary. **The murder-rate pair flipped for the WRONG
REASON** — the checker went quiet because 0023 lost its rationale, not because
the claims stopped contradicting each other — which is why it was renamed
rather than converted, and why R-3 exists.

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

1. ``test_no_published_verdict_ships_without_a_rationale`` — **biden_2022:0432**.
   Asserts the corpus has NO published verdict with an empty rationale, which
   the R-3 ruling makes publish-blocking. One claim fails it: 0432 was
   tie-routed to MISLEADING by the stage-2 discriminator in the phase-3 rebuild
   and again in the wave, and NO run in its lineage ever stored a rationale for
   it (the pre-remediation run has it as a split with no verdict at all). So
   unlike trump_2026:0023 there is nothing to adopt, and synthesizing text is
   the one thing R-3 forbids. Tied to: **a panel call that captures seat
   rationales, or an owner ruling.**
2. ``tests/test_bluesky.py::…::test_post_report_returns_none_when_unconfigured``
   — not a claim at all, and not tied to any decision. Asserts
   ``post_report`` returns ``None`` when the publisher is unconfigured; today it
   raises ``NotImplementedError``. Tied to: **the Bluesky v2 publisher landing
   in Phase 7.** Listed here only so the inventory is complete.

Retired by the wave: ``test_biden_deficit_half_stays_decided`` (now a plain
assertion — the panel decided it TRUE) and ``test_murder_rate_pair_is_coherent``
(renamed, NOT resolved — see the case itself).
Retired by the 2026-08-10 rulings: ``…0462_split_survives_its_panel_call``
(the escalation policy made the split the expected outcome, so the case asserts
it directly).
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
#:
#: As of 2026-08-10 they are the WAVE RULINGS artifacts: the wave's runs with
#: the 65 deferred newly-gated claims applied, trump_2026:0023's rationale
#: re-emitted from stored panel output, and the 0023/:0024 coherence conflict
#: annotated. Same rule as before — the gate reads the newest staged artifact,
#: or it stops measuring what would actually be published.
RUNS = {
    "gwbush_2006": "04738dd5",
    "clinton_1998": "393f7d06",
    "obama_2014": "c8008f2a",
    "biden_2022": "c733c283",
    "trump_2026": "83117c1a",
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


def test_beckstrom_0462_publishes_as_a_stable_models_split():
    """trump_2026:0462 "After a four-month deployment, she voluntarily extended
    her service, and her rank was going to be lifted." — ten items in the pack,
    six of them bearing, and it ships as a models-split with NO verdict.

    RESOLVED BY POLICY, 2026-08-10. This was a strict xfail asserting the claim
    should reach a substantive verdict, first pending the adjudication wave and
    then — when the wave's panel call split anyway — pending an escalation
    decision. The decision was made:

        **persistent-split-after-2 → PUBLISH the split.**

    Two independent panels judged this claim on the same evidence and reached
    the same three-way disagreement. That is not a pipeline failure waiting to
    be cleared; it is information about the claim, and "Models split" is a
    label in the published verdict vocabulary precisely so it can be reported.
    A third identical panel call would buy the same answer at the same price.

    So the assertion INVERTS. The claim is expected to publish as a split, and
    this case now guards that outcome as STABLE: if some future change quietly
    forces 0462 into a decided verdict, that is a regression against a ruled
    policy and it should fail here. The decision is owner-revisitable — the
    ruling is about how a durable split ships, not that this claim can never
    be revisited."""
    sid = "trump_2026:0462"
    r = row(sid)
    assert len(_run("trump_2026")["evidence"].get(sid, [])) >= 5

    # It publishes as a split with NO verdict — not decided, and not a
    # gate-forced withholding either (the pack is fine; the panel disagreed).
    assert r.get("status") == "disagreement"
    assert r.get("verdict") is None
    assert r.get("split") is True
    assert gate_code(sid) == ""
    assert not is_decided(sid)

    # The disagreement really is durable: three distinct seat labels, which is
    # what "no plurality" means and what makes a third call pointless.
    labels = {lbl for labels in (r.get("by_role") or {}).values()
              for lbl in labels}
    assert len(labels) >= 3, (
        f"{sid}: seats no longer disagree three ways ({sorted(labels)}) — the "
        f"persistent-split policy was ruled on a genuine deadlock, so this "
        f"needs re-reading rather than the assertion relaxing")

    # And the publish path renders it as a split rather than dropping it.
    from truthbot.verdict import bridge
    assert bridge.row_to_bundle(r).consensus.consensus_verdict == "Models split"


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


#: The ratified reading for obama_2014:0045 (2026-08-10), recorded verbatim so
#: the reason a claim is withheld sits next to the assertion that checks it.
#: Each limb is verified against the artifact below.
JOINING_FORCES_0045_RATIONALE = (
    "GATED is CORRECT, not a D15 false positive: the claim's two strongest "
    "supports are E3 (govinfo DCPD-201400050, the official PDF of this State "
    "of the Union) and E5 (CREC-2014-01-28, the Congressional Record's reprint "
    "of the same text) — both literally reprint the 'nearly 400,000' Joining "
    "Forces sentence under evaluation, which is the definition of an "
    "utterance-derivative record. Strip them and what remains is one "
    "obamalibrary.archives.gov document plus three obamawhitehouse.archives.gov "
    "press-office items, all the speaker's own record."
)


def test_obama_joining_forces_veterans_hiring_is_gated_on_utterance_records():
    """obama_2014:0045 — Joining Forces veterans hiring. The RATIFIED
    conversion (2026-08-10), same discipline as the Beckstrom 0469 conversion.

    This case used to assert the rebuild REPAIRED the claim — it was the
    control saying the rebuild fixed some of the T2.4 gate-forced class. The
    D15 utterance-record rule then flagged two of its pack items, and the
    owner's readout of those flags ruled them CORRECT: E3 and E5 are the
    official PDF of this speech and the Congressional Record's reprint of it.
    A document that reprints the sentence under evaluation cannot corroborate
    it, so withholding is the right answer here and the rule needs no change.

    The claim is therefore one of the 65 deferred newly-gated claims, applied
    2026-08-10, mechanism D15. The limbs are checked against the artifact
    rather than asserted from the prose, because a rationale nobody verifies
    is just a comment."""
    sid = "obama_2014:0045"

    # Limb 1: it is withheld, and withheld BY THE GATE — not by a panel that
    # looked at the evidence and ruled it uncheckable.
    assert verdict(sid) == "UNVERIFIABLE", JOINING_FORCES_0045_RATIONALE
    assert gate_code(sid) == "insufficient-qualifying-evidence"
    assert not is_decided(sid)

    # Limb 2: the withholding SUPERSEDES a decided TRUE, and the artifact says
    # so — the corrections ledger's entry has to be checkable against the run.
    superseded = row(sid).get("superseded") or {}
    assert superseded.get("verdict") == "TRUE", (
        "the applied gating must record what it replaced, or its ledger entry "
        "cannot be verified")

    # Limb 3: the two utterance-derivative supports really are in the pack and
    # really are what the readout says they are. If they ever stop being
    # reprints of this speech, the ratified reading needs revisiting — not
    # this assertion weakening.
    pack = _run("obama_2014")["evidence"].get(sid, [])
    urls = " ".join(str(e.get("source_url") or "") for e in pack)
    assert "DCPD-201400050" in urls, JOINING_FORCES_0045_RATIONALE
    assert "CREC-2014-01-28" in urls, JOINING_FORCES_0045_RATIONALE


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

    That was a blind spot, not a repair, and this test said so out loud rather
    than converting the old xfail into a green "coherent" assertion.

    REPAIRED 2026-08-10 (R-3 + D14 disposition), and the case KEEPS ITS NAME
    because what it proves is unchanged: an empty rationale silences the
    checker. It now proves it COUNTERFACTUALLY — blank 0023's restored
    rationale and the conflict disappears again — instead of by pointing at a
    live defect. The three things it asserts:

      1. 0023 says why again. The rationale was re-emitted from STORED panel
         output (the arbiter seat of the pre-wave run, which reached the same
         MISLEADING verdict), adopted verbatim and attributed. Nothing was
         written for it.
      2. The pair SHIPS with the contradiction annotated, not resolved. The
         D14 disposition for this publish is ANNOTATE: both rows carry a
         ``coherence_note`` naming the other claim. Forcing the two labels to
         agree was explicitly not the ruling.
      3. Blank the rationale again and the checker goes silent again — the
         blind spot is real, and it is the missing text that causes it."""
    run = _run("trump_2026")
    a, b = "trump_2026:0023", "trump_2026:0024"

    # 1. The contradiction is still published — both verdicts unchanged — and
    #    0023 now carries a rationale, adopted rather than authored.
    assert verdict(a) == "MISLEADING" and verdict(b) == "TRUE"
    assert row(a).get("split") is True
    reasoning_a = (row(a).get("reasoning") or "").strip()
    assert reasoning_a, "0023 is blank again — the R-3 re-emit regressed"
    prov = row(a).get("rationale_provenance") or {}
    assert prov.get("mode") == "adopted-verbatim"
    assert prov.get("synthesized") is False
    assert prov.get("adopted_verdict") == "MISLEADING", (
        "a rationale adopted from a seat that voted a DIFFERENT verdict is not "
        "this verdict's reason")

    # …and it is VERBATIM: the exact string the sourced run recorded.
    source = json.loads(
        (RUNS_DIR / f"{prov['adopted_from_run']}.json").read_text("utf-8"))
    sourced = next(r.get("reasoning") or "" for r in source["rows"]
                   if r["sid"] == a)
    assert reasoning_a == sourced.strip()

    # 2. The pair ships ANNOTATED. Both sides carry the note, and because they
    #    do, the unannotated-conflict checker is correctly quiet.
    for sid in (a, b):
        note = (row(sid).get("coherence_note") or "").strip()
        assert note, f"{sid} ships without the D14 coherence annotation"
        assert (b if sid == a else a) in note, (
            "the annotation must name the claim it conflicts with")
    assert va.adjacent_coherence_conflicts(run["claims"], run["rows"]) == []

    # 3. The blind spot is still real: strip the annotation AND the rationale
    #    and the checker sees nothing; strip only the annotation and it sees
    #    the conflict. That difference is the whole finding.
    unannotated = [{k: v for k, v in r.items() if k != "coherence_note"}
                   for r in run["rows"]]
    assert [c for c in va.adjacent_coherence_conflicts(run["claims"], unannotated)
            if c["sids"] == [a, b]], (
        "with the annotation removed the checker STILL finds no conflict — the "
        "pair may genuinely have been reconciled, in which case this case "
        "should be rewritten as a real coherence assertion")
    blanked = [dict(r, reasoning="") if r["sid"] == a else r
               for r in unannotated]
    assert [c for c in va.adjacent_coherence_conflicts(run["claims"], blanked)
            if c["sids"] == [a, b]] == [], (
        "an empty rationale no longer silences the checker — good news, but "
        "this case exists to document that it did")


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


# ── R-3: no published verdict may ship without a rationale ───────────────────

#: The one claim in the corpus that still publishes a verdict with no reason,
#: and cannot be repaired for $0. See the case below.
BLANK_RATIONALE_BLOCKERS = ("biden_2022:0432",)

BLANK_RATIONALE_PENDING = (
    "biden_2022:0432 was tie-routed to MISLEADING in the phase-3 rebuild and "
    "again in the wave, and no run in its lineage ever stored a rationale for "
    "it — the pre-remediation run has it as a split with no verdict at all. "
    "There is no stored panel output to adopt, and inventing one is exactly "
    "what the R-3 ruling forbids. Flips when the claim gets a panel call that "
    "captures seat rationales, or when the owner rules on it")


def _all_rows() -> list[dict]:
    return [r for speech in sorted(RUNS) for r in _run(speech)["rows"]]


def test_the_only_blank_rationale_is_the_known_blocker():
    """The publish-blocking lint, as a REGRESSION guard.

    Any NEW blank-rationale verdict fails here immediately, which is the point:
    the trump_2026:0023 defect survived a full wave because nothing was
    watching this. The one known blocker is named rather than tolerated
    silently, and the case below asserts the corpus-clean end state."""
    found = sorted(v["sid"] for v in va.blank_rationale_violations(_all_rows()))
    assert found == sorted(BLANK_RATIONALE_BLOCKERS), (
        f"blank-rationale set changed: {found}. A NEW entry means some "
        f"resolver started publishing verdicts that cannot say why; a MISSING "
        f"entry means a blocker cleared and this list should shrink.")


@pytest.mark.xfail(strict=True, reason=BLANK_RATIONALE_PENDING)
def test_no_published_verdict_ships_without_a_rationale():
    """PUBLISH-BLOCKING by the R-3 ruling: every published verdict, via every
    resolver path (panel, discriminator, tie-routing, evidence gate), must
    carry non-empty rationale text.

    A verdict with no rationale is a fact-check that cannot say why, and — as
    trump_2026:0023 proved — it silently removes the claim from the adjacent
    coherence checker, which links claims partly through their rationale text.

    0023 is repaired. biden_2022:0432 is not, and cannot be for $0: nothing in
    its lineage ever wrote a rationale for it. This stays a STRICT xfail so the
    day it clears is announced as an XPASS instead of passing unnoticed."""
    violations = va.blank_rationale_violations(_all_rows())
    assert violations == [], "\n".join(v["detail"] for v in violations)


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
    "test_beckstrom_0462_publishes_as_a_stable_models_split",
    "test_inflation_pair_discriminates_the_two_measures",
    "test_dei_claim_is_adverse_not_true",
    "test_biden_gdp_5_7_survives_as_decided_true",
    "test_biden_deficit_half_stays_decided",
    "test_obama_college_opportunity_summit_is_decided",
    "test_obama_joining_forces_veterans_hiring_is_gated_on_utterance_records",
    "test_murder_rate_pair_conflict_is_hidden_by_an_empty_rationale",
    "test_eggs_price_claim_discloses_its_framing",
    # R-3, added 2026-08-10 — publish-blocking by ruling.
    "test_the_only_blank_rationale_is_the_known_blocker",
    "test_no_published_verdict_ships_without_a_rationale",
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
