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

Current state (after the R-3 escape run, 2026-08-10):

===============================================  ========  ==================
case                                             status    resolved by
===============================================  ========  ==================
Beckstrom 0469 (purposive clause)                passing   ratified conversion
Beckstrom 0462 (was models-split)                passing   R-3 escape run — a
                                                           THIRD panel broke
                                                           the split 2-1
inflation pair 0030 + 0031                       passing   —
DEI claim 0056                                   passing   —
Biden 5.7% GDP 0115                              passing   —
Biden deficit-half 0244                          passing   adjudication wave
Obama College Opportunity Summit 0046            passing   —
Obama Joining Forces 0045                        passing   ratified conversion
                                                           — now asserts GATED
murder-rate pair 0023 + 0024 (COHERENCE)         passing   R-3 + D14 ANNOTATE
eggs framing disclosure 0219                     passing   —
no blank rationales (corpus)                     passing   R-3 escape run —
                                                           0432 got the panel
                                                           call it needed
biden 0432 says why                              passing   R-3 escape run
===============================================  ========  ==================

THE R-3 ESCAPE RUN (2026-08-10, later than the rulings above)
--------------------------------------------------------------
Two claims published with no rationale stored in ANY generation of their
lineage, so no deterministic re-gate could repair them and inventing text is
the one thing R-3 forbids. Both got a fresh panel call on their stored packs
through the audited ``wave_adjudicate.py --extra-sids`` escape (reason
recorded in ``metrics/remediation_v2/r3_report.json``, own tag, wave set
untouched), for $0.0602 against a $0.25 cap. Both MOVED:

  * **biden_2022:0432 — MISLEADING → Models split.** The old label came from
    the stage-2 discriminator tie-routing a three-way split, which stored a
    label and no text. The new panel split three ways again (True / False /
    Unverifiable) and this time every seat's reason is stored, so it publishes
    as a split that shows all three readings.
  * **trump_2026:0462 — Models split → UNVERIFIABLE.** This CONTRADICTS the
    premise of the persistent-split ruling, which reasoned that a third
    identical call would buy the same answer. It did not: the proposer moved
    from True to Unverifiable, leaving a 2-1 plurality. The policy is not
    overturned — its precondition stopped holding on this claim — but the
    change is owner-revisitable and is flagged in the case itself.

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
    ratification. It shipped as a models-split with no verdict at all, which no
    deterministic re-gate could settle — until the R-3 escape run's third panel
    call broke the split (see above).

XFAIL INVENTORY — the one remaining runtime xfail in the suite
---------------------------------------------------------------
Kept here so the whole set is legible from one place; the same table lives in
``metrics/remediation_v2/t2_xfail_inventory.md``.

1. ``tests/test_bluesky.py::…::test_post_report_returns_none_when_unconfigured``
   — not a claim at all, and not tied to any decision. Asserts
   ``post_report`` returns ``None`` when the publisher is unconfigured; today it
   raises ``NotImplementedError``. Tied to: **the Bluesky v2 publisher landing
   in Phase 7.**

Retired by the wave: ``test_biden_deficit_half_stays_decided`` (now a plain
assertion — the panel decided it TRUE) and ``test_murder_rate_pair_is_coherent``
(renamed, NOT resolved — see the case itself).
Retired by the 2026-08-10 rulings: ``…0462_split_survives_its_panel_call``
(the escalation policy made the split the expected outcome, so the case asserts
it directly).
Retired by the R-3 escape run: ``test_no_published_verdict_ships_without_a_
rationale`` — the last blank-rationale blocker got its panel call, so the
corpus-clean assertion is now plain. The 0462 case was RENAMED again
(``…publishes_as_a_stable_models_split`` → ``…third_panel_broke_the_split_and_
said_why``), which is the same discipline as the earlier renames: a case that
asserts something different gets a name that says so.
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
#: annotated. trump_2026 carries one further generation on top — the R-1 shape
#: correction and single-claim re-run of :0031. Same rule as before: the gate
#: reads the newest staged artifact, or it stops measuring what would actually
#: be published.
#: Later on 2026-08-10, biden_2022 and trump_2026 each gained one further
#: generation: the R-3 ESCAPE RUN. Two claims — biden_2022:0432 and
#: trump_2026:0462 — published with no stored rationale anywhere in their
#: lineage, which the R-3 ruling makes publish-blocking and which no
#: deterministic re-gate can repair, so they got a panel call through
#: ``wave_adjudicate.py --extra-sids`` (audited: reason recorded, own tag, wave
#: set untouched). Both artifacts are ``rebuild_of`` the runs above, which are
#: still on disk.
RUNS = {
    "gwbush_2006": "04738dd5",
    "clinton_1998": "393f7d06",
    "obama_2014": "c8008f2a",
    "biden_2022": "f570d45c",
    "trump_2026": "2d90a74b",
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


def test_beckstrom_0462_third_panel_broke_the_split_and_said_why():
    """trump_2026:0462 "After a four-month deployment, she voluntarily extended
    her service, and her rank was going to be lifted." — ten items in the pack,
    six of them bearing.

    THE HISTORY, because this case has now been three different assertions:

      1. xfail — "should reach a substantive verdict", pending the wave;
      2. the wave's panel call split three ways again, so the ruling became
         **persistent-split-after-2 → PUBLISH the split**, and the case
         asserted the split as stable;
      3. the R-3 escape run (2026-08-10) made a THIRD call — not to re-litigate
         the split, but because the published split carried no seat rationales
         and rendered the bare line "Panel split — no consensus verdict",
         which the R-3 ruling forbids. The seats came back **2-1
         Unverifiable**, and the split was gone.

    **This contradicts the ruling's premise and is flagged as such.** The
    escalation ruling reasoned that "a third identical panel call would buy the
    same answer at the same price". It did not: the proposer moved from True to
    Unverifiable, leaving a plurality. So the persistent-split POLICY is not
    overturned — its precondition (a durable three-way deadlock) stopped
    holding on this claim, which is exactly the kind of fact the ruling said
    was owner-revisitable.

    What this case now guards is the thing the R-3 ruling actually cares
    about: whichever way 0462 lands, the page must be able to say WHY. It
    asserts the verdict came from the seats (a real plurality, not a forced
    label), that it carries rationale text, and that the seats' own reasons are
    stored so a future re-split could still be published with both sides
    shown."""
    sid = "trump_2026:0462"
    r = row(sid)
    assert len(_run("trump_2026")["evidence"].get(sid, [])) >= 5

    # Decided by the PANEL, not by the gate and not by a tie-router.
    assert r.get("status") == "resolved"
    assert r.get("verdict") == "UNVERIFIABLE"
    assert gate_code(sid) == ""
    assert not (r.get("crm114") or {}), (
        f"{sid}: a stage-2 discriminator label would mean the split was "
        f"routed, not resolved — the plurality is what makes this publishable")

    # A genuine plurality: the seats moved, they were not overruled.
    votes = r.get("votes") or {}
    assert votes.get("UNVERIFIABLE", 0) >= 2 and len(votes) < 3, (
        f"{sid}: votes {votes} are back to a three-way deadlock — the "
        f"persistent-split ruling applies again and this case needs the "
        f"owner, not a weaker assertion")

    # R-3: it can say why, in its own text and in the seats' text.
    assert (r.get("reasoning") or "").strip()
    seats = [s for s in (r.get("seat_rationales") or [])
             if (s.get("reasoning") or "").strip()]
    assert len(seats) >= 2, (
        f"{sid}: {len(seats)} seat rationale(s) stored — the escape run "
        f"existed to capture these, and without them a re-split would render "
        f"the bare no-consensus line again")

    from truthbot.verdict import bridge
    bundle = bridge.row_to_bundle(r)
    assert bundle.consensus.consensus_verdict == "Unverifiable"
    assert bundle.consensus.explanation.strip()
    assert len(bridge.split_rationales(r)) >= 2


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


def test_0031_is_c_count_and_carries_the_computed_exhibit():
    """trump_2026:0031 — the R-1 SHAPE CORRECTION (2026-08-10), asserted as a
    shape correction and not as an outcome.

    The Layer-A backfill shaped this claim c-eval. Read on its own text — the
    basis the classifier is instructed to use — it has no superlative, no
    causal attribution and no comparison; those all belong to :0030 next door.
    It states a bare quantity against a published series, which is c-count.

    The correction MOVES THINGS, and the point of asserting it here is that the
    movement is visible rather than silent:

      * the quota branch — c-count is ministerial, so a SELF source becomes a
        PRIMARY_RECORD instead of ATTRIBUTION_ONLY at weight 0;
      * admissibility — c-eval is the one shape a computed exhibit may never
        attach to, so under the old shape the ratified exhibit was refused.

    The verdict did NOT move: TRUE before, TRUE after. That is the check that
    this was a shape correction rather than outcome-shopping — the shape was
    wrong on the text, and fixing it changed what the page can SHOW, not what
    it concludes. :0030 stays c-eval and gets no exhibit."""
    from truthbot.publish import computed_exhibit as ce

    claim31 = claim("trump_2026:0031")
    layer_a = claim31.get("layer_a") or {}
    assert layer_a.get("claim_shape") == "c-count"
    assert layer_a.get("claim_shape_corrected_from") == "c-eval"

    exhibit = row("trump_2026:0031").get("computed_exhibit") or {}
    assert ce.is_admissible(exhibit, claim_shape="c-count")
    assert not ce.is_admissible(exhibit, claim_shape="c-eval"), (
        "the exhibit must remain INADMISSIBLE on c-eval — that refusal is the "
        "reason the shape had to be correct before it could be attached")
    assert exhibit["series"] == "CPILFESL"
    assert exhibit["vintage_date"] == "2026-02-24"
    assert round(float(exhibit["result"]) * 100, 3) == 1.701

    # The DIRECTIONAL row: "down to" is a claim about direction, and one
    # window's rate cannot establish direction. Same series, same vintage,
    # same formula, prior window — so "down" rests on arithmetic over a
    # published series rather than on the panel's own recall.
    comp = exhibit.get("comparison") or {}
    assert comp, "the directional element has no second computed row"
    assert round(float(comp["result"]) * 100, 3) == 3.412
    assert float(comp["delta_pp"]) < 0
    assert set(comp["inputs"]) == {"2025-06-01", "2025-09-01"}

    # :0030 is correctly shaped and deliberately untouched.
    assert (claim("trump_2026:0030").get("layer_a") or {}).get(
        "claim_shape") in ("c-eval", None)
    assert not (row("trump_2026:0030").get("computed_exhibit") or {})


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

#: Claims that still publish a verdict with no reason. EMPTY as of the R-3
#: escape run (2026-08-10) — biden_2022:0432, the last one, was repaired by the
#: only honest means available: a fresh panel call. Kept as a named (empty)
#: list rather than deleted, so a regression has a place to show up.
BLANK_RATIONALE_BLOCKERS: tuple[str, ...] = ()


def _all_rows() -> list[dict]:
    return [r for speech in sorted(RUNS) for r in _run(speech)["rows"]]


def test_the_only_blank_rationale_is_the_known_blocker():
    """The publish-blocking lint, as a REGRESSION guard.

    Any NEW blank-rationale verdict fails here immediately, which is the point:
    the trump_2026:0023 defect survived a full wave because nothing was
    watching this. The known-blocker list is now EMPTY, so this and the case
    below assert the same corpus-clean state from two directions — the list is
    kept because a blocker that reappears should fail against a NAMED
    inventory, not just against zero."""
    found = sorted(v["sid"] for v in va.blank_rationale_violations(_all_rows()))
    assert found == sorted(BLANK_RATIONALE_BLOCKERS), (
        f"blank-rationale set changed: {found}. A NEW entry means some "
        f"resolver started publishing verdicts that cannot say why; a MISSING "
        f"entry means a blocker cleared and this list should shrink.")


def test_no_published_verdict_ships_without_a_rationale():
    """PUBLISH-BLOCKING by the R-3 ruling: every published verdict, via every
    resolver path (panel, discriminator, tie-routing, evidence gate), must
    carry non-empty rationale text.

    A verdict with no rationale is a fact-check that cannot say why, and — as
    trump_2026:0023 proved — it silently removes the claim from the adjacent
    coherence checker, which links claims partly through their rationale text.

    CLEARED 2026-08-10 (was a strict xfail). 0023 was repaired from stored
    panel output; biden_2022:0432 had no stored output anywhere in its lineage
    to adopt, and synthesizing text is the one thing R-3 forbids, so the only
    honest repair was a new panel call — which the R-3 escape run made. The
    corpus is clean, and this is now a plain assertion: the next blank
    rationale fails the publish gate outright."""
    violations = va.blank_rationale_violations(_all_rows())
    assert violations == [], "\n".join(v["detail"] for v in violations)


def test_0432_says_why_after_the_escape_run():
    """biden_2022:0432 "But cancer from prolonged exposure to burn pits ravaged
    Heath's lungs and body." — the claim that made no-blank-rationales a
    publish blocker, and the reason the audited ``--extra-sids`` escape exists.

    It shipped MISLEADING with an empty rationale: the phase-3 panel split
    three ways (Misleading / False / Unverifiable) and the stage-2
    discriminator tie-routed it to Misleading, which stored a LABEL and no
    text. Nothing in its lineage ever held a reason, so there was nothing to
    adopt.

    The fresh panel split three ways again (True / False / Unverifiable) — but
    this time every seat's rationale is stored, so the claim publishes as a
    models-split that SHOWS the three readings instead of asserting Misleading
    with nothing behind it. The verdict moved from a published label to no
    label, which is a withdrawal and is recorded in the corrections facts."""
    sid = "biden_2022:0432"
    r = row(sid)

    # No verdict is published, so the R-3 lint's scope does not include it —
    # what makes it publishable is the seats' own text.
    assert r.get("status") == "disagreement"
    assert r.get("verdict") is None
    assert r.get("split") is True
    assert gate_code(sid) == ""

    seats = [s for s in (r.get("seat_rationales") or [])
             if (s.get("reasoning") or "").strip()]
    assert len(seats) == 3, (
        f"{sid}: {len(seats)} seat rationale(s) with text — the split page "
        f"shows one reason per distinct verdict, and this claim's whole "
        f"defect was having none")

    from truthbot.verdict import bridge
    bundle = bridge.row_to_bundle(r)
    assert bundle.consensus.consensus_verdict == "Models split"
    sides = bridge.split_rationales(r)
    assert len(sides) == 3
    for side in sides:
        assert side["reasoning"].strip() in bundle.consensus.explanation


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
    "test_beckstrom_0462_third_panel_broke_the_split_and_said_why",
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
    "test_0432_says_why_after_the_escape_run",
    # R-1, added 2026-08-10 — the shape correction, asserted as one.
    "test_0031_is_c_count_and_carries_the_computed_exhibit",
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
