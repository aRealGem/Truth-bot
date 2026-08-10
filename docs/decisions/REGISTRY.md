# Decision registry — ruled, applied, and where to check them

**Status:** RATIFIED. Everything in this file has been decided and is in force.
**Companion:** `D17-candidates.md` holds the opposite kind of entry — things
deliberately *not* decided, logged with their evidence.

Numbered decisions with their own design documents (D15, D16) keep them; this
file is the index of **rulings that changed what ships** but are too small to
warrant a document each. Each entry states the ruling, what it changed, and
the assertion that will fail if the ruling is ever quietly reversed — because a
decision nobody can check is a preference, not a decision.

---

## R-2a — obama_2014:0045 is correctly WITHHELD (D15 is not misfiring)

**Ruled:** 2026-08-10. **Mechanism:** D15 (utterance-record).

The Joining Forces veterans-hiring claim was the acceptance suite's *control*
for the T2.4 repair: gate-forced Unverifiable before the rebuild, decided TRUE
after, and therefore the case that said the rebuild fixed some of the class.
D15 then flagged two of its pack items, which raised the obvious question of
whether the rule was over-firing on exactly the claim used to prove the
pipeline worked.

**The readout says it is not.** The two flagged items are:

| id | document | what it is |
|---|---|---|
| E3 | govinfo `DCPD-201400050` | the official government PDF of *this* State of the Union |
| E5 | `CREC-2014-01-28` | the Congressional Record's reprint of the same speech text |

Both literally reprint the "nearly 400,000" Joining Forces sentence under
evaluation. A document that reproduces the utterance cannot corroborate it, and
that is the whole content of D15. Strip the two and what remains is one
`obamalibrary.archives.gov` document plus three `obamawhitehouse.archives.gov`
press-office items — the speaker's own record, Political/self.

**Ruling:** gating is CORRECT. **No D15 bug; the rule is not to be changed.**

**Checked by:** `tests/acceptance/test_dc6_acceptance_gate.py::
test_obama_joining_forces_veterans_hiring_is_gated_on_utterance_records`,
which verifies the withholding, the superseded TRUE, and that E3/E5 really are
the documents named above.

---

## R-2b — the 65 deferred newly-gated claims are APPLIED

**Ruled:** 2026-08-10. **Applied by:** `scripts/apply_wave_rulings.py` ($0).

The adjudication wave recorded 65 claims that the ratified rules withhold and
left them un-applied, on the stated grounds that applying them was a separate
decision. It is now made: they are applied. Withholding needs no panel call, so
this cost nothing.

**Composition — measured, and NOT what the review's arithmetic said.** The
review proposed `27 + 50 − overlap = 65`, where 50 is D15's blast-radius
`gate_changed`. Those two numbers are not summable: the blast radius measures
D15 against the artifact's recorded gate *in isolation*, while 27 comes from
the composed gate on the re-scored stance. Their actual overlap is **zero** and
their union is 77, not 65.

Running all four rule configurations through the same consolidator the flip set
uses gives the composition that does add up:

| mechanism | claims |
|---|---:|
| the B1a+B2 re-score alone | 26 |
| D15 (utterance-record), on top of the re-score | 39 |
| D16(α) (statistical-release) | **0 — it never gates** |
| the two rules composing (neither alone suffices) | 0 |
| **total** | **65** |

So: **27 gated by the re-score alone, minus the 1 that D16(α) releases
(`clinton_1998:0006`), plus 39 that D15 adds = 65.** D16(α) is a release rule
and appears here only as a subtraction.

**Recorded in:** `metrics/remediation_v2/deferred_gated_mechanism.json`.
**Ledgered by:** `scripts/dc6_package.py --rulings`, which attributes the
mechanism **per claim** — a ledger that said "the ratified rules" for all 65
would be crediting 26 of them to rules that had nothing to do with it.

---

## R-3 — no published verdict may ship without a rationale

**Ruled:** 2026-08-10. **Publish-blocking.**

The stage-2 CRM-114 discriminator resolves an adverse-severity tie by naming a
label. It writes no prose, and the resolved row shipped with `reasoning` empty.
Two consequences, and the second is the serious one:

1. a published fact-check that cannot say why;
2. `verdict_audit.adjacent_coherence_conflicts` links claims partly through
   their rationale text, so an empty rationale **silently removes a claim from
   a detector that is supposed to be watching it.** The trump_2026:0023/:0024
   contradiction stopped being reported for exactly this reason, and it looked
   like a repair.

**Ruling — structural, no fabrication.** The discriminator ADOPTS the chosen
seat's stored rationale VERBATIM, attributed via `rationale_provenance` rather
than in the prose, so the sentence a reader sees is the sentence a model wrote.
It must never synthesize. Where no seat can supply text, nothing is invented
and the lint blocks the publish.

**Enabling change:** `pca.reduce` now records each seat's rationale
(`agreement.seat_rationales`). Before this the seats' text was discarded at
reduce time, so on a tie there was nothing to adopt even in principle.

**The lint:** `verdict_audit.blank_rationale_violations` — every published
verdict, via every resolver path (panel, discriminator, tie-routing, evidence
gate), must carry non-empty rationale text.

**Known blocker.** The lint found a second claim nobody had named:
**biden_2022:0432**, tie-routed to MISLEADING in the phase-3 rebuild and again
in the wave, with no rationale in ANY run of its lineage. It cannot be repaired
for $0 and is a strict xfail until it gets a panel call or an owner ruling.

**Checked by:** `test_the_only_blank_rationale_is_the_known_blocker` (passing —
a regression guard) and `test_no_published_verdict_ships_without_a_rationale`
(strict xfail — the end state).

---

## D14 disposition for this publish — ANNOTATE, never force agreement

**Ruled:** 2026-08-10, for **this publish only**.

With 0023's rationale restored the coherence checker sees the pair again:
trump_2026:0023 (MISLEADING) sits beside :0024 (TRUE) on the same statistic.
The ruling is that the pair **ships with the conflict annotated** — both rows
carry a `coherence_note` naming the other claim and the shared statistic — and
that the labels are **not** to be forced into agreement. The annotation
discloses; it does not adjudicate.

**Checked by:** `test_murder_rate_pair_conflict_is_hidden_by_an_empty_rationale`
(the case keeps its name: what it proves — that an empty rationale silences the
checker — is unchanged, and is now proved counterfactually) and
`tests/test_apply_wave_rulings.py::test_annotation_discloses_rather_than_adjudicates`.

---

## 0462 — persistent-split-after-2 PUBLISHES as Models-Split

**Ruled:** 2026-08-10. **Owner-revisitable.**

trump_2026:0462 was judged by two independent panels on the same evidence and
produced the same three-way split both times. The escalation policy:

> **A split that persists after two independent panel calls is PUBLISHED as
> Models-Split, with both sides' rationales shown. It is not a failure; it is a
> legitimate outcome in the verdict vocabulary.**

Consumed under the wave methodology, so no further spend is authorized against
it. Owner-revisitable — the ruling governs how a durable split ships, not
whether this claim may ever be revisited.

**Renderer.** It did NOT show both rationales before this ruling; a split
rendered as the bare line "Panel split — no consensus verdict." Fixed:
`bridge.split_rationales` selects one seat per distinct verdict and the
explanation carries each side's text verbatim, attributed by role.

**Standing gap, stated rather than papered over.** 0462's *own* seat rationales
do not exist. Every generation of the claim on disk is a split recorded before
seat-rationale capture, so its published page will show the fallback line until
the claim is judged by a panel that captures them. The capability is in place;
this one claim's data predates it.

**Checked by:** `test_beckstrom_0462_publishes_as_a_stable_models_split` and
`tests/verdict/test_bridge.py::test_published_split_carries_every_sides_rationale_verbatim`.

---

## R-1 — trump_2026:0031 is c-count, and that is a SHAPE CORRECTION

**Ruled:** 2026-08-10. **Applied and re-adjudicated the same day.**

`trump_2026:0031` — *"And in the last three months of 2025, it was down to 1.7
percent."* — carried `claim_shape=c-eval`, assigned by the Layer-A backfill.
Judged from the sentence alone, as the classifier's own instruction requires,
that is wrong: the sentence contains no superlative, no causal attribution and
no comparison. Every c-eval trigger in the neighbourhood belongs to the
PRECEDING sentence (:0030, *"my administration has driven core inflation down
to the lowest level in more than five years"*). What :0031 states is a bare
quantity measured against a published series — **c-count**.

**This is a shape correction, not outcome-shopping, and it is recorded as one.**
It moves the gate's quota branch: c-count is a MINISTERIAL shape, so
`evidential_role` routes SELF sources to PRIMARY_RECORD and PARTICIPANT sources
to CORROBORANT instead of the c-eval × SELF ATTRIBUTION_ONLY (weight 0). It
also makes the claim admissible for a computed exhibit, which c-eval is not.
Both effects are stated in the corrections-ledger entry explicitly.

**trump_2026:0030 stays c-eval** — superlative plus causal attribution,
correctly shaped, no exhibit, publishes as already adjudicated.

### What the re-run produced

One claim, one panel call, on the stored pack with the ratified exhibit
attached. **Verdict TRUE — unchanged.** That is the check that this was a shape
correction and not outcome-shopping: the shape was wrong on the text, and
fixing it changed what the page can SHOW, not what it concludes.

| | before | after |
|---|---|---|
| shape | c-eval | **c-count** |
| quota credit | 4 independent, 0 primary (SELF = attribution-only, weight 0) | 4 independent, **1 primary-record** |
| exhibit | REFUSED (inadmissible on c-eval) | **attached** |
| verdict | TRUE | TRUE |
| rationale | "…confirm the three-month annualized core inflation rate fell to ~1.7%…" | "The Sep→Dec 2025 three-month annualized core CPI **computes to 1.701%**, matching the claimed 1.7 percent." |

**The directional element.** "Down TO 1.7 percent" asserts a level *and* a
direction, and one window's rate cannot establish a direction. Rather than let
"down" ride on the panel's own arithmetic, the exhibit carries a **second
computed row** — same series, same pinned vintage, same annualization formula,
over the immediately preceding three months:

    (Sep/Jun)^4 - 1 = 3.412%   (Jul→Sep 2025)
    (Dec/Sep)^4 - 1 = 1.701%   (Oct→Dec 2025)      -1.71 pp

Same evidence class as the first row, so it belongs in the exhibit rather than
in the rationale. Renderer support: `computed_exhibit._comparison_html`, absent
and rendering "" on every exhibit without a directional element.

**Cost:** $0.0036 against a $0.25 cap and a ~$0.02 estimate. The runner's
in-run reading said $0.0000 — the proxy key's spend counter is written
asynchronously and had not caught up. Measured from the ledger
(17.964600 → 17.968165) and recorded in
`metrics/remediation_v2/r1_reshape_rerun_report.json`; the runner now settles
before re-reading.

**Recorded in:** `metrics/remediation_v2/shape_correction_trump_2026_0031.json`
(the correction and its justification) and `r1_corrections_entries.json` (the
ledger entry, which is emitted **even though the verdict did not move** —
because the shape did, and the shape moves the gate).

**Checked by:** `test_0031_is_c_count_and_carries_the_computed_exhibit`.

**Deferred:** exhibits as non-dispositive context on c-eval claims — logged as
**D17-e**, R-2 amendment territory, not this publish.
