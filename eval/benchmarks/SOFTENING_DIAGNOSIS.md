# FALSE→MISLEADING severity-softening — Phase 1 diagnosis (P67.2, 2026-07-19)

Two $0 diagnostics localize the softening that caps FALSE detection at ≤1/4 in every
config (Track B's roster-independent accuracy ceiling):

* `diagnose_softening.py` — offline; buckets every milder-than-gold outcome by the
  pipeline stage that produced it, across all committed eval artifacts + both
  full-publish runs (joined to gold by claim TEXT, not sid — F6).
* `audit_evidence_packs.py` — rebuilds the evidence packs for the 4 gold-FALSE sids
  exactly as the eval does (live Brave+FactCheck, no LLM) + counter-evidence probes.

Artifacts: `examples/diagnose-softening.json`, `examples/evidence-pack-audit.json`.

## 1a. Where the softening happens

Bucket counts over gold-decidable rows (see the JSON for the full table):

| config                         | UNANIMOUS_SOFT | ARBITER_SOFT | TIE_ABSTAIN | CRM114_SOFT |
|--------------------------------|:--:|:--:|:--:|:--:|
| openbook-crm114 (plain, dev)   | 6 | 0 | 1 | 0 |
| openbook-calib-crm114 (dev)    | 1 | 3 | 1 | 0 |
| frontier-openbook-crm114       | 3 | 0 | 0 | 0 |
| frontier-openbook-calib-crm114 | 3 | 0 | 0 | 0 |
| fullrun-trump_2026 (gold rows) | 3 | 0 | 1 | 0 |
| fullrun-biden_2022 (gold rows) | 2 | 0 | 0 | 0 |

* **UNANIMOUS_SOFT dominates.** The canonical miss is proposer+critic unanimously
  voting MISLEADING ({MISLEADING: 2}, never escalated) — the arbiter never runs, so
  only CRM-114 even sees these rows. Confirms F3.
* **Calib moves seats but the arbiter hedges them back.** Under dev+calib, 0342 and
  0556 escalate as {MISLEADING: 2, FALSE: 1} — the calibrated prompt got one seat to
  FALSE, and the arbiter sided with MISLEADING (2-1 ⇒ winner is provably the
  arbiter's own label, F2). UNANIMOUS_SOFT → ARBITER_SOFT is progress in the right
  direction that the arbiter currently cancels.
* **TIE_ABSTAIN is real but secondary** (trump_2026:0020 in plain + full-run: a
  correct FALSE vote dies in a {M:1, F:1, U:1} tie and the row bypasses CRM-114 —
  the F1 structural hole). The routing fix recovers these.
* **CRM-114 never soften-flips** (CRM114_SOFT = 0 everywhere). Its only bad flip in
  Track B (0052 M→F, gold M) was a HARSHER error. The discriminator's failure mode
  is *declining to fire*, not misfiring milder.

## 1b. The packs are NOT the problem — F4's inference is REFUTED

The audit hypothesis was that packs lack refuting evidence (why else would CRM-114,
explicitly told not to soften, decline to flip?). Live rebuild says otherwise:

| sid | refuting evidence in the cap-6 pack? |
|-----|--------------------------------------|
| trump_2026:0556 "obliterated Iran's nuclear program" | **YES, explicit** — E3 FactCheck.org *"damaged … NOT obliterated … didn't completely destroy"*; E1 NYT fact-check "strategy document far more circumspect" |
| biden_2022:0342 "only industry that can't be sued"   | **YES, explicit** — E1 FactCheck.org *"Biden repeats false claims"*, E2 PolitiFact *"Biden wrong claim about gun manufacturers"*, E3 Snopes PLCAA |
| trump_2026:0056 "we ended DEI in America"            | **YES, weaker in snippet** — E4 Fast Company *"here's why he's wrong"*; strongest snippets (Axios "courts disagree"; "only 19% of companies cut DEI") sit outside the pack |
| trump_2026:0020 "zero illegal aliens admitted"       | Pack is 6 fact-check pages quoting the claim; snippets carry the quote more than the refutation. (Panel already gets this one right in most configs.) |

Counter-evidence probes (systematic `+false`/`+debunked` and gist variants) surface
*more* refutations, but the base packs already contained enough to justify FALSE on
0556 and 0342 — and the seats, arbiter, and CRM-114 all still said MISLEADING **on
packs containing explicit "not X" fact-checks**.

## Root cause: the label rubric fights the gold on ABSOLUTE claims

All four gold-FALSE claims share one shape: a real underlying event wrapped in an
absolute/universal quantifier — "**obliterated**" (real strikes, real damage),
"**ended** DEI" (real federal rollback), "**only** industry" (PLCAA is real),
"**zero** illegal aliens" (real decline). Fact-checkers (and the gold) hold that
evidence contradicting the absolute core ⇒ FALSE. The models treat the real
underlying event as the core and the absolute as overstatement ⇒ MISLEADING — and
`_CALIB_PROCEDURE` rule (3) *explicitly instructs this*: "Overstating or distorting
a true underlying fact is MISLEADING — NOT FALSE." On absolute claims, the calib
prompt is arguing for the wrong label. The models are complying, not failing.

## Decision (per the pre-registered rule)

Refuting evidence **present but judged MISLEADING** ⇒ the lever is **judging, not
retrieval**: no query augmentation or FactCheck slot reservation. Phase 3 =

1. **Absolute-claim rule** in `_CALIB_PROCEDURE` (+ the CRM-114 discriminator
   criteria): when the core assertion is an absolute or universal (zero/none/only/
   ended/eliminated/completely destroyed), evidence of material counterexamples
   CONTRADICTS the core ⇒ FALSE — the underlying trend being real does not soften it.
2. **CRM-114 hardening** on the same rubric (it sees every UNANIMOUS_SOFT row and
   currently declines to flip even with explicit refutations in the pack); consider
   the tier bump only if the rubric fix is insufficient.
3. The two structural fixes regardless: **TIE_ABSTAIN → CRM-114 routing** (F1;
   recovers 0020-shaped rows) and **per-seat `by_role` vote capture** in
   `pca.reduce` (kills 2-1 tally ambiguity for good).

Caveats: n=4 gold-FALSE (Phase 2 expands +10–15 boundary rows before measuring);
live Brave is a reconstruction of the eval-time packs, not a replay.
