# Layer A check-worthiness — investigation + fix (2026-07-10 → 07-13)

Started from two false positives Jackie caught in the verdict-gold: a normative proposal
("let Medicare negotiate … like the VA already does") and a truism ("Thomas Jefferson drew
his last breath") were labeled **check-worthy**. The investigation went deeper than the two
claims; this is the record.

## What we learned (in order)

1. **It was a prompt gap, not (mainly) a model gap.** Layer A's A2 classifier runs on
   `claude-haiku`, single-pass, with a rubric that had **no few-shot examples**. It
   over-weighted "is there a fact here?" and under-applied the *dominant speech-act* and
   *importance* tests. → prompt **v2** (merged, PR #24): keeps the speech-act rule, drops the
   "well-known ⇒ truism" overreach, guards that *specific + consequential* facts stay
   check-worthy even when well-known/dramatic, + 5 examples.

2. **Built an adjudicated answer key** (`claim-set/checkworthy_gold.jsonl`, PR #24): 53
   sentences, sonnet+mistral panel + claude adjudication, 36 high-conf / 17 needs_review.
   Previously we were grading against single-pass-haiku labels — i.e. the buggy output itself.

3. **v2 scored on the gold:** haiku check-worthy recall 0.62 → **0.90**, F1 0.74 → **0.81**;
   anchors 2/4 → 3/4 (haiku), **4/4 (sonnet)**. Net win, but haiku-v2 precision dipped (0.73).

4. **Sonnet, same prompt, made the *same* clear misses as haiku** on the hardest cases
   (military-installation claim) — so those regressions were the *prompt*, not the model.
   On the reliable high-conf gold, sonnet ≈ 0.90 F1 vs haiku ≈ 0.84 — a real but small
   (~2-example) gap.

5. **Tiered-by-confidence does NOT work: haiku is *confidently* wrong.** Escalating
   low-confidence haiku → sonnet escalates **0%** at conf<0.7 and only reaches 4/4 anchors at
   conf<0.95, which escalates **74%** (≈ running sonnet on everything). Haiku's confidence
   does not flag its errors. (`classify_escalating` is committed as a utility, but confidence
   is not a usable trigger here.)

6. **THE KEY FINDING — the errors never reach A2 in production.** All 4 anchors and **all 21
   gold check-worthy claims route A1=`pass`.** In steady-state `run_layer_a`, A1-`pass` went
   **straight to the check-worthy queue with no LLM review** — so A2 prompt/model tuning
   couldn't fix the original false positives at all in production; A1's lexical mistakes
   (Jefferson, "should") sailed to the expensive PCA panel unchecked. A1 sends ~54% `pass`,
   ~20% `drop`, ~26% `ambiguous` (only the ambiguous band ever hit A2).

## The fix (option 1, this PR)

`run_layer_a(..., confirm_pass=True)` (new default): A1-`pass` **also** goes through A2, so
A2 (haiku-v2, which labels both original false positives correctly) can **veto** A1's lexical
false positives before anything reaches PCA. Only A1-`drop` skips A2. Rows carry `a1_pass` so
an A2 veto of an A1-pass is visible. `confirm_pass=False` restores the old shortcut.
Cost: A2 (cheap haiku) now runs on ~80% (pass+ambiguous) instead of ~26%; still cheap, and it
is the band that actually gates PCA spend.

## State / next steps

- **Merged (main):** prompt v2, `checkworthy_gold.jsonl`, eval harness (PR #24).
- **This PR (open, not merged):** `confirm_pass` flow fix + `tier` param + `classify_escalating`
  utility + tiered eval + this doc.
- **Next:** (a) wire the dev-lot/production runner to call A2 with `confirm_pass=True` and
  measure the check-worthy set before/after on a real speech; (b) decide if the
  ambiguous band uses haiku-v2 or sonnet (sonnet's edge is real but small — revisit with a
  bigger gold); (c) **expand `checkworthy_gold.jsonl` to ~150** (53 is a proof-of-concept
  answer key, not enough to gate a production run — this is the highest-value next step).
