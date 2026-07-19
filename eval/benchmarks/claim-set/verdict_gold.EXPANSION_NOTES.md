# Verdict-gold expansion — 2026-07-14 (Phase 3 gate #1)

Expands the canonical verdict-gold **17 → 21 rows** to tighten the Layer B / Layer C
accuracy estimate and close the "FALSE is Trump-only" gap. Verdicts here were
**adjudicated by jackie** (2026-07-14); rows she signed off are `needs_review: false`.
Every decidable row carries ≥1 authoritative source per the schema.

## Rows added / changed (this expansion)

| sid | verdict | notes |
|---|---|---|
| `biden_2022:0030` | TRUE | Putin invasion premeditated + unprovoked — Nov–Dec 2021 buildup + accurate US pre-invasion intel. |
| `biden_2022:0342` | **FALSE** | **Gun-liability line — the new Biden FALSE (see fixture rev below).** PolitiFact rates it False; gun makers CAN be sued (PLCAA has 6 exceptions) and are not the "only" protected industry (e.g. vaccine makers, NCVIA). |
| `trump_2026:0556` | **FALSE** | "Obliterated Iran's nuclear program" — DIA/NBC: one of three sites destroyed, program set back months. jackie's call (I had MISLEADING, FALSE-leaning). |
| `trump_2026:0592` | MISLEADING | Ukraine aid "through NATO, pay us in full" — PURL is real but "everything/in full" overstates; prior US direct aid unrepaid. |
| ~~`trump_2026:0600`~~ | — | **Dropped.** Context-poor fragment ("$1,775 … wanted my approval") — too under-specified to be a reliable gold item, even as UNVERIFIABLE. |
| ~~`trump_2026:0100`~~ | — | **Dropped.** "Luzon wound" is verifiable *in principle* but only with the honoree's identity (not in the claim) — under-specified for a standalone item. |

Final distribution: **TRUE 10 · MISLEADING 6 · FALSE 4 · UNVERIFIABLE 1** (n=21).
By speaker: `biden` TRUE 9 · MISLEADING 2 · FALSE 1 · `trump` TRUE 1 · MISLEADING 4 · FALSE 3 · UNVERIFIABLE 1.

## Fixture rev (jackie-approved option a) — the Biden FALSE

The "FALSE is Trump-only" gap was **fixture-limited, not annotation-limited**: none of the
14 check-worthy `biden_2022` claims in the frozen fixture is cleanly FALSE. The obvious
one — the gun-liability line — existed in `_sentences.jsonl` as **`biden_2022:0342`** but
was never sampled into `claim_set.jsonl` (train *or* heldout).

Per jackie's decision, `biden_2022:0342` was **manually injected into `claim_set.train.jsonl`**
(label `check-worthy`; verbatim text/context from `_sentences.jsonl`) and given a sourced
FALSE verdict-gold row. It is I6-clean (never in heldout) and **pinned to TRAIN**.

⚠️ **Reproducibility:** the injection is a post-build patch — `build_claim_set.py` does
*not* regenerate 0342 (it's not in `_candidates.jsonl`). A rebuild-from-scratch must
re-inject it (a NOTE in `build_claim_set.py` records this).

## Remaining gap

FALSE is now cross-speaker (biden 1 · trump 3) — the primary confound is fixed. But
**UNVERIFIABLE is still n=1 and Trump-only**: the two UNVERIFIABLE candidates in the first
draft (0600, 0100) were dropped as under-specified. A clean UNVERIFIABLE row needs a claim
that is genuinely unadjudicable *even with evidence* (not merely context-poor) — still open.

## Provenance

`annotator` records `claude-expand;jackie-adjudicated` (or `;jackie-directed` for 0342).
TRAIN-only (I6-safe); heldout untouched. No proxy spend (web research only).

---

# Expansion 3 — P67 Phase 2 boundary rows (2026-07-19, PR pending jackie review)

+14 rows, 21 → 35: **TRUE 11 / MISLEADING 11 / FALSE 11 / UNVERIFIABLE 2**. Purpose:
make the FALSE→MISLEADING severity-softening fix measurable — at n=4 gold-FALSE each
row was a 25% swing; at n=11 it's ~9%. Rows were deliberately selected on the
FALSE/MISLEADING boundary (absolute quantifiers, superlatives, exaggerated numbers)
per the Phase-1 diagnosis (`eval/benchmarks/SOFTENING_DIAGNOSIS.md`): the panel's
systematic miss is absolute claims wrapping a real underlying event.

- New FALSE (7): trump 0616 (drugs "virtually stopped completely" by sea), 0172
  ("biggest turnaround in history" — BEA shows deceleration), 0356 (shutdown "2
  points on GDP" — BEA says 1), 0252 (300–600% price differences — mathematically
  impossible as savings), 0312 (state fraud "even worse" than MN — off by orders of
  magnitude), 0568 (Iran missiles "soon reach the US" — DIA says ≥2035 if pursued),
  0452 (killer "came in through open borders" — born in Charlotte).
- New MISLEADING (5): trump 0024 (125-year murder-low superlative), 0248 (MFN
  "lowest price anywhere"), 0564 (Iran "32,000 protesters killed" as floor), 0400
  (Sage story agency inversion), 0620 (El Mencho credit-attribution).
- biden_2022:0385 TRUE (80 bipartisan bills — verified against the PL 117 list) and
  biden_2022:0045 UNVERIFIABLE (rhetorical superlative all checkers skipped —
  closes part of the UNVERIFIABLE gap noted above).

All rows `needs_review: true`, annotator `claude-expand-p67` — **verdict judgments
gated on jackie's adjudication** (no-auto-merge-verdicts rule). Several are marked
med confidence with the FALSE-vs-MISLEADING tension recorded in the rationale
(0172, 0356, 0252, 0568 lean FALSE per the schema's absolute-claim rule where
fact-checkers used the "misleading/overstated" register — flag if you disagree).
Claim text copied verbatim from `claim_set.train.jsonl` (exact sid+text join;
TRAIN-only, I6-safe). Sources: every URL fetched during research; spot-checked 200.
