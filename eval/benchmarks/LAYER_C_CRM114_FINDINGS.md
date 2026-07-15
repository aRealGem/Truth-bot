# Layer C — CRM-114 two-stage discriminator (Phase 3, P67.2)

The Phase-3 A/B showed a single 4-way open-book prompt can't hold both the FALSE and
MISLEADING boundaries (a see-saw). **CRM-114** isolates the hard boundary into its own
call: stage 1 is the normal open-book panel; **stage 2 re-decides only the claims the
panel put in the FALSE-or-MISLEADING bucket**, asking one binary question — is the core
assertion CONTRADICTED (FALSE) or merely OVERSTATED (MISLEADING)? The discriminator only
re-labels within {FALSE, MISLEADING}; it never touches a TRUE/abstained claim, and
stage-1 citations are preserved.

Opt-in via `score_layerb_vs_gold.py --crm114`. Default open-book is unchanged.

## v1 — single cheap seat (net-zero)

The first discriminator rode the `single` strategy (one cheap haiku completion). Live vs
the 21-row gold, it flipped only **2** stage-1 labels:

- `trump_2026:0020` MISLEADING→FALSE (gold FALSE ✅)
- `trump_2026:0592` MISLEADING→FALSE (gold MISLEADING ❌)

Net **0.55 → 0.55** — FALSE 0/4 → 1/4, MISLEADING 3/6 → 2/6. The two-stage *structure*
works (isolated to the bucket, no TRUE collateral, clean telemetry) but a single cheap
seat **carries the same MISLEADING bias the full panel showed** — it under-flips,
catching only 1 of 4 true-FALSE claims and making 1 wrong flip.

## v2 — 3-seat panel discriminator (NEGATIVE)

Diagnosis from v1: the boundary needs a vote, not one biased seat. v2 rides a mini `pca`
panel (roster.dev) on the BINARY question. Live vs the 21-row gold, it did **worse** than
v1: the discriminator flipped only **1** label — `trump_2026:0592` MISLEADING→FALSE, which
is **wrong** (gold MISLEADING). It recovered **0 of 4** gold-FALSE claims. decided-acc
0.526; FALSE 0/4 (unchanged from the softening baseline).

**Why the panel is worse than a single seat:** the three cheap seats (mistral / dsv4-flash
/ haiku) share the same MISLEADING lean, so a majority vote just **confirms the biased
majority** — voting amplifies a shared bias instead of correcting it. Stage 2 softens
exactly like stage 1, because it is the same cheap models.

## Synthesis — the boundary is not solvable on roster.dev

Across every Phase-3 lever:

| variant | FALSE correct | MISLEADING correct |
|---|---|---|
| default open-book | 0/4 | 3/6 |
| calib-v1 (aggressive "contradicted = FALSE", 4-way) | **4/4** | 0/6 |
| calib-v2 ("overstatement = MISLEADING", 4-way) | 1/4 | **4/6** |
| CRM-114 single-seat discriminator | 1/4 | — |
| CRM-114 panel discriminator (v2) | 0/4 | — |

The FALSE↔MISLEADING boundary is **not reliably solvable with roster.dev cheap models** —
not by prompt (see-saw), not by two-stage structure (both stages share the bias), not by
voting (amplifies it). The only lever that moved FALSE was *aggressive prompt framing*,
which just relocates the bias rather than calibrating it. The cheap models lack a stable
internal FALSE-vs-MISLEADING threshold.

**Recommendation (decisive next test):** point the CRM-114 discriminator at a **stronger
tier** (the truth-bot key is scoped to sonnet) — the one lever not yet tried, and the one
the diagnosis points at. If sonnet still can't discriminate, the boundary is genuinely
ill-posed at this gold size and the reader-facing coarse Truthy scale should merge
FALSE/MISLEADING rather than pretend to separate them. CRM-114 the *structure* is sound and
opt-in (default unchanged); it's the cheap *seats* that fail, so it's the right harness to
re-test with a stronger discriminator tier.

