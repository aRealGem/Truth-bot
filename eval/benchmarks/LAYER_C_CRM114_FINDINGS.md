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

## v3 — single SONNET discriminator (POSITIVE — the design works)

The diagnosis pointed at *tier*, not structure. v3 keeps the cheap 3-seat panel at stage 1
but runs a **single sonnet seat** as the stage-2 discriminator (economical: sonnet fires
only on the ~7-claim adverse bucket, not all 21). Measured **within one run** (same stage-1
labels, so the discriminator's effect is isolated — the rigorous read, immune to the n=21
cross-run swing):

| | decided-acc | FALSE correct | MISLEADING correct |
|---|---|---|---|
| stage 1 (cheap panel) | 0.550 | **0/4** | 3/6 |
| stage 2 (+sonnet CRM-114) | **0.600** | **2/4** | 2/6 |

+1 net hit, **no MISLEADING collapse** (unlike calib-v1's 0/6), no TRUE collateral (the
discriminator never touches TRUE rows). The **first lever that moved FALSE without a
see-saw.**

**The residual is gold uncertainty, not model failure.** Sonnet flipped exactly the *clean*
falses — `trump_2026:0020` ("zero illegal aliens", an absolute refuted) and `biden_2022:0342`
("only industry that can't be sued") → both correct. The two gold-FALSE it left as MISLEADING
(`trump_2026:0056` "ended DEI", `trump_2026:0556` "obliterated Iran's program") are the
genuinely *borderline* cases — `0556` is the very claim first rated MISLEADING before jackie
bumped it FALSE. Its one wrong flip (`0592`, Ukraine-through-NATO) is likewise a debatable
overstatement-vs-contradiction call. So sonnet discriminates *correctly*; the remaining
disagreement is where the FALSE/MISLEADING line is legitimately fuzzy.

## Bottom line

- Cheap-model PCA **cannot** hold the FALSE↔MISLEADING boundary — by prompt (see-saw),
  two-stage structure, or voting (amplifies the shared bias).
- **CRM-114 with a single sonnet stage-2 discriminator is the answer**: cheap panel does the
  bulk, one stronger judge adjudicates the ~7 hard claims. It's the only lever that improved
  decided-accuracy without breaking another category, and its errors sit on genuinely
  ambiguous claims.
- Opt-in via `--crm114` (stage-2 tier defaults to sonnet, `disc_tier`). **Default open-book
  is still unchanged** — adopting CRM-114-sonnet as the default is jackie's call (it changes
  verdict behavior and adds a little sonnet cost on the adverse bucket).
- Caveats: n=21, single-annotator; the +0.05 within-run gain is modest and real but not
  a large sample. Bigger, multi-annotator gold would sharpen it — and would clarify the
  borderline FALSE/MISLEADING rows the discriminator "misses".

