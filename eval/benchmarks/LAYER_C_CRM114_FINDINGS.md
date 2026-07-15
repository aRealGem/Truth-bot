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

## v2 — 3-seat panel discriminator (built; live A/B PENDING)

Diagnosis from v1: the boundary needs a vote, not one biased seat. v2 rides a mini `pca`
panel (roster.dev) on the BINARY question — because the choice is binary, the seats
can't hedge to a non-adverse label. Built + offline-tested (909→ suite green).

⚠️ **The v2 live A/B is not yet run** — the shared LiteLLM proxy hit a sustained rate
limit (429) after the day's many scoring runs, and CRM-114's extra discriminator calls
push each run over. The transport backoff (3×, ≤30s) correctly fails fast on a sustained
quota rather than retrying for an hour. Re-run when the quota resets:

```
score_layerb_vs_gold.py --open-book --crm114     # vs plain --open-book baseline
```

Success criterion: FALSE recovers (v1-calib got 4/4) **without** collapsing MISLEADING
(the see-saw failure) — i.e. the panel discriminator beats both the softening baseline
and the single-seat v1. If it doesn't, the boundary likely needs a stronger tier than
roster.dev, and the reader-facing coarse scale may merge FALSE/MISLEADING anyway.
