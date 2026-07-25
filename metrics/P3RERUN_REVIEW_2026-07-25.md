# Phase 3 Rerun — Review Package (2026-07-25)

**Status: COMPLETE, awaiting jackie's re-publish decision. Nothing deployed — site output is in scratch (`/tmp/p3rerun/site-*`), site-pca untouched.**

## Run configuration (both speeches)
- Roster **prod**: opus-worker proposer (subscription L-W, $0) / grok-4.3 critic / gpt-5.5 arbiter; CRM-114 stage-2 on.
- Evidence **shared_pack_v2**: R1 (opus native search, $0) + R2 (gpt-5-mini browsing) + R3 (grok-4.3 search) → deterministic consolidator, fact-checkers excluded, fair-game era gates, T2.4 quality gate (one targeted retry → forced UNVERIFIABLE).
- Chunked + journaled + budget-capped throughout; journals: `metrics/journals/{trump_2026,biden_2022}_p3rerun.jsonl`.
- Artifacts: trump `23939712` (183 rows), biden `7208bbbb` (111 rows). evidence_mode + roster recorded in both.

## Headline: verdict distributions shift toward True
| | published (dev roster, v1 evidence) | new (prod roster, v2 evidence) |
|---|---|---|
| trump_2026 | 178 claims; adverse (F+M) 102/150 decided ≈ 68% → "Mostly False" | 183 claims; adverse 74/168 decided ≈ **44%** |
| biden_2022 | 111 claims; True 76/99 decided ≈ 77% → "Largely True" | 111 claims; True 96/101 decided ≈ **95%** |

Per-claim diff (matched 225 of 289 published; `metrics/p3rerun_verdict_diff.json`): 127 unchanged, **98 changed**. Dominant transitions: Misleading→True 32, Unverifiable→True 16, split→True 10, Misleading→False 8. 69 claims are new (Layer A re-classification), 64 published claims didn't recur (mix of Layer A drift and text-normalization mismatches).

**The site characterizations will change materially — trump flips from "Mostly False" territory to roughly balanced.** This is the single biggest editorial consequence of the roster + evidence change.

## Gold benchmark says the shift is toward CORRECT (same-day, same gold, both runs)
Scored offline against `verdict_gold.train.jsonl` (35 gold claims, 31 matchable in both runs; heldout stays SEALED per I6 — say the word if you want it burned for the final number):

| | decided-accuracy | coverage | hits/misses | abstain-gap |
|---|---|---|---|---|
| published run | 0.643 | 0.903 | 18/10 | 2 |
| **new run** | **0.700** | **0.968** | 21/9 | 0 |

The new panel decides more claims AND is right more often. Miss profile (9): seven are FALSE↔MISLEADING severity calls within the adverse band; only two soften a gold-MISLEADING to TRUE (trump 0400 Sage anecdote, trump 0620 cartel kingpin) and one commits on a gold-UNVERIFIABLE (biden 0045 "Putin isolated"). n=31 is small — each claim ≈ 3 points.

## Cost reconciliation (ledgers, not log lines)
| Channel | Amount | Source |
|---|---|---|
| Proxy (panels + Layer A + CRM-114), rerun window | **$7.19** | /spend/logs sum (true ledger) |
| R2 gpt-5-mini browsing (341 calls, 24.6M in / 2.3M out) | **$10.80** | run-log token counts × list price |
| R3 grok-4.3 search (335 calls, 7.4M in / 0.55M out) | **$10.60** | run-log token counts × list price |
| R1 + opus proposer (subscription) | $0.00 | — |
| **Rerun total** | **≈ $28.6** | **inside the $30 hard cap, barely** |

Honest notes: (1) my in-flight estimate was $18–19 — low, because gpt-5-mini burned gpt-5.5-scale input tokens (~72k/call; price per token, not tokens, is where mini saves) and the T2.4 retry added ~16% extra retrieval calls; (2) ~$1 of the proxy figure is duplicate Layer A passes from the crash/resume cycles; (3) off-proxy figures are token×list-price — please eyeball the OpenAI/xAI consoles to confirm; xAI may add server-side search fees. Validation spend earlier in the week (pilots, metered leg, probes) was separate, ≈ $9.
**For the Nixon+ program this rerun re-prices a speech at ~$9.7 with grok-always-on; the shipped grok-fallback mode cuts that to ~$6 — the $5.30–5.80 projection needs the caveat that mini's input burn runs hot.**

## Ops notes from the run
- One transient upstream 500 and one REAL proxy bug found: litellm's budget enforcer double-counts every request (enforcement = exactly 2× the true ledger). Worked around (2× ceiling), documented in ops memory; fix properly at next litellm upgrade.
- Small UNVERIFIABLE cluster in trump chunks 15–17 (thin-evidence gates, provenance-coded) — visible in the diff for review.
- The 17 published corrections are inside the matched set; their new verdicts appear in the diff file.

## Decisions for jackie
1. **Re-publish gate:** approve regenerating site-pca from the new artifacts (his deploy + githack re-point follows per plan)? The trump characterization change is the headline call.
2. **Heldout (I6, read-once):** burn it now for the definitive benchmark, or hold?
3. T3.3 tagline ("primary sources") can proceed after era_lint --strict green + zero fact-check pack items — verified during re-render if (1) approved.
