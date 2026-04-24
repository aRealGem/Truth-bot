# Truth-bot 2026 SOTU — Consolidated Findings Review

**Report evaluated:** Donald Trump — February 24, 2026 — truth-bot Pipeline v0.2.0
**Run ID:** `8406cdc6-37b1-4d5e-a887-6cca51087488`
**Scope:** 117 claims, 4 models (Anthropic Claude Opus 4.7, OpenAI GPT-4.1 fallback, Gemini 2.5 Pro, xAI Grok 4)
**Cost:** $4.02 total / $0.0344 per claim
**Last verified:** 2026-04-24 11:33 UTC
**Review compiled:** 2026-04-24
**Sources:** pipeline telemetry (`metrics/adapter_calls.jsonl`) + internal pipeline-engineer review + external deep-think model review (Medium ~85% confidence, sample-based on highest-materiality disputes, not a full 117-claim re-score)
**Paired plan:** [.cursor/plans/fix-sotu-run-findings_d3e2088a.plan.md](../../.cursor/plans/fix-sotu-run-findings_d3e2088a.plan.md)

---

## Executive Summary

The Truth-bot architecture — atomic claim decomposition, multi-model consensus, tiered sourcing, visible caveats — is a genuinely strong concept and more transparent than any major fact-checking outlet's methodology. However, this specific 2026 SOTU report is **not publication-grade reliable** on material current-affairs claims. Four claims carry materially wrong final labels. The 18% model consensus rate confirms the pipeline isn't yet producing trustworthy automated verdicts. The dominant failure mode is **temporal grounding** — models evaluating 2025–2026 claims against 2017–2018 data, or treating real recent events as fictional.

**Bottom line:** today, the site's best use is **analyst triage**, not final publication. Read the per-model reasoning and sources; do not trust the single final badge without human review.

---

## Part A — Materially Wrong Final Labels (Priority 1)

Four claims wrong enough to undermine the report's credibility as a standalone fact-check. In each case primary sources directly contradict the published verdict.

| # | Claim Summary | Published | Best-Supported | Why It's Wrong |
|---|---|---|---|---|
| **99** | Marco Rubio received 100% of Senate confirmation votes | False | **Truthy** | Senate roll call shows 99-0. One senator absent, so technically 99% of full chamber, but 100% of participating senators voted yes. Report's own caveat admits this. |
| **109** | MFN drug pricing available via TrumpRx.gov | False | **True** | White House announced TrumpRx.gov on Feb 5, 2026. Site exists and presents MFN pricing. Report conflates "does the program work well?" with "does the program exist?" |
| **107** | Venezuela raid involved Russian/Chinese military tech | False | **Truthy** or **Falsey** | Reuters reported Operation Absolute Resolve including China- and Russia-supplied Venezuelan military systems. Report treated the raid as fictional — catastrophic temporal grounding failure. "Thousands of soldiers" is the only genuinely disputable element. |
| **108** | Helicoide prison closure / hundreds of political prisoners released | False | **Truthy** | Reuters reported Delcy Rodríguez's announcement that Helicoide would be repurposed and hundreds of prisoners were in the release process post-Maduro capture. Same root cause as #107. |

---

## Part B — Too-Harsh Labels (Adjudication Error, Not Core Fact Reversal)

Claims where the pipeline identified real problems but collapsed nuance the models themselves surfaced, producing labels too harsh for the underlying reality.

| # | Claim Summary | Published | Better Label | Rationale |
|---|---|---|---|---|
| **24** | BBB: "no tax on tips" | False | **Truthy** | IRS guidance describes the provision branded as "No Tax on Tips." Legal mechanism is a deduction, not literal elimination. Directionally correct, phrasing oversells. |
| **25** | BBB: "no tax on overtime" | False | **Truthy** | Same pattern as #24 — real provision, sloganized framing. |
| **26** | No tax on Social Security for seniors | False | **Truthy** or **Falsey** | White House framed as "No Tax on Social Security" and said 88% of seniors would pay no tax on benefits. Actual mechanism: temporary $6,000 senior deduction. Tax on benefits not repealed. Most legitimately debatable of the three. |
| **6** | Murder rate lowest in 125 years | False | **Truthy** | CCJ projects ~4.0/100K for 2025, which would be lowest since 1900 per available data. PolitiFact cautious but not "False." Pre-1960 data reliability caveat is real but doesn't justify "False." |
| **7** | Core inflation lowest in 5+ years | False | **Truthy** | BLS November 2025 core CPI at 2.6% YoY = lowest since March 2021 (~4 years 8 months). "More than five years" is slight overstatement, directionally correct. Three models evaluated the wrong presidential term. |
| **12** | Mortgage rates at lowest in 4 years | False | **Truthy** | Freddie Mac reported 5.98% in late Feb 2026, lowest since Sept 2022 (~3.4 years). "Four years" overstated by ~6 months. Temporal confusion in OpenAI and Gemini contaminated consensus. |
| **110** | Trump Accounts tax-free for every child | False | **Falsey** | "False" is roughly defensible because "tax-free" and "every American child" overstate the law. But OpenAI and xAI denied the program existed entirely, which is wrong — IRS, Treasury, and the official site clearly describe Trump Accounts. Right answer is reached partly by accident. |

---

## Part C — Consolidated Findings (15)

Our original 9 findings (internal review) merged with 6 net-new findings from the deep-think external review.

### C1. OpenAI batch web_search behavior on this run — unresolved, pending empirical test
All 117 OpenAI verdicts in this run showed `tool_call_count=0` and produced training-data-only verdicts on current events. Original conclusion was "Batch API silently drops hosted tools" — **this is NOT confirmed.** Three confounding factors make the zero-count ambiguous:

1. The run used `gpt-4.1` (stale pin — see C2), not the current flagship `gpt-5.4`.
2. The payload used the legacy `web_search_preview` tool variant, not the GA `web_search` tool.
3. The multi-claim parser (the only path used on this run) does not count `web_search_call` items at all — see C6. So `tool_call_count=0` may mean "tools stripped" OR "we didn't count them" — the telemetry was blind.

Current OpenAI docs (as of 2026-04) explicitly list `web_search: Supported` for gpt-5.4, and `/v1/responses` is a supported Batch endpoint. Net: the cheap path may actually work once C2 + C6 + tool-variant are fixed. The accuracy fix plan now includes a Phase 2.5 empirical test before committing to the expensive live-mode migration. Root cause of Part A wrong labels on 99, 107, 108, 109 is still OpenAI being blind to post-cutoff reality on this run — just whether that's architectural or correctable stays TBD.

### C2. OpenAI adapter pinned to `gpt-4.1` + provenance drift
Stale workaround in [src/truthbot/verify/adapters/openai.py:35-36](../../src/truthbot/verify/adapters/openai.py) (`"gpt-5.4 returns 500s; gpt-4.1 is stable + fast"`). gpt-5.4 has been GA for months and is already wired into [src/truthbot/metrics/costs.py:56-85](../../src/truthbot/metrics/costs.py) and [eval/gpt_eval.py](../gpt_eval.py). Compounds C1 — we're frozen to ~Oct 2024 knowledge on a system meant to factcheck 2026 claims.

**Provenance drift:** About page says **GPT-5.4**, STATUS.md admits **GPT-4.1 fallback**, GitHub README says integrations are **"stubbed."** The published report does not expose which model actually ran. Trust eroder.

### C3. Gemini rejects post-cutoff search results as fictional
Gemini runs web search (376 tool calls this run) but dismisses real reporting if dates are past its training cutoff. Confirmed in the wild on the Operation Midnight Hammer claim: *"articles are consistently dated in the future (2025, 2026) and appear to be part of a war game scenario or a work of speculative fiction."* Also manifests on claims 107, 108, 109, 110. External review grades Gemini **C** because of this.

### C4. Consensus opacity — tie-break + docs-vs-behavior gap
Two overlapping issues:

- **Intra-family dissent mis-flag:** `_TIE_BREAK_ORDER` in [src/truthbot/verify/engine.py:43-145](../../src/truthbot/verify/engine.py) picks the most conservative label on ties; UI then flags everyone above that label as "dissent." `[Mostly True, Mostly True, True, True]` → consensus = Mostly True, both `True` voters marked as dissent despite directional agreement.
- **Behavior doesn't match docs:** About page says verdicts are produced by majority vote with "Models split" shown on ties. Actual report shows definitive "False" on 2-2 splits (claim 109), definitive "False" when one model failed to return (claims 24, 25), and definitive "False" when all four models returned different ratings (claim 4). The actual tie-breaking logic is opaque to readers.

### C5. `OPENAI_SYNTHESIS_SYSTEM` prompt bloat
~260 lines in [src/truthbot/verify/adapters/base.py:104-297](../../src/truthbot/verify/adapters/base.py), ~130 US-domestic statistical sources. Misdirects models on foreign-policy / program claims ("not on bls.gov → False"). Primary driver of xAI's $1.92 input-token bill (largest line item this run).

### C6. Telemetry parse gaps — `tool_call_count` not counted in the MULTI-CLAIM path
Nuance: the single-claim [`parse_batch_response`](../../src/truthbot/verify/adapters/openai.py) at openai.py:117-120 **does** count `web_search_call` items correctly. The multi-claim [`parse_multi_batch_response`](../../src/truthbot/verify/adapters/openai.py) at openai.py:214-226 — which was the only path exercised on this SOTU run — does NOT. The Anthropic equivalent also does not count `tool_use` blocks. Returns `tool_call_count=0` even when search ran. This gap is what made C1's root cause ambiguous (see C1). Prerequisite for the Phase 2.5 empirical test in the accuracy plan.

### C7. OpenAI URLs without tool calls
186 URLs emitted across OpenAI verdicts with `tool_call_count=0` — by definition model-emitted, not retrieved. Either recalled from training data or hallucinated. No reachability validation anywhere.

### C8. Caveat block has no model attribution
[src/truthbot/publish/site.py:1051-1065](../../src/truthbot/publish/site.py) walks `bundle.model_verdicts`, collects every non-empty `caveats` field, and `" ".join()`s them into a single "Caveat" callout labeled with no speaker.

### C9. Caveat dedup is exact-string only
`if cav and cav not in seen` is literal string equality. Semantically identical caveats from different models double up; opposed caveats (e.g. "Tier 1 confirms the operation" + "appears to be speculative fiction") get stitched into one paragraph that contradicts itself — the exact Midnight Hammer failure.

### C10. Temporal grounding Pattern A — wrong presidential term (NEW)
Distinct from C3. Models — especially Gemini, sometimes OpenAI — evaluate claims against Trump-I (2017–2018) data instead of Trump-II (2025–2026). Root cause: no explicit "today's date / term number / inauguration date" anchor in prompts. The training-data era wins over the claim era.

| Claim # | Model | Error |
|---|---|---|
| 2 | OpenAI + Gemini | Both reference "January 2017" inflation data |
| 7 | Gemini | Evaluates "January 2017 to January 2018" timeframe |
| 13 | Gemini | Discusses CoreLogic mortgage data from 2017 |
| 14 | Gemini | Counts record highs during "2017 calendar year" |
| 17 | Gemini | References "210,000 construction jobs during 2017" |
| 18 | Gemini | Analyzes "January to December 2017" oil production |

External review calls this **the #1 blocker** — more consequential than C3 because Pattern A produces correct-looking but term-wrong reasoning that is harder for readers to spot.

### C11. 6-bin verdict scale creates gradient disputes (NEW)
Current scale (`True / Mostly True / Exaggerated / Misleading / False / Unverifiable`) creates two problems:

1. Models can't consistently distinguish "Mostly True" from "Exaggerated" — both describe directionally-correct-but-overstated claims. This artificially suppresses consensus.
2. "Misleading" is underused (6/117 claims) because it's poorly defined relative to "False."

Proposed **5-bin scale**: `True / Truthy / Unverifiable / Falsey / False`. See Part H for definitions and projected redistribution.

### C12. No standing professional fact-checker calibration (NEW)
Review did an ad-hoc compare against FactCheck.org / CNN / PolitiFact / ABC News — 7 claims aligned, 4 diverged. We have no benchmark harness, no calibration metric, and no standing credibility anchor. See Part G.

### C13. No source-tier or temporal-accuracy weighting (NEW)
Current consensus treats a T1 BLS 2025 citation and a T6 blog from the wrong presidential term equally when counting votes. A model citing stale-term data should weigh less than one citing on-window primary sources.

### C14. "Unverifiable" is not guarded (NEW)
Should mean evidence is (a) non-public, (b) genuinely insufficient, or (c) authoritative sources materially conflict. Currently sometimes used as a soft default when a model didn't look hard enough. No definition enforced in prompt or post-validation.

### C15. 18% consensus rate is a symptom, not a cause (NEW)
Emerges from C10 (temporal confusion → spurious disagreement across timeframes) + C11 (6-bin gradient ambiguity). Worth tracking as a pipeline health metric. Target: consensus rate should climb meaningfully once C10 and C11 are addressed.

---

## Part D — What Works (Preserve)

Do not regress these during fixes:

1. **Atomic claim decomposition** — 117 discrete, evaluable claims from a single speech is impressive granularity.
2. **Per-model reasoning transparency** — Exposing each model's chain of thought makes the report auditable in a way no major fact-checking outlet matches.
3. **Source tiering** (T1=Gov → T6=Other) — Explicitly marking source authority is a real methodological innovation.
4. **Visible caveats** — Every claim card notes data limitations, proxy measures, and freshness risks. The *concept* is sound even though the *display* has the bugs in C8/C9.
5. **Combined evidence/source lists** — 8–10 sources per claim with domain tagging.
6. **Multi-claim live batching (Phase E)** — xAI 24 calls / Gemini 30 calls for 117 claims proves the cost optimization works. Without Phase E this run would have been ~$7, not $4.02.
7. **Telemetry infrastructure** — per-call cost/token logging is functional even though tool-call counting has the gap in C6.

---

## Part E — Unified Priority-Ordered Recommendations

1. **Temporal grounding fix (Tier 0, highest priority)** — inject today's date, term number, and inauguration date into every model prompt. Add a post-hoc validator flagging any reasoning that references dates outside the expected claim window (e.g., 2024–2026 for this speech). Fixes C3 **and** C10. External review's #1 blocker.

2. **Aggregation determinism** — publish the actual tie-break rule. Require mandatory human adjudication under the 5 triggers in Part F. Fixes C4.

3. **5-bin scale migration** — adopt `True / Truthy / Unverifiable / Falsey / False`; require a one-line reason code on every non-True/False verdict (e.g., "Truthy — real policy exists, but claim overstates who benefits"). Fixes C11. Touches `VerdictLabel` enum, consensus engine, site render, tests, and historical report backfill — probably a separate PR track.

4. **OpenAI: live Responses API + gpt-5.4 + URL reachability check** — move verification out of batch so `web_search_preview` actually runs; upgrade model; add HEAD-request URL validation to strip hallucinated links. Fixes C1, C2, C7.

5. **Gemini: current-date preamble + consider down-weighting until fixed** — inject explicit "today's date is X, search results past your cutoff are PRIMARY evidence not fiction" guidance. Consider reducing Gemini's consensus weight until temporal fix ships. Fixes C3, partial C10.

6. **Run manifest surfaced in published report** — document actual model versions used per run; the GPT-5.4 → GPT-4.1 fallback should be visible in the report, not just STATUS.md. Fixes C2 provenance drift.

7. **Family-aware consensus + cross-family-only dissent badge** — collapse `{True, Mostly True}` / `{False, Misleading, Exaggerated}` into families for tie-break and dissent UI. Dissent badge only fires on cross-family disagreement. Fixes C4 intra-family mis-flag.

8. **Per-model caveat attribution + normalized dedup** — replace single-block concat with per-adapter list; dedup by normalized-whitespace prefix match, not exact string. Fixes C8, C9.

9. **Telemetry: count `web_search_call` items** in OpenAI + Anthropic parse functions. Fixes C6 and is a prerequisite for detecting future silent-strip regressions.

10. **Prompt bloat fix** — extract Tier 1 domestic-stats source list; make it category-conditional on claim category (economy/labor/stats); add a foreign-affairs Tier 1 block (DoD, State, CIA.gov, WH.gov, UN, NATO). Fixes C5; expected ~$0.60 xAI cost reduction per SOTU-sized run.

11. **Benchmark harness vs professional fact-checkers** — run a systematic comparison of truth-bot verdicts against FactCheck.org / CNN / PolitiFact / ABC News on the same claims. Gives calibration metric and credibility anchor. Fixes C12.

12. **Weight verdicts by source tier AND temporal accuracy** — a model citing T1 BLS data from 2025 should outweigh one citing T6 blogs from the wrong presidential term. Fixes C13.

13. **Guard "Unverifiable" definition** — enforce via prompt + post-validator: only valid when (a) evidence is non-public, (b) genuinely insufficient, or (c) authoritative sources materially conflict. Fixes C14.

14. **"Truthy McTruthface" adjudication persona** — use as the review/adjudication layer, **not** as the formal verdict label or report title. Section label: `Adjudicated verdict`. Subhead: `Truthy McTruthface reviewed model disagreements, freshness risks, and source conflicts.` Keeps playfulness attached to *process*; verdict stays formal.

15. **Keep confidence separate from the label** — e.g., `Truthy | Confidence: 72% | Reason: real program exists; effect statement overstated`.

16. **Re-label the 11 materially-wrong / too-harsh claims** from this run (Parts A + B) as a one-time data fix after schema migration.

---

## Part F — Mandatory Adjudication Triggers

Human review is required when ANY of the following is true:

1. No majority label exists across the provider panel.
2. The claim concerns events in the last 6–12 months.
3. Models cite conflicting high-quality sources (two or more T1/T2 sources in direct contradiction).
4. Any model returns `Unverifiable`.
5. The claim turns on a specific quantity, date, or named individual/event.

These triggers drive the `Truthy McTruthface` adjudication queue.

---

## Part G — Professional Fact-Checker Cross-Check

### Aligned with professionals

| # | Topic | Truth-Bot | Professionals | Aligned |
|---|---|---|---|---|
| 2 | Inherited "record inflation" | False | False | yes |
| 10 | Gas below $2.30 most states | False | False | yes |
| 15 | Biden < $1T investment | False | False | yes |
| 16 | $18T in investment | False | False | yes |
| 9 | Gas over $6 under Biden | True | True | yes |
| 102 | Worst inflation in history | False | False | yes |
| 37 | Chicken/butter/fruit costs down | False | False (CNN: groceries up 2.1%) | yes |

### Diverged from professionals

| # | Topic | Truth-Bot | Professional Nuance | Gap |
|---|---|---|---|---|
| 3 | Zero illegal aliens admitted | False | Border Patrol releases were zero for 8 months (PolitiFact / WRAL) | Anthropic's "Mostly True" was arguably closer |
| 6 | Murder rate lowest in 125 yrs | False | Plausible per CCJ data; PolitiFact cautious but not "False" | Too harsh |
| 11 | Iowa gas at $1.85 | False | Likely E-85 ethanol pricing (~$1.92), not regular gas (~$2.45) | "Misleading" more precise |
| 24–26 | Tax claim trilogy | All False | Real provisions exist; IRS describes them; sloganized framing oversells | Too harsh |

---

## Part H — Proposed 5-Bin Scale

### Definitions

| Label | Definition | Typical Pattern | User Takeaway |
|---|---|---|---|
| **True** | Core factual claim supported; any imprecision is minor and non-material | Right event, right quantity, right timeframe | "This is substantively correct." |
| **Truthy** | Core gist supported, but wording is sloganized, overstated, too broad, or missing key qualifiers | Real policy/event exists, but effects or scope are oversold | "There's something real here, but the phrasing pushes it too far." |
| **Unverifiable** | Public evidence insufficient, unavailable, non-authoritative, or materially conflicting | Private conversations, unpublished numbers, unresolved current reporting | "We can't responsibly score this either way yet." |
| **Falsey** | Kernel of truth, but central claim is materially wrong, distorted, or misattributed | Real program exists, but claim gets the effect, scale, or outcome wrong | "This leans false despite having a small factual anchor." |
| **False** | Core factual predicate is contradicted or invented | Wrong outcome, fictional event, clear contradiction | "This is substantively wrong." |

### Why this scale works

1. **Collapses the ambiguous gradient.** `Mostly True` + `Exaggerated` merge into `Truthy` — models that agree "directionally correct but overstated" now land in the same bin, boosting consensus.
2. **Isolates deceptive framing.** `Misleading` becomes `Falsey` — claims with a factual kernel weaponized to deceive get their own clear bucket.
3. **Programming double-entendre is a feature.** Every developer knows truthy/falsy evaluation: a value that *behaves* as true in a boolean context without *being* literally `true`. Exactly what a SOTU claim like "murder rate lowest in 125 years" is.
4. **Colbert "truthiness" callback** positions the project in a lineage of accountability humor without undermining rigor.

### Projected 2026 SOTU redistribution

| Old Category | Count | → New Category | Projected Count |
|---|---|---|---|
| True | 42 | **True** | ~42 |
| Mostly True (18) + Exaggerated (13) | 31 | **Truthy** | ~31 |
| Unverifiable | 6 | **Unverifiable** | ~8 (absorbs 2 current "False" that are really unverifiable) |
| Misleading | 6 | **Falsey** | ~10 (absorbs some current "False" with real kernels, e.g. 24–26) |
| False | 32 | **False** | ~26 (sheds claims that are really Truthy, Falsey, or Unverifiable) |

---

## Part I — Gap Analysis vs Existing Fix Plan

What this review surfaces that [.cursor/plans/fix-sotu-run-findings_d3e2088a.plan.md](../../.cursor/plans/fix-sotu-run-findings_d3e2088a.plan.md) does **not** yet cover:

| Gap | Rec # | Notes |
|---|---|---|
| Temporal grounding Pattern A (wrong term) | 1 | Existing plan's Gemini date-preamble (Tier 2b) only addresses Pattern B (C3). Pattern A (C10) needs explicit term/inauguration anchoring + post-hoc validator — applies to ALL adapters, not just Gemini. External review calls this the #1 blocker. |
| 6-bin → 5-bin scale migration | 3 | Existing plan doesn't touch `VerdictLabel` enum or the scale. This is a large cross-cutting change (enum, consensus, site, tests, historical report backfill) — probably a separate PR track. |
| Run manifest in published report | 6 | Existing plan upgrades OpenAI to gpt-5.4 but doesn't expose actual-model-used in the site. |
| Professional fact-checker benchmark harness | 11 | Brand new capability — nothing in the plan today. |
| Source-tier + temporal-accuracy weighting | 12 | Existing plan's Tier 2c family-aware consensus doesn't weight by source tier or temporal correctness. |
| Explicit adjudication trigger matrix | Part F | Existing plan fixes dissent UI but doesn't define the 5 mandatory-human-review triggers. |
| Guard "Unverifiable" definition | 13 | Not in plan today. |
| `Truthy McTruthface` adjudication persona | 14 | Not in plan today. |
| Confidence-separated-from-label | 15 | Existing plan preserves current schema; this is part of schema migration in rec 3. |
| Re-label 11 wrong/harsh claims as data fix | 16 | Post-schema-migration one-time backfill; not in plan. |

Existing plan **does** cover: C1 (OpenAI live), C2 (gpt-5.4 upgrade, partial), C3 (Gemini preamble), C4 (family tie-break, partial), C5 (prompt trim), C6 (tool-call counting), C7 (URL reachability), C8 (caveat attribution), C9 (caveat dedup).

Suggested sequencing for a refreshed plan:

- **Phase 1 (1 PR, days):** existing Tier 1 of current plan — gpt-5.4 upgrade, telemetry fix, caveat attribution. Ship fast.
- **Phase 2 (2–3 PRs, 1 week):** temporal grounding (Tier 0, NEW), OpenAI live mode, Gemini preamble, family-aware consensus with source-tier weighting.
- **Phase 3 (dedicated track, weeks):** 5-bin scale migration + run manifest + "Unverifiable" guard + `Truthy McTruthface` adjudication queue + 11-claim relabel.
- **Phase 4:** benchmark harness vs pro fact-checkers as a standing test.

---

## Uncertainty Drivers (Preserve)

1. Not all 117 claims were externally re-scored — external review is sample-based on highest-materiality disputes.
2. Some labels are inherently judgment-sensitive between `Truthy` / `Falsey` (especially claims 24–26 and 107).
3. Only public artifacts and telemetry were reviewed, not raw per-claim JSON for every model.
4. Professional fact-checker comparisons are based on the subset of claims those outlets chose to cover — not all 117.

**Minimal data that would raise confidence:** raw per-claim JSON outputs for all providers, the consensus/tie-break implementation code (already available but not cross-referenced in review), and a human-adjudicated benchmark on the full 117 claims.
