# Truth-bot Status — 2026-04-28

## Session note — 2026-04-28 (FitnessScorer coarse-axis verdict-agreement metric + Run 5)

Overnight follow-up to the 2026-04-27 5-bucket projection layer (commit
[`dc64ca0`](https://github.com/aRealGem/Truth-bot/commit/dc64ca0)). Plan:
[.cursor/plans/fitnessscorer_coarse-axis_b49ee82d.plan.md](.cursor/plans/fitnessscorer_coarse-axis_b49ee82d.plan.md).
Goal: re-score Run 4's verdict-agreement on the 5-bucket Truthy scale to disentangle
genuine quality drift from fine-axis label noise, since the GPT 5.4 Pro reference set
is already roughly coarse-axis-shaped.

### What shipped

- **`FitnessScorer` axis parameter** ([`eval/evolver/fitness.py`](eval/evolver/fitness.py)):
  `verdict_distance(..., axis="fine")` accepts `"coarse_lenient"` / `"coarse_strict"`
  alongside the historical `"fine"` default. Both projections mirror
  `LENIENT_PROJECTION` / `STRICT_PROJECTION` in
  [`src/truthbot/verify/engine.py`](src/truthbot/verify/engine.py); coarse axis uses a
  5-bucket label order (`true → truthy → unverifiable → falsey → false`) with the same
  `[0, 1]` normalised distance ceiling so verdict-agreement scores are directly
  comparable across axes. `axis="fine"` is byte-identical to the pre-axis-param form
  (round-trip test pins this).
- **`score_run.py` CLI extension** ([`eval/sotu-2026/score_run.py`](eval/sotu-2026/score_run.py)):
  new `--axis {fine,coarse_lenient,coarse_strict}` flag (default `fine`) + `--all-axes`
  flag that prints a 3-column scorecard with a one-line headline like
  `Verdict agreement: 62.8% fine -> 60.3% coarse_lenient (-2.4pp) -> 69.0% coarse_strict (+6.2pp)`.
- **Tests** ([`eval/tests/test_fitness_coarse_axis.py`](eval/tests/test_fitness_coarse_axis.py)):
  25 cases covering identity-on-every-axis, Lenient-lifts MT/Excg, Strict-separates
  Excg-from-MT, symmetric distance, max-distance preserved across axes, "Models split"
  handled as unverifiable on every axis, default-axis round-trip byte-identical, unknown
  axis raises `ValueError`, and `FitnessScorer.score(axis=...)` actually re-scores
  verdict_agreement (not just decorates the dict).
- **Run 5 documented** ([`eval/sotu-2026/BENCHMARK.md`](eval/sotu-2026/BENCHMARK.md)):
  multi-axis re-score of Run 4, full scorecard, label-distribution analysis, and
  architectural takeaway.

### Headline empirical finding (counter to the going-in hypothesis)

The plan predicted Lenient would *lift* verdict-agreement; the data flipped the sign:

| Axis              | Verdict agreement | Δ vs fine | Fitness | Δ vs 0.679 baseline |
|-------------------|------------------:|----------:|--------:|--------------------:|
| fine (6-bucket)   | 62.8%             | —         | 0.5413  | −0.1377             |
| **coarse_lenient**| 60.3%             | **−2.4pp**| 0.5341  | −0.1449             |
| **coarse_strict** | **69.0%**         | **+6.2pp**| **0.5599** | **−0.1191**     |

The data explains why. Reference label distribution: 9 `MISLEADING`-family + 7 `MISLEADING`
plain + 4 `TRUE` + 4 `UNSUPPORTED` + 2 `PARTLY TRUE` + 1 `UNVERIFIABLE`. Consensus on the
same 29 claims: 9 `False` + 6 **`Exaggerated`** + 5 `True` + 4 `Mostly True` + 3
`Unverifiable` + 2 `Models split` + **0 `Misleading`**. The reference has no
`EXAGGERATED` bucket and the consensus has no `Misleading` verdicts — that's the editorial-bar
disagreement.

- Under Lenient, consensus `Exaggerated → Truthy` vs reference `Misleading → Falsey`
  pushes the pair onto opposite sides of the Truthy/Falsey divide (worse than fine's
  partial-credit adjacency).
- Under Strict, both `Exaggerated` and `Misleading` collapse onto `Falsey`, which is
  exactly where the reference's editorial bar sits.

### Architectural takeaway: two defaults for two audiences

| Surface | Default | Why |
|---|---|---|
| Public-facing site (headline pill) | **Lenient** (shipped 2026-04-27) | Reader-facing — Lenient surfaces consensus agreement within the panel and respects the consensus's own editorial choice that "Exaggerated" sits closer to "Mostly True" than to "Misleading." |
| Regression scorecard (`score_run.py`) | **Strict** *(interpretation default; CLI keeps `--axis fine` for byte-identical Run 4 backward compat)* | Reference-facing — the reference set labels rhetorical claims `MISLEADING`. Strict puts both `Exaggerated` and `Misleading` on `Falsey` and isolates real verdict disagreement from label-vocabulary drift. |

The fine-axis score remains useful as a fine-grained drift signal between
`Mostly True` ↔ `Exaggerated` ↔ `Misleading`, which Strict deliberately collapses.

### What this rules out — and what it doesn't

- **Rules out:** "the −18pp Run 4 gap is mostly fine-axis label drift between
  Mostly True and Exaggerated." It isn't. Lenient (which collapses MT + Excg → Truthy)
  makes the gap *worse*, so MT/Excg drift is small.
- **Confirms:** the gap is partially driven by the **Misleading vs Exaggerated** editorial-threshold
  divergence between consensus and the GPT 5.4 Pro reference. Strict closes ≈14% of it
  (`−0.1377 → −0.1191`).
- **Doesn't rule out:** that the remaining ≈86% of the gap (still −0.1191 fitness even
  under Strict) is genuine quality drift between the 2026-04-19 single-model baseline and
  the current 4-adapter consensus pipeline. The Anthropic-only single-claim re-score
  (originally listed under [`BENCHMARK.md`](eval/sotu-2026/BENCHMARK.md) calibration gap
  #3) is the right next experiment to nail that down — but the surprise-magnitude case for
  it weakened once the lens decomposition landed.

### Test health

- 25/25 new coarse-axis tests pass.
- 106/106 existing eval tests pass (no fine-axis regression).
- Default-axis CLI run produces byte-identical Run 4 numbers (Fitness 0.5413,
  verdict_agreement 62.8%) — confirming pure-additive surface.

---

## Session note — 2026-04-27 (5-bucket coarse-axis projection — Lenient default + Strict toggle)

Implemented the 5-bucket "Truthy scale" coarse-axis projection layer first proposed in
Part H of [`eval/sotu-2026/findings-review.md`](eval/sotu-2026/findings-review.md). Plan:
[.cursor/plans/5-bucket_coarse-axis_projection_ba2d0050.plan.md](.cursor/plans/5-bucket_coarse-axis_projection_ba2d0050.plan.md).
Board: Backlog → WIP this session.

### Why now (empirical motivation)

Same-session analysis across the 84 published claims (4 adapters, 7 reports) compared the
status quo against the two candidate projection rubrics:

| Rubric | Strong consensus | Models split | Δ vs status quo |
|---|---|---|---|
| Status quo (6 buckets) | 39% | 23% | — |
| **Lenient projection (Part H default)** | **55%** | 18% | **+16pp strong** |
| Strict projection | 39% | 23% | (no change) |

16 / 84 claims (19%) shift consensus strength under Lenient vs Strict — that's exactly the
editorial-tension cohort the toggle is meant to surface. Lenient delivers the agreement-lift
the projection layer was designed for; Strict preserves the editorial bar where Exaggerated
reads as Falsey, and lets readers flip between them.

### What shipped

- **Engine layer** ([`src/truthbot/verify/engine.py`](src/truthbot/verify/engine.py)):
  module-level `LENIENT_PROJECTION` + `STRICT_PROJECTION` constants, a `_project_consensus`
  helper that re-runs the same plurality/strength logic on the projected panel, and a
  split-projection guardrail (no plurality on the projected axis ⇒ "Models split", never
  tie-broken — prevents the projection from manufacturing false agreement). `_build_consensus`
  now emits four extra fields: `coarse_lenient_label`/`coarse_lenient_strength` and
  `coarse_strict_label`/`coarse_strict_strength`.
- **Data model** ([`src/truthbot/models.py`](src/truthbot/models.py)): `ConsensusVerdict`
  extended with the four optional fields, all defaulted to empty strings / `"none"` so legacy
  cached bundles deserialize cleanly.
- **Site render** ([`src/truthbot/publish/site.py`](src/truthbot/publish/site.py)): headline
  `.claim-pill-headline` defaults to the Lenient label and carries `data-fine-label`,
  `data-coarse-lenient`, `data-coarse-strict` attrs (plus `*-css` slugs) so the JS toggle can
  flip without DOM rebuild. Per-model strip is unchanged (still 6-bucket). Status bar gains an
  Editorial-lens chip; About page documents both lenses + the guardrail. `claims.json` now
  publishes the coarse fields too.
- **Assets** ([`src/truthbot/publish/assets/styles.css`](src/truthbot/publish/assets/styles.css),
  [`src/truthbot/publish/assets/truthbot.js`](src/truthbot/publish/assets/truthbot.js)):
  `--v-truthy` (#84cc16) / `--v-falsey` (#ea580c) CSS vars + matching `.v-truthy` / `.v-falsey`
  pill classes; lens chip styling; client-side toggle persists choice to
  `localStorage["editorial-lens"]` and no-ops on pages with no headline pills.
- **Tests**: [`tests/test_consensus_projection.py`](tests/test_consensus_projection.py) (18
  cases — projection-mapping invariants, both-lens unanimity/strong/weak, the 2-2 and
  all-different guardrail paths, single-adapter, all-Unverifiable, empty panel, and the
  round-trip-no-fine-axis-change guarantee). [`tests/test_site_render_projection.py`](tests/test_site_render_projection.py)
  (5 cases — headline data-attrs, default-Lenient render, per-model strip retains 6-bucket
  classes, legacy-bundle fallback, dissent counter stays on fine axis). All 23 new tests pass;
  existing site-social and verify suites unchanged.

### What's explicitly out of scope

Model-side `SYNTHESIS_SYSTEM` rubric is unchanged — frontier prompts and prompt-cache hits are
preserved. Historical 84 claims keep their 6-bucket headlines until a re-publish happens
(non-blocking; the projection is computed at consensus time so any re-render with current code
emits the new headline). `VerdictLabel` enum stays 6-bucket. FitnessScorer coarse-axis verdict
agreement is a future calibration row, not this layer.

---

## Session note — 2026-04-27 (P0 #3 — reference-set regression scoring, current baseline established)

First execution of the P0 #3 reference-set regression scorecard (board: WIP→Done this session).
Wires `eval/evolver/fitness.py:FitnessScorer` against the published 29-claim run output (run
`258b5758`) so every release now has a single-line accuracy signal instead of relying on
cost-diff + spot-check.

### Current baseline (Run 4 in [`eval/sotu-2026/BENCHMARK.md`](eval/sotu-2026/BENCHMARK.md))

| Metric | Score | Weight | Notes |
|---|---|---|---|
| Claim recall | **100.0%** (29/29) | 0.25 | Matches the 2026-04-19 baseline exactly. |
| Verdict agreement | **62.8%** | 0.30 | -18pp vs the 2026-04-19 single-model baseline (~81%) — see caveat below. |
| Explanation quality | 39.5% | 0.20 | Sidecar adapters only (OpenAI/Gemini/xAI); Anthropic explanations live in claim HTML and aren't in a canonical JSON. |
| Source citation | 16.1% | 0.15 | Same caveat. |
| Parsimony | 0.0% | 0.10 | 262k tokens sidecar-only vs target_max=30k calibrated for single-model standalone — not meaningful here, see calibration gap #2. |
| **Fitness** | **0.5413** | — | vs 0.679 baseline (Δ −0.1377). |

Cost: $4.97. Reproduce: `python eval/sotu-2026/score_run.py --run-id 258b5758-8e25-4bf0-8f34-63778d2f976e --report-id e81546a0-6371-4e96-9e94-3d6213864d5a`.

### Important interpretation caveat (the −18pp is not a clean regression signal)

The 2026-04-19 baseline was **single-model standalone** (claude-opus-4-7 only). Run 4 is the
**4-adapter consensus** pipeline (Anthropic + OpenAI + Gemini + xAI). The verdict-agreement
gap conflates:

1. Any genuine quality drift since 2026-04-19.
2. Consensus-voting label drift — the consensus algorithm can land on a label different from
   the strongest individual model's vote, even when that model matches the reference.
3. The new "Models split" verdict state (2/29 in this run, mapped to `unverifiable` in the
   scorer) that didn't exist in single-model baselines.

Untangling requires capturing an Anthropic-only single-claim re-score against the same 29-claim
transcript and adding it as Run 5. Tracked as P1 follow-up in the calibration-gaps section of
[`eval/sotu-2026/BENCHMARK.md`](eval/sotu-2026/BENCHMARK.md).

### Plumbing shipped

- New: [`eval/sotu-2026/score_run.py`](eval/sotu-2026/score_run.py) — argparse CLI joining
  extractions + sidecar + `claims.json` to feed `FitnessScorer.score()`. Trivial to repeat per
  release: `--run-id`, `--report-id`, optional `--metrics-dir`, `--site-data-dir`,
  `--baseline-fitness` overrides.
- Modified: [`eval/evolver/fitness.py`](eval/evolver/fitness.py) — added `"Models split" →
  "unverifiable"` to `_TRUTHBOT_LABEL_NORMALIZE` so the scorer recognizes the production
  consensus's no-majority state instead of warning + defaulting. All 81 existing eval tests
  still pass (`.venv/bin/python -m pytest eval/tests/`).
- Updated: [`eval/sotu-2026/BENCHMARK.md`](eval/sotu-2026/BENCHMARK.md) — Run 4 added as
  current baseline with full scorecard, plus a "Known calibration gaps" section enumerating
  Anthropic-explanation persistence, parsimony target_max recalibration, and the apples-to-
  oranges baseline mismatch.

### Action items surfaced by the first scorecard

1. **62.8% verdict agreement merits investigation.** Even after accounting for the consensus-vs-
   standalone caveat, this is the single biggest accuracy lever. Check whether the
   underperformance is concentrated in specific verdict categories (e.g., are Exaggerated/
   Misleading consistently mis-mapped?) before attributing it to consensus drift.
2. **Anthropic explanations need a canonical persistence path** so explanation_quality and
   source_citation become honest 4-adapter scores instead of sidecar-only.
3. **Parsimony target needs production-pipeline recalibration** (e.g., 100k–500k window) but
   only after we lock the new baseline so the recalibration doesn't invalidate trend tracking.

These will be added as P1/P2 backlog rows in the next board pass.

---

## Session note — 2026-04-26 (Pre-SOTU Grok cap + multi-claim backfill, 29-claim fire)

### Shipped today

Two pre-SOTU fixes from the
[`pre-sotu-grok-cap-and-backfill_7b176093`](.cursor/plans/pre-sotu-grok-cap-and-backfill_7b176093.plan.md)
plan plus a clean 29-claim SOTU fire on the validated stack.

- **Fix 1 — Grok `max_tool_calls` cap.** `GrokAdapter._call_with_search` now
  passes `max_tool_calls = _max_tool_calls_per_claim() * n` on every
  `client.responses.create`. Default is 8/claim, override via
  `TRUTHBOT_GROK_MAX_TOOL_CALLS`. Defensive `TypeError`/server-rejection
  fallback retries without the kwarg if the xAI SDK or server doesn't
  recognize it (xAI's Responses endpoint is undocumented for this param).
  Tests: 4 new in `tests/test_grok_multi_claim.py` (default cap, env override,
  rejection fallback, single-claim default).
- **Fix 2 — Multi-claim `model_reported_sources` defensive backfill.**
  `build_multi_verdicts` in `src/truthbot/verify/adapters/base.py` now
  populates `model_reported_sources` for **all** chunk indices with
  `tool_retrieved_urls` whenever a multi-claim provider drops attribution
  (`web_sources` either missing or explicit `[]`), and additionally
  populates `web_sources` for the index-0 "call owner" so reports keep at
  least one visible cited source per chunk. Closes the
  attribution-fidelity-vs-visible-grounding trade-off documented in the
  prior session note. Tests: 7 new in `tests/test_multi_batch_base.py` plus
  one each in `tests/test_grok_multi_claim.py` and
  `tests/test_gemini_multi_claim.py`.

Test suite: 437 passed / 1 xfailed (was 432 / 1).

- **Approval-budget rule.** Codified the operator preference for upfront
  approval-prompt counts as a Cursor rule
  ([`.cursor/rules/approval-budget.mdc`](.cursor/rules/approval-budget.mdc),
  `alwaysApply: true`). Future sessions in this repo will lead with
  `Approval budget: N prompts -- <summary>` before non-trivial tasks so
  the operator can decide whether to babysit or step away.

### 10-claim validation rerun (gate before SOTU fire)

Baseline: `run_id=128597ce-6e83-44a3-8811-8811e1fa219e` (yesterday, no caps).
Rerun:    `run_id=146ee42a-97c8-443b-ae4a-8511dda0916d` (today, both fixes).

| Metric | Baseline | Rerun | Delta |
| --- | --- | --- | --- |
| Total cost | $4.15 | **$1.69** | **−59%** |
| xAI cost | $3.19 | **$1.22** | **−62%** |
| xAI tools / claim | unbounded | **4.60 (cap=8)** | capped |
| MRS non-empty (multi-claim) | <50% | **100%** all 3 sidecar providers | fixed |
| Anthropic gold-standard verdicts | 10/10 ok | 10/10 ok | unchanged |

All five pass criteria green — fire authorized.

### SOTU 29-claim fire — 2026-04-26

Run ID: `258b5758-8e25-4bf0-8f34-63778d2f976e`
Report: `site-test/reports/2026-02-24-donald-trump-e81546.html`
Wall clock: ~12 min submit + ~24 min Anthropic batch reconcile.
Command: `truthbot publish --transcript eval/sotu-2026/transcript.txt
--mode batch --triage --max-claims 29` (with `TRUTHBOT_OPENAI_LIVE=1`).
Pipeline routing: 16 claims short-circuited via cache HIT / triage,
13 claims dispatched to Anthropic batch + OpenAI/Gemini/xAI live sidecar.

| Provider | Cost | vs pre-fix baseline (10× scale) | Tools/claim (frontier) | MRS non-empty | WS non-empty |
| --- | --- | --- | --- | --- | --- |
| Anthropic | $0.76 | n/a (already golden standard) | n/a (batch) | 100% | 100% |
| OpenAI    | $0.25 | unchanged | 0.69 | 92% | 15% |
| Gemini    | $0.17 | unchanged | 3.62 | **100%** (was ≈0%) | 0% (all hallucinated, stripped) |
| xAI       | $3.79 | $14.62 projected → **$3.79 actual = −74%** | **2.15** (cap=8, never hit) | 100% | 100% |
| **Total** | **$4.97** | $14.62 projected → **−66%** | — | — | — |

URL fabrication rate (frontier sidecar): xAI 1.5% (1/66 stripped),
Anthropic 0%, OpenAI 100%, Gemini 100% — i.e. OpenAI and Gemini still
hallucinate citations in `web_sources`, but the defensive backfill ensures
`model_reported_sources` carries the verifiable `tool_retrieved_urls` so
audit and consensus see grounding signal. xAI now consistently emits
real, retrieved URLs.

Consensus distribution across all 29 claims (all four adapters answered
100% of claims):
- False: 9 / Exaggerated: 6 / True: 5 / Mostly True: 4 / Unverifiable: 3 / Models split: 2
- Strength: strong 12 / weak 15 / none 2

### Cost-diff vs Phase E baseline

Linear-scale projection from pre-cap 10-claim Phase 3a baseline
(`10764cdb`, $5.04) to 29 claims = **$14.62 expected**. Actual = $4.97
= **−66%**. Reproducibility: 10-claim validation predicted $1.69 × 2.9 =
$4.90 → 29-claim actual $4.97 (within 1.4%). The Grok cap is the dominant
saving lever; the MRS backfill cost is zero — it only labels existing
tool-retrieved URLs.

### Explicitly still deferred (unchanged from yesterday)

- **Bug C — triage-tier URL grounding.** OpenAI / Gemini triage still
  strip 100% of model-reported URLs at the triage tier; needs its own
  planning session.
- **Anthropic / OpenAI `--mode live` multi-claim overrides.** Phase E
  extension; intentionally deferred to preserve A/B baseline.
- **111-claim full-transcript run.** Optional follow-up; 29-claim
  benchmark is sufficient for code-path coverage and reference-set
  comparison.

---

## Session note — 2026-04-25 (Calibration fixes + 10-claim rerun)

### Shipped today

Three high-leverage fixes uncovered by the Phase 3a calibration, plus the 10-claim rerun that proved/disproved each.

- **Bug A — multi-claim `web_sources` schema (partial fix).** Strengthened
  the multi-claim user-message preamble in
  ``src/truthbot/verify/adapters/base.py`` to require per-claim
  ``web_sources`` attribution explicitly, instead of relying on inheritance
  from the single-claim CITATION DISCIPLINE rubric. Also fixed a JSON-shape
  bug introduced mid-fix: an inline ``// REQUIRED ...`` comment placed inside
  the JSON example block, which made Anthropic and Gemini respond with prose
  explaining the schema instead of valid JSON arrays (every batch row in the
  v1 rerun logged ``parse_error`` / ``api_error``). Comment moved to
  surrounding free-text. New regression test
  ``test_build_multi_user_message_schema_block_is_valid_json_shape`` asserts
  the bracketed schema example contains no ``//`` substring; new
  ``test_build_multi_user_message_demands_per_claim_web_sources`` pins the
  strengthened preamble.

- **Bug — Anthropic triage cost $0.18/call (fixed).** Audit of
  ``metrics/adapter_calls.jsonl`` showed 9/10 Anthropic triage calls in the
  Phase 3a run used ``claude-opus-4-7`` despite the ``TriageAnthropic``
  subclass overriding ``model_id = "claude-haiku-4-5"``. Root cause:
  ``AnthropicAdapter._call_with_fallback`` iterated the hard-coded
  ``_FALLBACK_MODELS`` list (Opus-first) and ignored ``self.model_id``
  entirely. Fix prepends ``self.model_id`` to the fallback chain (with
  dedup) so triage subclasses and ``TRUTHBOT_TRIAGE_ANTHROPIC_MODEL`` env
  overrides actually drive the first request. New regression tests in
  ``tests/test_anthropic_fallback.py``:
  ``test_fallback_chain_starts_with_subclass_model_id`` and
  ``test_fallback_chain_does_not_duplicate_model_id``.

- **Bug B — Gemini cache/model mismatch (fixed).** All Phase 3a Gemini
  frontier multi-claim calls 400'd with ``Model used by GenerateContent
  request (models/gemini-2.5-pro) and CachedContent (models/gemini-2.5-flash)
  has to be the same``. Root cause: ``_cached_content_name`` was a single
  class-level slot, so triage (flash) populated it first and every later
  frontier (pro) call reused the flash-bound cache name. Fix replaces the
  scalar with ``_cached_content_names: dict[str, str]`` keyed by
  ``self._active_model``; cross-instance reuse within a tier is preserved.
  Migrated all callers (``test_gemini_cache.py``,
  ``test_gemini_multi_claim.py``, ``tests/smoke/test_smoke_submit*.py``).
  New regression: ``test_cache_is_keyed_by_active_model`` (instantiates
  triage and frontier subclasses, asserts each gets its own cache entry
  and the second call within a tier reuses).

### 10-claim calibration deltas (SOTU 2026 transcript)

Baseline: ``run_id=10764cdb-9b4a-489a-b76d-8f9d3fd7ba59`` (Phase 3a).
Rerun:    ``run_id=128597ce-6e83-44a3-8811-8811e1fa219e`` (today).

| Metric | Baseline | Rerun | Delta |
| --- | --- | --- | --- |
| Total cost | $5.04 | **$4.15** | **−18%** |
| Anthropic triage cost | $1.80 (9× Opus) | **$0.55 (10× Haiku)** | **−69%** |
| Anthropic frontier status | ok (gold standard) | ok | unchanged |
| Anthropic frontier reported / retrieved | 34/34 | 42/42 | preserved |
| Gemini frontier API status | 100% 400 (cache mismatch) | **100% ok** | fixed |
| Gemini frontier reported / retrieved | 0/0 (all errored) | 0/16 | partial |
| OpenAI frontier reported / retrieved | 0/1 | 0/0 | unchanged |
| xAI frontier reported / retrieved | 0/85 | 0/160 | unchanged |
| URL classification (verified / broken) | 36 / 4 | **131 / 0** | broader + cleaner |

### Outcome vs plan

- **Step 1 (multi-claim ``web_sources``):** ✅ for Anthropic batch (42
  reported, gold standard preserved). ❌ for OpenAI / Gemini / xAI live
  multi-claim — they still emit ``web_sources: []`` despite invoking the
  search tool 6–27 times per chunk. The strengthened prompt was insufficient
  for these providers; their tool-result ergonomics differ from Anthropic's
  inlined-citation model. Likely follow-up: change
  ``build_multi_verdicts`` so when ``web_sources`` is missing/empty AND
  ``tool_retrieved_urls`` is non-empty, populate ``model_reported_sources``
  with the tool URLs as a defensive backfill (vs the current contract where
  only index-0 ``web_sources`` is backfilled). Trade-off documented:
  attribution fidelity vs visible grounding.
- **Step 2 (Anthropic triage cost):** ✅ confirmed; cost fell 3.3× and
  ``model_id`` is now ``claude-haiku-4-5`` for all triage rows.
- **Step 3 (Gemini cache pin):** ✅ confirmed; zero 400s, frontier verdicts
  return.
- **Step 4 (rerun + STATUS.md update):** ✅ this note.

### Explicitly still deferred

- **Bug C — triage-tier URL grounding** (OpenAI + Gemini triage strip 100%
  of model-reported URLs, 24/24 and 31/31 in the rerun). Triage runs a
  separate single-claim code path; the strengthened multi-claim prompt does
  not affect it. Reasoning unchanged from the original plan: triage
  verdicts are consensus inputs, not published, so the strip is an audit
  signal rather than a production failure. Worth revisiting only if we
  start surfacing triage citations downstream.
- **Grok unbounded tool-call budget.** xAI triage spent $2.92 today (70%
  of total cost) on 500K input tokens and 83 tool calls across 10 claims.
  Real concern but not a regression and not on this plan's scope.

## Session note — 2026-04-23 (Phase E Grok/Gemini live claim-batching)

### Shipped today
- **Live multi-claim claim-batching for Grok + Gemini.** Both adapters gained a
  ``call_multi(claims, evidence_by_claim, ...)`` override that folds
  ``SYNTHESIS_SYSTEM`` over N claims in a single API call — one
  ``client.responses.create`` for Grok, one ``client.models.generate_content``
  for Gemini. ``LLMAdapter.call_multi`` landed with a default that loops
  ``self.call`` so Anthropic/OpenAI ``--mode live`` behavior is byte-identical.
  New caps: ``GrokAdapter.max_claims_per_request = 6`` and
  ``GeminiAdapter.max_claims_per_request = 4``. Telemetry attributes the
  whole call's usage to index-0 (``build_multi_verdicts`` contract) so
  ``costs.estimate_cost`` bills once per API call, not N times.
- **Sidecar dispatcher refactor.** ``BatchDispatcher.submit`` now chunks
  sidecar/live adapters by their cap, issues one ``adapter.call_multi`` per
  chunk, and falls back to per-claim ``adapter.call`` if the multi-claim
  call raises. Closes the "multi-claim chunk failure → single-claim retry"
  backlog row for the sidecar path.
- **Engine multi-claim entry point.** ``VerificationEngine.verify_bundles_batch``
  fans out **per adapter** (concurrent) instead of per-claim, each adapter
  looping its chunks via ``call_multi``. ``Pipeline._run_publish`` now
  routes ``--mode live`` through this path when
  ``TRUTHBOT_CLAIMS_PER_REQUEST > 1`` AND at least one active adapter has
  ``max_claims_per_request > 1`` (default today with Grok/Gemini enabled).
  Legacy per-claim fan-out is still used when ``TRUTHBOT_CLAIMS_PER_REQUEST=1``.
- **Tests.** 432 passed / 1 xfailed (up from 426 / 1). Added:
  - ``tests/test_grok_multi_claim.py`` (6 tests) — fake ``openai.OpenAI``
    client, verdict ordering, index-0 usage attribution, malformed JSON
    fallback, URL backfill, cap-raised assertion, single-request invariant.
  - ``tests/test_gemini_multi_claim.py`` (7 tests) — fake ``google.genai``
    module, CachedContent singleton reuse across two multi-claim calls,
    regression guard against passing ``system_instruction`` / ``tools`` when
    the cache is active, cached-token attribution to index-0, grounding
    URL backfill, malformed JSON fallback, cap-raised assertion.
  - ``tests/test_sidecar_multi_claim.py`` (5 tests) — ``call_multi`` invoked
    exactly once per chunk, chunking math (7 claims × cap=4 → 4+3),
    per-claim fallback on chunk failure, sidecar JSONL round-tripping
    ``batch_call_index`` / ``batch_call_id``, legacy cap=1 adapters still
    produce verdicts.
  - ``tests/test_verify_bundles_batch.py`` (5 tests) — mixed multi-capable
    and single-claim adapters, chunking math across adapter caps, per-claim
    fallback when ``call_multi`` raises, empty-input edge case, no-adapters
    fallback builds ``UNVERIFIABLE`` bundles.
  - ``tests/smoke/test_smoke_submit.py`` — new ``TestXAILiveMulti`` and
    ``TestGeminiLiveMulti`` smoke classes that assert both ground-truth
    labels come back from a single multi-claim API call and record
    ``request_count=1`` in ``smoke_summary.jsonl`` for post-run cost diffs
    against the existing 2-request single-claim baseline.

### Next — full-SOTU cost-diff (operator-run, pending API keys)

To confirm the real-world savings on a ~29-claim SOTU run, re-run the
publish pipeline under multi-claim mode and diff telemetry against the
pre-Phase-E baseline. Procedure:

1. Pick a baseline run_id already in ``metrics/run_summaries/`` (the most
   recent SOTU before this push). Capture its
   ``total_cost_usd`` / total ``input_tokens`` from
   ``truthbot metrics summary --run-id <baseline>``.
2. Re-run the same transcript with multi-claim enabled:
   ```bash
   set -a && . ./.env && set +a
   TRUTHBOT_CLAIMS_PER_REQUEST=6 .venv/bin/truthbot publish \
     --transcript eval/sotu-2026/transcript.txt \
     --speaker "Donald Trump" --role "President" \
     --date 2026-02-24 --venue "State of the Union" \
     --site-root site-test --mode batch --triage --max-claims 0
   .venv/bin/truthbot batch reconcile <new_run_id>
   ```
   (``--mode live`` also works now — Grok/Gemini will multi-claim on the
   sidecar; Anthropic/OpenAI inherit the default ``call_multi`` loop in
   live mode, i.e. behavior-identical to pre-Phase-E.)
3. Diff the two runs:
   ```bash
   .venv/bin/truthbot metrics summary --run-id <baseline>
   .venv/bin/truthbot metrics summary --run-id <new>
   ```
   Expected: Grok + Gemini input_tokens drop ~4-5× on their share of the
   run (one SYNTHESIS_SYSTEM send per chunk instead of per claim); total
   cost delta dominated by those two providers because Anthropic/OpenAI
   already multi-claim in batch mode.
4. Append the before/after deltas to this file under a new session note.

## Session note — 2026-04-22 evening (batch-API push)

### Shipped tonight
- **Real batch API implementation.** ``truthbot publish --mode batch`` now submits per-claim requests to each provider's native batch API instead of falling through to live calls:
  - Anthropic Message Batches (``client.messages.batches.create/retrieve/results``) with ``web_search_20250305`` tool + ephemeral prompt caching.
  - OpenAI Batch API against ``/v1/responses`` with ``web_search_preview`` (writes JSONL, uploads via ``/files``, creates batch, downloads output file at reconcile time).
  - Gemini has ``build_batch_payload``/``parse_batch_response`` wired but ``supports_batch=False`` pending a pinned google-genai with verified ``batches.create`` + GoogleSearch support; it falls out of batch-mode consensus until then.
  - xAI (Grok): no public batch API — runs **live during submit** as a sidecar, verdicts spooled to ``metrics/batch_sidecar/<run_id>.jsonl`` and merged at reconcile.
- **Two-phase CLI.** ``truthbot batch poll <run_id>`` returns a short status string; the new ``truthbot batch reconcile <run_id>`` polls → fetches results → parses → merges sidecar → builds consensus → caches ``VerdictBundle``s → publishes the site. Pipeline ``_run_publish`` split cleanly so ``--mode live`` is byte-for-byte unchanged.
- **Cost-reporting honesty.** ``costs.estimate_cost`` now gates the 50% batch discount on a real ``batch_job_id`` (not just the ``mode`` string). ``telemetry.measure`` threads the provider batch ID through. Scaffolding/misconfigured calls are billed full price. Existing `test_estimate_cost_batch_multiplier` updated to cover both gates.
- **Total-claim preservation.** ``--max-claims 0`` (the new default) means "verify every checkable claim". Extractor safety cap raised to 500; transcript budget raised to 200 K chars (covered a full SOTU instead of truncating to ~25% of one). Pipeline always prints ``N claims extracted total, M checkable, K selected``.
- **Tests.** 263 passed / 1 xfailed (up from 257 / 1). New ``tests/test_batch_mode.py`` adds mocked Anthropic submit→poll→reconcile round-trip, pending-path early-return, real-``batch_job_id`` telemetry assertion, sidecar JSONL round-trip, and error-row handling.
- **Backlog updated.** Added entries for "Multi-claim batching", "Gemini batch transport", and "xAI Grok batch" in ``PROJECT_BOARD.md``.

### Pre-submission blocker — **API keys in `.env` are truncated**
Three keys got eaten by a terminal line-wrap on paste (all end at column 80 with a stray ``>`` redirection char). Until these are re-pasted, **no live or batch API call can succeed** — any SOTU run right now would burn only the preamble request and 401 before anything useful happens.

Offending keys (check lengths, not values):
- ``ANTHROPIC_API_KEY`` — currently ~55 char body, real keys are ~108.
- ``OPENAI_API_KEY`` — currently ~55 char body, real ``sk-proj-*`` keys are ~156.
- ``XAI_API_KEY`` — currently ~59 char body, real ``xai-*`` keys are ~84.
- ``GEMINI_API_KEY`` looks intact (50 chars, no trailing ``>``).

### Morning runbook — SOTU 2026-02-24 overnight batch

Before running anything:

1. **Re-paste** the three keys above into ``.env`` (one line each, no line breaks, no trailing ``>``). Terminal paste works best if you widen the window first or use an editor.
2. **Verify** with a cheap ping per provider (these each cost well under a cent):
   ```bash
   set -a && . ./.env && set +a
   .venv/bin/python -c "import anthropic; anthropic.Anthropic().messages.create(model='claude-haiku-4-5', max_tokens=8, messages=[{'role':'user','content':'ping'}]); print('anthropic ok')"
   .venv/bin/python -c "import openai; openai.OpenAI().responses.create(model='gpt-5.4-nano', input='ping', max_output_tokens=16); print('openai ok')"
   .venv/bin/python -c "import openai; openai.OpenAI(api_key=__import__('os').environ['XAI_API_KEY'], base_url='https://api.x.ai/v1').chat.completions.create(model='grok-4', messages=[{'role':'user','content':'ping'}], max_tokens=8); print('xai ok')"
   ```
3. **Submit** the SOTU run (all 29 claims, no cap, triage on):
   ```bash
   set -a && . ./.env && set +a
   .venv/bin/truthbot publish \
     --transcript eval/sotu-2026/transcript.txt \
     --speaker "Donald Trump" --role "President" \
     --date 2026-02-24 --venue "State of the Union" \
     --site-root site-test --mode batch --triage --max-claims 0
   ```
   The command prints the ``run_id`` + both poll commands and exits immediately (no waiting).
4. **Reconcile & publish** after batches complete (Anthropic batches usually finish in 5–60 min, OpenAI up to 24 h):
   ```bash
   .venv/bin/truthbot batch poll <run_id>         # quick status check
   .venv/bin/truthbot batch reconcile <run_id>    # merge + site regen
   .venv/bin/truthbot metrics summary --run-id <run_id>
   ```
   Append totals (claims extracted, cost, duration) to this file.

## Current State

## Current State (pre-batch push, retained for history)
- **Pipeline:** Phases 1–5 are complete. Multi-adapter verification (Anthropic + OpenAI fallback to gpt-4.1), telemetry logging (`metrics/adapter_calls.jsonl`), and the static site publisher are all working end-to-end via `truthbot publish`.
- **Latest publish:** `site-test/` contains the 2026-03-04 Trump SOTU run (5 claims, 3 False / 2 True-ish verdicts). Material is ready to rsync to ExpressionPi once a live document root is configured.
- **Telemetry:** Anthropic calls average ~19s / $0.65 each; OpenAI adapter currently lands on gpt-4.1 fallback because gpt-5.4 isn’t GA yet. `truthbot metrics summary` reports cleanly from the JSONL log.
- **Working tree:** `src/truthbot/publish/site.py` + the regenerated `site-test` assets contain a WIP UI refresh (provider/model labels inside claim cards). The content renders fine locally, but the file picked up mojibake (“Ã¢ÂÂ” etc.) and needs to be re-encoded before shipping.

## Next Steps
1. **Phase 6 — ExpressionPi deploy:** set `TRUTHBOT_SITE_ROOT` to the nginx docroot on ExpressionPi, rerun `truthbot publish`, and wire nginx to serve `/truthbot` with daily rsync/cron refresh.
2. **Adapter coverage:** add valid OPENAI/GEMINI/XAI keys (or better fallbacks) so that consensus verdicts once again include more than Anthropic+OpenAI fallback, and update telemetry cost tables as models change.
3. **Automated runs:** define a nightly job (cron or systemd timer) that ingests the latest transcript, runs `truthbot publish`, and copies reports to ExpressionPi + Bluesky once Phase 7 lands.
4. **Site refresh cleanup:** fix the encoding on `publish/site.py` (re-save as UTF-8) and commit the provider/model labeling improvements alongside regenerated assets.

## Blockers / Risks
- **Provider keys:** GEMINI_API_KEY and XAI_API_KEY are still unset. OPENAI_API_KEY works but flagship `gpt-5.4` is unavailable, so we’re stuck on gpt-4.1 fallback until GA.
- **ExpressionPi availability:** we need a confirmed docroot + nginx stanza before pushing the static site live.
- **Bluesky publisher:** intentionally stubbed (`NotImplementedError`) until the site deployment is stable; Phase 7 is still queued.

## Session note — 2026-04-21 evening
- Fresh clone of `aRealGem/Truth-bot` landed on `main @ b1c3c7b`; existing `.env` preserved.
- `.venv` created with Python 3.13.12; `pip install -e ".[dev]"` succeeded.
- (Superseded by 2026-04-22 morning) Earlier run was **175 passed, 9 failed** due to test/schema drift; see morning note for resolution.

## Session note — 2026-04-22 morning
- **Tests:** Schema drift resolved in `tests/test_verify.py` and `tests/test_pipeline.py` (`ConsensusVerdict` / `.consensus_label` vs legacy `Verdict` on engine returns; `DummyEngine` returns `ConsensusVerdict` to match `Pipeline.run()`).
- **Bluesky:** `test_post_report_returns_none_when_unconfigured` is **xfail (strict)** until Phase 7 implements `post_report` (see `TODO.md`).
- **Baseline:** `pytest` → **183 passed, 1 xfailed, 0 failed**. No `--deselect` workaround needed.

### Recommendation (today)
1. **Next Steps #1** — Phase 6 ExpressionPi deploy (docroot + nginx + `TRUTHBOT_SITE_ROOT`).
2. **Next Steps #4** — UTF-8 cleanup in `src/truthbot/publish/site.py` (mojibake) before shipping UI refresh.
3. **Phase 7** — implement Bluesky `post_report`, then remove the strict xfail on the unconfigured test.
