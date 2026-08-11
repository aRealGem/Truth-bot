# DC-B1 spend estimate — evidence re-scoring and re-adjudication

**NOTHING HERE HAS BEEN RUN. NO SPEND HAS OCCURRED.**
No `score_evidence()` call was made against a live lane, no key file was sourced,
no model or API call of any kind was issued in producing this document. Both
code deliverables are written, tested offline against stubs, and remain gated:
the re-score script refuses to act without `--go --budget`, and the v2 builder's
scorer is `None` by default with both production flags shipping OFF.

Generated 2026-08-07. Machine-readable sibling: `dcb1_estimate.json`.

---

## THE CLICK — one scope, pinned

There is exactly one thing to approve. It is not a menu.

| | |
|---|---|
| **Scope** | **(a)** re-score all **4,344** stored evidence items + **(b)** re-adjudicate the **full gate-forced pool (87 claims)** plus the **6 named extras** = **93 claims** |
| **Estimated spend** | **$6.41 – $7.40** |
| **Authorized ceiling** | **$10.00** — a CEILING, not a forecast |
| **Days** | one; the whole job fits inside a single $20 day |

**The ceiling is not the expectation.** Actual spend lands *below* it, for three
reasons that all cut the same way:

1. **Adjudication only fires for claims the repaired gate actually releases.**
   The 87 is the maximum candidate pool — every claim currently gate-forced
   *could* flip once its evidence is scored. Every one that does not, costs $0.
2. **Claims that stay legitimately gated cost $0.** If scoring confirms a pack
   genuinely lacks two bearing Tier-1..3 items, the gate was right and no panel
   runs.
3. **Claims the repair newly WITHHOLDS also cost $0.** Scoring can move a
   currently-decided claim the other way — an item that looked like support
   turns out not to bear on the claim, dropping the pack below quota.
   Withholding a verdict needs no panel call. It is a correction that costs
   nothing to make.

So $7.40 is the price of the world where *everything* flips at the high
per-claim rate. The expected outcome is nearer $4.

### Why the full pool, and not a cheaper slice

Partial coverage is the option that looks thrifty and is not. If we
re-adjudicate some gate-forced claims and not others, then **which claims got a
second look depends on where we stopped, not on the evidence** — and the
stopping line (first N, or the "newly gated" subset, or a 50% sample) has
nothing to do with the merits of any individual claim. That is an
evenhandedness failure of exactly the kind **M-6** exists to prevent: the
published corpus would carry two classes of gate-forced Unverifiable, one
reviewed under the repaired gate and one not, distinguishable only by budget
history. The saving is about $3. It is not worth buying an asymmetry we would
then have to disclose.

Priced alternatives are retained in the appendix **for audit trail only**. They
are not choices on this click.

---

## Why this spend is being requested

`build_evidence_pack_v2` wired the R1/R2/R3 retriever shortlists straight into
`consolidate()`, so `verify/relevance.py::score_evidence` — the only writer of
`relevance_score` and `supports_claim` — was never on the v2 path. Measured over
the five rebuilt runs:

| | value |
|---|---|
| items stored | **4,344** |
| items carrying the 0.5 relevance default | **4,344 (100.00%)** |
| items with a null stance | **1,136 (26.2%)**, per speech 20.5–30.2% |

`consolidator._bearing()` requires stance `True`/`False`, so a null item can
never credit `MIN_BEARING_T13 = 2`. Packs holding good evidence therefore
gate-force Unverifiable. The verified example is `trump_2026:0469`, where NPR is
the only crediting item while AP, NBC and two govinfo records sit unscored.

**The A1 fitness finding, with its denominator.** `a1_fitness_report.json`
records that *every* stored run is unfit to gate. "Every" means **17 stored run
artifacts = 5 published (live on the site) + 5 rebuilt (staged, unpublished) +
7 superseded (retained per archive-never-delete)** — not 17 published reports.
The site publishes five. The finding covers the whole stored record, of which
the published corpus is one cohort; this spend repairs the rebuilt cohort, which
is the one queued for publication.

Owner ruling Q-1 approved the permanent fix (**B1b**, wire the scorer into the
v2 builder — done, `scorer=` hook, default off) plus a one-off re-score of the
stored items (**B1a**, this estimate). Both are model spend.

## (a) B1a — re-score the 4,344 stored items

Per item: one cheap Haiku call per **sid** (not per item) through the LiteLLM
proxy, over the pack's items in a single prompt. Stored v2 packs cap at
`PACK_CAP_V2 = 10`, comfortably under the `DEFAULT_SCORE_CAP = 16`, so every sid
is exactly one call and nothing is truncated.

**Lane: `claude-haiku` via the LiteLLM proxy — ON-PROXY.** That matters for
honesty: the funded run's real cost will be **ledger-true**
(`proxy_lane.proxy_key_spend()`), not estimated, and the budget breaker reads
that same ledger. Unlike the phase-3 rebuild, there is no off-proxy R2/R3
component to approximate.

### What the $0.4391 actually prices — both directions

The figure is not a per-claim guess scaled up. `estimate_speech()` in
`scripts/rescore_stored_packs.py` reconstructs the real payloads and prices
**both halves of every call**:

* **Input.** For each sid it builds the *exact* `score_evidence` prompt via
  `relevance.score_payload(claim_text, items)` — the same function the funded
  path calls — and counts `len(_SCORE_SYSTEM) + len(score_payload(...))`
  characters. System prompt plus payload, per call, measured.
* **Output.** The reply shape is fixed, so it is modelled rather than guessed:
  `REPLY_CHARS_OVERHEAD (16) + REPLY_CHARS_PER_ITEM (46) × len(items)` — the
  `{"i", "relevance", "supports"}` record per item plus the `{"scores": [...]}`
  wrapper.
* **Conversion and rates.** Characters → tokens at `CHARS_PER_TOKEN = 4.0`,
  then priced at
  `hydramind.models.RATE_TABLE_USD_PER_MTOK["claude-haiku"] = (0.80, 4.00)`
  USD per Mtok in / out.

**Measured totals: 288,406 input tokens + 52,072 output tokens, over 529 calls
covering 4,344 items.**

| speech | calls | items | tok_in | tok_out | est USD |
|---|---:|---:|---:|---:|---:|
| gwbush_2006 | 48 | 396 | 26,789 | 4,746 | 0.0404 |
| clinton_1998 | 92 | 792 | 52,335 | 9,476 | 0.0798 |
| obama_2014 | 96 | 799 | 52,134 | 9,572 | 0.0800 |
| biden_2022 | 111 | 885 | 58,600 | 10,622 | 0.0894 |
| trump_2026 | 182 | 1,472 | 98,548 | 17,656 | 0.1495 |
| **TOTAL** | **529** | **4,344** | **288,406** | **52,072** | **0.4391** |

**≈ $0.44 for the whole corpus re-score**, i.e. **$0.00010 per item** — which
is not a suspiciously round number, it is what 66 input tokens and 12 output
tokens per item cost at Haiku rates. The per-item figure is low because the
work per item genuinely is small: one line of the payload in, one small JSON
record out, amortised across a batched per-sid call.

**Uncertainty.** Two sources, both small and both bounded upward:

1. *Tokenization.* `chars / 4` is an **approximation, not a tokenizer run**.
   JSON scaffolding and URLs tokenize worse than prose, so the true input count
   is plausibly 10–25% higher. Even at **+50% this line is under $0.70.**
2. *Rate table.* The `claude-haiku` entry is flagged in `hydramind/models.py` as
   a fallback estimate, because Haiku is proxy-priced in practice. **Haiku is
   on-proxy, so the funded run is ledger-true**: the proxy's own price is
   authoritative, `proxy_key_spend()` reports what was actually billed, and this
   estimate only has to be good enough to set a cap.

The output half is the confident half: the reply is a fixed, tiny JSON shape, so
volume scales exactly with item count.

## (b) PCA re-adjudication — the pinned scope

**The flip set is UNKNOWN until B1a runs.** A claim only needs re-adjudication
if scoring gives it ≥2 bearing Tier-1..3 items where it previously had fewer.
Which claims those are is exactly what B1a discovers. That is why the pinned
scope is the whole candidate pool priced as a ceiling: it is bounded, not
predicted.

### Correction to the brief

The brief called this "the 66 currently gate-forced claims". The artifacts do
not support that number as the candidate pool — **66 is the `newly_gated` count**
(claims the rebuild newly gated relative to the published run), summed from the
`phase3_*_verdict_diff.json` files. The count of claims **currently** gate-forced
in the rebuilt runs is **87**:

| speech | claims | gate-forced now | of which newly gated |
|---|---:|---:|---:|
| gwbush_2006 | 48 | 7 | 5 |
| clinton_1998 | 92 | 14 | 11 |
| obama_2014 | 96 | 12 | 7 |
| biden_2022 | 111 | 16 | 12 |
| trump_2026 | 182 | 38 | 31 |
| **TOTAL** | **529** | **87** | **66** |

**87 is the pinned pool.** Every currently gate-forced claim could in principle
flip, and each one that does not costs nothing. The 66 is a meaningful subset
(the regressions specifically) but it is not the ceiling and it is not a scope.

### Per-claim cost, verified

From the committed phase-3 run logs, cost / claims per speech:

| speech | claims | run cost | $/claim | note |
|---|---:|---:|---:|---|
| gwbush_2006 | 48 | $3.0823 | **$0.0642** | clean single-session run |
| clinton_1998 | 92 | $6.8453 | **$0.0744** | clean single-session run |
| trump_2026 | 182 | $13.6136 | **$0.0748** | clean single-session run |
| obama_2014 | 96 | $1.8350 | $0.0191 | resumed — **undercounts** |
| biden_2022 | 111 | $4.0042 | $0.0361 | resumed — **undercounts** |

This confirms the brief's **$0.064–0.075/claim** on gpt-5-mini, from the three
runs that completed in one session. The two resumed legs are excluded on
purpose, and the reason is a real accounting gap — see the carry-forward
obligation below.

### The 6 named extras

Named by Fable for re-adjudication independent of the gate outcome:
`trump_2026:0030`, `trump_2026:0031` (with the exhibit), `trump_2026:0023`,
`trump_2026:0024`, `trump_2026:0343`, plus `clinton_1998:0313` (the CW2-decisive
sid from A3).

**All six are currently DECIDED, not gate-forced** — verified against the
artifacts, in the order listed: FALSE, TRUE, MISLEADING, TRUE, TRUE, TRUE. They
are therefore strictly **additive** to any flip set — no double-counting.

6 claims → **$0.39 – $0.45**.

### Pinned scope, priced

| line | claims | low | high |
|---|---:|---:|---:|
| Full gate-forced pool | 87 | $5.59 | $6.51 |
| + the 6 named extras | 6 | $0.39 | $0.45 |
| **B1b total — the pinned scope** | **93** | **$5.97** | **$6.96** |

(The total is 93 × the per-claim rate, computed once; adding the two rounded
line items gives $5.98 at the low end. The $0.01 is rounding, not a missing
claim.)

## (c) Combined total, the cap, and scheduling

| line | low | high |
|---|---:|---:|
| (a) B1a re-score, all 4,344 items | $0.44 | $0.44 |
| (b) B1b re-adjudication, 93 claims (87 pool + 6 named) | $5.97 | $6.96 |
| **PINNED SCOPE, ALL IN** | **$6.41** | **$7.40** |

### Authorized ceiling: **$10.00**

Raised from the $8.00 first proposed. $7.40 against $8.00 is **8% headroom**,
and a cap that breaks mid-run is not a saving — it is its own incident: a halted
job leaves the corpus half-re-adjudicated, which is the same evenhandedness
problem as a deliberate partial, arrived at by accident. **$10.00 gives 35%
headroom over the worst case and still fits inside a single $20 day**, with $10
left for anything else that day.

Stress-tested: if the chars/4 approximation is 50% low **and** all 93 claims
adjudicate at the high per-claim rate, the run lands at ≈ **$7.62** — still
$2.38 under the ceiling.

### Sequencing — one day, three steps, separately budgeted

1. **B1a re-score, ~$0.44.** Cheap enough to run whole. Start with
   `--speech gwbush_2006 --go --budget 0.25` as calibration (48 sids, est.
   $0.04): if ledger-true cost lands far from estimate, stop and re-price before
   the other four.
2. **Join the sidecars, identify the actual flip set.** $0 — pure local analysis
   against `rescored_<speech>.json`. This is the step that turns the pinned
   ceiling into a real number, and it is where the "actual lands below the
   ceiling" claim gets tested rather than asserted.
3. **B1b re-adjudication of the flip set + the 6 named claims**, budget set from
   step 2's actual count at $0.075/claim, capped at **$9.00** (the ceiling less
   what step 1 spent).

Step 2's finding can only shrink step 3. If the flip set comes back empty, step
3 costs nothing but the 6 named claims (~$0.45) — and that outcome is a finding
in its own right, not a wasted click.

## Carry-forward obligation → the DC-6' final ledger

Recorded here so it is inherited **explicitly**, not silently:

> The corpus spend total is a **mixed basis** — proxy spend is ledger-true (the
> LiteLLM proxy key was billed), off-proxy spend is a list-rate **estimate** —
> and the two **resumed** legs (`obama_2014`, `biden_2022`) **undercount their
> off-proxy portion**. `phase3_rebuild` banks only proxy spend in the chunk
> journal (`append_chunk_journal` is handed the `proxy_key_spend` delta), so a
> resumed session carries the prior leg's proxy cost forward and drops that
> leg's off-proxy estimate. The DC-6 table reconstructs both legs from the run
> logs, which recovers what the runner's own SPEND line lost — but only down to
> the last *banked* chunk; both legs died inside the following chunk after its
> retrieval had already run. **The corpus total is a lower bound on the
> off-proxy component.**

That is fine for pricing this estimate — it is exactly why the per-claim rate
basis uses the three single-session runs only. It is not fine to inherit
quietly. The disclosure is implemented in `scripts/dc6_package.py`
(`SPEND_BASIS_DISCLOSURE`), emitted into both `dc6_review.json` and the "6.
Spend + provenance" section of `dc6_review.md`, and asserted by
`tests/test_dc6_package.py` so a future regeneration cannot drop it.

## What was built, and what remains gated

| deliverable | state |
|---|---|
| `scripts/rescore_stored_packs.py` | written, 19 offline tests, **not run** (only `--estimate`/plan, both $0) |
| `build_evidence_pack_v2(scorer=...)` | wired, 10 offline tests with a stub scorer, default `None` |
| `pipeline.py --score-evidence` | present, **defaults OFF**, asserted by test |
| `phase3_rebuild.py --score-evidence` | present, **defaults OFF**, asserted by test |

Nothing above spends until someone passes `--go --budget` (B1a) or
`--score-evidence` (B1b), and DC-B1 is signed.

---

## Appendix — priced alternatives, retained for audit trail

**These are NOT choices.** They were priced while the scope was open, and are
kept so the record shows what was considered and what it would have cost. The
decision section above pins the full pool; anything here would reintroduce the
M-6 evenhandedness problem described there.

| alternative (not offered) | claims | low | high | why not |
|---|---:|---:|---:|---|
| Newly-gated subset only | 66 | $4.24 | $4.94 | "newly gated" is a property of the *diff against the published run*, not of the claim's evidence — a claim gate-forced in both runs is no less wrongly gated |
| Plausible partial — 50% of the pool | 44 | $2.82 | $3.29 | the 50% would be chosen by budget, not by evidence |
| 50% of the newly-gated subset | 33 | $2.12 | $2.47 | both objections at once |
| Combined at the 44-claim partial (+6 named, +B1a) | 50 | $3.65 | $4.18 | the "midpoint scenario" the earlier draft offered as a live option |

The earlier draft also proposed an **$8.00** cap. Superseded by $10.00 for the
headroom reason given above; recorded here so the change is visible rather than
silent.
