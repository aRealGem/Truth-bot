# D17-d structural probe — rules and per-rule confusion

Companion to `d17d_structural_probe.json`, produced by
`scripts/d17d_structural_probe.py`. Analysis only: $0, no model, no network, no
pipeline change, no verdict moved. Deterministic — re-running is byte-identical.

**What "agree" means.** The desk pass (`d17d_triage.json`) is one careful human
read used as an **audit fixture**. "Agree" = "matches the desk", not "is correct
about the world". Where the probe and the desk differ, the desk is the more
likely to be right, but neither is ground truth.

**Population.** All 128 claims stamped `insufficient-qualifying-evidence` across
the five frozen publishing heads.

---

## The six rules

Evaluated in order; first match wins. Reads only `claim_type`, `claim_shape`,
whether any evidence item carries `series_rows`, and evidence tiers. **No claim
text is read as prose** — the desk doc is explicit that separating
permanently-undecidable from merely under-retrieved "needs this desk pass, not a
regex", so the probe abstains where structure is blind instead of guessing.

| rule | disposition | signal | predicts / residual range |
|---|---|---|---|
| **R1-series-attached** | commit | an evidence item carries `series_rows` | `series-core` |
| **R2-attribution-type** | commit | `claim_type == attribution` | `substantive` |
| **R3-eval-shape** | commit | `claim_shape == c-eval` | `substantive` |
| **R4-statistical-unattached** | abstain | `claim_type == statistical`, no series | {series-core, web-tier1} |
| **R5-narrative-type** | abstain | `claim_type ∈ {personal-anecdote, historical, comparison, other}` | {web-tier1, substantive} |
| **R6-no-signal** | abstain | no usable structured field | all four |

---

## Totals

| | n |
|---|---:|
| gate-withheld | 128 |
| **committed** | **37** — 7 match the desk, **30 do not** |
| **abstained** | **91** — residual range holds the desk class 87×, misses 4× |

---

## Per-rule confusion

### R1-series-attached — commit `series-core` — fired 1
| desk class | n |
|---|---:|
| series-core | 1 |

**1 agree, 0 error.** The only unambiguous "this is checkable" signal in the
artifact, and it fired **once in 128**. That one claim (`obama_2014:0189`) is the
already-recorded `window_period_mismatch` case. Perfect precision, negligible
recall.

### R2-attribution-type — commit `substantive` — fired 17
| desk class | n |
|---|---:|
| web-tier1 | 12 |
| substantive | 4 |
| compound-split | 1 |

**4 agree, 13 error.** Error direction — **every error the same way**:

| direction | n |
|---|---:|
| predicted `substantive` → desk `web-tier1` | 12 |
| predicted `substantive` → desk `compound-split` | 1 |

The rule predicts "undecidable" and is wrong 13 times out of 17 (**precision
0.235**). In this pipeline `attribution` means *"X said / announced / is-alleged
Y"*, which is overwhelmingly a **documentable public act**: a DOJ task force
announcement, four former Joint Chiefs chairmen endorsing, a judge's custody
ruling, a named driver's licence status. The four it got right are the genuine
article — an adversary's *aim* (`gwbush_2006:0033`), an unmeasured mass
attribution (`trump_2026:0514`), a rhetorical-breadth quantifier
(`biden_2022:0373`), and the private Michael Dell conversation
(`trump_2026:0153`).

> **Correction to `docs/decisions/D17-d-rederive.md` §4(a):** that write-up said
> "only 3 of the attribution claims were truly substantive" and named three. The
> computed figure is **4** — `biden_2022:0373` belongs in that list. The
> disagreement count of 13 was correct.

`attribution` conflates *attribution of a public act* (checkable) with
*attribution of private intent* (undecidable).

### R3-eval-shape — commit `substantive` — fired 19
| desk class | n |
|---|---:|
| web-tier1 | 14 |
| series-core | 2 |
| substantive | 2 |
| compound-split | 1 |

**2 agree, 17 error** (**precision 0.105** — the worst rule). Error direction,
again all one way:

| direction | n |
|---|---:|
| predicted `substantive` → desk `web-tier1` | 14 |
| predicted `substantive` → desk `series-core` | 2 |
| predicted `substantive` → desk `compound-split` | 1 |

`c-eval` bundles **checkable superlatives and counts** in with genuinely
evaluative cores: "today there are 122 [democracies]", "$880 billion in the hands
of…", "drug use among youth down 19 percent", "240 trade agreements". Two of its
errors are desk `series-core` — claims a named series would settle outright.

Note `claim_shape` is present on only **31 of 128** claims (clinton_1998 and
gwbush_2006 only; absent on biden/obama/trump), so this rule cannot fire at all
on three of the five speeches.

### R4-statistical-unattached — abstain {series-core, web-tier1} — fired 22
| desk class | n |
|---|---:|
| web-tier1 | 15 |
| series-core | 4 |
| compound-split | 2 |
| substantive | 1 |

Residual range **holds the desk class 19×, misses 3×**
(`trump_2026:0057`, `trump_2026:0334`, `trump_2026:0343`). Two misses are
compound claims whose checkable fragment reads as statistical; one
(`trump_2026:0334`, "many, if not most, illegal aliens…") is an
unmeasured-population quantifier that looks like a plain number.

### R5-narrative-type — abstain {web-tier1, substantive} — fired 69
| desk class | n |
|---|---:|
| web-tier1 | 40 |
| substantive | 28 |
| compound-split | 1 |

Residual range **holds the desk class 68×, misses 1×** (`trump_2026:0130`). This
is the largest rule and the honest one: it narrows correctly almost every time
and **cannot pick within the range**. The 40/28 split *is* the open problem —
documentable valor citations and private hospital-room moments in one bucket with
no field separating them.

### R6-no-signal — abstain, all four — fired 0
Never fired: every gate-withheld claim carries at least a `claim_type`.

---

## The requested overlap — anecdote-precedence ∩ desk-substantive

`anecdote-precedence` is reported two ways because R5 spans four narrative types,
so neither reading has to be inferred:

| reading | fired | ∩ desk-substantive | precision if treated as substantive | recall of desk substantive (n=35) |
|---|---:|---:|---:|---:|
| R5, all narrative types | 69 | **28** | 0.406 | 0.800 |
| R5, `personal-anecdote` only | 47 | **22** | 0.468 | 0.629 |

**Either way it is a coin flip that loses.** Treating narrative type as a
substantive signal would be wrong 59% of the time (all-narrative) or 53% of the
time (anecdote-only) — and wrong in the direction of telling a reader that a
documented fact cannot be verified.

---

## The finding that governs step 5

**At the commit layer, all 30 errors run one direction: predicted `substantive`
when the desk found the claim documentable.** A render keyed on these signals
would not under-claim; it would **stamp "cannot be verified" on 30 documentable
claims**, which is precisely the defect D17-d exists to remove, relocated into a
new mechanism. The two commit rules that produce it (R2, R3) have precision
0.235 and 0.105 against the fixture.

**Scope correction — one-directionality is a COMMIT-LAYER property only.** An
earlier draft of this finding said "zero errors run the other way" and glossed it
as "the probe never once called a genuinely-undecidable claim checkable." That
second sentence is **false**, and the counterexample is recorded here rather than
dropped:

> **`trump_2026:0334`** — *"Many, if not most, illegal aliens do not speak
> English…"* — an unmeasured-population quantifier the desk classed
> `substantive`. R4 reads it as a plain statistic and narrows to
> **{series-core, web-tier1}**, a residual range that **excludes `substantive`
> entirely**. The abstain layer therefore does run the reverse direction: it
> ruled out "undecidable" for a claim that is undecidable.

It is the only such case (1 of 91 abstentions), but it matters for step 5,
because a render that treats an abstention's residual range as a *narrowing* —
"we know it is at least one of these" — would inherit that error and present a
permanently-undecidable claim as merely unretrieved. The safe reading of a
residual range is "structure failed here", not "structure narrowed it".

Both layers are pinned in `tests/test_d17d_structural_probe.py`
(`test_every_committed_error_runs_one_direction`,
`test_residual_layer_has_one_reverse_miss`).

### Denominator note — the anecdote figures

The 47 in the overlap table is **anecdotes that reach R5**, not anecdotes.
Population-wide there are **48** `personal-anecdote` claims; `clinton_1998:0225`
never reaches R5 because it also carries `claim_shape = c-eval` and is consumed
by R3 first. Rule precedence changes the denominator, so the two numbers must not
be quoted interchangeably — 22/47 is an R5-scoped precision, and a
population-wide anecdote statistic would have 48 as its base. Pinned in
`test_anecdote_denominator_is_r5_scoped_not_population_wide`.

**Structurally inexpressible desk classes:**
- `compound-split` — 5 desk claims, **0 recovered**. No segmentation signal
  exists in the artifact; they scatter across four rules.
- `series-core` — 7 desk claims, **1 recovered**. The D17-c series machinery
  works; it simply never ran on these gated packs. The other 6 read as ordinary
  statistical (4) or evaluative (2) claims.

**Conclusion.** No current structured field, alone or in combination, supports a
"cannot be verified" label. The pipeline needs a **public/private (retrievable
vs undecidable) axis** it does not have, and `attribution` and `c-eval` each want
splitting before either can carry weight.

---

# R7 — pack anatomy

R1–R6 used only claim-level fields and never looked at the **pack**. R7 asks
whether that was a miss: does pack anatomy — how much was retrieved, at what
tiers, how much bore on the claim — carry a decidability signal the claim fields
do not? Artifact: `d17d_pack_anatomy_probe.json`, from
`scripts/d17d_pack_anatomy_probe.py`. 128 packs, 969 evidence items.

**No threshold was fitted to the desk pass.** A cut chosen to maximise agreement
would launder the fixture into the classifier and report its own reflection as a
finding. R7 reports distributions and lets the separation speak.

## Which per-item fields survive into the artifact

| field | items carrying | present |
|---|---:|:--:|
| `source_tier` | 969 | yes |
| `supports_claim` (bearing flag) | 668 | yes |
| `one_line_why` | 287 | yes |
| `arithmetic_hinge` | 20 | yes |
| `series_rows` | 1 | yes |
| `role` | 0 | **no** |
| `era_note` | 0 | **no** |
| `utterance_rule` | 0 | **no** |
| `quota_credit` | 0 | **no** |
| `disqualification_code` | 0 | **no** |
| `gate_code` | 0 | **no** |

**There are no per-item disqualification codes.** Nothing in a stored item
records why it failed to count. Tier and bearing survive; the gate's own
reasoning does not.

## The stored artifact cannot reproduce the gate it is a record of

All 128 packs were rejected by the real gate. Reconstructing quota credit from
the surviving fields (`source_tier ∈ {Government, Wire, Established}` **and**
`supports_claim is not None`) scores **78 of 128 packs at ≥2 credits — 61% would
"have passed."**

The reconstruction is a **proxy, not the gate**: `consolidator._quota_credit`
also consults `role` (D11.2 role-aware credit), `utterance_rule` (D15 — credits
0), the post-speech band, and era mode. **None of those survive into the stored
evidence item.** Any pack-anatomy feature is therefore computed over a strictly
poorer field set than the decision it is trying to explain.

## Anatomy by desk class — no separation

| desk class | n | items (mean) | Tier-1..3 | bearing | proxy credits |
|---|---:|---:|---:|---:|---:|
| web-tier1 | 81 | 7.53 | 3.83 | 5.46 | 2.43 |
| substantive | 35 | 7.60 | 3.89 | 5.11 | 2.34 |
| series-core | 7 | 7.57 | 4.29 | 4.29 | 2.14 |
| compound-split | 5 | 8.00 | 4.60 | 3.40 | 1.80 |

**The two classes a render must never confuse are indistinguishable.**
`web-tier1` (a retrieval backlog) and `substantive` (permanent abstention) differ
by 0.07 items, 0.06 Tier-1..3 sources, and 0.09 proxy credits. The tier
difference even runs mildly *backwards*: substantive packs carry marginally
**more** Tier-1..3 sources than documentable ones, so the intuition "more
qualifying evidence ⇒ more checkable" is not merely weak here, it is faintly
inverted.

**Answer to "was ignoring pack anatomy a miss?" — no.** Pack anatomy adds no
usable decidability signal on this corpus. That is not because the packs are
uninformative, but because they are all *failures of the same gate*: the corpus
is conditioned on rejection, so the anatomy that would discriminate has already
been flattened by the selection.

## Egress note — the generational assumption (S-12)

Every figure above is measured on packs from a single methodology generation:
**`v2.3-role-axis-s5cap`** (S5 political tier, ≤3 saturation cap, role axis, era
fail-closed), which is the generation of all five heads and the current one.

S-12 exists because a constant measured on one payload was applied to another and
ran the escalation 8.2× over ($0.3266 against $0.0396). The same failure mode is
available here in non-monetary form: **these anatomy numbers are a property of
this generation's retrieval, not of the pipeline.** The tier mix in particular is
generation-specific — `pre-s5-tiering` runs classified every `.gov` host as
top-tier Government, and `pre-s5-cap` runs had no per-claim political saturation
cap, so both would yield materially different Tier-1..3 counts on the same
claims. A threshold derived from the table above and carried across a generation
boundary would be exactly the S-12 error: a number that cannot name the payload
it measured.

Accordingly, **no R7 figure may be used as a constant, a threshold, or a cost
basis outside `v2.3-role-axis-s5cap`.** If retrieval changes — a new retriever,
a widened tier set, re-retrieval of the backlog — this probe must be re-run
before any of its numbers are quoted again. The 61% gate-reproduction gap should
be treated as the floor on how much the artifact under-describes the decision,
not as a measured constant.
