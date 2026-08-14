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

**All 30 committed errors run in one direction: predicted `substantive` when the
desk found the claim documentable. Zero errors run the other way.** The probe
never once called a genuinely-undecidable claim checkable.

That asymmetry is not reassuring — it is the dangerous polarity. A render keyed
on these signals would not under-claim; it would **stamp "cannot be verified" on
30 documentable claims**, which is precisely the defect D17-d exists to remove,
relocated into a new mechanism. The two commit rules that produce it (R2, R3)
have precision 0.235 and 0.105 against the fixture.

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
