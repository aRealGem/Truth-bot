# D17-d step 1 — can pipeline structure re-derive the decidability desk pass?

**Status: a $0 probe result for owner review. No verdict moved, nothing
published, no model called. Input to the step-5/6 decision about what a page may
honestly assert.**

Evidence — the committed, canonical artifacts are on branch `d17d-probe0`:
- `metrics/remediation_v2/d17d_structural_probe.json` — per claim: disposition
  (committed / abstained), predicted class, the `rule_id` that fired, the desk
  class, agreement, and the residual class range for every abstention.
- `metrics/remediation_v2/d17d_probe_rules.md` — the six rules with per-rule
  confusion counts and the direction of every committed error.
- Built by `scripts/d17d_structural_probe.py`; locked by
  `tests/test_d17d_structural_probe.py` (12 tests, deterministic, offline).

This document is the narrative reading of those artifacts. An earlier
uncommitted pass (`build_d17d_rederive.py` → `d17d_rederive.json` +
`d17d_disagreement.json`) produced identical per-claim results — verified 0
mismatches across all 128 — and is superseded by the rule-id'd probe above.

---

## 1. The question

The D17-d desk pass (`d17d_triage.json`) hand-classified all **128 gate-withheld
claims** into four decidability classes. That is human judgement. This probe asks
a narrower, falsifiable question:

> Using **only** the structured fields the pipeline already records
> (`claim_type`, `claim_shape`, whether any evidence item carries `series_rows`,
> evidence tiers) — and reading **no claim text as prose** — how much of that
> judgement can be reproduced?

The desk pass is the **audit fixture**. This re-derivation is the first draft of
its automated replacement. Where structure cannot pick a single class it
**abstains loudly** into a named `*-undetermined` bucket rather than guessing —
because the D17-d doc is explicit that separating a permanently-undecidable claim
from a merely under-retrieved one "needs this desk pass, not a regex."

---

## 2. Headline

> Of 128 desk calls, structure will commit to a single class on **37**. Measured
> against the fixture, that commitment is right **7 times and wrong 30**. The
> other **91** are structurally undetermined (87 consistent with the desk's
> class, 4 pointed at the wrong range).

**The pipeline carries no reliable signal for "this claim is undecidable."** The
one clean structural signal for a *checkable* claim — an attached `series_rows` —
fired on exactly **1 of the 7** desk `series-core` claims (and that one is the
already-recorded window-mismatch case, `obama_2014:0189`). Everything else the
classifier "confidently" decided, it decided by two fields that turn out to be
conflations (§4). The large undetermined majority is not a failure of this
probe — it *is* the measurement: it is the signal the pipeline does not yet hold.

---

## 3. Confusion matrix — desk class × derived bucket (all 128)

| desk ↓ / derived → | series-core | substantive-leaning | series-or-web *(abstain)* | web-or-substantive *(abstain)* |
|---|---:|---:|---:|---:|
| **web-tier1** (81) | – | 26 | 15 | 40 |
| **substantive** (35) | – | 6 | 1 | 28 |
| **series-core** (7) | 1 | 2 | 4 | – |
| **compound-split** (5) | – | 2 | 2 | 1 |

Confident buckets are the two leftmost; the two rightmost are honest abstentions
that name the desk classes they are consistent with.

---

## 4. The 30 disagreements are two conflations, nothing else

Every confident-but-wrong call comes from one of two structured fields meaning
less than it appears to:

**(a) `claim_type = attribution` — 13 wrong of 17 fired.** In the pipeline
"attribution" means *"X said / announced / is alleged Y."* That is overwhelmingly
a **documentable public act or statement**, which is why the desk put it in
web-tier1:
- `biden_2022:0051` — DOJ "assembling a dedicated task force" (a department announcement)
- `clinton_1998:0236` — "four former Chairmen of the Joint Chiefs… endorsed" (their own statements)
- `trump_2026:0326` / `:0342` / `:0403` — a named driver's status, a killer's prior arrest, a judge's ruling (court and agency records)

The classifier read `attribution` as the signature of *no-retrieval-reaches-it*.
It is not. Only **4** of the 17 were truly substantive — an adversary's *aim*
(`gwbush_2006:0033`), an unmeasured mass attribution (`trump_2026:0514`), a
rhetorical-breadth quantifier (`biden_2022:0373`), and a private conversation
with Michael Dell (`trump_2026:0153`). That is **precision 0.235**.
`attribution` conflates *attribution of a public act* (checkable) with
*attribution of private intent* (undecidable).

**(b) `claim_shape = c-eval` — 17 wrong of 19 fired (precision 0.105).**
`c-eval` bundles **checkable superlatives and counts** in with genuinely
evaluative cores:
- `gwbush_2006:0025` "today there are 122 [democracies]", `:0147` "$880 billion in the hands of…", `:0217` "drug use among youth down 19 percent"
- `clinton_1998:0090` "240 trade agreements", `:0195` "largest antidrug budget"

These are documentable (or series-checkable); the desk classed them web-tier1 /
series-core. The classifier saw the evaluative flag and called them substantive.
`claim_shape` is also present on only 31 of 128 claims (clinton_1998 and
gwbush_2006 only), so this signal cannot fire at all on three of five speeches.

### The polarity that governs the decision

**At the commit layer, all 30 errors run one direction: predicted `substantive`
for a claim the desk found documentable.** A render keyed on these fields would
not under-claim. It would **stamp "cannot be verified" on 30 documentable
claims** — the exact defect D17-d exists to remove, relocated into a new
mechanism. The two rules a render would lean on have precision **0.235** and
**0.105** against the fixture.

**One caveat, against my own first framing.** I originally wrote that the probe
"never once called a genuinely-undecidable claim checkable." That is false at the
abstain layer. `trump_2026:0334` ("many, if not most, illegal aliens do not speak
English…") is desk-`substantive`, and R4 narrows it to
{series-core, web-tier1} — a residual range that **excludes** `substantive`. One
case in 91, but it kills the blanket claim, and it warns against reading a
residual range as a genuine narrowing: for step 5 the safe reading is "structure
failed here", not "we know it is at least one of these."

**Consequence for step 5.** Neither `attribution` nor `c-eval` can drive a render
that says "cannot be verified."

---

## 5. What is structurally invisible

- **`compound-split` (5): 0 recovered.** No segmentation signal exists in the
  artifact, so not one compound claim can be found as compound. It scatters
  across three buckets. Deciding these still needs the logged utterance-
  segmentation work before anything else.
- **`series-core` (6 of 7): invisible.** D17-c attached `series_rows` to only one
  gated pack. The other six read as ordinary `statistical` claims
  (`series-or-web-undetermined`) or, when superlative-shaped, as
  `substantive-leaning`. The series machinery exists; it simply never ran on
  these gated claims.
- **4 undetermined-misses** — structure narrowed to the wrong range:
  `trump_2026:0334` ("many, if not most, illegal aliens…" — an unmeasured-
  population quantifier that reads as plain `statistical`), and three
  `compound-split` claims whose checkable fragment made them look statistical.

---

## 6. What signal each undetermined bucket would need

This is the bridge to steps 5 and 6 — it names precisely what the pipeline must
gain before a render can assert decidability on its own:

| bucket | claims | the signal it lacks |
|---|---:|---|
| `web-or-substantive-undetermined` | 69 | a **public/private (retrievable/undecidable) axis** on the claim — the documented-citation vs private-room distinction no current field carries |
| `series-or-web-undetermined` | 22 | a **named-series resolver** — does a nameable statistical series settle the number, or only web retrieval? |
| `compound-split` (invisible) | 5 | an **utterance-segmentation marker** — already logged as a structural item |
| `attribution` conflation | 13 | split `attribution` into **public-act vs private-intent** |
| `c-eval` conflation | 17 | split `c-eval` into **checkable-superlative/count vs genuinely-evaluative** |

---

## 7. Bounds

- **$0.** No model, no network, no clock. Deterministic — re-running the build is
  byte-identical (fixture-locked in the test).
- The five `pca_runs` heads were read only; none was rewritten.
- **Owner still holds every downstream click.** Step 6 (ratifying the 35
  `substantive` classifications) must precede any page that says "cannot be
  verified"; step 5 (render) follows this probe, not the other way around; steps
  3–4 (metered retrieval) need an explicit click. This document changes nothing
  on the page.
