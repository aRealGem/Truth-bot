# B2 — the scoring-prompt fix, and what it actually found

**Spend:** $0.5404 (Haiku on the LiteLLM proxy, cap $1.50, never approached).
**Programme total:** $1.6038 of the $10.00 ceiling.
**Subset:** 115 claims / 1,028 items, derived deterministically and printed before any money moved (`scripts/b2_primary_series.py`, `metrics/remediation_v2/b2_subset.json`).

---

## 1. The headline: the prompt was not the binding constraint

The brief's diagnosis was that Haiku mis-classified raw statistical series as
"context", so the best evidence in the pack credited nothing. The prompt now
tells the scorer that a primary series carrying the figure at issue must take a
side. It was re-run over exactly the items that diagnosis predicts it would fix.

**It fixed 17 of 227 of them — 7.5%.**

| | items |
|---|---:|
| targeted items (Tier-1..3, stance-None, primary data/record source) | 227 |
| ...that now carry a stance | **17** |
| ...marked `arithmetic_hinge` | 23 |

And within the 115 re-scored packs, the stance-null rate went **up**, from 36.3%
to 38.0%.

The reason is written in the model's own `one_line_why` lines, over and over:

> "BLS time-series for employed persons 16+ back to 1948 is the authoritative
> historical record needed to judge whether January 2026 represents an all-time
> high, **but snippet does not state the January 2026 level**."

> "Primary appropriations law for FY1998 provides actual funding level in effect
> near the claim date, **but snippet does not state the 1993 baseline or
> comparison**."

`relevance.score_payload` sends the scorer a source name and **400 characters of
snippet**. It never sends the table. The pipeline retrieved the URL of the
series and never fetched its contents. So the instruction "read the number and
take a side" produces the only honest answer available: *I cannot, because I was
not shown the number.*

**The mis-classification was a symptom; the missing fetch is the disease.** A
data series cannot be scored from its own metadata, and no prompt can close that
gap. The correct fix is a retrieval-side one — fetch and excerpt the relevant
rows of a primary series into the snippet before scoring — and it is out of
scope here.

### Why the null rate rising is a quality improvement

Those 18 net new nulls are items that previously carried a stance the scorer had
no basis for. B1a's prompt did not ask whether the snippet contained the figure,
so the model supplied a plausible direction from the source's identity and
topic. The B2 prompt asks for the comparison in writing, and the answer is
frequently "there is no comparison to state." Replacing an unfounded stance with
an honest abstention makes the corpus smaller and more trustworthy at the same
time. The headline metric moved the wrong way; the epistemics moved the right
way.

The two cases where the fix worked are exactly the cases where it should:
`trump_2026:0054`'s ALFRED and TradingEconomics items both **refuted** the
"most Americans ever working" claim, because their snippets *do* carry the
figure ("all-time high was 163,992K in December 2025").

## 2. The flip set, versus the previous 32 / 55 / 23 / 419

| | released | still gated | newly gated | unchanged |
|---|---:|---:|---:|---:|
| B1a only | 32 | 55 | 23 | 419 |
| **B1a + B2** | **33** | **54** | **27** | **415** |

Gate reproduction is 529/529 on every speech, so the delta is attributable to
the re-score and not to drift. Net movement is small and slightly toward
withholding — consistent with a pass whose main effect was to retract stances
rather than add them.

Reproducible both ways: `scripts/regate_from_rescore.py --no-b2` reproduces the
B1a-only column exactly.

## 3. Stance-null rate per run, before and after

"Before" is the artifact as rebuilt; "after" is B1a + B2 merged.

| speech | items | null before | null after (B1a only) | null after (B1a+B2) |
|---|---:|---|---|---|
| gwbush_2006 | 396 | 102 · 25.8% | 55 · 13.9% | 57 · 14.4% |
| clinton_1998 | 792 | 222 · 28.0% | 108 · 13.6% | 113 · 14.3% |
| obama_2014 | 799 | 187 · 23.4% | 91 · 11.4% | 93 · 11.6% |
| biden_2022 | 885 | 181 · 20.5% | 106 · 12.0% | 107 · 12.1% |
| trump_2026 | 1472 | 444 · 30.2% | 301 · 20.4% | 309 · 21.0% |
| **corpus** | **4344** | **1136 · 26.2%** | **661 · 15.2%** | **679 · 15.6%** |

## 4. The arithmetic hinges — 64 items across 33 claims

The reviewer-mandated guard. Where the stance depends on arithmetic the *scorer*
performed over a series — a maximum, a ratio, a real-terms deflation, a
comparison the source does not itself make — the item is marked
`arithmetic_hinge: true`, persisted in the sidecar, and surfaced in the pack
payload so the panel is told the stance is a hypothesis.

**It has deliberately been given no gate effect.** These claims are collected and
reported for routing to computed-exhibit treatment (R-2,
`publish/computed_exhibit.py`, `metrics/computed_exhibits/`); nothing here
promotes or demotes a verdict on its own.

Both claims the brief named as hinge-shaped were caught: `trump_2026:0054`
("most Americans ever working") and `obama_2014:0189` ("minimum wage worth 20%
less").

**The 33 claims:**

`biden_2022`: 0114, 0169, 0211, 0216, 0266, 0305
`clinton_1998`: 0032, 0035, 0091, 0101
`gwbush_2006`: 0134, 0163, 0187, 0217
`obama_2014`: 0001, 0087, 0177, 0189, 0202
`trump_2026`: 0022, 0030, 0031, 0043, 0046, 0052, 0055, 0057, 0137, 0161, 0208, 0219, 0295, 0583

Full per-item detail, with each item's `one_line_why`, is in
`regate_flipset.json` under `arithmetic_hinges`.

## 5. Spend

| pass | gwbush | clinton | obama | biden | trump | total |
|---|---:|---:|---:|---:|---:|---:|
| B1a | 0.0840 | 0.1949 | 0.1965 | 0.2158 | 0.3720 | **1.0632** |
| B2 | 0.0463 | 0.1080 | 0.0980 | 0.0805 | 0.2077 | **0.5404** |

**Programme total: $1.6038 against the $10.00 ceiling — $8.3962 remaining.**

B2's estimate was $0.2299 and the actual was $0.5404, 2.4× over. The estimator
priced the reply at three times the B1a shape to allow for `one_line_why`; the
real replies are longer and more discursive than that. The cap was never
approached, but the multiplier should be raised before the next contract change
that adds a free-text field.

## 6. Sidecar layout — and why it is two files

B2 wrote `rescored_b2_<speech>.json` alongside B1a's `rescored_<speech>.json`
rather than merging in place. `score_evidence` rewrites **every** item in a pack,
not just the targeted ones, so writing into B1a's file would have silently
replaced B1a scores for items B2 was never asked about, with no way to tell
afterwards which vintage a row came from.

That decision earned its keep: B2 did in fact retract stances on non-targeted
items in those 115 packs (373 → 391 nulls), and because both files survive, the
B1a-only view is still exactly reproducible with `--no-b2`. The merge is per-sid
— the unit that was actually re-scored as a whole — with B2 taking precedence,
and the two spends stay separately attributable.

## 7. What this changes about the plan

1. **Do not re-run this prompt over the rest of the corpus.** The binding
   constraint is the 400-character snippet, not the wording. Another pass buys
   more honest abstentions, not more stances.
2. **The real fix is retrieval-side:** fetch primary series and excerpt the
   relevant rows into the snippet before scoring. That is a new piece of work,
   and it should be scoped before any further scoring spend.
3. **B1a's stance numbers should be read with more suspicion than they were.**
   On primary-source items its stances were frequently produced without the
   figure in view. The 15.2% null rate it reported was partly bought with
   unfounded stances.
4. **33 claims need computed-exhibit routing** before their verdicts can be
   trusted, independent of anything above.
