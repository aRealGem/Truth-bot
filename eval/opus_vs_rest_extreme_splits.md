# Opus vs the rest — extreme-split scan

**Source:** [`site-test/data/claims.json`](../site-test/data/claims.json) · **Generated:** Truth-bot/`eval/opus_vs_rest_extreme_splits.md`

## Method

Each fine-axis verdict label is mapped to a Truthy-axis score:

| Label | Score |
|-------|------:|
| True | +2 |
| Mostly True | +1 |
| Unverifiable | +0 |
| Models split | +0 |
| Exaggerated | -1 |
| Misleading | -1 |
| False | -2 |

For each claim with Anthropic on the panel and at least two other adapters, we compute `|opus_score - median(rest_scores)|`. A claim is **extreme** when that diff ≥ 3 points (roughly: Opus calls it Mostly True/True while ≥half of the rest call it False/Misleading, or vice versa).

## Headline

* **5 distinct extreme splits** out of **40 distinct claims** with Anthropic on the panel (12.5%).
* Opus is the **more-truthy** voice in **4** of those splits.
* Opus is the **more-falsey** voice in **1** of those splits.

> **Asymmetry note.** Opus is the lone optimist roughly 4× more often than the lone pessimist. That matches the user's 2026-04-29 hunch and is consistent with Claude's tendency toward charitable interpretation of partisan claims; worth a closer prompt-engineering look if it persists across speakers/topics.


## Top splits (deduped by claim text, sorted by magnitude then recurrence)

### 1. diff = 3 — Opus says **Mostly True** · seen in **5** report runs of this speech

_Donald Trump · 2026-02-24_

> Trump claims that in the past nine months, zero illegal aliens have been admitted to the United States.

**anthropic**: Mostly True | **openai**: False | **gemini**: False | **xai**: False

[Open claim page](../site-test/claims/e1153466-d40b-4c24-b4f7-034e8e8d2e1f.html)

---

### 2. diff = 3 — Opus says **Mostly True** · seen in **4** report runs of this speech

_Donald Trump · 2026-02-24_

> Trump claims his administration drove core inflation down to the lowest level in more than five years within 12 months.

**anthropic**: Mostly True | **openai**: False | **gemini**: False | **xai**: False

[Open claim page](../site-test/claims/a3bfdf9b-366d-4fba-80bc-8a2f976dff79.html)

---

### 3. diff = 3 — Opus says **Mostly True**

_Donald Trump · 2026-02-24_

> Trump claims that when he last spoke in the chamber 12 months prior, he had inherited a nation with inflation at record levels.

**anthropic**: Mostly True | **openai**: False | **gemini**: False | **xai**: False

[Open claim page](../site-test/claims/aed0b384-6404-46fe-b5f1-c055a8dd4f08.html)

---

### 4. diff = 3 — Opus says **False**

_Donald Trump · 2026-02-24_

> Trump claims his administration drove core inflation down to its lowest level in more than five years within 12 months.

**anthropic**: False | **openai**: Mostly True | **gemini**: Mostly True | **xai**: Exaggerated

[Open claim page](../site-test/claims/9f978c90-6c62-4e2d-94b3-fc0137fbdf41.html)

---

### 5. diff = 3 — Opus says **Mostly True**

_Donald Trump · 2026-02-24_

> Trump claims mortgage rates are the lowest in four years.

**anthropic**: Mostly True | **openai**: Mostly True | **gemini**: False | **xai**: False

[Open claim page](../site-test/claims/0334719e-890d-4f97-bea3-cf9ff2f97c35.html)

---

## Caveats

* This is **pure label scoring** — it doesn't read the per-model explanations. A diff-3 split where Opus has a citation the others missed is qualitatively different from a diff-3 split where Opus is over-charitable. Spot-check the worst offenders by hand.
* Median-of-rest is robust to a single outlier on the rest side, but with only 3 peers (OpenAI + Gemini + xAI) one outlier still shifts the median noticeably. Read the row of labels, not just the diff.
* The reference set in [`eval/sotu-2026/reference.json`](sotu-2026/reference.json) is the only ground truth we trust; if a flagged split lines up with a reference claim, cross-check there before concluding which side is wrong.
