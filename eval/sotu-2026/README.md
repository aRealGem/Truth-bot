# Test Case: 2026 State of the Union

**Event:** State of the Union Address  
**Speaker:** President Donald Trump  
**Date:** February 24, 2026  
**Duration:** ~108 minutes (described by PolitiFact and FactCheck.org as unusually long)  
**Venue:** Joint Session of Congress  

## Reference Analysis

**Produced by:** GPT 5.4 Pro with extended thinking  
**Analysis date:** February 26, 2026  
**Claims extracted:** 29 checkable factual assertions  

The model excluded value judgments, vague boasts, proposals, promises, and predictions — only scored measurable factual claims (numbers, measurable outcomes, historical claims).

## Scorecard Summary

| Verdict Category | Count | Percentage |
|---|---|---|
| **TRUE** (absolute truths) | 4 / 29 | 14% |
| **FALSE** (absolute lies) | 11 / 29 | 38% |
| **Other** (partly true / misleading / unsupported / unverifiable) | 14 / 29 | 48% |

Among decisively classifiable claims only (TRUE or FALSE, excluding partly true / misleading / unsupported):
- TRUE: 4/15 = **27%**
- FALSE: 11/15 = **73%**

## Verdict Distribution

| Verdict | Count |
|---|---|
| TRUE | 4 |
| FALSE | 7 |
| FALSE / MISLEADING | 2 |
| PARTLY TRUE | 2 |
| MISLEADING | 7 |
| UNSUPPORTED | 3 |
| UNVERIFIABLE | 1 |
| UNSUPPORTED | 3 |

## Files

| File | Description |
|---|---|
| `reference.md` | Full fact-check in clean markdown (converted from PDF) |
| `reference.json` | Structured 29-claim array for automated comparison |
| `transcript_urls.md` | Source transcript URLs (R1–R3) |
| `sources.md` | All reference sources R1–R8 with descriptions and URLs |

## Using This Test Case

Feed truth-bot the transcript from one or more URLs in `transcript_urls.md`, then compare its output against `reference.json`. Key metrics:
- **Claim recall** — how many of the 29 claims did it identify?
- **Verdict agreement** — how often does its verdict match the reference?
- **Explanation quality** — does it cite comparable sources?
