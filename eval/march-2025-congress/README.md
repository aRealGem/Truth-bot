# Test Case: March 2025 Congressional Address — Transgender Claims

**Event:** Joint Address to Congress
**Speaker:** President Donald Trump
**Date:** March 4, 2025
**Topic subset:** Claims regarding transgender issues only

## Reference Analysis

**Produced by:** Claude (with web search), cross-referenced against NBC News, NPR, PBS, CNN,
PolitiFact, FactCheck.org, and Snopes.
**Analysis date:** April 19, 2026
**Claims extracted:** 10 checkable factual/ideological assertions related to transgender issues

## Evaluation Prompt

The intended evaluation prompt for this test case is:

> Objectively evaluate the veracity of the President's claims regarding transgender issues in
> the March 2025 Congressional Address. Summarize in a table and give an overall truth/lie score.

## Claim-by-Claim Scorecard

| # | Topic                                     | Verdict                              |
|---|-------------------------------------------|--------------------------------------|
| 1 | Two-sex EO                                | MISLEADING                           |
| 2 | Trans sports ban EO                       | TRUE (action only; scale overstated) |
| 3 | School curriculum EO                      | MISLEADING                           |
| 4 | Payton McNabb injury                      | PARTLY TRUE / MISLEADING             |
| 5 | Cycling "women's record"                  | MISLEADING                           |
| 6 | $8M Lesotho LGBTQ+ spending               | FALSE (~4x inflation)                |
| 7 | $8M "making mice transgender"             | FALSE (~17x inflation)               |
| 8 | Littlejohn "child abuse"                  | MISLEADING                           |
| 9 | Gender-affirming care = "mutilation"      | FALSE / contradicts medical consensus|
|10 | "Big lie" that kids can be born in wrong body | FALSE / contradicts medical consensus|

## Overall Truth/Lie Score: ~2 / 10

**Precise breakdown:**

- **0** claims fully true with honest framing
- **2** literally true as to administrative action (executive orders exist), but paired with
  misleading or unsupported premises
- **4** misleading — real facts stripped of disqualifying context
- **3** materially false on the numbers or characterization (Lesotho $8M, mice $8M, "mutilation")
- **1** ideological assertion that contradicts the consensus of every major US medical association

## Pattern Analysis

The pattern is consistent: verifiable executive actions used to anchor a string of inflated dollar
figures, decontextualized anecdotes, and medical characterizations rejected by the AMA, AAP, APA,
and Endocrine Society.

The two most quantifiable claims — the Lesotho and "transgender mice" dollar figures — are off by
roughly 4x and 17x respectively, both in the direction that inflates outrage.

## Status

⚠️ **NOT YET INTEGRATED** — This test case is stored as reference data only.
Do not use for active benchmarking or run through the evolver until explicitly instructed.
The SOTU 2026 benchmark remains the primary evaluation target.

## Files

| File             | Description                                         |
|------------------|-----------------------------------------------------|
| `reference.json` | Structured 10-claim array for automated comparison  |
| `README.md`      | This file                                           |
| `sources.md`     | Reference sources used                              |

## Notes on This Test Case

- This covers only the transgender-related subset of the March 2025 address, not the full speech.
- Claim count (10) is smaller than SOTU 2026 (29); raw recall metrics will be less stable.
- Several claims sit at the checkability boundary — "child abuse," "mutilation," and "big lie"
  are partly ideological, partly empirically falsifiable. Tests extraction genome filtering.
- Claims #6 and #7 (dollar figures) are the cleanest verifiable numeric assertions in this set.
- Claim #10 (the "big lie" framing) is a meta-claim about medical consensus. Most extraction
  prompts will likely skip it as non-checkable — which may be correct behavior, but worth
  examining explicitly.
- The directional bias in the numeric errors (both inflated, both toward outrage maximization)
  is a potentially interesting signal for the scoring/analysis layer.
