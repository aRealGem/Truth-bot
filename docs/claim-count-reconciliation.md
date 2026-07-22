# Claim-count reconciliation — 289 vs 277 (vs the retired 274)

*P67.9 T3.4 (pulled forward), 2026-07-21. Documents the canonical counts the
2026-07-21 external audit found drifting across surfaces (finding F4).*

## The three numbers

| Count | What it is | Source of truth |
|---|---|---|
| **289** | Published claims on the site: 178 (Trump SOTU 2026) + 111 (Biden SOTU 2022) — the check-worthy output of the live PCA runs | `site-pca/data/claims.json` (canonical for everything on-site since T0.7; index stat and consistency checker derive from it) |
| **277** | The curated gold **evaluation** corpus: `claim_set.jsonl` = 184 train + ~94 heldout rows (I6 read-once) | `eval/benchmarks/claim-set/claim_set.jsonl` |
| **274** | RETIRED. The v1 model-insights table counted only claims with a non-empty `model_verdicts_summary`, silently dropping the 15 Models-split claims (289 − 15). The v1 page was removed in PR-1 and rebuilt from per-seat provenance in PR-7 | nothing — do not cite |

## Why 289 ≠ 277 (and why that's correct)

They are **different corpora over different segmentations**:

- The published 289 come from the live pipeline's own sentence segmentation
  (783 Trump / 480 Biden sentences) and its A1/A2 check-worthiness routing.
- The gold 277 were curated separately for evaluation on an earlier
  segmentation pass (722 Trump sentences in `_sentences.jsonl`), with human
  labeling via the shard/label workflow. Sentence indices between the two
  segmentations DO NOT correspond — a sid like `trump_2026:0522` names
  *different sentences* in the two corpora (this bit the T1.2 audit tooling;
  the run artifacts are self-contained for this reason).

Neither number is a miscount of the other. Site copy must cite **289** (or
whatever `claims.json` holds after future runs); eval writeups must cite
**277** and say "gold evaluation corpus".

## Attribution audit (Mora/Rivera class)

`scripts/attribution_audit.py` (read-only; heldout loaded texts-only, labels
never materialized) scans both corpora for cross-speech verbatim and
near-verbatim duplicates (word 4-gram Jaccard ≥ 0.6, sentences ≥ 6 words).
**Result 2026-07-21: zero exact and zero near-verbatim cross-speech pairs**
in the gold corpus (226 comparable claims) and in the published claims (285).
Artifact: `metrics/attribution_audit_2026-07-21.json`.
