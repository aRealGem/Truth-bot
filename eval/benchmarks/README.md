# C1 benchmarks — check-worthiness claim set + scorer

Gating eval material for **C1 (HydraMind L2 / truth-bot v2)**. This directory
holds the **check-worthiness claim set** (public SOTU-derived) and the shared
**scorer**. It was built 2026-07-02 and moved under version control 2026-07-04
(the working tree at `~/cc-host/benchmarks` is unversioned and outside backup
coverage).

## `claim-set/` — check-worthiness triage (277 sentences)

Sentence-level triage **upstream** of truth-bot's verifier: *should this
sentence enter the fact-checking pipeline at all?* Labels map onto the extractor
contract in `src/truthbot/extract/claims.py`:

- `check-worthy` — verifiable factual assertion of public importance
  (+ `claim_type`: statistical/historical/attribution/comparison/other)
- `opinion` — opinion / rhetoric / value judgment / future prediction
- `unimportant` — literally factual but trivial (not worth a fact-check budget)

| label | n | % | train / heldout |
|---|---|---|---|
| opinion | 151 | 55% | 100 / 51 |
| check-worthy | 73 | 26% | 48 / 25 |
| unimportant | 53 | 19% | 35 / 18 |

Source: `Historical-SOTU-Transcripts` (Miller Center / UVA). Primary =
`trump_2026_sotu.txt` (the TB-00 SOTU 2026 material); secondary =
`biden_2022_sotu.txt`. See `claim-set/LABELING_GUIDE.md` for label definitions
and edge cases. `_`-prefixed files are build provenance (segmenter output,
sampled candidates, per-shard labeler outputs).

Complements `eval/sotu-2026/reference.json` (29 veracity verdicts): that grades
*verdict accuracy given claims*; this grades *which sentences are worth checking*.

## `scorer/`

- `segment_sotu.py` — deterministic SOTU sentence segmenter
- `score.py` — unified scorer; `score.py claims preds.jsonl` reports accuracy +
  confusion against `claim_set.jsonl`
- `secret_scan.py` — pre-push secret/PII gate (0-hit required)

## Reproduce / run

```bash
python3 claim-set/build_claim_set.py            # rebuild claim_set*.jsonl + secret scan
python3 scorer/score.py claims preds.jsonl      # {"sid","pred"} per line -> accuracy + confusion
python3 scorer/secret_scan.py claim-set scorer  # pre-push gate
```

## Companion (not here by design)

The **Cass-task benchmark** contains real operational/infra agent prompts and is
kept in a **private offsite repo**, never public. `score.py`'s `cass`/`demo`
paths therefore have no data in this public repo — the `claims` path is
self-contained.
