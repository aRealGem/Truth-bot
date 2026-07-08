# Verdict-gold — schema, rubric, and scoring semantics (Layer B)

Layer A gold (`claim_set.*.jsonl` `label`) answers *"should this sentence be
fact-checked?"*. **Verdict-gold** answers the next question for the check-worthy
subset: *"what is the evidence-backed verdict?"* — the ground truth truth-bot v2
Layer B is scored against.

It is an **overlay** keyed by `sid`, kept in a separate file so the frozen claim
set is never edited: `verdict_gold.train.jsonl` (and, later, a **separately
guarded** `verdict_gold.heldout.jsonl` — see *Discipline* below).

## Why a separate rubric from the model's contract

Closed-book Layer B emits the reduced 4-label set `TRUE | FALSE | MISLEADING |
UNVERIFIABLE`, where `UNVERIFIABLE` is an **abstention** ("can't adjudicate from
general knowledge"). Gold, by contrast, is **evidence-backed**: an annotator with
sources decides the real verdict. So gold and prediction live in the same 4-label
space, but scoring must treat a model abstention as a *coverage gap*, not a wrong
answer (see *Scoring semantics*).

## Row schema

```json
{
  "sid": "biden_2022:0245",
  "claim": "…the claim text, for readability…",
  "gold_verdict": "TRUE | FALSE | MISLEADING | UNVERIFIABLE",
  "confidence": "high | med | low",
  "sources": ["https://authoritative-source/…", "…"],
  "rationale": "one or two clauses citing what the sources establish",
  "annotator": "who/what assigned it (e.g. claude-seed, jackie)",
  "date": "YYYY-MM-DD",
  "needs_review": true,
  "edge_case": "optional rule name, else null"
}
```

## Label rubric (evidence-backed)

- **TRUE** — the claim's checkable content is supported by authoritative evidence.
- **FALSE** — contradicted by authoritative evidence (includes absolute claims,
  e.g. "zero", that the evidence refutes).
- **MISLEADING** — the literal number/fact may be accurate but the framing omits
  or distorts context a fact-checker would flag (the classic "true but misleading").
  This is where the 4-label set is coarser than the reader-facing scale on purpose;
  `EXAGGERATED` and `MOSTLY_TRUE` from `models.py::VerdictLabel` fold in here until
  Layer C evidence justifies the finer gradations.
- **UNVERIFIABLE** — genuinely not adjudicable even *with* evidence (unknowable,
  private, or no authoritative source exists). Distinct from a model abstention:
  here the *annotator with sources* concludes it can't be settled.

Map up to the 6-bucket `models.py::VerdictLabel` is deferred to Layer C.

## Sourcing policy (no fabricated labels)

- **Every** decidable row (TRUE/FALSE/MISLEADING) MUST carry ≥1 authoritative
  `source` URL and a `rationale` naming what it establishes. No source ⇒ the row
  stays `UNVERIFIABLE` or out of the set.
- Prefer primary/institutional sources (government stats: BLS/CDC/CBO/Census;
  major fact-checkers: PolitiFact, FactCheck.org, CRFB) over partisan outlets.
- `needs_review: true` marks a row assigned by a single annotator (e.g. the seed
  below) that has not yet been independently adjudicated. The mature process should
  mirror Layer A's multi-annotator pass (`_labels_0..4.json` → adjudication) before
  a row is treated as settled gold.

## Scoring semantics (see `scorer/score_verdict.py`)

A closed-book abstaining system must not be scored like a forced-choice classifier:

- **committed** = a `resolved` item with a verdict in `{TRUE, FALSE, MISLEADING}`.
  `UNVERIFIABLE`, `disagreement`, and `no_label` all count as **abstain**.
- **decided-accuracy** = `hit / (hit + miss)` over committed items — *when the model
  commits, is it right?* The primary quality signal for closed-book.
- **coverage** = committed / total gold-with-prediction.
- **abstain_gap** = decidable gold where the model abstained — the coverage the
  Layer C evidence lane is meant to close (not a "miss").
- **abstain_ok** = gold `UNVERIFIABLE` where the model rightly abstained.

So a closed-book run legitimately shows low coverage + high decided-accuracy; that
is the *expected* shape, and rising coverage is the Layer C win condition.

## Discipline

- **TRAIN only** for now. Build and iterate gold on `verdict_gold.train.jsonl`.
- A `verdict_gold.heldout.jsonl` is a **separate, deliberate** artifact: creating it
  means reading heldout claims, so it should be done independently of pipeline
  authorship (leakage hygiene) and a scored heldout pipeline pass still consumes a
  fresh `rc_id` under the I6 `HeldoutGuard`.

## Status

`verdict_gold.train.jsonl` currently holds a **3-row sourced seed** (Biden-2022
claims, chosen because they are past and well-documented) that exercises the scorer
end-to-end. Expanding to a full, adjudicated set is the tracked follow-up — ideally
multi-annotator and evidence-assisted once Layer C lands.
