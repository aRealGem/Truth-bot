# Layer A speech-act taxonomy (spec §4)

**Locked v2 dependency — landed before the A2 classifier prompt** (build order §4 step 0).
Layer A decides *check-worthiness*: for each sentence, should it enter the verification
pipeline (Layer B), or flow to the characterization stream (the product, Principle 4)?

## Grounding

- **Searle, *A Taxonomy of Illocutionary Acts* (1975).** Five illocutionary classes —
  **assertives** (commit speaker to truth of a proposition), **directives**, **commissives**
  (commit speaker to future action), **expressives**, **declarations**. Only **assertives**
  carry a truth-evaluable proposition. Directives/commissives/expressives/declarations do not,
  and route to characterization, not verification.
- **Benoit, functional theory of political campaign discourse.** Political utterances are
  **acclaims / attacks / defenses**. These are *functions*, orthogonal to truth-value: an
  attack can be assertive (checkable) or expressive (rhetoric). We record the function as
  metadata but gate on the Searle assertive/proposition test, never on the function.
- **Hassan et al., ClaimBuster (KDD 2017)** and **CLEF CheckThat!** — check-worthiness is a
  *gate-first* task: cheaply score every sentence, spend the verification budget only on
  check-worthy factual claims (Principle 3, RAND firehose). A1's feature set (numerics,
  comparatives, assertion verbs, named entities) is the ClaimBuster feature family.

## Mapping to the 277-row label contract

The three labels are exactly those in `eval/benchmarks/claim-set` and
`src/truthbot/extract/claims.py` (`is_checkable` + skip opinion/rhetoric/prediction):

| label | Searle class | truth-bot extractor | Layer routing |
|---|---|---|---|
| `check-worthy` | **assertive** with a **publicly material, verifiable** proposition | `is_checkable: true`, extracted; `claim_type ∈ {statistical, historical, attribution, comparison, other}` | → Layer B queue |
| `opinion` | directive / commissive / expressive, OR assertive that is subjective / a future prediction | skipped (rule 2: "skip pure opinion, rhetorical framing, value judgments, and predictions about the future") | → characterization stream |
| `unimportant` | assertive but **trivial** (no public stakes): ceremonial, procedural, personal aside | not worth budget | → characterization stream |

## Decision procedure (mirrors LABELING_GUIDE)

1. Ceremonial / greeting / procedural / personal aside, no public stakes → `unimportant`.
2. Subjective judgment, rhetoric, aspiration, promise, or **future** prediction → `opinion`
   (fails the assertive-proposition test, or is not yet truth-evaluable).
3. Specific, verifiable proposition (number / event / comparison / attribution) → `check-worthy`
   (+ `claim_type`).
4. Otherwise → `opinion`.

## Neutrality (Principle 1 / I3)

The taxonomy conditions on the **proposition and its speech-act form**, never on **who is
speaking**. Speaker identity is payload metadata; no rule, threshold, or prompt branch may
read it (enforced by the I3 template linter over the A2 prompt).
