# Claim-set labeling guide — check-worthiness triage

This set trains/evaluates the **check-worthiness gate** that sits *upstream* of
truth-bot's verifier. The verifier (see `Truth-bot/src/truthbot/extract/claims.py`)
already decides TRUE/FALSE for claims it is handed; this set decides the prior
question: **for each sentence, should it enter the fact-checking pipeline at
all?** The three labels map directly onto truth-bot's extractor contract
(extract every factual assertion; skip opinion / rhetoric / value-judgment /
future-prediction).

## Labels

### 1. `check-worthy`
A **factual, verifiable assertion of public importance** — the kind of thing a
fact-checker would pull and verify against evidence.

Includes: statistics; historical events; quantitative comparisons; causal
attributions; claims about what a person/administration/entity did, said, or
caused; current, checkable states of the world.

Maps to truth-bot `is_checkable: true` (would be extracted). Assign a
`claim_type`: `statistical` | `historical` | `attribution` | `comparison` | `other`.

### 2. `opinion`  (opinion / rhetoric)
**Not verifiable against evidence.** Pure opinion, value judgment, subjective
characterization, rhetorical framing, aspiration/exhortation, promises, calls to
action, and **predictions/promises about the future** (unfalsifiable now).

Maps to truth-bot extractor rule 2 ("skip pure opinion, rhetorical framing,
value judgments, and predictions about the future"). `claim_type: null`.

### 3. `unimportant`  (unimportant fact)
**Literally factual/verifiable but trivial** — negligible public-interest
stakes, not worth a fact-check budget. Greetings, thanks, acknowledgements,
ceremonial address, procedural remarks, guest introductions, personal
anecdotes, self-evident truisms. `claim_type: null`.

## Decision order (apply top-down)
1. Is it ceremonial / thanks / greeting / procedural / a personal aside with no
   public stakes? → **`unimportant`**.
2. Is its core proposition a subjective judgment, pure rhetoric, an aspiration,
   a promise, or a future prediction? → **`opinion`**.
3. Does it assert something specific and verifiable against evidence (a number,
   an event, a comparison, an attribution)? → **`check-worthy`**.
4. Otherwise → **`opinion`** (default for non-propositional rhetoric).

## Edge cases (decided, for consistency)
- **Fact wrapped in rhetoric** — label by the dominant *checkable* content. A
  specific verifiable stat/event inside a rhetorical sentence → `check-worthy`.
- **Superlatives**: "greatest economy ever" (no measurable referent) → `opinion`;
  "lowest murder rate in 125 years" (measurable) → `check-worthy` (`comparison`).
- **Future tense**: "we will build…", "we are going to win" → `opinion`
  (promise/prediction). A statement about something *already done* → `check-worthy`.
- **Attribution**: "the last administration did X" → `check-worthy` if X is a
  verifiable act, `opinion` if X is a characterization ("did a terrible job").
- **Vague magnitudes**: "millions and millions poured across" — if it asserts a
  real, checkable magnitude → `check-worthy` with `confidence: low`; if purely
  emphatic → `opinion`.
- **Guest intro / "joining us tonight is X"** → `unimportant`, unless it asserts
  a checkable fact about the person (a specific act/record) → `check-worthy`.
- **Quotes / self-quotes** → judge the propositional content being asserted.
- **Compound sentence, two clauses of different classes** → take the higher tier
  in the order check-worthy > opinion > unimportant, and note it.

## Per-sentence output schema
```json
{
  "sid": "trump_2026:0042",
  "label": "check-worthy | opinion | unimportant",
  "claim_type": "statistical|historical|attribution|comparison|other|null",
  "confidence": "high|med|low",
  "rationale": "one clause explaining the call",
  "edge_case": "optional: name the edge-case rule applied, else null"
}
```

## Provenance
Sentences segmented deterministically by `scorer/segment_sotu.py` from the
Historical-SOTU-Transcripts corpus (Miller Center / UVA). Primary source:
`trump_2026_sotu.txt` (TB-00 SOTU 2026 material). Secondary: `biden_2022_sotu.txt`
for speaker/style diversity. Audience stage directions ([Applause], etc.) were
stripped at segmentation.
