# PCA provenance preservation + reconciled-judge rendering

**Status:** in progress (spec + persistence layer). Author: ccagent (for Jackie).
**Branch:** `claude/pca-provenance`. Follow-up to PR-C (#36, PCA wired into publisher)
and PR-D (#37, first live PCA SOTU publish).

Merges three backlog items that share one root cause and one fix surface:

1. **Single-judge vs multi-model UI collision** (top priority)
2. **Provenance dropped** (Layer A label, per-seat votes, CRM-114 stages not structurally recorded)
3. **Runs are not replayable** (latent third gap, surfaced while speccing #1/#2)

## Root cause

The PCA lane produces a rich per-claim adjudication `row`, but
`bridge._consensus_and_panel` (`src/truthbot/verdict/bridge.py:182`) collapses it into a
`ConsensusVerdict` that keeps only two derived scalars — `agreement: bool` and
`consensus_strength: str` — plus 0–1 reconciled `ModelVerdict` cards. Everything else
(`votes`, `split`, `escalated`, structured `crm114`, and the upstream Layer A label) is
discarded at that boundary and persisted nowhere.

### Symptom A — UI collision

The publisher speaks legacy-panel vocab. `site.py` computes
`total_models = len(bundle.model_verdicts)` (`~1964`) and renders
`"{agreeing} of {total_models} agree"` (`~2026`). Under PCA that is:

- **"1 of 1 agree"** on the 274 resolved claims (vacuous), and
- **"0 of 0"** on the 7 "Models split" claims, which render with an **empty model strip**
  (0 model cards).

### Symptom B — provenance dropped

Proven empirically (2026-07-16): after the live SOTU runs, the per-seat vote tally was
**unrecoverable** — the disk caches (`truthbot_cache/{cache,bundles/cache}.db`) were empty,
`metrics/adapter_calls.jsonl` logs a different (legacy gemini/xai) pipeline with synthetic
IDs, and no bundle store exists. Only the report-level `model_agreement_rate` scalar and
per-claim `consensus_strength` survived into `site-pca/data/`.

### Symptom C — no replayability

`run_pca_verify` bridges rows in-memory and returns bundles; the raw adjudication rows are
never written to disk. A re-publish (e.g. to validate a render change) therefore requires a
**fresh live run** (~1hr, real API spend). Contrast the legacy path, which persists extracted
claims via `_persist_extracted_claims` precisely so a failed downstream step costs nothing to
retry.

## Empirical findings that motivate this (2026-07-16 SOTU data, n=281)

| Finding | Value |
|---|---|
| Reconstructed unanimous-label claims (from frozen `model_agreement_rate`) | Trump ~117/168 (0.696), Biden ~99/113 (0.876) |
| Non-unanimous claims (≥1 dissenting seat) | ~65 combined — **far more than** the 7 "Models split" |
| `consensus_strength` distribution | `weak` 274 · `none` 7 · **`strong` 0 · `single` 0** |
| Claims with 0 model cards | 7 — exactly the 7 "Models split" claims |
| Lenient vs strict coarse label divergence | 0 (projection is currently a no-op on this data) |

Two independent "disagreement" metrics that don't align: `agreement` = *all seats voted the
identical label*; `consensus_strength = weak` = *the top label got exactly 2 votes* (regardless
of dissent). ~65 claims are simultaneously `weak` and non-unanimous. The **zero `strong`**
result (top label never reached 3 concurring seats) is the fingerprint of the
`strength_from_votes` weak-on-everything calibration issue — but it can only be *diagnosed*
once `votes` is persisted, which is why calibration is a follow-up, not part of this change.

## Decision: Option (b) — reconciled-judge mode + structured provenance

- **Option (a) — bridge synthesizes N `ModelVerdict`s (one per seat).** *Rejected.* The row
  carries only a vote *tally* (`{label: count}`), not per-seat reasoning/sources. Fabricating
  N cards would invent explanations and citations that don't exist and misrepresent PCA seats
  as the legacy named-model panel. It "fixes" the N-of-M render by lying.
- **Option (b) — publisher reconciled-judge mode + pipeline provenance.** *Chosen.* Honest to
  what PCA is: one reconciled judge, backed by a vote tally and a provenance chain. Nothing
  fabricated.

## Design

### 1. Data model — structured provenance (`models.py`)

New default-empty nested model on `ConsensusVerdict` (same legacy-clean pattern as the existing
`coarse_*` fields; old bundles deserialize with empty defaults, renderer falls back):

```python
class VerdictProvenance(BaseModel):
    layer_a_label: str = ""            # check-worthy routing label
    layer_a_source: str = ""           # "A1" | "A2"
    panel_votes: dict[str, int] = {}   # {"True": 2, "Misleading": 1}
    panel_split: bool = False
    panel_escalated: bool = False
    crm114_stage1: str = ""            # was only interpolated into explanation text
    crm114_final: str = ""
```
`ConsensusVerdict.provenance: VerdictProvenance = Field(default_factory=VerdictProvenance)`

### 2. Bridge — populate instead of discard (`bridge.py`)

`_consensus_and_panel` already reads `votes = row.get("votes")`. Extend it to build a
`VerdictProvenance` from `row["votes"]`, `row["split"]`, `row["escalated"]`, the structured
`row["crm114"]` (`{stage1, final}` — already parsed, currently only interpolated into `expl`),
and the Layer A label threaded from the claim dict. Populate `consensus.provenance` in **both**
the resolved and non-resolved (split / no_label) branches. The reconciled `ModelVerdict` is
unchanged.

### 3. Threading + persistence (`publish_pipeline.py`, `pipeline.py`)

- `run_pca_verify` enriches each claim dict with `layer_a` provenance pulled from the
  check-worthy queue rows, then passes it through the bridge.
- `PcaVerifyResult` gains `rows` (raw adjudication rows) + `claims` so the orchestrator can
  persist them.
- `_run_publish_pca` writes a replay artifact (`metrics/pca_runs/<run_id>.json`) holding
  `{rows, claims, characterization}` — the minimum to re-bridge and re-publish **without LLM
  spend**. Mirrors `_persist_extracted_claims`.

### 4. Reporting — serialize provenance into `claims.json` (`site.py` `_claim_meta`)

Emit `panel_votes`, `split`, `layer_a_label`, `crm114_*` alongside `model_verdicts_summary`, so
per-claim agreement is reconstructable from published data (the thing that was unrecoverable on
2026-07-16).

### 5. Publisher — reconciled-judge render mode (`site.py`) — **DONE**

Detect PCA mode (`_is_pca_bundle`: `len(model_verdicts) <= 1 and provenance.panel_votes`).
In that mode:
- Replaced `"N of M agree"` with a panel-vote line: **"2 of 3 seats agree"** for resolved,
  **"Panel split — False ×1, True ×1"** for splits (previously blank).
- Added a provenance strip under the header:
  **`Layer A: check-worthy (A2) → PCA panel: Misleading ×2, False ×1 → CRM-114: MISLEADING→FALSE`**.
- Split claims (zero cards) now render a "No single verdict — panel did not converge"
  placeholder instead of an empty grid.
- Legacy N-of-M path unchanged when `len(model_verdicts) >= 2` (empty provenance).

### 6. Adjacent bug — report distribution folds "Models split" into "Unverifiable"

`verdict_distribution` reports `Unverifiable 29 / split 0` for Trump (actual `24 + 5`). Give
"Models split" its own bucket in the 6-bucket distribution (the lenient/strict dists already
keep it separate).

## Out of scope (follow-ups)

- **`strength_from_votes` recalibration** — prerequisite (persisted `votes`) lands here; the
  recalibration is separate.
- **Biden severity-softening** (85% True) — unrelated verdict-quality thread.

## Validation

- Unit: bridge populates `VerdictProvenance` for resolved / split / escalated / CRM-flip rows;
  legacy bundle without provenance deserializes clean and renders via fallback.
- Serialization: `claims.json` carries `panel_votes` + `layer_a_label`; split claims carry a
  non-empty vote tally.
- Replay: `metrics/pca_runs/<run_id>.json` round-trips through the bridge to identical bundles.
- Full suite stays green.
