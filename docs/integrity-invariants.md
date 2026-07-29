# Engine-enforced integrity invariants (I1–I6)

**Status:** reference. Author: ccagent (for Jackie).
Companion to the wiki design note `projects:truthbot:claim-eval-v3`. Written while
scoping Claim Evaluation v3 tranche 1, which modifies an invariant-guarded field.

These are **hard guards, not config**. A YAML typo cannot defeat them, and the load-time
guards *fail* (raise) rather than warn. All six live in one module —
`hydramind/invariants.py` — so the rules cannot drift between call sites. Every guard
raises a subclass of `InvariantError`, each carrying a `code` attribute (`"I1"`…`"I6"`).

This document records what the code enforces **today**. It is descriptive, not
aspirational: if an invariant is only partially enforced, that is stated plainly.

## Summary

| # | Rule | Enforcement point | Guard |
|---|------|-------------------|-------|
| **I1** | grok never proposes or arbitrates (critic-only if present) | load time — `registry.load` | `check_i1_grok_pool` |
| **I2** | material tie ⇒ `disagreement_flagged` | runtime — `strategy.reduce` | `is_material_disagreement` / `is_escalation_split` |
| **I3** | no speaker/source conditioning anywhere | load time — spec keys *and* prompt templates | `check_i3_no_speaker_conditionals`, `lint_template_for_speaker_conditionals` |
| **I4** | verdict citations ⊆ evidence pack | runtime — verdict emit | `check_i4_citations` |
| **I5** | evidence carries provenance, enters via Layer C only | structural — evidence entry | `check_i5_provenance` |
| **I6** | heldout read once per release candidate | release — heldout access | `HeldoutGuard.read` |

## I1 — grok never proposes or arbitrates

`grok ∉ providers(proposer|arbiter)`. Grok may serve as a **critic** only.

Checked at `hydramind/registry.py:48`, during spec load. A registry that names grok in
either the proposer or arbiter provider pool fails to load — the process does not start
with a violating roster, so this cannot be tripped mid-run.

## I2 — material disagreement must be flagged

Unlike the others, I2 is enforced *in* the strategies' `reduce()`. The helpers in
`invariants.py` are the single source of truth those call sites share, so the definition
of "material" cannot drift between strategies.

Two definitions exist, and the distinction matters:

- **`label_mismatch`** — the **decided PCA policy** (P96.2.1). Escalate if and only if the
  proposer and a critic disagree on the **label**. Confidence is *not* part of the trigger.
- **`material_disagreement`** — the legacy rule: label mismatch **OR** `|Δconfidence| ≥ threshold`.

The active criterion is named in `pca.yaml` (`escalation.criterion`), so the policy is
explicit in the spec and manifest rather than implicit in gate code. Valid values live in
`ESCALATION_CRITERIA` (`invariants.py:146`). Dispatched at `hydramind/strategies/pca.py:88`.

## I3 — no speaker or source conditioning

Nothing anywhere may branch on **who is being analyzed**. This is the invariant that makes
truth-bot's output defensible as non-partisan, and it is guarded twice:

1. **Spec keys** — `check_i3_no_speaker_conditionals` (`registry.py:43`) walks every key in
   the raw spec and rejects `speaker`, `per_source`, `source_id`, `by_speaker`,
   `persona_of_subject`, and similar.
2. **Prompt templates** — `lint_template_for_speaker_conditionals` runs at *import* of every
   module holding a template, so a violating prompt breaks the build, not a run:
   - `src/truthbot/checkworthy/classifier.py:72` — Layer A `A2_SYSTEM`
   - `src/truthbot/verdict/discriminator.py:55` — `CRM114_SYSTEM`
   - `src/truthbot/verdict/prompts.py:124` — every PCA seat prompt, looped by role

**Deliberate exception:** conditioning on *which model produced an output* is allowed
(model provenance, Principle 2). The regex keys on speaker/source vocabulary precisely so
model-provenance conditionals stay legal.

## I4 — citations ⊆ evidence pack

A verdict may only cite evidence that was actually in its pack. `check_i4_citations`
raises `I4CitationError` listing the unknown ids.

Enforced at `hydramind/strategies/pca.py:153`, against the item's `evidence_pack_ids`.

**Scope note:** `bridge.py:138` deliberately *skips* rather than raises on this condition —
the bridge is a display adapter, not an invariant checkpoint. The real gate is the
strategy. Do not add a second enforcement point in the bridge without deciding which one
owns the failure.

## I5 — evidence provenance

```python
_REQUIRED_PROVENANCE = ("url", "retrieved_at", "sha256", "tier")
```

Every evidence item must carry all four. The check is **falsy**, not merely present —
an empty string or `None` fails. Evidence enters only via Layer C.

Fails closed at pack entry, in both pack builders:

- `src/truthbot/verdict/evidence_pack.py:282` (v1)
- `src/truthbot/verdict/evidence_pack_v2.py:156` (v2)

### `tier` is provenance, not decoration

This is the part most easily forgotten. **`tier` is one of exactly four mandatory
provenance fields.** It is not a display hint and not a ranking convenience — it is part
of the integrity record that I5 refuses to let through when missing.

Two consequences for anyone changing tiering:

1. **Any new tier value must be truthy.** A tier whose value is `""` or `0` silently fails
   I5 for every item carrying it.
2. **The render layer currently recomputes tier instead of reading it.**
   `src/truthbot/publish/site.py:622` (`_tier_bucket()`) re-classifies URLs from scratch to
   draw its badges, ignoring the `source_tier` the pipeline computed and I5 validated. So
   the provenance record and the published badge are produced by two independent
   implementations and can disagree. Any change to tier assignment must update **both**, or
   the pipeline and the site will tell readers different things. Collapsing these onto one
   implementation is the correct long-term fix and is not yet done.

## I6 — heldout read once per release candidate

`HeldoutGuard` tracks `(dataset_id, rc_id)` pairs; a second read of the same heldout split
under the same release-candidate id raises `I6HeldoutReuseError`. This is what keeps a
heldout split honest — it cannot be quietly consulted twice while tuning toward it.

Used at `eval/benchmarks/run_g1.py:109`.

**Known limitation — enforcement is per-process.** `HeldoutGuard._seen` is an in-memory
instance set with no persistence. A fresh process constructs a fresh guard and the history
resets, so I6 constrains reuse *within* a run, not *across* runs. Treat it as a guard
against accidental double-reads in one pipeline, not as a durable seal on the heldout set.
Making it durable would require persisting the seen-pairs outside the process.

## Relationship to the `T`-numbered rules

`I`-numbers and `T`-numbers are different schemes and are easy to confuse.

- **`I1`–`I6`** — engine-level integrity invariants, defined here, enforced by
  `hydramind/invariants.py`, fail closed.
- **`T0.x`–`T3.x`** — truth-bot pipeline requirements (e.g. `T1.1` era window, `T2.1`
  fact-checker exclusion, `T2.4` evidence quota gate). These are product rules implemented
  across the verdict modules; they are not routed through `InvariantError`.

A `T`-rule can be relaxed by a product decision. An `I`-rule cannot be relaxed without an
explicit design change here.
