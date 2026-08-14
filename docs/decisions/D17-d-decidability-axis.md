# D17-d — decidability is RECORDED, not derived

**Status: schema landed, nothing ratified, nothing rendered.** The registry ships
with 128 rows at `provenance: desk` and **0 publishable**. Step 6 (owner
ratification) is what makes any of it visible. $0.

---

## 1. The ruling

Record `decidability` as a **first-class per-claim property** carrying its own
provenance. Do **not** derive it from `claim_type`, `claim_shape`, pack anatomy,
or any residual range.

**The axis — four values, naming the claim's relation to the public record:**

| value | meaning |
|---|---|
| `retrievable-pending-lane` | retrieval would settle it; the lane has not been run |
| `retrieved-insufficient` | retrieval ran and what came back did not qualify |
| `undecidable-from-public-record` | no public record reaches it — **requires a review trigger** |
| `needs-decomposition` | a checkable core is buried in a compound utterance |

**Provenance — four values, one publishable:** `desk`, `rule`, `model` are
working notes; only **`owner-ratified`** may reach a page.

## 2. Why not derive it

The D17-d probes tried, and failed measurably. All three routes are closed:

- **Utterance-form fields are anti-correlated with decidability.** Where
  `attribution` and `c-eval` committed at all, precision was **0.235** and
  **0.105** against a 0.633 majority-class prior — and *every* committed error
  ran one way, predicting "undecidable" for a documentable claim. A render keyed
  on them would stamp "cannot be verified" on 30 documentable claims.
- **Pack anatomy (R7) separates nothing.** `web-tier1` vs `substantive` differ by
  0.07 items and 0.09 proxy quota credits, and the tier signal is faintly
  *inverted*. The corpus is conditioned on gate rejection, so the discriminating
  anatomy is already flattened.
- **The desk's own classes are not recoverable:** `compound-split` 0 of 5,
  `series-core` 1 of 7.

Deriving "cannot be verified" from utterance shape is **ruling (d) relocated** —
admissibility keyed on the wrong axis — and the S-12 proxy pattern with
empirically broken generational assumptions. Hence: recorded, with provenance.

## 3. Three properties the code enforces

**Fail closed.** `publishable_entries()` returns only `owner-ratified`. A `desk`
row is stored and auditable but invisible to any render. This mirrors the wave-2
badge rule (no classification record → no badge) and is what stops ccagent's
judgement being published as the system's.

**Never says never.** `undecidable-from-public-record` **requires** a
`review_trigger` naming what would reopen the question; the loader rejects the
file otherwise. A fact-checker does not get to call a question permanently closed
without saying what would change its mind — `trump_2026:0153` carries its own
re-adjudication flag as the standing reminder.

**Undroppable.** The lookup is **keyed by sid in a registry**, not carried on an
object. `series_rows` vanished at render because three places rebuild a
`PackItem` from an `Evidence`; bundles come back with `speaker='Unknown'` because
the offline artifact path rebuilds claims as `{sid, text, context, layer_a}`. A
sid-keyed registry is immune to that whole class of bug, because nothing has to
carry it. Pinned by `test_lookup_survives_the_offline_artifact_path`.

## 4. What shipped

| file | role |
|---|---|
| `src/truthbot/publish/decidability.py` | schema, validation, fail-closed accessors — mirrors `publish/corrections.py` |
| `data/decidability.json` | system of record; 128 rows, all `desk`, 0 publishable |
| `scripts/seed_decidability_from_desk.py` | deterministic seeder from the desk pass |
| `tests/test_decidability_axis.py` | 20 tests |

Seeded distribution: `retrievable-pending-lane` 88 (81 web-tier1 + 7
series-core), `undecidable-from-public-record` 35, `needs-decomposition` 5.
`retrieved-insufficient` is **reserved and deliberately unseeded** — it describes
a pack that was retrieved and fell short, and the desk's whole point is that
these lanes were never run.

## 5. Explicitly deferred

- **No render wiring.** Step 5 keys only on the recorded axis + `provenance_code`
  — never on type, shape, residual ranges, or any derived feature. It cannot
  start before step 6, because with 0 publishable rows it would render nothing.
- **No derived subset ships.** The 4 claims where a rule happened to concur with
  the desk (`gwbush_2006:0033`, `trump_2026:0514`, `trump_2026:0153`,
  `biden_2022:0373`) go through step 6 as owner-ratified rows. A rule wrong 13 of
  its 17 fires adds ~zero evidential weight when it concurs; treating
  "structure + desk agree" as confidence is the correlated-error pattern this
  project already banned for the panel.
- **Splitting `attribution` (public-act vs private-state) and reshaping
  `c-eval` → `c-count`** is schema hygiene that improves retrieval routing and
  seeds the axis's first priors. It rides with the metered batch (it needs a
  Layer-A re-run), and **neither field ever carries the axis.**
