# Aggregation Pages — Scope & Design Document

> **Status:** Design/scoping only. No implementation code included.  
> **Audience:** Pat (product decisions required throughout).  
> **Produced by:** truth-bot pipeline round 6.

---

## Data Model Questions

### Does the current Report schema have fields for `speaker_id` and `speech_type`?

**No.** The current `Transcript` model (and the `SiteReport` dataclass derived from it) has:

| Field | Type | Notes |
|-------|------|-------|
| `speaker` | `str` | Free-form name string, e.g. `"Donald Trump"` |
| `venue` | `str \| None` | Free-form, e.g. `"U.S. Capitol, State of the Union"` |
| `metadata` | `dict` | Unstructured; anything can be stuffed here |

There is no `speaker_id`, no `speech_type` enum, no canonical speaker slug, and no taxonomy for speech type.

**Additions needed for aggregation (minimum):**

- `Transcript.speaker_id: Optional[str]` — stable canonical ID for a speaker
- `Transcript.speech_type: Optional[str]` — category label for the speech
- `SiteReport.speaker_id: Optional[str]` — propagated through to the site layer
- `reports.json` entries would need both fields stored so `regen_site.py` can use them without re-running the pipeline

---

### How should speakers be normalized?

The same person can appear as "Donald Trump", "Donald J. Trump", "President Trump", "Trump", "POTUS", etc. Three options:

**Option A — Exact string match (no normalization)**

- Accept only the canonical display name at ingest time; enforce it via validation.
- Tradeoff: simple, zero ambiguity, but requires whoever adds a transcript to enter the name correctly every time. Typos fragment the speaker page silently.

**Option B — Slug-based deduplication (normalize at slug generation)**

- Generate a URL slug from the speaker string (strip titles/suffixes, lowercased, hyphens). Two different name strings that produce the same slug are treated as the same speaker.
- Tradeoff: low setup cost, survives minor name variants. Fails on genuine collisions (two people with similar names) and on cases where the full name is needed but only a short form was stored. "President Trump" -> `president-trump`, not `donald-trump`.

**Option C — Explicit speaker entity registry**

- Maintain a `data/speakers.json` or similar mapping: `{ "donald-trump": { "display_name": "Donald Trump", "aliases": ["Donald J. Trump", "President Trump", "POTUS 47"] } }`.
- Ingest or a post-processing step resolves raw speaker strings against the registry and writes `speaker_id` onto each transcript/report.
- Tradeoff: most robust and most maintainable long-term. Requires upfront authoring of the registry and a resolution step. Aliases list must be kept current. **Highest implementation cost of the three.**

---

### What speech-type taxonomy exists?

**None currently.** The `venue` field is free-form and conflates location with occasion (e.g., "U.S. Capitol, State of the Union").

Three taxonomy options:

**Option A — Fixed enum**

Examples: `state_of_the_union`, `press_conference`, `campaign_rally`, `congressional_hearing`, `tweet`, `interview`, `floor_speech`, `executive_order_signing`.

- Tradeoff: predictable, easy to filter and display. Requires agreement on the full list upfront. Adding a new type later is a schema migration. Works well if corpus stays focused on a small set of speech forms.

**Option B — Free-form tags**

Each report can have `speech_type: list[str]`, e.g. `["state-of-the-union", "joint-session"]`.

- Tradeoff: flexible, no migration needed for new forms. Tag proliferation is a real risk (`sotu` vs `state-of-the-union` vs `state_of_the_union`). Aggregation queries become fuzzy. Tagging discipline required from whoever adds transcripts.

**Option C — Hierarchical categories**

Two-level taxonomy: category -> subcategory. E.g., `Executive / State of the Union`, `Legislative / Floor Speech`, `Campaign / Rally`, `Social Media / Tweet`.

- Tradeoff: expressible nuance (a campaign rally is different from a White House press conference even though both are "presidential"). Higher UI complexity — aggregation pages need to decide whether to show at category level or subcategory level, or both. Most future-proof for a growing corpus.

---

### Is the current `report.id` sufficient as an aggregation key?

**Yes for per-report disambiguation; no for aggregation.**

Each `report.id` is a UUID — unique per run. For speaker and speech-type aggregation pages, we need to query *across* reports, which means we need a stable `speaker_id` (and/or `speech_type` tag) stored in `reports.json` and readable without re-running the pipeline.

The current `reports.json` already stores `speaker` (string) and `venue` (string), so a slug-based approach (Option B above) could work without adding new fields — at the cost of the ambiguity described there.

---

## URL Structure

### Options for speaker pages

**Option 1 — `/speakers/donald-trump.html`**

- Tradeoff: flat, easy to generate, simple links. If we later want `/speakers/` as a directory listing, the file conflicts with the directory name on some servers (NGINX handles it, but static hosts vary). Works cleanly on GitHub Pages and most static hosts.

**Option 2 — `/speakers/donald-trump/index.html`**

- Tradeoff: the "clean URL" pattern; `/speakers/donald-trump/` works without a trailing `.html`. Requires every speaker to be a subdirectory. Generator creates more directories. Better for SEO canonical URLs. Slightly more complex build logic.

**Option 3 — `/people/donald-trump.html`**

- Tradeoff: `/people/` is a more neutral namespace than `/speakers/` (a senator gives speeches, but so does an executive branch official, a corporate figure, etc.). Semantic choice only; tradeoffs otherwise same as Option 1.

---

### Slug generation rules

For names with edge cases:

| Input | Suggested slug | Notes |
|-------|---------------|-------|
| `Donald J. Trump` | `donald-trump` | Strip middle initial + period |
| `Ted Cruz` | `ted-cruz` | Simple |
| `O'Brien` | `obrien` | Strip apostrophe |
| `Angel Garcia` (with diacritics) | `angel-garcia` | NFKD normalize + strip diacritics |
| `Martin Luther King Jr.` | `martin-luther-king-jr` | Keep "jr" (strip period); avoids collision with MLK Sr. |
| `William Bradford Jr.` vs `William Bradford Sr.` | `william-bradford-jr` / `william-bradford-sr` | Suffix must be preserved |

**Decisions for Pat:**
- Should middle initials be stripped or kept? (Affects collision between e.g. "John F. Kennedy" and a hypothetical "John Kennedy" in the corpus.)
- How to handle the "Unknown" speaker (current pipeline default)? Suppress aggregation page, or create a `/speakers/unknown.html` catch-all?

For speech types (if using slugs):

| Input | Suggested slug |
|-------|---------------|
| `State of the Union` | `state-of-the-union` |
| `Press Conference` | `press-conference` |
| `Tweet` | `tweet` |
| `Q&A` | `q-and-a` |

---

## UI Scoping

### Discovery: how do users find aggregation pages?

**Option A — Top-nav entry only** ("Speakers" tab in the masthead nav)

- Tradeoff: always visible, authoritative. Nav becomes cluttered as more aggregation dimensions are added (speakers + speech types + date ranges). Currently the nav has 3 links; adding "Speakers" keeps it manageable.

**Option B — Linked from each report card only**

- Tradeoff: contextual discovery — user finds Trump's report, clicks his name to see all Trump reports. Requires no nav change. Users who want to browse by speaker have no obvious entry point unless they've already seen a report.

**Option C — Both**

- Tradeoff: highest discoverability. Requires nav + report card link + the aggregation index page (a listing of all speakers). Most UI surface area to maintain.

**Decision for Pat:** Which discovery model matches the audience's browsing behavior? Are users arriving via direct links to reports (social media) or browsing the site organically?

---

### `data-claim-count` on aggregation pages

The Truthy widget reads `data-claim-count` to decide singular vs. plural phrasing. On aggregation pages, options:

**Option A — Total claims across all reports for this speaker/type**

- E.g., Trump's page: 47 claims across 3 reports -> `data-claim-count="47"`
- Truthy's bubble uses multi-claim phrasing. Makes sense as a cumulative score.

**Option B — Count of reports**

- E.g., Trump's page: 3 reports -> `data-claim-count="3"`
- Truthy's mood reflects the aggregate of all reports. Count reflects reports, not claims. Slightly confusing given the widget's existing semantics.

**Option C — Omit Truthy from aggregation pages**

- Truthy is designed for a single-speech verdict. Aggregating across speeches strains the metaphor. Could omit Truthy and show just the verdict bar + stats panel.
- Tradeoff: simpler, avoids semantic confusion, but loses visual consistency.

**Decision for Pat:** Does Truthy belong on aggregation pages? If yes, should the mood reflect the speaker's overall truthfulness across all speeches (Option A) or the most recent speech only?

---

### Aggregate verdict distribution visualization

**Option A — Per-report bars (one bar per report, stacked vertically)**

```
Report 1  [################]  March 2026 SOTU
Report 2  [################]  April 2026 Press Conf
Report 3  [################]  May 2026 Rally
```
(Where each bar is color-coded by verdict categories.)

- Tradeoff: shows trend over time; user can see if a speaker is getting more or less accurate. More vertical space. Harder to summarize at a glance.

**Option B — Cumulative bar (one combined bar)**

```
All speeches  [################]  22 True / 8 False / 5 Misleading
```

- Tradeoff: compact, scannable at a glance. Loses temporal information. Two outlier speeches can wash out trend data.

Both options are not mutually exclusive — could show the cumulative bar at the top and per-report breakdown below. **Decision for Pat:** Which is the primary unit of information on a speaker page?

---

### Speech-type pages with multiple speakers

If a speech-type page (e.g., `/speech-types/state-of-the-union.html`) lists multiple presidents who gave SOTUs, options:

**Option A — Per-speaker breakdown**

Each speaker gets their own row with their own verdict bar, sorted by some criteria (date, most claims, alphabetical). Clean, readable. Tall page if many speakers.

**Option B — Combined stats with speaker color-coding**

A single stacked bar where each segment is colored by both verdict and speaker. Adds a second color dimension; requires a speaker color palette in addition to the verdict palette. Note: the current design doc states verdict colors are "the ONLY chromatic colors" — this would require revisiting that constraint.

**Option C — Tabbed view (one tab per speaker)**

JS-dependent. Adds interactivity but breaks the current "no JS for content" assumption (Truthy already requires JS for audio/animations, but adding tab logic is a new category of JS dependency for core content navigation).

**Decision for Pat:** Should speech-type pages primarily answer "how truthful is this speech form across speakers?" (Option B) or "how does each speaker do in this speech form?" (Option A)?

---

## Implementation Sketch (text only, no code)

### Python modules needing changes

1. **`src/truthbot/models.py`** — Add `speaker_id: Optional[str]` and `speech_type: Optional[str]` to `Transcript`. Potentially add a `SpeakerEntity` model if using Option C normalization.

2. **`src/truthbot/publish/site.py`** — Add:
   - `_render_speaker_page(speaker_id, reports)` function
   - `_render_speech_type_page(speech_type, reports)` function
   - A new `SitePublisher.publish_aggregations()` method that iterates all reports in the index, groups by speaker and speech type, and writes aggregation HTML files
   - Extend `_ensure_structure()` to create `/speakers/` and `/speech-types/` directories
   - Extend `_report_meta()` to write `speaker_id` and `speech_type` into `reports.json`

3. **`src/truthbot/publish/assets/styles.css` / `CSS` constant in site.py** — New CSS for aggregation page layout (speaker header, report list, aggregate bar). Likely a new section added after current section [22].

4. **`regen_site.py`** — Add a call to `publisher.publish_aggregations()` after all individual reports are regenerated.

5. **`src/truthbot/ingest/transcript.py`** (possibly) — If speaker normalization is done at ingest, this module handles it. If post-hoc, a new `normalize.py` utility.

### New generator flow (top-down)

1. `regen_site.py` loads `reports.json`
2. For each report: `publisher.publish(site_report)` (existing, unchanged)
3. After all reports: `publisher.publish_aggregations(reports_index)`
   - Group reports by `speaker_id`
   - For each speaker group: `_render_speaker_page()` -> write to `/speakers/{slug}.html`
   - Group reports by `speech_type`
   - For each speech-type group: `_render_speech_type_page()` -> write to `/speech-types/{slug}.html`
   - Regenerate `index.html` with links to aggregation pages (if nav Option A or C chosen)

### Estimated scope and risk

| Sub-task | Scope | Risk |
|----------|-------|------|
| Add `speaker_id` / `speech_type` to models | Small | Low — backward-compatible optional fields |
| Backfill existing `reports.json` entries | Small | Medium — manual data entry, errors fragment pages silently |
| Speaker normalization (Option C registry) | Medium | Medium — registry maintenance burden grows with corpus |
| `_render_speaker_page()` template | Medium | Low — similar pattern to existing report rendering |
| `_render_speech_type_page()` template | Medium | Low |
| Aggregation index page (`/speakers/index.html`) | Small | Low |
| Nav changes + responsive | Small | Low |
| Truthy on aggregation pages | Small-Medium | Medium — semantic mismatch; needs product decision first |
| Per-report timeline visualization | Large | High — new chart type, no existing pattern in CSS |

**Overall scope: Medium.** Largest risk is data hygiene (speaker normalization) rather than code complexity. The code is straightforward static generation; the hard part is deciding on the data model and keeping `reports.json` consistent as the corpus grows.

---

## Open Questions for Pat

1. **Speaker normalization strategy** (Options A/B/C above) — which approach matches your operational workflow? Are you comfortable maintaining a speaker registry (`data/speakers.json`), or would slug-based deduplication suffice for the current corpus size?

2. **Speech-type taxonomy** — fixed enum, free-form tags, or hierarchical? If fixed enum, what is the complete list for the current and near-future corpus?

3. **Discovery model** — top-nav, report card links, or both? Is the primary use case organic browsing or deep-link sharing?

4. **Truthy on aggregation pages** — include or omit? If include, which `data-claim-count` semantics?

5. **Verdict visualization on aggregation pages** — cumulative bar, per-report bars, or both?

6. **URL namespace** — `/speakers/` vs `/people/`? And flat `.html` vs directory-style `/index.html`?

7. **Unknown speaker handling** — the current pipeline defaults `speaker` to `"Unknown"`. Should transcripts with no speaker be excluded from aggregation, or collected in a catch-all page?

8. **Backfill priority** — should existing reports be backfilled with `speaker_id` and `speech_type` before implementation begins, or will aggregation pages go live with partial data and fill in over time?
