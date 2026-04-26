---
owner: truth-bot
created: 2026-04-24
status: active
---

# v-p1-p2 Calibration — Follow-ups Pending OpenAI Batch

**Run ID:** `ed7be4ad-3f2e-4010-a674-be2f8a17589e`
**OpenAI batch ID:** `batch_69eba50283648190b919495f69366411`
**Submitted:** 2026-04-24 13:14 EDT
**Observed status as of 2026-04-24 15:35 EDT:** `in_progress` — `0 / 2` completed

Other three providers finished inside ~3 minutes; OpenAI is the outlier, consistent
with the small-batch off-peak latency pattern we've seen.

## Must-do once the batch lands

1. **Reconcile + publish.**
   `truthbot batch reconcile ed7be4ad-3f2e-4010-a674-be2f8a17589e --publish --site-root site-test`

2. **Inspect OpenAI verdicts for temporal grounding quality.** Specifically
   verify the OpenAI path produced 2025-2026-anchored reasoning, matching the
   Anthropic/Gemini/xAI quality confirmed from the sidecar.

3. **Verify Phase 2b telemetry on OpenAI batch rows.** Read
   `metrics/adapter_calls.jsonl` rows tagged with this run_id: `tool_call_count`
   must now be populated (non-zero) for the OpenAI batch verdicts. If zero,
   the `web_search_call` counting in `parse_multi_batch_response` isn't firing
   and we have a bug.

4. **Record the end-to-end OpenAI batch elapsed time.** This is the empirical
   signal for Phase 2.5b's decision rule: if median elapsed > ~30min for small
   batches, we promote Phase 3a (live Responses API) from CONDITIONAL to
   required.

5. **Re-run the refined validator (Phase 1c refinement) across the full
   4-adapter × all-claims sidecar.** The current sidecar was scanned
   pre-refinement. After OpenAI lands, re-scan all 12-14 verdicts with the
   claim-text-aware validator and record the final `temporal_flags` state.

6. **Decide Phase 3a go/no-go.** If the OpenAI batch also returned rich
   `web_search_call` invocations AND ran in a reasonable SLA, batch may
   actually be viable. If latency is the dominant issue, flip Phase 3a to
   required.

## Important caveat about the in-flight batch

- **The in-flight batch (`batch_69eba50283648190b919495f69366411`) still
  uses the legacy `web_search_preview` tool.** The GA swap (Phase 2.5a,
  landed 2026-04-24 15:40 EDT) only affects future submissions.
- Therefore: the Phase 2.5b **capability test** (does OpenAI batch actually
  invoke web_search during a batch run?) must be re-submitted AFTER this
  in-flight batch terminates, using the updated payload with
  `{"type": "web_search"}`.
- The in-flight batch can still tell us: (a) small-batch SLA,
  (b) whether `_preview` variant fires tool calls in batch mode at all.

## Learnings already captured (do not re-derive)

- **Phase 1 temporal preamble** is in the real OpenAI batch payload: verified
  in `metrics/batch_inputs/openai-ed7be4ad-…jsonl` — preamble leads every
  `input[1]` user content block with `TEMPORAL CONTEXT (authoritative — …)`
  plus the 47th/2nd-term anchor.
- **Phase 2a (`gpt-5.4`)** is the `model` field in the submitted batch body.
- **Phase 2b tool-call telemetry** is non-zero for Anthropic / Gemini / xAI
  (7, 16, 23 invocations respectively across the 6 API calls). Pre-Phase-2b
  the batch path hardcoded 0.
- **Phase 1c validator false-positive rate = 5/12** on raw pre-refinement
  heuristic. 0/12 after the claim-text-aware refinement landed.
- **Anthropic opus-4-7 verdicts** all cite 2025-2026 evidence
  (BEA Feb 2026, BLS Dec 2025 core CPI, CCJ 2025 year-end homicide,
  CBP FY2026 Q1 border encounters). Strong temporal grounding.
- **Intra-provider dissent observed on claim `a11e4bdb`** (3-month
  annualized core inflation 1.7% in last 3 months of 2025):
  Gemini → False, xAI → True, both citing 2025 BLS data, zero temporal
  flags. Real interpretation disagreement on "annualized" term — NOT a
  temporal issue. This is the exact pattern Phase 3c (family-aware
  consensus + source-tier weighting) is designed to resolve.

## Phase 3b (URL reachability) findings — added 2026-04-24

Module landed: `src/truthbot/verify/url_validation.py` + CLI
`truthbot urls check <sidecar>`. Tested against the 45 unique URLs
cited by the 12 non-OpenAI sidecar verdicts:

| Classification | Count | % | Notes |
|---|---|---|---|
| `ok` | 21 | 47% | HEAD/GET → 2xx |
| `bot-blocked` | 9 | 20% | 403 from trusted domain (bls.gov, bbc.com, etc.) — likely real |
| `dead-4xx` | 7 | 16% | Real hallucinations (404 path on real domain) |
| `unknown` | 3 | 7% | All `vertexaisearch.cloud.google.com/grounding-api-redirect/...` — Gemini session URLs, not durable citations |
| `dns` | 2 | 4% | `counciloncriminaljustice.org`, `930wfmd.com` (may be sandbox-specific) |
| `transient` | 1 | 2% | Timeout |
| `cert-error` | 1 | 2% | `jeffa.net` hostname mismatch |
| `malformed` | 1 | 2% | `httpshttps://www.ebc.com/...` — double-scheme concat bug |

**"Likely real" total: 30/45 (67%)**
**"Almost certainly hallucinated or dead": 11/45 (24%)**

### New follow-ups driven by Phase 3b data

A. **Malformed URL concatenation bug.** The `httpshttps://www.ebc.com/...`
   output suggests either an adapter-side concatenation bug or a bad
   model output that we should sanitize in
   `verify/adapters/<provider>.py` URL extraction. Add a pre-persist
   regex that rejects any source URL matching `r"^https?[^:/]*https?://"`.

B. **Gemini grounding-redirect URLs are useless as citations.** The
   `vertexaisearch.cloud.google.com/grounding-api-redirect/...` URLs are
   Gemini's internal API-session redirects; they 403 outside the session
   and are not durable citation targets. Options:
     1. Resolve them via the Gemini grounding metadata to get the real
        destination URL before storing in `web_sources`.
     2. Strip them entirely in `verify/adapters/gemini.py` and rely on
        the `grounding_chunks` field's actual source URIs.
   Recommend option 2; open an item under Phase 3c prep.

C. **Publish-layer rendering (Phase 3b wiring).** Currently
   `url_validation.annotate_verdicts` is a library only. The publish
   path (`src/truthbot/publish/site.py::_evidence_list_html`) should:
     * For `ok` URLs: render as today.
     * For `bot-blocked`: render as today but add a small "WAF·bot-
       blocked — click to verify manually" annotation.
     * For `dead-4xx`/`malformed`/`dns`/`cert-error`: render with
       strike-through + "dead link — reported by model but unreachable".
     * For `transient`/`unknown`: render as today (don't penalize flaky
       infra / novel failures).
   Not wired yet — decision deferred until the full 117-claim SOTU rerun
   so we have a larger sample to tune thresholds. Tracked as `p3b-publish-wire`.

D. **User-Agent rotation for `.gov` HEAD checks.** 7 of the 9
   bot-blocks are `.bls.gov`. A browser-like UA (Chrome) typically
   gets through. If the bot-block rate on the full 117-claim rerun
   climbs too high, add a secondary GET with a Chrome UA before
   classifying as bot-blocked. Low priority; current classification
   is already correct in spirit.
