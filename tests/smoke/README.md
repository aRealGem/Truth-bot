# Live smoke suites

These tests spend **real money** against provider APIs. They are marked
`@pytest.mark.live` and excluded from default `pytest` invocations via
the `addopts = "-m 'not live'"` line in [`pyproject.toml`](../../pyproject.toml).

There are two variants of the smoke suite:

- **2-claim smoke** (original): minimal round-trip with a single chunk
  on the batch providers. Proves everything works end-to-end; does
  **not** exercise the multi-chunk path.
- **Paginated 5-claim smoke**: uses 5 trivial claims at
  `claims_per_request=2` so batch providers produce **3 chunks
  (2 + 2 + 1)**. Validates multi-chunk submit, reconcile aggregation
  across multiple `batch_call_id`s, and per-chunk `custom_id` uniqueness.

Run the 2-claim smoke:

```bash
# Phase A: submit (and, for live providers, complete). ~2-3 min.
pytest tests/smoke/test_smoke_submit.py -m live -v

# Phase B: poll + reconcile the two batch providers. Up to 2.5h.
pytest tests/smoke/test_smoke_reconcile.py -m live -v
```

Run the paginated smoke (safely coexists with the 2-claim smoke on disk
via the `*_pg` manifest keys — both can be run in the same tree):

```bash
# Phase A: submit (xAI + Gemini complete live with 5 calls each). ~3-4 min.
pytest tests/smoke/test_smoke_submit_paginated.py -m live -v

# Phase B: poll + reconcile the two batch providers' 3-chunk jobs.
pytest tests/smoke/test_smoke_reconcile_paginated.py -m live -v
```

Cost per full run:

- 2-claim:  **<$0.10** across all four providers.
- Paginated 5-claim: **~$0.20-0.30**, dominated by xAI's 5 sequential
  live calls (~$0.032 each). Anthropic and OpenAI stay cheap even with
  3 chunks because the system prompt cache amortizes across claims in
  each chunk.

## Architecture

Two pairs of pytest files (2-claim + paginated) that talk to each other
through a manifest on disk:

```
metrics/smoke/
  manifest.json         # per-provider: run_id, batch_id, status, verdicts
  smoke_summary.jsonl   # one row per completed verify (timing + cost)
```

The 2-claim smoke uses manifest keys `anthropic` / `openai` / `xai` /
`gemini`. The paginated smoke uses `*_pg`-suffixed keys
(`anthropic_pg`, `openai_pg`, `xai_pg`, `gemini_pg`) so the two suites
coexist without state collision.

`test_smoke_submit.py` (2-claim):
- Anthropic / OpenAI: submit a 2-claim batch (`claims_per_request=2`,
  collapses to 1 chunk) and return as soon as the batch_id is stamped.
  Writes the manifest entry. Does NOT wait for batch completion.
- xAI / Gemini: run live (no batch API exists for these). Completes
  within the test, asserts verdicts against ground truth, writes manifest.

`test_smoke_reconcile.py` (2-claim):
- Reads the manifest for `anthropic` / `openai` entries.
- Polls with an SLA-driven cadence (60s early, 2min middle, 5min late).
- On completion, calls `reconcile_run` and asserts verdicts.
- If the manifest entry is missing, the test is SKIPPED (not failed) —
  reconcile-only runs with a stale tree are therefore legal.

`test_smoke_submit_paginated.py`:
- Anthropic / OpenAI: submit 5 claims at `claims_per_request=2`, asserting
  `chunk_size=2, request_count=3`. Writes manifest under `anthropic_pg`
  / `openai_pg`.
- xAI / Gemini: 5 sequential live calls each, wrapped in a single
  `telemetry_run_context` so all 5 rows share one `run_id`. Asserts
  labels match the truth pattern (strict for xAI; Gemini allows 1 flake).

`test_smoke_reconcile_paginated.py`:
- Reads `anthropic_pg` / `openai_pg` entries and calls the same
  poll + reconcile helper as the 2-claim smoke
  (`tests.smoke.conftest._run_reconcile_n`).
- Asserts `len(bundles) == 5` and each claim's bundle has at least one
  label matching its truth polarity.

## Ground truth

Two claims with opposite, unambiguous truth values used by the 2-claim smoke:

- **TRUE**:  "The United States landed astronauts on the Moon in 1969."
- **FALSE**: "The Eiffel Tower is located in Berlin, Germany."

The paginated smoke adds three more equally-unambiguous claims (defined
in `conftest.CLAIM_EXTRAS`):

- **TRUE**:  "Water boils at 100 degrees Celsius at standard atmospheric pressure."
- **FALSE**: "The Great Wall of China is visible from the Moon with the naked eye."
- **TRUE**:  "The Pacific Ocean is the largest ocean on Earth."

Any reasonable fact-checker should label these correctly on the first
try. Tests assert "True-ish" (`TRUE` or `MOSTLY_TRUE`) and "False-ish"
(`FALSE`, `MISLEADING`, or `EXAGGERATED`) respectively — not exact
label matches, because consensus adapters have legitimate label
latitude for edge-ish cases.

## Provider SLAs (2026-04)

| Provider  | Endpoint type   | Vendor SLA                                       | Observed (tiny jobs) | Smoke default cap |
| --------- | --------------- | ------------------------------------------------ | -------------------- | ----------------- |
| Anthropic | Message Batches | "Most batches complete within an hour"; 24h hard expiry | 5-60 min          | 2.5 h             |
| OpenAI    | Batch API       | `completion_window="24h"` is the only option; "often faster" | 5-30 min | 2.5 h             |
| xAI       | `/v1/responses` (live)  | Per-request, 10-30s each with tool calls       | 30-60s for 2 claims  | 3 min             |
| Gemini    | `generate_content` (live) | Per-request, 5-15s each with GoogleSearch    | 15-30s for 2 claims  | 3 min             |

## Automated watch cap vs. vendor expiry

The suite has two distinct time horizons:

1. **Automated cap (2.5 h).** If the reconcile test is still polling
   past 2.5 h, it fails with a clear message. This is our "longest
   reasonable" line: past that, continuing to poll on autopilot is
   more likely to mask a real problem than to yield a result.

2. **Vendor expiry (24 h).** Both Anthropic and OpenAI discard
   batches server-side 24 h after creation. Past that, the data
   is gone regardless of client tooling.

Between those two lines, the manifest stays on disk and you can
resume any time:

```bash
# Quick status check for the still-pending batch.
.venv/bin/truthbot batch poll <run_id>

# When complete, merge results + regenerate site.
.venv/bin/truthbot batch reconcile <run_id>
```

The `run_id` is in `metrics/smoke/manifest.json`, keyed by provider.

## Env overrides

All times are integer seconds. Prefix: `TRUTHBOT_SMOKE_TIMEOUT_`.

| Var                                    | Default  | Meaning                                   |
| -------------------------------------- | -------- | ----------------------------------------- |
| `TRUTHBOT_SMOKE_TIMEOUT_ANTHROPIC_BATCH` | `9000`  | Automated poll cap for Anthropic reconcile (2.5 h) |
| `TRUTHBOT_SMOKE_TIMEOUT_OPENAI_BATCH`   | `9000`  | Automated poll cap for OpenAI reconcile   (2.5 h) |
| `TRUTHBOT_SMOKE_TIMEOUT_XAI_LIVE`       | `180`   | Live call timeout for xAI (3 min)         |
| `TRUTHBOT_SMOKE_TIMEOUT_GEMINI_LIVE`    | `180`   | Live call timeout for Gemini (3 min)      |

Example — give OpenAI its full 24 h before giving up:

```bash
TRUTHBOT_SMOKE_TIMEOUT_OPENAI_BATCH=86400 \
  pytest tests/smoke/test_smoke_reconcile.py::TestReconcileOpenAIBatch -m live -v
```

## CI considerations

This suite is not intended to run on every PR. Budget-friendly options:

- Add a GitHub Actions workflow triggered by manual dispatch or a
  nightly cron, not on `push`.
- Use a dedicated "smoke budget" API key per provider with a strict
  monthly cap.
- The suite takes up to 2.5 h; GitHub-hosted runners have a 6 h limit,
  so the default cap leaves generous headroom.
