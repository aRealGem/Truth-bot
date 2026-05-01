# Desk trace: tool URLs vs visible `web_sources` (OpenAI + Gemini pattern)

This note documents the **post-processing path** that explains “empty visible sources” on two frontier adapters. Repository checkouts often omit `metrics/batch_sidecar/*.jsonl`; the pipeline behavior is pinned by unit tests.

## Data flow (multi-claim batch / live)

1. Adapter walks the Responses (or provider) envelope and collects **tool-retrieved URLs** (`tool_retrieved_urls` in [`build_multi_verdicts`](../../src/truthbot/verify/adapters/base.py)).
2. Model JSON may list extra URLs in `sources` / `evidence` fields — **model-reported** URLs.
3. [`apply_url_grounding`](../../src/truthbot/verify/adapters/base.py) computes:
   - **`web_sources`**: intersection / validation of model-reported vs tool-retrieved + HTTP reachability (HEAD) rules.
   - **`model_reported_sources`**: surviving raw model URLs; may be **backfilled** from tool URLs when frontier adapters fabricate or omit lists (defensive backfill in `build_multi_verdicts`, 2026-04-26).

## Synthetic classification table (fixture-level)

| Scenario | `tool_retrieved` | Model-reported URL | Typical `web_sources` | `model_reported_sources` (after backfill) |
|----------|------------------|--------------------|------------------------|-------------------------------------------|
| Aligned + live host | `https://www.bls.gov/cpi.htm` | same | `[bls…]` | `[bls…]` or subset |
| Fabricated host | `https://api.example/...` | `https://halluc.example/y` | often `[]` (HEAD fail / no intersection) | tool URLs may **backfill** MRS for audit |
| No model URLs, tools ran | non-empty list | none / empty | May stay empty on card until policy change; MRS backfill still lands in JSON | tool slice (≤10) |

**Regression anchors:** [`tests/test_adapter_url_grounding.py`](../../tests/test_adapter_url_grounding.py), [`tests/test_multi_batch_base.py`](../../tests/test_multi_batch_base.py) (`build_multi_verdicts` backfill cases).

## Operational note

Published HTML merges **DataHoover / evidence tiers** for reader-facing links; per-model `web_sources` may be empty in JSON while the combined evidence block is rich — compare before attributing “no sources” to a single adapter.

If you have a local `metrics/batch_sidecar/<run_id>.jsonl`, join rows for `openai` / `google` with reconciled `claims.json` and tabulate `tool_call_count`, `tool_retrieved_urls` length, `web_sources` length, and `stripped_source_count` per verdict.
