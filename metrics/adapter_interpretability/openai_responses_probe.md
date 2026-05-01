# OpenAI Responses API — single-claim interpretability probe

**Goal:** Decide hypothesis (a) model skips search vs (b) tool path under-fires, using one cheap live call and one grep-friendly log line.

## Environment

```bash
export TRUTHBOT_OPENAI_RESPONSES_PROBE=1
export OPENAI_API_KEY=…   # required
```

Truthy values for the probe flag: `1`, `true`, `yes`, `on` (case-insensitive).

## Run

From the repo root:

```bash
TRUTHBOT_OPENAI_RESPONSES_PROBE=1 uv run python scripts/openai_responses_probe.py \
  "White House announced TrumpRx.gov for MFN drug pricing in February 2026."
```

(Claim paraphrased from SOTU findings materiality list — any **post-cutoff factual** claim works.)

The script prints JSON verdict fields to stdout. **Stderr** emits a single WARNING:

`OpenAIAdapter RESPONSES_PROBE context=… tier=… model=… web_search_calls=… tool_urls=… max_tool_calls_kwarg=…`

## Interpret

- **`web_search_calls=0`** with a claim that obviously needs the web → favors engineering / tool invocation (hypothesis b).
- **`web_search_calls≥1`** but explanation still reads like stale training data → favor prompt / synthesis (hypothesis a or rubric issue).
- Cross-check `metrics/adapter_calls.jsonl` for the same run_id (if pipeline logging is enabled).

## Related code

- [`src/truthbot/verify/adapters/openai.py`](../../src/truthbot/verify/adapters/openai.py): `_log_openai_responses_probe`, `_walk_output_for_urls`.
