# Grok triage vs frontier — 10-claim calibration A/B (operator checklist)

Use the same transcript strip as PROJECT_BOARD / STATUS references (baseline run `146ee42a` **or** current 10-claim calibration file if path changed).

## Before (control)

1. Compare against **pre-change behavior** (same `TRUTHBOT_GROK_MAX_TOOL_CALLS` frontier default 8): historically **triage** `GrokAdapter.call()` also used **8** (`_max_tool_calls_per_claim()` only). After this work, triage defaults to **3** unless `TRUTHBOT_GROK_TRIAGE_MAX_TOOL_CALLS` overrides.
2. Run with `--triage` enabled, batch or live per your stack, **10 claims**.
3. Capture:
   - `metrics/run_summaries/<run_id>.json` → **`by_tier.triage`** vs **`by_tier.frontier`** (or xAI-only lines in cost breakdown)
   - `metrics/adapter_calls.jsonl` rows with `tier=="triage"` and `adapter_name=="xai"` → histogram of **`tool_call_count`**
   - Transcript meta **`claims_triaged_auto`** and list **`triaged_claim_ids`**

## After (treatment — current mainline)

1. Default triage Grok cap is **3** tool calls/claim unless `TRUTHBOT_GROK_TRIAGE_MAX_TOOL_CALLS` is set.
2. Repeat the same 10-claim run (same transcript hash, same `--max-claims` cap, same RNG seed if triage shadow sampling is on).
3. Re-capture the three bullet blocks above.

## Compare (interpretability)

| Signal | What improves if triage cap helps |
|--------|-----------------------------------|
| Triaged xAI USD | Should drop materially if tool turns were the cost driver |
| `tool_call_count` @ triage | Should cluster at or below the configured triage cap × successful tool rounds |
| Unanimous short-circuit rate | May **decrease** slightly (less search → less agreement) — acceptable trade; watch for wild swings |
| stderr | Grep **`GrokAdapter: xAI SDK rejected max_tool_calls`** — if present, cap is **not** enforced and upstream retry path is unbounded |

## Optional

- Set `TRUTHBOT_GROK_TRIAGE_MAX_TOOL_CALLS=2` for an aggressive second datapoint; validate triage–frontier verdict agreement on held-out claims if budget allows.
