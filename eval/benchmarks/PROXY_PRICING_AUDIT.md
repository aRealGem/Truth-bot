# LiteLLM proxy pricing audit (2026-07-09)

Staged for review — **not applied**. Applying means editing `~/litellm/config.yaml`
and `systemctl --user restart litellm`, which briefly blips **Cass**; do it when Cass
is idle. The proxy is ccagent-owned (see wiki `infra:software`).

## Why this matters beyond truth-bot

LiteLLM computes `response_cost` (and increments per-key `spend`) only for models in
its **built-in cost map**. The DeepInfra **DeepSeek** models are not in it, so every
call to them reports **$0** — which means:

- **Cass's spend tracking is blind.** Cass runs `deepseek-v4-flash` (primary) →
  `deepseek-v4-pro` (fallback), both unpriced → the proxy shows ~$0 for Cass, and the
  `cost-guard-4h` cron (`cost_guard.py --threshold 5.00`, see `infra:processes`) can
  never trip. The DeepInfra dashboard shows the real spend; the proxy doesn't.
- truth-bot's `dsv4-flash` seat was the same gap — **already fixed** (priced 2026-07-09,
  verified `cost_source=proxy`). This audit closes the rest.

## Gap 1 — unpriced models (add cost fields)

Prices are DeepInfra list, verified 2026-07-09. Add under each entry's `litellm_params`
(per-token = $/Mtok ÷ 1e6):

| upstream model | model_name entries affected | input | output |
|---|---|---|---|
| DeepSeek-V4-Flash | `deepseek-v4-flash`, `deepseek-v4-flash-think`, `deepseek-ai/DeepSeek-V4-Flash`, `custom-api-deepinfra-com/deepseek-ai/DeepSeek-V4-Flash` (**not** `dsv4-flash` — done) | 0.00000009 | 0.00000018 |
| DeepSeek-V4-Pro | `deepseek-v4-pro`, `deepseek-v4-pro-think`, `deepseek-ai/DeepSeek-V4-Pro`, `custom-api-deepinfra-com/deepseek-ai/DeepSeek-V4-Pro` | 0.0000013 | 0.0000026 |
| DeepSeek-V3 | `deepseek-chat` | 0.00000032 | 0.00000089 |
| DeepSeek-R1-0528 | `deepseek-r1` | 0.0000005 | 0.00000215 |

Add cache pricing where relevant (V4-Flash `cache_read_input_token_cost: 0.000000018`).
Pattern per entry (matches the applied `dsv4-flash` fix):

```yaml
  litellm_params:
    model: deepinfra/deepseek-ai/DeepSeek-V4-Pro
    api_key: os.environ/DEEPINFRA_API_KEY
    input_cost_per_token: 0.0000013
    output_cost_per_token: 0.0000026
```

**Priority order:** `deepseek-v4-flash` (Cass primary) and `deepseek-v4-pro` (Cass
fallback) first — those unblock the cost-guard. The `-think`, `deepseek-ai/*`, and
`custom-api-*` aliases are duplicates of the same upstream; price them for consistency.

## Gap 2 — stale aliases (reconcile, not a cost bug)

These resolve and get priced by LiteLLM's built-in map, but point at **older** models
than a 2026 roster probably intends. Flagging for reconciliation:

| alias | currently -> | latest as of 2026-07-09 |
|---|---|---|
| `grok` | `xai/grok-2-latest` | grok-4.5 ($2/$6) |
| `gemini-pro` | `gemini/gemini-1.5-pro` | gemini 3.1 pro ($2/$12) / 2.5 pro |
| `gemini-flash` | `gemini/gemini-2.0-flash` | gemini 3.5 flash ($1.50/$9) |
| `gpt-4o` / `gpt-4o-mini` | `openai/gpt-4o*` | GPT-5.x series (4o superseded) |

If truth-bot's prod roster or Cass targets any of these, update the `model:` line to the
intended version (and re-check its price). Not urgent for the dev roster.

## Apply (when Cass idle)

1. `cp ~/litellm/config.yaml ~/litellm/config.yaml.bak.$(date +%Y%m%d_%H%M%S)`
2. Add the cost fields above; `python3 -c "import yaml; yaml.safe_load(open('$HOME/litellm/config.yaml'))"` to validate.
3. `systemctl --user restart litellm` (XDG_RUNTIME_DIR=/run/user/1002); wait for `/health/readiness` healthy.
4. Verify: a call to each newly-priced alias returns a non-zero `x-litellm-response-cost`.
