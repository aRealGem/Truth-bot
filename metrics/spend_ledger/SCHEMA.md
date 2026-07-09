# truth-bot spend ledger

Independent, append-only run ledger for truth-bot's live proxy runs. One JSON
object per line (`truthbot.jsonl`, git-ignored — runtime telemetry). Written by
`eval/benchmarks/ledger.py::append_run`, called by the dev-lot runners after
`hm.run(...)`. It records the **client config + test-run settings**, the run
outcome, and the **cost LiteLLM reported** — decoupled from key provisioning.

## Identity model

The spender is a **client** (`truth-bot`, HydraMind `project=`), not a strategy —
`pca` is a HydraMind strategy the client runs. The LiteLLM virtual key is a client
key (label `truth-bot`, env `LITELLM_TRUTHBOT_KEY`); strategy/roster are recorded
as fields, not baked into the key. See `eval/benchmarks/proxy_client.py`.

## Cost source of truth

`cost.total_cost_usd` is summed from the manifest's per-call **proxy-reported**
cost (`cost_source="proxy"`, from the `x-litellm-response-cost` header — see
`hydramind/transport.extract_cost`). `cost.cost_source_tally` shows the provenance
mix, so a `table`/`none`-sourced (estimated) total is never mistaken for a
LiteLLM-authoritative one.

## Row fields

| field | meaning |
|---|---|
| `ts` | ISO-8601 UTC timestamp of the ledger write |
| `run_id` | short unique id for the run |
| `client` | spend-attribution identity — `truth-bot` (`manifest.project`) |
| `key_label` | LiteLLM virtual-key label used (`truth-bot`) |
| `strategy` | HydraMind strategy (`pca`, `single`, …) |
| `roster` | roster name (`dev`, `prod`) if any |
| `task` | task bundle name (`verdict`, `classify`) |
| `n_items` | items in the run |
| `dataset_hash` | hash of the input bundle (reproducibility) |
| `budget_ceiling_usd` | `cost.ceiling_usd` from the resolved spec |
| `base_url` | proxy base URL used |
| `result.halted` / `result.halt_reason` | cost-ceiling halt state |
| `result.flagged` / `result.split_rate` | disagreement-flagged count / P/C split rate |
| `result.escalation` | escalation monitor `{escalated,total,rate,watermark,over_watermark}` |
| `cost.total_cost_usd` | run total (proxy-sourced; see above) |
| `cost.tokens_in` / `cost.tokens_out` | token totals |
| `cost.cost_source_tally` | `{proxy,table,none}` call-count mix |
| `cost.lanes` | per-lane spend rows (`manifest.to_spend_records()`) |
| `model_mismatches` | silent-fallback / unregistered-model detections |
