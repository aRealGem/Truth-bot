# Provisioning a truth-bot proxy key (live dev-lot runs)

The dev-lot runners (`run_layer_b_devlot.py`, `run_pca_devlot.py`) and the
CN-sensitivity probe reach the LiteLLM proxy (L-P lane) with a **client** virtual
key. Identity is the consumer — **`truth-bot`** — not the strategy (`pca` is a
HydraMind strategy the client runs). See `proxy_client.py`.

## Env vars

Add to `~/truth-bot/.env` (0600) — these are **not** yet in `.env.example`; add
them there too (`.env.example` is edited by a human — the agent's permission
sandbox denies writes under that path):

```
# LiteLLM proxy — truth-bot client key (label "truth-bot"), budget-capped
LITELLM_TRUTHBOT_KEY=sk-...
LITELLM_BASE_URL=http://127.0.0.1:4141
```

Resolution order (soft migration): `LITELLM_TRUTHBOT_KEY` → legacy
`LITELLM_PCA_KEY` → generic `LITELLM_KEY` (`proxy_client.resolve_key_env`).

## Minting the key

The virtual key is minted **once** (persistent, not per-run) on the proxy admin
side (clawd/master-key holder) with a budget cap and a model allowlist covering
the roster.dev seats (`claude-haiku`, `mistral`, `dsv4-flash`). The proxy holds
the upstream provider creds — Anthropic via OAuth/subscription, DeepInfra key — so
this **one** client key reaches every seat over L-P (no per-provider keys needed
for the closed-book dev-lot). Attribute it `project=truth-bot`.

> The agent identity (ccagent) does not hold the proxy master key and does not
> hunt it from credential stores; minting is a privileged step done on request.

## Spend accounting

Every live run appends a row to the independent ledger
`metrics/spend_ledger/truthbot.jsonl` (git-ignored) via `ledger.append_run` —
client config + settings + the LiteLLM-reported cost. Schema:
`metrics/spend_ledger/SCHEMA.md`.

## Verify (free)

```
curl -s -H "Authorization: Bearer $LITELLM_TRUTHBOT_KEY" \
  http://127.0.0.1:4141/v1/models | python3 -m json.tool
```
Confirm `claude-haiku`, `mistral`, `dsv4-flash` are listed before spending.
