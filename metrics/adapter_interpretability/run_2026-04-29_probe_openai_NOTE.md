# OpenAI Responses probe — 2026-04-29 UTC session

Probe command (approx — see shell history):

```bash
set -a && . ./.env && set +a
TRUTHBOT_OPENAI_RESPONSES_PROBE=1 uv run python scripts/openai_responses_probe.py \
  "White House announced TrumpRx.gov for MFN drug pricing in February 2026."
```

## RESPONSES_PROBE line (stderr → combined log)

```
OpenAIAdapter RESPONSES_PROBE context=live_single tier=frontier model=gpt-5.4 claim_id=<uuid> batch_call_id= web_search_calls=1 tool_urls=0 max_tool_calls_kwarg=2
```

SDK also warned: `response_format not supported by SDK; falling back to text output`.

## Interpretation vs plan rubric

- **web_search_calls=1**: tools **did** run — does **not** support “engineering / tool invocation completely dead” hypothesis (hypothesis **b**) for this claim + model combo.
- **tool_urls=0** in probe line reflects URLs harvested in `_walk_output_for_urls`; verdict still carries **model_reported_sources** (WhiteHouse.gov URLs) with **stripped_source_count=2** and empty **web_sources** — aligns with attribution/HEAD-strip path, worth a separate attribution audit.
- Full structured verdict: [run_2026-04-29_probe_openai_verdict.json](run_2026-04-29_probe_openai_verdict.json).

Raw tee (stdout+stderr): [run_2026-04-29_probe_openai_combined.txt](run_2026-04-29_probe_openai_combined.txt).
