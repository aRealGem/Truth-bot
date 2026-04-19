"""
Per-provider per-token cost rates (USD).

Sources (verify and update when models GA):
  Anthropic: https://www.anthropic.com/pricing
  OpenAI:    https://openai.com/api/pricing
  Google:    https://ai.google.dev/pricing
  xAI:       https://x.ai/api

Rates stored as cost-per-single-token (per-MTok price / 1_000_000).
"""
from __future__ import annotations

COST_TABLE: dict[tuple[str, str], tuple[float, float]] = {
    ("anthropic", "claude-opus-4-7"):   (15.00 / 1_000_000, 75.00 / 1_000_000),
    ("openai",    "gpt-5.4-pro"):       (10.00 / 1_000_000, 30.00 / 1_000_000),
    ("gemini",    "gemini-2.5-pro"):    (3.50  / 1_000_000, 10.50 / 1_000_000),
    ("xai",       "grok-4"):            (5.00  / 1_000_000, 15.00 / 1_000_000),
}
FALLBACK_COST_PER_TOKEN: tuple[float, float] = (5.00 / 1_000_000, 15.00 / 1_000_000)


def estimate_cost(adapter_name: str, model_id: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = COST_TABLE.get((adapter_name, model_id), FALLBACK_COST_PER_TOKEN)
    return (input_tokens * in_rate) + (output_tokens * out_rate)
