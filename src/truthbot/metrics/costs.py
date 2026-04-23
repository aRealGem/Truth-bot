"""
Per-provider per-token cost rates (USD).

Sources (rates verified 2026-04-22; re-verify when models or prices change):
  Anthropic: https://www.anthropic.com/pricing (API tab)
  OpenAI:    https://openai.com/api/pricing
  Google:    https://ai.google.dev/gemini-api/docs/pricing
  xAI:       https://docs.x.ai/docs/models

Rates stored as cost-per-single-token (per-MTok price / 1_000_000).

Each entry is (input_rate, cached_input_rate, output_rate) per token.
- input_rate: full-price input tokens (non-cached).
- cached_input_rate: discounted input for cache hits / OpenAI cached prefix / Gemini context cache.
  For Anthropic, cache *read* tokens are billed at 0.10 × input_rate (stored explicitly here).
  Cache *write* tokens use cache_creation_input_tokens × 1.25 × input_rate in estimate_cost
  (e.g. Opus 4.7: $6.25/MTok cache-write vs $5/MTok fresh input on the current page).

Gemini 2.5 Pro: table uses the ≤200K-token bucket; prompts here stay well under that.
  Long-context bucket (>200K) is approximately $2.50 / $0.25 / $15.00 per MTok if needed later.
"""
from __future__ import annotations

from typing import Optional

# (input_per_token, cached_input_per_token, output_per_token)
COST_TABLE: dict[tuple[str, str], tuple[float, float, float]] = {
    # --- Anthropic (frontier + fallbacks) ---
    ("anthropic", "claude-opus-4-7"): (
        5.00 / 1_000_000,
        0.50 / 1_000_000,  # cache read ≈ 0.10 × input
        25.00 / 1_000_000,
    ),
    ("anthropic", "claude-opus-4-5"): (
        5.00 / 1_000_000,
        0.50 / 1_000_000,
        25.00 / 1_000_000,
    ),
    ("anthropic", "claude-3-5-sonnet-20241022"): (
        3.00 / 1_000_000,
        0.30 / 1_000_000,
        15.00 / 1_000_000,
    ),
    # --- Anthropic triage / cheap ---
    ("anthropic", "claude-3-5-haiku-20241022"): (
        0.80 / 1_000_000,
        0.08 / 1_000_000,
        4.00 / 1_000_000,
    ),
    ("anthropic", "claude-haiku-4-5"): (
        1.00 / 1_000_000,
        0.10 / 1_000_000,
        5.00 / 1_000_000,
    ),
    # --- OpenAI ---
    ("openai", "gpt-5.4"): (
        2.50 / 1_000_000,
        0.25 / 1_000_000,
        15.00 / 1_000_000,
    ),
    ("openai", "gpt-4.1"): (
        2.00 / 1_000_000,
        0.50 / 1_000_000,  # cached prompt tier
        8.00 / 1_000_000,
    ),  # legacy: current code _PRIMARY_MODEL fallback; row unused if OpenAI drops the endpoint
    ("openai", "gpt-4o"): (
        2.50 / 1_000_000,
        1.25 / 1_000_000,
        10.00 / 1_000_000,
    ),  # legacy fallback SKU
    ("openai", "gpt-4o-mini"): (
        0.15 / 1_000_000,
        0.075 / 1_000_000,
        0.60 / 1_000_000,
    ),  # legacy mini; current cheap tier is gpt-5.4-mini
    ("openai", "gpt-5.4-mini"): (
        0.75 / 1_000_000,
        0.075 / 1_000_000,
        4.50 / 1_000_000,
    ),
    ("openai", "gpt-5.4-nano"): (
        0.20 / 1_000_000,
        0.02 / 1_000_000,
        1.25 / 1_000_000,
    ),
    # --- Gemini ---
    ("gemini", "gemini-2.5-pro"): (
        1.25 / 1_000_000,
        0.125 / 1_000_000,
        10.00 / 1_000_000,
    ),
    ("gemini", "gemini-2.5-flash"): (
        0.30 / 1_000_000,
        0.03 / 1_000_000,
        2.50 / 1_000_000,
    ),
    # --- xAI ---
    ("xai", "grok-4"): (
        2.00 / 1_000_000,
        0.20 / 1_000_000,  # aligned with grok-4.20 alias resolution
        6.00 / 1_000_000,
    ),
    ("xai", "grok-3-mini"): (
        0.30 / 1_000_000,
        0.075 / 1_000_000,
        0.50 / 1_000_000,
    ),  # best-effort legacy estimate; xAI no longer publishes a public list rate
    ("xai", "grok-4-1-fast-reasoning"): (
        0.20 / 1_000_000,
        0.05 / 1_000_000,
        0.50 / 1_000_000,
    ),
}
FALLBACK_COST_PER_TOKEN: tuple[float, float, float] = (
    5.00 / 1_000_000,
    5.00 / 1_000_000,
    15.00 / 1_000_000,
)

# 50% batch discount vs live list prices — Anthropic, OpenAI, Google, and xAI all still
# advertise 50% batch pricing (see module docstring URLs).
BATCH_DISCOUNT: dict[str, float] = {
    "anthropic": 0.5,
    "openai": 0.5,
    "gemini": 0.5,
    "xai": 0.5,
}


def estimate_cost(
    adapter_name: str,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    openai_cached_prompt_tokens: int = 0,
    gemini_cached_content_tokens: int = 0,
    mode: str = "live",
    batch_job_id: Optional[str] = None,
) -> float:
    """
    Compute estimated USD cost for one adapter call.

    ``input_tokens`` should be the API-reported total input / prompt tokens when available.
    Provider-specific cache fields subtract from full-rate input where applicable.

    The 50% batch discount is only applied when ``mode == "batch"`` AND a real
    ``batch_job_id`` is present. Calls labeled batch without an actual provider
    batch ID (scaffolding / misconfiguration) are billed at live list prices so
    telemetry stays honest.
    """
    in_rate, cached_in_rate, out_rate = COST_TABLE.get(
        (adapter_name, model_id), FALLBACK_COST_PER_TOKEN
    )

    if adapter_name == "anthropic":
        # Anthropic: cache reads billed at cached_in_rate (= 0.1 × in_rate in table);
        # cache writes at 1.25 × in_rate.
        fresh = max(0, input_tokens - cache_read_input_tokens - cache_creation_input_tokens)
        cost = (
            fresh * in_rate
            + cache_creation_input_tokens * in_rate * 1.25
            + cache_read_input_tokens * cached_in_rate
            + output_tokens * out_rate
        )
    elif adapter_name == "openai" and openai_cached_prompt_tokens > 0:
        uncached = max(0, input_tokens - openai_cached_prompt_tokens)
        cost = (
            uncached * in_rate
            + openai_cached_prompt_tokens * cached_in_rate
            + output_tokens * out_rate
        )
    elif adapter_name == "gemini" and gemini_cached_content_tokens > 0:
        uncached = max(0, input_tokens - gemini_cached_content_tokens)
        cost = (
            uncached * in_rate
            + gemini_cached_content_tokens * cached_in_rate
            + output_tokens * out_rate
        )
    else:
        cost = input_tokens * in_rate + output_tokens * out_rate

    if mode == "batch" and batch_job_id:
        mult = BATCH_DISCOUNT.get(adapter_name, 1.0)
        cost *= mult

    return cost
