"""
Per-provider per-token cost rates (USD).

Sources (rates verified 2026-04-22; the gpt-5.5 / gpt-5-mini / gpt-5.4 / gpt-4o
and grok-4.3 rows re-verified against the live pricing pages 2026-08-26.
Re-verify when models or prices change — a missing or stale row does not fail,
it silently mis-bills, which is how the Jul/Aug 2026 undercount happened):
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

import logging
from typing import Optional

logger = logging.getLogger(__name__)

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
    # R2 evidence-retrieval default (TRUTHBOT_R2_MODEL, else this). Until
    # 2026-08-26 this row was missing, so every R2 call fell through to
    # FALLBACK_COST_PER_TOKEN and billed output at $15/MTok against a true
    # $30 — a silent 2x undercount on the highest-volume OpenAI path.
    ("openai", "gpt-5.5"): (
        5.00 / 1_000_000,
        0.50 / 1_000_000,
        30.00 / 1_000_000,
    ),
    # Distinct SKU from gpt-5.4-mini below; pinned by the phase3/rescue/headret
    # scripts via TRUTHBOT_R2_MODEL.
    ("openai", "gpt-5-mini"): (
        0.25 / 1_000_000,
        0.025 / 1_000_000,
        2.00 / 1_000_000,
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
    # R3 retrieval lane (TRUTHBOT_R3_MODEL). <200K-prompt bucket; the >=200K
    # bucket is 2.50 / 0.40 / 5.00 per MTok if retrieval prompts ever grow.
    ("xai", "grok-4.3"): (
        1.25 / 1_000_000,
        0.20 / 1_000_000,
        2.50 / 1_000_000,
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

# (adapter, model) pairs already reported as unpriced, so the error fires once
# per process rather than once per call.
_WARNED_UNPRICED: set[tuple[str, str]] = set()


def is_priced(adapter_name: str, model_id: str) -> bool:
    """True when this (adapter, model) has a real rate row.

    Callers use this to record *how* a dollar figure was derived. A
    fallback-priced call is a guess, and the July/August 2026 reconciliation
    showed the guess can be wrong by 2x in either direction.
    """
    return (adapter_name, model_id) in COST_TABLE


def _rates(adapter_name: str, model_id: str) -> tuple[float, float, float]:
    """Look up per-token rates, complaining loudly on a miss.

    Deliberately does not raise: this runs inside ``TelemetryLogger.measure``'s
    ``finally`` block, where an exception would surface at the API call site and
    turn a metering gap into an outage.
    """
    rates = COST_TABLE.get((adapter_name, model_id))
    if rates is not None:
        return rates
    key = (adapter_name, model_id)
    if key not in _WARNED_UNPRICED:
        _WARNED_UNPRICED.add(key)
        logger.error(
            "No cost row for (%s, %s) — pricing this call at the generic "
            "fallback (%.2f/%.2f/%.2f per MTok). Spend telemetry for this "
            "model is a GUESS until a row is added to COST_TABLE.",
            adapter_name,
            model_id,
            FALLBACK_COST_PER_TOKEN[0] * 1_000_000,
            FALLBACK_COST_PER_TOKEN[1] * 1_000_000,
            FALLBACK_COST_PER_TOKEN[2] * 1_000_000,
        )
    return FALLBACK_COST_PER_TOKEN


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
    in_rate, cached_in_rate, out_rate = _rates(adapter_name, model_id)

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
