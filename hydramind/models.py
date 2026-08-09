"""
Logical tier→provider→model alias table. These aliases MUST match model_name
entries registered on the LiteLLM proxy (L-P), otherwise a call 400s or silently
falls back — the "Flash-registration gap" class of bug. Verified against the
proxy config 2026-07-04: anthropic = claude-{haiku,sonnet,opus}; openai =
gpt-4o{,-mini}; gemini = gemini-{flash,pro}; grok = grok.

NOTE: `mistral` (the decided cross-vendor critic for pca) has NO registered
alias on the proxy yet — a pca/Layer-B live run needs it added first. Flagged;
not needed for the Layer-A `single` runs (builds 5–6).
"""
from __future__ import annotations

from typing import Optional

from .types import ModelBinding

TIER_MODELS: dict[tuple[str, str], str] = {
    ("anthropic", "cheap"): "claude-haiku",
    ("anthropic", "standard"): "claude-sonnet",
    ("anthropic", "frontier"): "claude-opus",
    ("openai", "cheap"): "gpt-4o-mini",
    ("openai", "standard"): "gpt-4o-mini",
    ("openai", "frontier"): "gpt-4o",
    ("gemini", "cheap"): "gemini-flash",
    ("gemini", "standard"): "gemini-pro",
    ("gemini", "frontier"): "gemini-pro",
    ("grok", "standard"): "grok",
    # mistral intentionally absent — unregistered on the proxy (see module note).
}

# Family tokens used to detect silent fallback: the returned `model` must carry
# the requested alias's family token, else the proxy served something else.
_FAMILY_TOKENS = ("haiku", "sonnet", "opus", "gpt-4o", "gpt", "gemini",
                  "grok", "deepseek", "dsv4", "mistral")

# Alias → (provider, tier). Aliases are the proxy model_names (verified live
# 2026-07-04). Used to build a ModelBinding directly from a roster seat alias.
ALIAS_META: dict[str, tuple[str, str]] = {
    "claude-haiku": ("anthropic", "cheap"),
    "claude-sonnet": ("anthropic", "standard"),
    "claude-opus": ("anthropic", "frontier"),
    "gpt-4o-mini": ("openai", "standard"),
    "gpt-4o": ("openai", "frontier"),
    "gpt-5.5": ("openai", "frontier"),
    "gemini-flash": ("gemini", "cheap"),
    "gemini-pro": ("gemini", "standard"),
    "grok": ("xai", "standard"),
    "grok-4.3": ("xai", "frontier"),
    "mistral": ("deepinfra", "standard"),
    "dsv4-flash": ("deepinfra", "cheap"),
    "opus-worker": ("anthropic", "frontier"),
}

# Aliases that ride L-W (Claude worker, subscription auth) instead of the
# proxy. NOT proxy model_names — the transport routes them before any proxy
# dispatch (P67.9 / T3.1). The "opus" token keeps returned_ok() family
# checking honest against the worker's reported claude-opus-* id.
WORKER_ALIASES: frozenset[str] = frozenset({"opus-worker"})


def binding_from_alias(alias: str) -> ModelBinding:
    """Build a ModelBinding from a proxy model alias (roster seat value)."""
    provider, tier = ALIAS_META.get(alias, ("unknown", "standard"))
    return ModelBinding(provider=provider, model=alias, tier=tier)


def resolve_model(provider: str, tier: str) -> str:
    m = TIER_MODELS.get((provider, tier))
    if m is None:
        for t in ("frontier", "standard", "cheap"):
            if (provider, t) in TIER_MODELS:
                return TIER_MODELS[(provider, t)]
        return f"{provider}:{tier}"
    return m


def binding_for(provider: str, tier: str) -> ModelBinding:
    return ModelBinding(provider=provider, model=resolve_model(provider, tier), tier=tier)


def _family(name: str) -> str | None:
    name = (name or "").lower()
    for tok in _FAMILY_TOKENS:
        if tok in name:
            return tok
    return None


def returned_ok(requested_alias: str, returned_model: str) -> bool:
    """True if `returned_model` plausibly IS the requested alias (same family).
    A False here means a silent fallback / unregistered-model reroute — callers
    (equivalence, G5) must fail on it."""
    if not returned_model:
        return True                     # nothing reported; can't judge here
    rf, gf = _family(requested_alias), _family(returned_model)
    if rf is None:
        return True
    return rf == gf


# ── fallback cost rate table (P96.2.1) ────────────────────────────────────────
# USD per 1M tokens, (input, output). This is a FALLBACK only: the LiteLLM
# proxy-reported cost is always preferred (cost_source="proxy"); the table is
# consulted only when the proxy reports nothing (cost_source="table"), e.g. an
# L-B lane response or a model the proxy has no price for.
#
# roster.dev seats (P=mistral, C=dsv4-flash, A=claude-haiku). The two DeepInfra
# rates are the published list prices (deepinfra.com, verified 2026-07-09) — the
# live dev-lot found the proxy prices claude-haiku + mistral but NOT dsv4-flash,
# so dsv4-flash actually rides this table; its rate must be right.
# NOTE: DeepInfra also bills *cached* input cheaper (dsv4-flash $0.018/Mtok); this
# flat (in,out) table ignores caching — fine for closed-book (≈0 cache), revisit
# if a shared-context prod roster leans on prompt caching.
#
# claude-haiku CORRECTED 2026-08-09: it was (0.80, 4.00), a "rough fallback"
# guess, and every $0 estimator read it as if it were the price. The LiteLLM
# proxy that actually bills this lane is configured input_cost_per_token
# 0.000001 / output_cost_per_token 0.000005 = (1.00, 5.00), so the old entry was
# a flat 1.25x low on both sides — one of the three errors behind the B1a/B2
# cost misses. See truthbot.costs, which reads this table rather than copying it.
RATE_TABLE_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku": (1.00, 5.00),    # Anthropic claude-haiku-4-5; matches the proxy config
    "mistral":      (0.075, 0.20),   # DeepInfra Mistral-Small-3.2-24B (list, 2026-07-09)
    "dsv4-flash":   (0.09, 0.18),    # DeepInfra DeepSeek-V4-Flash (list, 2026-07-09)
    # roster.prod seats (P67.9): rates pinned in the proxy config, so these are
    # fallback-only. opus-worker is absent by design — L-W is subscription
    # auth, cost_source="subscription", never table-priced.
    "grok-4.3":     (1.25, 2.50),    # xAI grok-4.3 (litellm price map, 2026-07-23)
    "gpt-5.5":      (5.00, 30.00),   # OpenAI gpt-5.5 (litellm price map, 2026-07-23)
    # OFF-PROXY seat: the phase-3 economy config's R2 browsing retriever bills
    # OpenAI directly, so nothing here is ledger-checkable. Moved in from
    # scripts/phase3_rebuild.MODEL_RATES so the repo prices a model in one place.
    "gpt-5-mini":   (0.25, 2.00),    # OpenAI gpt-5-mini (litellm price map, 2026-07-23)
}


def cost_from_table(model: str, tokens_in: int, tokens_out: int) -> Optional[float]:
    """Estimate a call's USD cost from captured tokens and the local rate table.
    Returns None when the model has no table entry (caller records "none")."""
    rates = RATE_TABLE_USD_PER_MTOK.get(model)
    if rates is None:
        return None
    r_in, r_out = rates
    return ((tokens_in or 0) * r_in + (tokens_out or 0) * r_out) / 1_000_000.0
