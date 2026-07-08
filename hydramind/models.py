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
    "gemini-flash": ("gemini", "cheap"),
    "gemini-pro": ("gemini", "standard"),
    "grok": ("xai", "standard"),
    "mistral": ("deepinfra", "standard"),
    "dsv4-flash": ("deepinfra", "cheap"),
}


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
# L-B lane response or a proxy without cost-tracking configured.
#
# Seats named by the task and used by roster.dev (P=mistral, C=dsv4-flash,
# A=claude-haiku). Values are PROVISIONAL list-price approximations — treat a
# table-sourced total as an estimate and reconcile against the proxy's reported
# cost before any budget decision leans on it. Models absent here resolve to
# cost_source="none" (0.0) rather than a guessed rate.
RATE_TABLE_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku": (0.80, 4.00),
    "mistral":      (0.20, 0.60),
    "dsv4-flash":   (0.30, 0.88),
}


def cost_from_table(model: str, tokens_in: int, tokens_out: int) -> Optional[float]:
    """Estimate a call's USD cost from captured tokens and the local rate table.
    Returns None when the model has no table entry (caller records "none")."""
    rates = RATE_TABLE_USD_PER_MTOK.get(model)
    if rates is None:
        return None
    r_in, r_out = rates
    return ((tokens_in or 0) * r_in + (tokens_out or 0) * r_out) / 1_000_000.0
