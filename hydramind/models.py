"""
Logical tier→provider→model alias table. These are *aliases* the LiteLLM proxy
(L-P) and the native SDKs (L-B) resolve to concrete upstreams; HydraMind never
hard-codes a raw upstream model id in strategy code.
"""
from __future__ import annotations

from .types import ModelBinding

TIER_MODELS: dict[tuple[str, str], str] = {
    ("anthropic", "cheap"): "claude-haiku-4-5",
    ("anthropic", "standard"): "claude-sonnet-4-6",
    ("anthropic", "frontier"): "claude-opus-4-8",
    ("openai", "standard"): "gpt-5.4",
    ("openai", "frontier"): "gpt-5.4",
    ("gemini", "standard"): "gemini-2.5-pro",
    ("gemini", "frontier"): "gemini-2.5-pro",
    ("mistral", "standard"): "mistral-large-latest",
    ("grok", "standard"): "grok-4",
}


def resolve_model(provider: str, tier: str) -> str:
    m = TIER_MODELS.get((provider, tier))
    if m is None:
        # Fall back to the provider's richest tier we know, else a synthetic id.
        for t in ("frontier", "standard", "cheap"):
            if (provider, t) in TIER_MODELS:
                return TIER_MODELS[(provider, t)]
        return f"{provider}:{tier}"
    return m


def binding_for(provider: str, tier: str) -> ModelBinding:
    return ModelBinding(provider=provider, model=resolve_model(provider, tier), tier=tier)
