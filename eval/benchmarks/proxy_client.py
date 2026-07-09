"""
Proxy CLIENT identity for truth-bot's use of the LiteLLM L-P lane.

The virtual key is a *client* identity (truth-bot), not a *strategy*: `pca` is a
HydraMind strategy, not a consumer. Env convention is `LITELLM_<CLIENT>_KEY`; for
truth-bot that is `LITELLM_TRUTHBOT_KEY`. The legacy strategy-scoped
`LITELLM_PCA_KEY` and the generic `LITELLM_KEY` are accepted as fallbacks during
migration. The proxy holds the upstream provider creds (Anthropic via OAuth /
subscription, DeepInfra key for mistral/dsv4-flash), so ONE client key reaches
every roster.dev seat over L-P.
"""
from __future__ import annotations

import os
from typing import Mapping, Optional

CLIENT = "truth-bot"            # HydraMind project= / spend-attribution identity
KEY_LABEL = "truth-bot"        # LiteLLM virtual-key label at the proxy
BASE_URL_ENV = "LITELLM_BASE_URL"
DEFAULT_BASE_URL = "http://127.0.0.1:4141"

# Preferred first; legacy names kept for a soft migration off the strategy key.
KEY_ENV_CANDIDATES = ("LITELLM_TRUTHBOT_KEY", "LITELLM_PCA_KEY", "LITELLM_KEY")
CANONICAL_KEY_ENV = KEY_ENV_CANDIDATES[0]


def _env(environ: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def resolve_key_env(environ: Optional[Mapping[str, str]] = None) -> str:
    """Env var NAME to use for the truth-bot proxy key: the first candidate that
    is set, else the canonical `LITELLM_TRUTHBOT_KEY` (so the guard message names
    the right var even when nothing is set)."""
    env = _env(environ)
    for name in KEY_ENV_CANDIDATES:
        if env.get(name):
            return name
    return CANONICAL_KEY_ENV


def key_present(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = _env(environ)
    return any(env.get(n) for n in KEY_ENV_CANDIDATES)


def base_url(environ: Optional[Mapping[str, str]] = None) -> str:
    return _env(environ).get(BASE_URL_ENV, DEFAULT_BASE_URL)


BLOCKED_MSG = (
    f"BLOCKED: no truth-bot proxy key in env (set {CANONICAL_KEY_ENV}; legacy "
    f"{KEY_ENV_CANDIDATES[1]}/{KEY_ENV_CANDIDATES[2]} also accepted). Source the "
    f"repo .env. No spend attempted.")
