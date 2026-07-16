"""Shippable proxy-lane construction for the v2 (PCA) publish path.

The eval tree has ``eval/benchmarks/proxy_client.py`` (the canonical truth-bot proxy
identity), but it doesn't ship with the package. The ``publish --engine pca`` path
needs the same L-P lane at runtime, so this module mirrors that identity (env-var
resolution + base URL) and adds a ``build_hydramind`` factory. Kept dependency-light
and side-effect-free at import; the HydraMind/transport imports are local to the
factory so ``verdict`` importers that never build a live lane don't pay for them.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Optional

CLIENT = "truth-bot"                 # HydraMind project= / spend-attribution identity
BASE_URL_ENV = "LITELLM_BASE_URL"
DEFAULT_BASE_URL = "http://127.0.0.1:4141"

# Preferred first; legacy names kept for a soft migration off the strategy key.
KEY_ENV_CANDIDATES = ("LITELLM_TRUTHBOT_KEY", "LITELLM_PCA_KEY", "LITELLM_KEY")
CANONICAL_KEY_ENV = KEY_ENV_CANDIDATES[0]

BLOCKED_MSG = (
    f"BLOCKED: no truth-bot proxy key in env (set {CANONICAL_KEY_ENV}; legacy "
    f"{KEY_ENV_CANDIDATES[1]}/{KEY_ENV_CANDIDATES[2]} also accepted). Source the "
    f"repo .env. No spend attempted.")


def _env(environ: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def resolve_key_env(environ: Optional[Mapping[str, str]] = None) -> str:
    """Env var NAME holding the truth-bot proxy key: first candidate set, else the
    canonical one (so guard messages name the right var even when nothing is set)."""
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


def build_hydramind(*, response_parser: Callable[[Any], dict]) -> Any:
    """Construct the HydraMind engine bound to the truth-bot proxy lane.

    ``response_parser`` is the per-call raw-JSON parser (``adjudicator.parse_verdict``
    for the verdict panel). Raises the same way the transport would if the proxy is
    unreachable — the caller should have checked ``key_present`` first and printed
    ``BLOCKED_MSG`` to fail loudly rather than silently spend."""
    from hydramind import HydraMind
    from hydramind.manifest import NullSpendSink
    from hydramind.registry import load_registry
    from hydramind.transport import ProxyCompletion, Transport

    return HydraMind(
        load_registry(),
        Transport(completion_fn=ProxyCompletion(
            key_env=resolve_key_env(),
            base_url=base_url(),
            response_parser=response_parser,
        )),
        spend_sink=NullSpendSink(),
        project=CLIENT,
    )
