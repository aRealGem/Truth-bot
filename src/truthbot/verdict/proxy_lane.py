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


def build_hydramind(*, response_parser: Optional[Callable[[Any], dict]] = None) -> Any:
    """Construct the HydraMind engine bound to the truth-bot proxy lane.

    ``response_parser`` is the per-call response transform the transport applies to
    each model's JSON. The two v2 lanes need DIFFERENT parsers, so build one engine
    per lane:
      * Layer A classify → ``None`` (identity): ``classifier.parse_a2`` reads the raw
        ``{"label", "claim_type", …}`` JSON itself.
      * Layer B/CRM-114 verdict → ``adjudicator.parse_verdict``: the panel/normalizer
        read ``{"verdict", "confidence", "citations", "reasoning"}``.
    Passing the verdict parser to the classify lane (or vice-versa) yields a dict with
    the wrong keys and a fail-closed parse error downstream.

    Raises the same way the transport would if the proxy is unreachable — the caller
    should have checked ``key_present`` first and printed ``BLOCKED_MSG`` to fail
    loudly rather than silently spend.

    Uses the transport's stock retry defaults (3×/0.5–30 s). A sustained proxy 429 is
    almost always the virtual key's LiteLLM budget cap (a permanent BudgetExceededError,
    not a per-minute RPM window) — retrying that harder only delays an inevitable
    failure, so the burst is bounded on the client side by Layer A pacing
    (``build_pca_lane_fns``) instead."""
    from hydramind import HydraMind
    from hydramind.manifest import NullSpendSink
    from hydramind.registry import load_registry
    from hydramind.transport import ProxyCompletion, Transport

    return HydraMind(
        load_registry(),
        Transport(completion_fn=ProxyCompletion(
            key_env=resolve_key_env(),
            base_url=base_url(),
            response_parser=response_parser,   # None → transport's identity passthrough
        )),
        spend_sink=NullSpendSink(),
        project=CLIENT,
    )


def proxy_key_spend(environ: Optional[Mapping[str, str]] = None) -> float:
    """Current proxy-DB spend for the truth-bot key, in USD (P67.3 budget
    probe). The proxy DB is the authoritative ledger — self-reported
    telemetry undercounts ~7x. Raises on transport errors so the budget
    probe fails LOUD, never silently open."""
    import json
    import urllib.request

    env = _env(environ)
    key = env.get(resolve_key_env(env)) or ""
    if not key:
        raise EnvironmentError(BLOCKED_MSG)
    base = env.get(BASE_URL_ENV, DEFAULT_BASE_URL).rstrip("/")
    req = urllib.request.Request(f"{base}/key/info",
                                 headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        doc = json.loads(resp.read().decode("utf-8"))
    info = doc.get("info", doc)
    return float(info.get("spend") or 0.0)
