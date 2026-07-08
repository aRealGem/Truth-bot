"""
L-B backend — Anthropic Message Batches (design §2).

50% off input+output, ≤24 h SLA, custom_id reconciliation, results retained
29 days. Multi-turn agentic loops are NOT batchable; every L2 wave is single
completions by construction, so it fits.

The anthropic SDK and the ANTHROPIC_API_KEY (repo-scoped .env, CW-12) are
required only at run time — lazily imported so the rest of hydramind imports
and unit-tests with zero network deps. L-B does NOT go through the proxy: the
proxy is routine-completions-only.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

from .types import Call, CallResult, Lane
from .models import cost_from_table
from .transport import call_key, _loads_or_text

# Anthropic batch custom_id must match ^[a-zA-Z0-9_-]{1,64}$ — our call_keys
# ("solo:trump_2026:0007") contain ':' and can exceed 64 chars. Sanitize for the
# wire and keep a reverse map so results reconcile back to the real call_key.
_CID_RX = re.compile(r"[^a-zA-Z0-9_-]")


def _cid(key: str) -> str:
    return _CID_RX.sub("_", key)[:64]


# Proxy aliases → real Anthropic model ids. L-B bypasses the proxy and calls the
# Anthropic API directly, so it must translate the proxy alias (what strategies
# carry) to the concrete upstream id — mirrors the proxy's own model_list. Keeps
# L-P and L-B pointed at the SAME model so the equivalence check is meaningful.
ALIAS_TO_ANTHROPIC = {
    "claude-haiku": "claude-haiku-4-5-20251001",
    "claude-sonnet": "claude-sonnet-4-6",
    "claude-opus": "claude-opus-4-8",
}


class AnthropicBatchBackend:
    def __init__(self, key_env: str = "ANTHROPIC_API_KEY", max_tokens: int = 1024,
                 poll_interval: float = 5.0, poll_timeout: float = 24 * 3600,
                 response_parser=None, alias_map: dict | None = None):
        self.key_env = key_env
        self.alias_map = alias_map or ALIAS_TO_ANTHROPIC
        self.max_tokens = max_tokens
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.response_parser = response_parser or (lambda d: d)

    def _client(self):
        key = os.environ.get(self.key_env)
        if not key:
            raise RuntimeError(
                f"L-B key env '{self.key_env}' not set; source it from the "
                f"repo-scoped .env (CW-12).")
        try:
            import anthropic  # lazy
        except ImportError as e:  # pragma: no cover - env-dependent
            raise RuntimeError("anthropic SDK not installed; `uv add anthropic` "
                               "to enable L-B.") from e
        return anthropic.Anthropic(api_key=key)

    def _request(self, c: Call) -> dict:
        return {
            "custom_id": _cid(call_key(c)),
            "params": {
                "model": self.alias_map.get(c.binding.model, c.binding.model),
                "max_tokens": self.max_tokens,
                "temperature": 0,
                "system": c.prompt.template,
                "messages": [{"role": "user", "content": json.dumps(c.inputs or {})}],
            },
        }

    def run_batch(self, calls: list[Call]) -> dict[str, CallResult]:
        client = self._client()
        by_cid = {_cid(call_key(c)): c for c in calls}   # sanitized custom_id → Call
        batch = client.messages.batches.create(
            requests=[self._request(c) for c in calls])
        deadline = time.monotonic() + self.poll_timeout
        while True:
            b = client.messages.batches.retrieve(batch.id)
            if b.processing_status == "ended":
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"L-B batch {batch.id} did not end in time")
            time.sleep(self.poll_interval)

        out: dict[str, CallResult] = {}       # keyed by ORIGINAL call_key
        for entry in client.messages.batches.results(batch.id):
            call = by_cid[entry.custom_id]
            key = call_key(call)
            if entry.result.type != "succeeded":
                out[key] = CallResult(call=call, output={"error": entry.result.type},
                                      lane=Lane.L_B, raw=entry)
                continue
            msg = entry.result.message
            text = "".join(blk.text for blk in msg.content if blk.type == "text")
            usage = getattr(msg, "usage", None)
            t_in = getattr(usage, "input_tokens", 0) if usage else 0
            t_out = getattr(usage, "output_tokens", 0) if usage else 0
            # L-B has no LiteLLM cost surface, so cost is the rate-table estimate on
            # the captured tokens (cost_source="table"; "none" for untabled models).
            est = cost_from_table(call.binding.model, t_in, t_out)
            out[key] = CallResult(
                call=call,
                output=self.response_parser(_loads_or_text(text)),
                lane=Lane.L_B,
                tokens_in=t_in, tokens_out=t_out,
                cost_usd=est or 0.0,
                cost_source=("table" if est is not None else "none"),
                returned_model=getattr(msg, "model", ""),
                raw=entry,
            )
        return out
