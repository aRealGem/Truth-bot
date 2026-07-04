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
import time
from typing import Optional

from .types import Call, CallResult, Lane
from .transport import call_key, _loads_or_text


class AnthropicBatchBackend:
    def __init__(self, key_env: str = "ANTHROPIC_API_KEY", max_tokens: int = 1024,
                 poll_interval: float = 5.0, poll_timeout: float = 24 * 3600,
                 response_parser=None):
        self.key_env = key_env
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
            "custom_id": call_key(c),
            "params": {
                "model": c.binding.model,
                "max_tokens": self.max_tokens,
                "temperature": 0,
                "system": c.prompt.template,
                "messages": [{"role": "user", "content": json.dumps(c.inputs or {})}],
            },
        }

    def run_batch(self, calls: list[Call]) -> dict[str, CallResult]:
        client = self._client()
        by_key = {call_key(c): c for c in calls}
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

        out: dict[str, CallResult] = {}
        for entry in client.messages.batches.results(batch.id):
            cid = entry.custom_id
            call = by_key[cid]
            if entry.result.type != "succeeded":
                out[cid] = CallResult(call=call, output={"error": entry.result.type},
                                      lane=Lane.L_B, raw=entry)
                continue
            msg = entry.result.message
            text = "".join(blk.text for blk in msg.content if blk.type == "text")
            usage = getattr(msg, "usage", None)
            out[cid] = CallResult(
                call=call,
                output=self.response_parser(_loads_or_text(text)),
                lane=Lane.L_B,
                tokens_in=getattr(usage, "input_tokens", 0) if usage else 0,
                tokens_out=getattr(usage, "output_tokens", 0) if usage else 0,
                raw=entry,
            )
        return out
