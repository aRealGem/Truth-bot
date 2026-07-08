"""
Transport router (design §2, §4.4). The engine hands the router a Wave; the
router groups calls by lane and dispatches:

  L-P  proxy single completions (LiteLLM, routine-completions-only)  — LIVE
  L-B  native provider batch (Anthropic Message Batches interface)   — interface
  L-T  native provider tools (web_search)                            — stub (C1)
  L-W  Claude Code worker (Max sub), agentic TOOL_TASK               — on hold (C1)

Batch is a property of the WAVE: a wave routes to L-B only when it is batchable,
its tag is in spec.batch.eligible_waves, and the lot size ≥ min_lot. Otherwise
each completion goes L-P. Prompts/schemas are identical across L-P and L-B — the
lane is purely a cost/latency choice (§2 semantic invariant).
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable, Optional, Protocol

from .types import Call, Wave, Lane, Kind, CallResult, Spec


# ── lane backends (injectable for tests) ──────────────────────────────────────

CompletionFn = Callable[[Call], CallResult]   # L-P single completion


def call_key(c: Call) -> str:
    """Unique key for reconciling a call within a wave/batch. A single item can
    appear under multiple roles in one wave (pca wave1: proposer + critic), so
    the key must include the role — this is the custom_id used by L-B."""
    return f"{c.role}:{c.item_id}"


class BatchBackend(Protocol):
    """L-B interface: submit a homogeneous lot, poll, reconcile by custom_id."""
    def run_batch(self, calls: list[Call]) -> dict[str, CallResult]: ...


# ── L-P: LiteLLM proxy client (OpenAI-compatible) ─────────────────────────────

class ProxyCompletion:
    """Live L-P client against the LiteLLM proxy. Uses stdlib urllib (no extra
    deps). The virtual key is read from the env var named by `key_env` — sourced
    from the repo-scoped .env (CW-12), never hunted from credential stores. If
    the key is absent, calls raise (fail closed) rather than silently no-op."""

    def __init__(self, base_url: str = "http://127.0.0.1:4141",
                 key_env: str = "LITELLM_KEY", timeout: float = 60.0,
                 response_parser: Optional[Callable[[dict], dict]] = None):
        self.base_url = base_url.rstrip("/")
        self.key_env = key_env
        self.timeout = timeout
        self.response_parser = response_parser or (lambda d: d)

    def __call__(self, call: Call) -> CallResult:
        key = os.environ.get(self.key_env)
        if not key:
            raise RuntimeError(
                f"L-P proxy key env '{self.key_env}' not set; source it from the "
                f"repo-scoped .env (CW-12) before a live run.")
        body = {
            "model": call.binding.model,
            "messages": [
                {"role": "system", "content": call.prompt.template},
                {"role": "user", "content": json.dumps(call.inputs or {})},
            ],
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        parsed = self.response_parser(_loads_or_text(content))
        return CallResult(
            call=call, output=parsed, lane=Lane.L_P,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            cost_usd=float(usage.get("response_cost", 0.0) or 0.0),
            returned_model=data.get("model", ""),
            raw=data,
        )


import re as _re
_FENCE_RX = _re.compile(r"```(?:json)?\s*(.*?)\s*```", _re.DOTALL)
_OBJ_RX = _re.compile(r"\{.*\}", _re.DOTALL)


def _loads_or_text(s: str) -> dict:
    """Parse a model reply into a dict, tolerating markdown fences and prose
    around the JSON object (models don't always honor 'JSON only')."""
    for candidate in (s,
                      (_FENCE_RX.search(s).group(1) if _FENCE_RX.search(s) else None),
                      (_OBJ_RX.search(s).group(0) if _OBJ_RX.search(s) else None)):
        if not candidate:
            continue
        try:
            v = json.loads(candidate)
            if isinstance(v, dict):
                return v
        except Exception:
            continue
    return {"text": s}


# ── L-T / L-W: not built in C1 ────────────────────────────────────────────────

class LaneNotAvailable(RuntimeError):
    pass


def _l_t_stub(call: Call) -> CallResult:
    raise LaneNotAvailable("L-T (native web_search) is a C1 stub — evidence "
                           "acquisition is Layer C's job, not built here.")


def _l_w_stub(call: Call) -> CallResult:
    raise LaneNotAvailable("L-W (Claude Code worker) is ON HOLD for C1.")


# ── router ────────────────────────────────────────────────────────────────────

class Transport:
    def __init__(self, completion_fn: CompletionFn,
                 batch_backend: Optional[BatchBackend] = None):
        self.completion_fn = completion_fn
        self.batch_backend = batch_backend

    def lane_for_wave(self, wave: Wave, spec: Spec) -> Lane:
        """Wave-level lane decision for COMPLETION calls (TOOL_TASK always L-W)."""
        eligible = set(spec.batch.get("eligible_waves", []))
        min_lot = int(spec.batch.get("min_lot", 10**9))
        completions = [c for c in wave.calls if c.kind == Kind.COMPLETION]
        if (wave.batchable and wave.tag in eligible
                and self.batch_backend is not None
                and len(completions) >= min_lot):
            return Lane.L_B
        return Lane.L_P

    def dispatch(self, wave: Wave, spec: Spec) -> list[CallResult]:
        results: list[CallResult] = []

        tool_calls = [c for c in wave.calls if c.kind == Kind.TOOL_TASK]
        completions = [c for c in wave.calls if c.kind == Kind.COMPLETION]

        for c in tool_calls:                       # L-W (on hold)
            results.append(_l_w_stub(c))

        if not completions:
            return results

        lane = self.lane_for_wave(wave, spec)
        if lane == Lane.L_B:
            reconciled = self.batch_backend.run_batch(completions)  # keyed by call_key
            for c in completions:
                cr = reconciled[call_key(c)]
                cr.lane = Lane.L_B
                results.append(cr)
        else:
            for c in completions:
                results.append(self.completion_fn(c))
        return results
