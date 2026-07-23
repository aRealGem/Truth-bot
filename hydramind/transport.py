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
import subprocess
import time
import urllib.error
import urllib.request
from typing import Callable, Optional, Protocol

from .types import Call, Wave, Lane, Kind, CallResult, Spec

# Proxy HTTP statuses worth retrying: 429 (rate limit) + transient 5xx. A 4xx
# other than 429 is a caller error (bad key/model/body) — fail fast, don't retry.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


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
                 response_parser: Optional[Callable[[dict], dict]] = None,
                 max_retries: int = 3, backoff_base: float = 0.5,
                 backoff_cap: float = 30.0,
                 sleep_fn: Callable[[float], None] = time.sleep):
        self.base_url = base_url.rstrip("/")
        self.key_env = key_env
        self.timeout = timeout
        self.response_parser = response_parser or (lambda d: d)
        # Bounded exponential backoff on a rate-limited/transient proxy: a burst of
        # 429s (e.g. back-to-back runs sharing one virtual key) shouldn't kill a live
        # run mid-flight. Honors Retry-After when the proxy sends it. sleep_fn is
        # injectable so tests don't actually sleep.
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self._sleep = sleep_fn

    def _retry_delay(self, err: urllib.error.HTTPError, attempt: int) -> float:
        """Seconds to wait before retry ``attempt`` (0-based). Prefers the proxy's
        Retry-After (integer seconds); else exponential backoff capped."""
        retry_after = err.headers.get("Retry-After") if err.headers else None
        if retry_after:
            try:
                return min(float(int(retry_after)), self.backoff_cap)
            except (TypeError, ValueError):
                pass    # HTTP-date form or garbage → fall through to backoff
        return min(self.backoff_base * (2 ** attempt), self.backoff_cap)

    def _post(self, req: urllib.request.Request) -> tuple[str, dict]:
        """POST with bounded retry on rate-limit/transient errors. Returns
        (body_text, lowercased_headers); re-raises after exhausting retries."""
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return (resp.read().decode("utf-8"),
                            {k.lower(): v for k, v in resp.headers.items()})
            except urllib.error.HTTPError as err:
                if err.code not in _RETRYABLE_STATUS or attempt >= self.max_retries:
                    raise
                self._sleep(self._retry_delay(err, attempt))
            except urllib.error.URLError:
                # Connection-level blip (proxy restarting, transient network) — retry
                # with plain backoff; a persistent failure still raises after the cap.
                if attempt >= self.max_retries:
                    raise
                self._sleep(min(self.backoff_base * (2 ** attempt), self.backoff_cap))
        raise RuntimeError("unreachable: _post retry loop exited without return/raise")

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
        body, headers = self._post(req)
        data = json.loads(body)
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)
        parsed = self.response_parser(_loads_or_text(content))
        cost_usd, cost_source = extract_cost(
            data, headers, usage, tokens_in, tokens_out, call.binding.model)
        return CallResult(
            call=call, output=parsed, lane=Lane.L_P,
            tokens_in=tokens_in, tokens_out=tokens_out,
            cost_usd=cost_usd, cost_source=cost_source,
            returned_model=data.get("model", ""),
            raw=data,
        )


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def extract_cost(data: dict, headers: dict, usage: dict,
                 tokens_in: int, tokens_out: int, model: str) -> tuple[float, str]:
    """Resolve a call's (cost_usd, cost_source) from a LiteLLM proxy reply.

    LiteLLM exposes the computed cost of a request in more than one place; over
    the HTTP proxy the canonical surface is the `x-litellm-response-cost`
    response header, with `_hidden_params.response_cost` and a usage-embedded
    cost as body-side fallbacks (the shape varies by proxy version, so we probe
    all three before falling back to the local rate table). Preference order:

      1. `x-litellm-response-cost` header      -> "proxy"
      2. `_hidden_params.response_cost` (body) -> "proxy"
      3. `usage.{response_cost,cost,total_cost}`-> "proxy"
      4. local rate table on captured tokens   -> "table"
      5. nothing resolvable                     -> "none" (0.0)

    A proxy-reported 0.0 is honored as "proxy" (it is what the proxy said), not
    downgraded to a table estimate."""
    c = _to_float(headers.get("x-litellm-response-cost"))
    if c is not None:
        return c, "proxy"
    hidden = data.get("_hidden_params") or {}
    c = _to_float(hidden.get("response_cost"))
    if c is not None:
        return c, "proxy"
    for key in ("response_cost", "cost", "total_cost"):
        c = _to_float(usage.get(key))
        if c is not None:
            return c, "proxy"
    from .models import cost_from_table
    c = cost_from_table(model, tokens_in, tokens_out)
    if c is not None:
        return c, "table"
    return 0.0, "none"


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


# ── L-T: not built in C1 ──────────────────────────────────────────────────────

class LaneNotAvailable(RuntimeError):
    pass


def _l_t_stub(call: Call) -> CallResult:
    raise LaneNotAvailable("L-T (native web_search) is a C1 stub — evidence "
                           "acquisition is Layer C's job, not built here.")


def _l_w_stub(call: Call) -> CallResult:
    raise LaneNotAvailable("L-W agentic TOOL_TASK is ON HOLD; only worker "
                           "COMPLETIONS (worker_models aliases) ride L-W.")


# ── L-W: Claude worker completions (P67.9 / T3.1) ─────────────────────────────

class WorkerCallError(RuntimeError):
    """A worker completion failed after retries. Fail LOUD: a seat that
    silently returns nothing would corrupt the PCA quorum; the P67.3
    journal/resume machinery is the recovery path, not a fake result."""


class ClaudeWorkerCompletion:
    """L-W completion backend: the ``claude`` CLI headless on the Max
    subscription (T3.1) — zero marginal cost, so the prod roster's frontier
    proposer seat doesn't ride API billing. Mirrors the validated R1
    ``ClaudeWorkerRetriever`` pattern: ANTHROPIC_API_KEY is STRIPPED from the
    child env so the CLI can never fall back to API-key billing.

    Semantic note (§2): L-P sends ``template`` as the system message and the
    JSON-rendered inputs as the user message; the CLI takes one prompt string,
    so this lane concatenates the same two parts in the same order. No tools
    are granted (headless auto-denies) — seat completions are closed over
    their inputs; evidence acquisition stays Layer C's job."""

    def __init__(self, model: str = "opus", timeout_s: float = 300.0,
                 response_parser: Optional[Callable[[dict], dict]] = None,
                 max_retries: int = 2,
                 run_fn: Optional[Callable[..., "subprocess.CompletedProcess"]] = None):
        self.model = model
        self.timeout_s = timeout_s
        self.response_parser = response_parser or (lambda d: d)
        self.max_retries = max_retries
        self._run = run_fn or subprocess.run

    def _invoke(self, prompt: str) -> "subprocess.CompletedProcess":
        env = dict(os.environ)
        env.pop("ANTHROPIC_API_KEY", None)   # subscription auth, never API billing
        return self._run(
            ["claude", "-p", prompt, "--output-format", "json",
             "--model", self.model],
            capture_output=True, text=True, timeout=self.timeout_s, env=env)

    def __call__(self, call: Call) -> CallResult:
        prompt = (call.prompt.template + "\n\nINPUT:\n"
                  + json.dumps(call.inputs or {}, ensure_ascii=False))
        last_err = ""
        for _attempt in range(self.max_retries + 1):
            try:
                proc = self._invoke(prompt)
            except (OSError, subprocess.TimeoutExpired) as exc:
                last_err = f"invocation failed: {exc}"
                continue
            if proc.returncode != 0:
                last_err = f"exit {proc.returncode}: {(proc.stderr or '')[-300:]}"
                continue
            try:
                envelope = json.loads(proc.stdout)
            except json.JSONDecodeError:
                envelope = {}
            text = envelope.get("result") if isinstance(envelope, dict) else None
            if not text:
                last_err = f"no result in worker envelope: {(proc.stdout or '')[:200]}"
                continue
            usage = envelope.get("usage") or {}
            model_usage = envelope.get("modelUsage") or {}
            returned = next(iter(model_usage), "") if isinstance(model_usage, dict) else ""
            return CallResult(
                call=call, output=self.response_parser(_loads_or_text(text)),
                lane=Lane.L_W,
                tokens_in=int(usage.get("input_tokens") or 0),
                tokens_out=int(usage.get("output_tokens") or 0),
                # Max-subscription lane: no marginal spend, and the envelope's
                # total_cost_usd is an API-rate hypothetical — never bank it.
                cost_usd=0.0, cost_source="subscription",
                returned_model=returned, raw=envelope,
            )
        raise WorkerCallError(
            f"L-W worker completion failed after {self.max_retries + 1} "
            f"attempts ({call.role}:{call.item_id}): {last_err}")


# ── router ────────────────────────────────────────────────────────────────────

class Transport:
    def __init__(self, completion_fn: CompletionFn,
                 batch_backend: Optional[BatchBackend] = None,
                 worker_fn: Optional[CompletionFn] = None,
                 worker_models: frozenset[str] = frozenset()):
        self.completion_fn = completion_fn
        self.batch_backend = batch_backend
        # L-W completion routing (P67.9): calls whose binding.model is in
        # worker_models ride worker_fn instead of the proxy. The alias set
        # lives with the caller (hydramind.models.WORKER_ALIASES) so the
        # router stays policy-free.
        self.worker_fn = worker_fn
        self.worker_models = frozenset(worker_models)

    def _is_worker(self, c: Call) -> bool:
        return c.binding.model in self.worker_models

    def lane_for_wave(self, wave: Wave, spec: Spec) -> Lane:
        """Wave-level lane decision for COMPLETION calls (TOOL_TASK always L-W).
        Worker-alias completions are excluded — they always ride L-W and never
        count toward L-B lot size."""
        eligible = set(spec.batch.get("eligible_waves", []))
        min_lot = int(spec.batch.get("min_lot", 10**9))
        completions = [c for c in wave.calls
                       if c.kind == Kind.COMPLETION and not self._is_worker(c)]
        if (wave.batchable and wave.tag in eligible
                and self.batch_backend is not None
                and len(completions) >= min_lot):
            return Lane.L_B
        return Lane.L_P

    def dispatch(self, wave: Wave, spec: Spec) -> list[CallResult]:
        results: list[CallResult] = []

        tool_calls = [c for c in wave.calls if c.kind == Kind.TOOL_TASK]
        worker_calls = [c for c in wave.calls
                        if c.kind == Kind.COMPLETION and self._is_worker(c)]
        completions = [c for c in wave.calls
                       if c.kind == Kind.COMPLETION and not self._is_worker(c)]

        for c in tool_calls:                       # L-W agentic (on hold)
            results.append(_l_w_stub(c))

        for c in worker_calls:                     # L-W completions
            if self.worker_fn is None:
                raise LaneNotAvailable(
                    f"call {call_key(c)} binds worker alias "
                    f"'{c.binding.model}' but no worker_fn is wired — refuse "
                    f"to reroute a subscription seat onto a billed lane.")
            results.append(self.worker_fn(c))

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
