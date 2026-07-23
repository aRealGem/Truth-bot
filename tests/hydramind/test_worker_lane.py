"""L-W worker completions (P67.9 / T3.1): ClaudeWorkerCompletion + the
transport's worker-alias routing. The prod roster's opus-worker proposer must
ride the subscription CLI (key-stripped), never the proxy, and never L-B."""
import json
import subprocess

import pytest

from hydramind.models import WORKER_ALIASES, binding_from_alias
from hydramind.transport import (ClaudeWorkerCompletion, LaneNotAvailable,
                                 Transport, WorkerCallError)
from hydramind.types import Call, CallResult, Kind, Lane, PromptRef, Spec


def _call(alias: str, role: str = "proposer", item_id: str = "c1") -> Call:
    return Call(role=role, item_id=item_id,
                prompt=PromptRef.of("p", "You are seat {role}."),
                binding=binding_from_alias(alias),
                inputs={"claim": "x"})


def _spec() -> Spec:
    from hydramind.registry import load_registry
    return load_registry()["pca"]


def _envelope(result_obj, usage=None, model="claude-opus-4-8"):
    return json.dumps({
        "type": "result", "subtype": "success",
        "result": json.dumps(result_obj),
        "usage": usage or {"input_tokens": 100, "output_tokens": 20},
        "modelUsage": {model: {}},
        "total_cost_usd": 1.23,   # API-rate hypothetical — must NOT be banked
    })


def _fake_run(envelope_stdout, seen_env=None, returncode=0):
    def run(cmd, capture_output, text, timeout, env):
        if seen_env is not None:
            seen_env.update(env)
            seen_env["__cmd__"] = cmd
        return subprocess.CompletedProcess(cmd, returncode,
                                           stdout=envelope_stdout, stderr="boom")
    return run


def test_worker_completion_parses_and_never_bills():
    seen = {}
    fn = ClaudeWorkerCompletion(
        run_fn=_fake_run(_envelope({"verdict": "TRUE", "confidence": 0.9}),
                         seen_env=seen))
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-should-be-stripped")
    r = fn(_call("opus-worker"))
    assert r.lane is Lane.L_W
    assert r.output == {"verdict": "TRUE", "confidence": 0.9}
    assert r.cost_usd == 0.0 and r.cost_source == "subscription"
    assert r.tokens_in == 100 and r.tokens_out == 20
    assert "opus" in r.returned_model
    # subscription auth: the CLI child env must not carry the API key
    assert "ANTHROPIC_API_KEY" not in seen
    assert seen["__cmd__"][0] == "claude" and "--output-format" in seen["__cmd__"]


def test_worker_completion_applies_response_parser():
    fn = ClaudeWorkerCompletion(
        run_fn=_fake_run(_envelope({"verdict": "true"})),
        response_parser=lambda d: {"verdict": str(d.get("verdict", "")).upper()})
    r = fn(_call("opus-worker"))
    assert r.output == {"verdict": "TRUE"}


def test_worker_completion_fails_loud_after_retries():
    calls = {"n": 0}

    def run(cmd, capture_output, text, timeout, env):
        calls["n"] += 1
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="dead")

    fn = ClaudeWorkerCompletion(run_fn=run, max_retries=2)
    with pytest.raises(WorkerCallError, match="exit 1"):
        fn(_call("opus-worker"))
    assert calls["n"] == 3                     # initial + 2 retries


def test_transport_routes_worker_alias_to_worker_fn():
    routed = {"worker": [], "proxy": []}

    def worker_fn(c):
        routed["worker"].append(c.binding.model)
        return CallResult(call=c, output={}, lane=Lane.L_W,
                          cost_usd=0.0, cost_source="subscription")

    def proxy_fn(c):
        routed["proxy"].append(c.binding.model)
        return CallResult(call=c, output={}, lane=Lane.L_P)

    t = Transport(completion_fn=proxy_fn, worker_fn=worker_fn,
                  worker_models=WORKER_ALIASES)
    from hydramind.types import Wave
    wave = Wave(calls=[_call("opus-worker", role="proposer"),
                       _call("grok-4.3", role="critic")], batchable=False)
    results = t.dispatch(wave, _spec())
    assert routed["worker"] == ["opus-worker"]
    assert routed["proxy"] == ["grok-4.3"]
    lanes = {r.call.binding.model: r.lane for r in results}
    assert lanes["opus-worker"] is Lane.L_W and lanes["grok-4.3"] is Lane.L_P


def test_transport_refuses_worker_alias_without_worker_fn():
    t = Transport(completion_fn=lambda c: (_ for _ in ()).throw(
        AssertionError("proxy lane must not see a worker alias")),
        worker_models=WORKER_ALIASES)
    from hydramind.types import Wave
    with pytest.raises(LaneNotAvailable, match="opus-worker"):
        t.dispatch(Wave(calls=[_call("opus-worker")], batchable=False), _spec())


def test_worker_calls_do_not_count_toward_batch_lot():
    """A wave of worker completions must never tip the L-B min_lot check."""
    class Boom:
        def run_batch(self, calls):
            raise AssertionError("worker calls must not reach L-B")

    t = Transport(completion_fn=lambda c: CallResult(call=c, output={}, lane=Lane.L_P),
                  batch_backend=Boom(),
                  worker_fn=lambda c: CallResult(call=c, output={}, lane=Lane.L_W,
                                                 cost_source="subscription"),
                  worker_models=WORKER_ALIASES)
    spec = _spec()
    from hydramind.types import Wave
    tag = (spec.batch.get("eligible_waves") or ["wave1"])[0]
    calls = [_call("opus-worker", item_id=f"c{i}") for i in range(50)]
    wave = Wave(calls=calls, batchable=True, tag=tag)
    assert t.lane_for_wave(wave, spec) is Lane.L_P   # no proxy completions ⇒ no L-B
    results = t.dispatch(wave, spec)
    assert all(r.lane is Lane.L_W for r in results)
