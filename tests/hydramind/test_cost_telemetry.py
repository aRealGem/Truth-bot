"""P96.2.1 cost telemetry: proxy-cost parse, rate-table fallback, cost_source
provenance in the manifest, and a synthetic over-budget ceiling trip. All offline
— no proxy is contacted (the one ProxyCompletion test mocks urllib)."""
import json

import pytest

from hydramind import HydraMind, TaskItem
from hydramind.types import Call, CallResult, Lane
from hydramind.registry import load_registry, SPECS_DIR
from hydramind.transport import Transport, ProxyCompletion, extract_cost, call_key
from hydramind.manifest import NullSpendSink
from hydramind.models import cost_from_table


# ── extract_cost: surface precedence ──────────────────────────────────────────

def test_extract_cost_prefers_proxy_header():
    cost, src = extract_cost(
        data={"_hidden_params": {"response_cost": 0.99}},
        headers={"x-litellm-response-cost": "0.0123"},
        usage={"response_cost": 0.5}, tokens_in=10, tokens_out=5, model="claude-haiku")
    assert src == "proxy" and cost == pytest.approx(0.0123)


def test_extract_cost_hidden_params_fallback():
    cost, src = extract_cost(
        data={"_hidden_params": {"response_cost": 0.042}},
        headers={}, usage={}, tokens_in=10, tokens_out=5, model="claude-haiku")
    assert src == "proxy" and cost == pytest.approx(0.042)


def test_extract_cost_usage_embedded_fallback():
    cost, src = extract_cost(
        data={}, headers={}, usage={"cost": 0.007},
        tokens_in=10, tokens_out=5, model="claude-haiku")
    assert src == "proxy" and cost == pytest.approx(0.007)


def test_extract_cost_table_fallback_when_proxy_silent():
    # mistral is in the rate table → table-sourced estimate from tokens
    cost, src = extract_cost(
        data={}, headers={}, usage={}, tokens_in=1_000_000, tokens_out=0,
        model="mistral")
    assert src == "table"
    assert cost == pytest.approx(cost_from_table("mistral", 1_000_000, 0))
    assert cost > 0.0


def test_extract_cost_none_when_unknown_model_and_no_proxy():
    cost, src = extract_cost(
        data={}, headers={}, usage={}, tokens_in=100, tokens_out=100,
        model="some-unregistered-model")
    assert src == "none" and cost == 0.0


def test_extract_cost_zero_from_proxy_stays_proxy():
    # a legitimately-reported 0.0 is proxy truth, not a table estimate
    cost, src = extract_cost(
        data={}, headers={"x-litellm-response-cost": "0"}, usage={},
        tokens_in=10, tokens_out=5, model="mistral")
    assert src == "proxy" and cost == 0.0


# ── ProxyCompletion end-to-end (mocked urllib, reads the header) ───────────────

class _FakeResp:
    def __init__(self, body: dict, headers: dict):
        self._body = json.dumps(body).encode("utf-8")
        self.headers = headers

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_proxycompletion_reads_response_cost_header(monkeypatch):
    from hydramind import transport
    body = {"choices": [{"message": {"content": '{"label": "opinion"}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            "model": "claude-haiku"}
    headers = {"X-LiteLLM-Response-Cost": "0.00456"}   # mixed case → lowercased
    monkeypatch.setattr(transport.urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeResp(body, headers))
    monkeypatch.setenv("LITELLM_KEY", "sk-test")
    pc = ProxyCompletion()
    call = Call(role="solo", item_id="s1", prompt=_dummy_prompt(),
                binding=_dummy_binding("claude-haiku"))
    cr = pc(call)
    assert cr.cost_source == "proxy"
    assert cr.cost_usd == pytest.approx(0.00456)
    assert cr.tokens_in == 12 and cr.tokens_out == 3


# ── cost_source lands in the manifest + coverage ──────────────────────────────

def _table_costed(tokens_in=100, tokens_out=50):
    """A fake L-P completion fn that stamps a table-sourced cost per call."""
    def fn(call: Call) -> CallResult:
        c = cost_from_table(call.binding.model, tokens_in, tokens_out) or 0.0
        return CallResult(call=call, output={"label": "opinion"}, lane=Lane.L_P,
                          cost_usd=c, cost_source="table",
                          tokens_in=tokens_in, tokens_out=tokens_out,
                          returned_model=call.binding.model)
    return fn


def test_manifest_records_cost_source_and_coverage():
    reg = load_registry(SPECS_DIR)
    hm = HydraMind(reg, Transport(completion_fn=_table_costed()),
                   spend_sink=NullSpendSink())
    items = [TaskItem("s1", {"text": "a"}), TaskItem("s2", {"text": "b"})]
    _, manifest = hm.run("classify", items, "single",
                         tune={"roles.solo.providers": ["anthropic"],
                               "roles.solo.tier": "cheap"})
    assert all(c.cost_source == "table" for c in manifest.cost_records)
    assert manifest.cost_source_tally.get("table") == 2
    assert manifest.cost_source_coverage() == {"table": 2}


# ── ceiling trips on a synthetic over-budget run (real accumulated cost) ───────

def test_ceiling_trips_on_table_costed_over_budget():
    reg = load_registry(SPECS_DIR)
    # single ceiling is $2.00; 100 huge table-costed calls blow past it
    hm = HydraMind(reg, Transport(completion_fn=_table_costed(tokens_in=10_000_000)),
                   spend_sink=NullSpendSink())
    items = [TaskItem(f"s{i}", {"text": "t"}) for i in range(100)]
    _, manifest = hm.run("classify", items, "single",
                         tune={"roles.solo.providers": ["anthropic"],
                               "roles.solo.tier": "cheap"})
    assert manifest.halted
    assert "ceiling" in manifest.halt_reason
    assert manifest.total_cost_usd > 2.00
    assert manifest.cost_source_tally.get("table")   # cost was real (table-sourced)


# ── helpers ────────────────────────────────────────────────────────────────────

def _dummy_prompt():
    from hydramind.types import PromptRef
    return PromptRef.of("t", "{input}")


def _dummy_binding(alias):
    from hydramind.models import binding_from_alias
    return binding_from_alias(alias)
