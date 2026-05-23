"""Pin ``tool_choice='required'`` on every OpenAI web_search call site.

2026-05-23 substance-track fix for the temporal-dismissal failure mode
documented in eval/sotu-2026/temporal-regressions-runbook.md. The 0/4
first live run on the regression set had OpenAI returning 0 model-reported
URLs on post-cutoff claims — the model silently skipped invoking
``web_search`` because the tool was attached with the default
``tool_choice='auto'``. Forcing the tool via ``tool_choice='required'``
makes refusing-to-search a hard contract violation rather than a quiet
verdict-quality regression.

OpenAI Responses API docs (developers.openai.com/api/docs/guides/tools-web-search):
"With ``tool_choice: 'auto'``, search is optional. Use ``tool_choice:
'required'`` or a specific web search tool choice when search must run."
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from truthbot.models import Claim
from truthbot.verify.adapters.openai import OpenAIAdapter


@pytest.fixture(autouse=True)
def _api_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _claim(text: str = "test claim") -> Claim:
    return Claim(transcript_id="t1", text=text, speaker="Test")


def test_batch_payload_forces_tool_choice_required() -> None:
    adapter = OpenAIAdapter()
    payload = adapter.build_batch_payload(_claim(), evidence=[])

    assert payload["tool_choice"] == "required", (
        "Single-claim batch payload must force web_search to fire — otherwise "
        "the temporal-dismissal failure mode silently returns Unverifiable "
        "without ever invoking the tool."
    )
    assert payload["tools"] == [{"type": "web_search"}]


def test_multi_batch_payload_forces_tool_choice_required() -> None:
    adapter = OpenAIAdapter()
    claims = [_claim("claim a"), _claim("claim b"), _claim("claim c")]
    payload = adapter.build_multi_batch_payload(claims, evidence_by_claim={})

    assert payload["tool_choice"] == "required"
    assert payload["tools"] == [{"type": "web_search"}]


def test_live_call_passes_tool_choice_required(monkeypatch) -> None:
    """``_call_with_search`` (the live Responses path) must also force the tool.

    Captures the ``kwargs`` actually handed to ``client.responses.create`` so a
    silent removal of ``tool_choice`` from the kwargs dict in
    ``_call_with_search`` would fail the test.
    """
    captured: dict[str, Any] = {}

    class _FakeResponses:
        def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                status="completed",
                output=[
                    SimpleNamespace(
                        type="message",
                        content=[
                            SimpleNamespace(
                                type="output_text",
                                text='{"label":"True","confidence":"Medium","explanation":"x","caveats":"","web_sources":[]}',
                                annotations=[],
                            )
                        ],
                    )
                ],
                usage=SimpleNamespace(
                    input_tokens=10,
                    output_tokens=5,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=0),
                ),
            )

    class _FakeClient:
        def __init__(self) -> None:
            self.responses = _FakeResponses()

    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda **_kw: _FakeClient())

    adapter = OpenAIAdapter()
    # First attempt: response_format may not be supported; the adapter
    # retries without it. Either path must still pass tool_choice.
    adapter._call_with_search(_FakeClient(), user_msg="hello")

    assert captured.get("tool_choice") == "required", (
        f"live path dropped tool_choice; captured kwargs keys: {sorted(captured)}"
    )
    assert captured.get("tools") == [{"type": "web_search"}]
