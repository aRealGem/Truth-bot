"""Evidence-v2 retriever tests (P67.8 / PR-5, remediation T2.5-T2.6).

Pins: the shared retrieval prompt is speaker-blind and era-scoped; shortlist
items convert to Evidence with retrieval-side fact-checker exclusion; R1
strips ANTHROPIC_API_KEY so Lane-Worker can only run on subscription auth;
R2 falls down its model chain; R3 refuses until D1 is decided; the T2.6
contamination guard raises (not warns) on gold leakage.
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from truthbot.models import SourceTier
from truthbot.verify.retrievers import (
    ClaudeWorkerRetriever,
    ContaminationError,
    OpenAIBrowsingRetriever,
    PendingDecisionError,
    R3Retriever,
    assert_no_contamination,
    build_retrieval_prompt,
    items_to_evidence,
    _parse_shortlist_json,
)

UTT = date(2026, 2, 24)
WINDOW = (date(2024, 1, 1), date(2026, 5, 1))


def test_prompt_is_speaker_blind_and_era_scoped() -> None:
    p = build_retrieval_prompt("Gas is below $2.30 in most states.",
                               utterance=UTT, window=WINDOW)
    assert "2024-01-01" in p and "2026-05-01" in p
    assert "2026-03-03" in p          # fair-game end stated
    assert "fact-check" in p.lower()  # instructed to avoid fact-checkers
    # no speaker parameter even exists — the prompt builder takes none
    import inspect
    assert "speaker" not in inspect.signature(build_retrieval_prompt).parameters


def test_items_to_evidence_converts_and_enforces_exclusion() -> None:
    items = [
        {"url": "https://www.bls.gov/cpi/latest.htm", "date": "2026-02-11",
         "stance": "refutes", "one_line_why": "CPI shows prices higher"},
        {"url": "https://www.politifact.com/factchecks/2026/x/", "date": "2026-02-25",
         "stance": "refutes", "one_line_why": "ruling"},
        {"url": "not-a-url", "date": None, "stance": "context", "one_line_why": ""},
        {"url": "https://blog.example.com/post", "date": "bad-date",
         "stance": "supports", "one_line_why": "anecdote"},
    ]
    evs = items_to_evidence(items, retriever_label="R1")
    urls = [e.source_url for e in evs]
    assert "https://www.politifact.com/factchecks/2026/x/" not in urls  # T2.1
    assert evs[0].source_tier == SourceTier.GOVERNMENT
    assert evs[0].supports_claim is False
    assert evs[0].published_at.date() == date(2026, 2, 11)
    assert evs[0].snippet.startswith("[2026-02-11]")
    assert evs[1].published_at is None  # bad date tolerated
    assert len(evs) == 2


def test_parse_shortlist_handles_fences_and_garbage() -> None:
    good = '```json\n{"items": [{"url": "https://a.com/x"}]}\n```'
    assert _parse_shortlist_json(good)[0]["url"] == "https://a.com/x"
    assert _parse_shortlist_json("no json here") == []


def test_r1_strips_api_key_and_parses_cli_envelope(monkeypatch) -> None:
    captured = {}

    class FakeProc:
        returncode = 0
        stderr = ""
        stdout = json.dumps({"result": json.dumps(
            {"items": [{"url": "https://apnews.com/article/x",
                        "date": "2026-02-20", "stance": "supports",
                        "one_line_why": "wire report"}]})})

    def fake_run(cmd, capture_output, text, timeout, env):
        captured["cmd"] = cmd
        captured["env"] = env
        return FakeProc()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setattr("subprocess.run", fake_run)
    evs = ClaudeWorkerRetriever().shortlist("claim", utterance=UTT, window=WINDOW)
    assert "ANTHROPIC_API_KEY" not in captured["env"]  # subscription-only lane
    assert captured["cmd"][0] == "claude" and "-p" in captured["cmd"]
    assert "WebSearch" in captured["cmd"]
    assert [e.source_url for e in evs] == ["https://apnews.com/article/x"]
    assert evs[0].source_tier == SourceTier.WIRE


def test_r2_falls_down_model_chain(monkeypatch) -> None:
    calls = []

    def fake_post(self, model, prompt):
        calls.append(model)
        if model == "gpt-5.5":
            raise RuntimeError("model_not_found")
        return {"output": [{"content": [{"type": "output_text", "text": json.dumps(
            {"items": [{"url": "https://www.reuters.com/world/story",
                        "date": "2026-02-19", "stance": "context",
                        "one_line_why": "background"}]})}]}]}

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(OpenAIBrowsingRetriever, "_post", fake_post)
    evs = OpenAIBrowsingRetriever().shortlist("claim", utterance=UTT, window=WINDOW)
    assert calls == ["gpt-5.5", "gpt-5.4"]
    assert evs[0].source_tier == SourceTier.WIRE
    assert evs[0].supports_claim is None  # context stance


def test_r3_refuses_until_d1() -> None:
    with pytest.raises(PendingDecisionError, match="D1"):
        R3Retriever().shortlist("claim")


def test_contamination_guard_raises_on_gold_leak() -> None:
    prompt = build_retrieval_prompt("NATO was created to secure peace.",
                                    utterance=UTT, window=WINDOW)
    # clean prompt passes
    assert_no_contamination(prompt, ["Matches 1949 treaty's stated purpose"])
    with pytest.raises(ContaminationError):
        assert_no_contamination(
            prompt + " Matches 1949 treaty's stated purpose",
            ["Matches 1949 treaty's stated purpose"])
    # short fragments (< 12 chars) are ignored — "true" appears everywhere
    assert_no_contamination(prompt, ["true"])


def test_brave_classify_tier_is_shared_module_level() -> None:
    from truthbot.verify.sources.brave import classify_tier
    assert classify_tier("https://www.bls.gov/x") == SourceTier.GOVERNMENT
    assert classify_tier("https://www.govtech.com/x") == SourceTier.OTHER
