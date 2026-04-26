"""Layer 1d adapter wiring tests for ground-truth URL intersection.

Verifies that all four adapters (Anthropic, OpenAI, Gemini, Grok) correctly
populate ``web_sources`` (post-intersection), ``model_reported_sources``
(raw model output), and ``stripped_source_count`` across both single-claim
and multi-claim parse paths. Each adapter has 4 test cases:

  1. exact tool/model URL match → kept
  2. fabricated URL (model emits, tool didn't return) → stripped
  3. mixed (some real, some fabricated) → kept ∩ tool only
  4. model omits web_sources entirely → fallback / empty path

Total: 16 tests.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from truthbot.models import Claim


# ── Helpers ───────────────────────────────────────────────────────────────

def _claim(idx: int, text: str = "Test claim") -> Claim:
    return Claim(
        id=f"c-{idx:08d}",
        transcript_id="t-test",
        text=text,
        speaker="Test Speaker",
    )


def _verdict_json(claims: list[Claim], web_sources_per_claim: list[list[str] | None]) -> str:
    rows = []
    for c, ws in zip(claims, web_sources_per_claim):
        row = {
            "claim_id": c.id,
            "label": "True",
            "confidence": "High",
            "explanation": "Test.",
        }
        if ws is not None:
            row["web_sources"] = ws
        rows.append(row)
    return json.dumps(rows)


# ── Anthropic ─────────────────────────────────────────────────────────────

def _fake_anthropic_response(verdict_text: str, retrieved_urls: list[str]) -> Any:  # type: ignore[name-defined]
    """Build a minimal fake Anthropic ``message`` response."""
    blocks = []
    for u in retrieved_urls:
        blocks.append({"type": "server_tool_use", "name": "web_search"})
        blocks.append({
            "type": "web_search_tool_result",
            "content": [{"url": u, "title": "x", "page_age": "1d"}],
        })
    blocks.append({"type": "text", "text": verdict_text})
    return {
        "content": blocks,
        "usage": {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 0},
        "model": "claude-opus-4-7",
    }


from typing import Any  # noqa: E402  (placed after fakes for narrative flow)


class TestAnthropicMultiVerdictGrounding:
    """Layer 1d: AnthropicAdapter.parse_multi_batch_response."""

    def _adapter(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from truthbot.verify.adapters.anthropic import AnthropicAdapter
        return AnthropicAdapter()

    def test_exact_match_kept(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        claims = [_claim(1)]
        retrieved = ["https://www.bls.gov/cpi.htm"]
        body = _verdict_json(claims, [["https://www.bls.gov/cpi.htm"]])
        v = adapter.parse_multi_batch_response(
            _fake_anthropic_response(body, retrieved), claims
        )[0]
        assert v.web_sources == ["https://www.bls.gov/cpi.htm"]
        assert v.model_reported_sources == ["https://www.bls.gov/cpi.htm"]
        assert v.stripped_source_count == 0

    def test_fabricated_stripped(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        claims = [_claim(1)]
        retrieved = ["https://www.bls.gov/cpi.htm"]
        body = _verdict_json(claims, [["https://fake.example/halluc"]])
        v = adapter.parse_multi_batch_response(
            _fake_anthropic_response(body, retrieved), claims
        )[0]
        assert v.web_sources == []
        assert v.model_reported_sources == ["https://fake.example/halluc"]
        assert v.stripped_source_count == 1

    def test_mixed_intersection(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        claims = [_claim(1)]
        retrieved = ["https://www.bls.gov/cpi.htm", "https://cbo.gov/x"]
        body = _verdict_json(
            claims, [["https://bls.gov/cpi.htm", "https://halluc.example", "https://cbo.gov/x"]]
        )
        v = adapter.parse_multi_batch_response(
            _fake_anthropic_response(body, retrieved), claims
        )[0]
        # bls.gov and cbo.gov match (www-prefix collapsed); halluc stripped.
        assert v.web_sources == ["https://bls.gov/cpi.htm", "https://cbo.gov/x"]
        assert v.stripped_source_count == 1

    def test_omitted_falls_back_to_index_zero(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        claims = [_claim(1), _claim(2)]
        retrieved = ["https://www.bls.gov/cpi.htm"]
        body = _verdict_json(claims, [None, None])  # no web_sources emitted
        verdicts = adapter.parse_multi_batch_response(
            _fake_anthropic_response(body, retrieved), claims
        )
        # index-0 backfill from grounding URLs
        assert verdicts[0].web_sources == ["https://www.bls.gov/cpi.htm"]
        assert verdicts[1].web_sources == []  # don't fabricate per-sibling


# ── OpenAI ────────────────────────────────────────────────────────────────

def _fake_openai_response(verdict_text: str, retrieved_urls: list[str]) -> dict:
    """Build a minimal fake OpenAI Responses-API body."""
    output: list[dict] = []
    for _ in retrieved_urls:
        output.append({"type": "web_search_call"})
    annotations = [{"url": u} for u in retrieved_urls]
    output.append(
        {
            "type": "message",
            "content": [
                {
                    "type": "output_text",
                    "text": verdict_text,
                    "annotations": annotations,
                }
            ],
        }
    )
    return {
        "output": output,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        "model": "gpt-5.4-2026-03-05",
    }


class TestOpenAIMultiVerdictGrounding:
    """Layer 1d: OpenAIAdapter.parse_multi_batch_response."""

    def _adapter(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        from truthbot.verify.adapters.openai import OpenAIAdapter
        return OpenAIAdapter()

    def test_exact_match_kept(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        claims = [_claim(1)]
        retrieved = ["https://www.bls.gov/cpi.htm"]
        body = _verdict_json(claims, [["https://www.bls.gov/cpi.htm"]])
        v = adapter.parse_multi_batch_response(
            _fake_openai_response(body, retrieved), claims
        )[0]
        assert v.web_sources == ["https://www.bls.gov/cpi.htm"]
        assert v.stripped_source_count == 0

    def test_fabricated_stripped(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        claims = [_claim(1)]
        retrieved = ["https://www.bls.gov/cpi.htm"]
        body = _verdict_json(claims, [["https://halluc.example/y"]])
        v = adapter.parse_multi_batch_response(
            _fake_openai_response(body, retrieved), claims
        )[0]
        assert v.web_sources == []
        assert v.stripped_source_count == 1
        assert v.model_reported_sources == ["https://halluc.example/y"]

    def test_mixed_intersection(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        claims = [_claim(1)]
        retrieved = ["https://www.bls.gov/cpi.htm"]
        body = _verdict_json(
            claims, [["https://www.bls.gov/cpi.htm", "https://halluc/x"]]
        )
        v = adapter.parse_multi_batch_response(
            _fake_openai_response(body, retrieved), claims
        )[0]
        assert v.web_sources == ["https://www.bls.gov/cpi.htm"]
        assert v.stripped_source_count == 1

    def test_omitted_falls_back_to_index_zero(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        claims = [_claim(1), _claim(2)]
        retrieved = ["https://www.bls.gov/cpi.htm"]
        body = _verdict_json(claims, [None, None])
        verdicts = adapter.parse_multi_batch_response(
            _fake_openai_response(body, retrieved), claims
        )
        assert verdicts[0].web_sources == ["https://www.bls.gov/cpi.htm"]
        assert verdicts[1].web_sources == []


# ── Gemini (multi via call_multi seam) ────────────────────────────────────


class TestGeminiMultiVerdictGrounding:
    """Layer 1d: GeminiAdapter.call_multi (via fake genai client)."""

    def _adapter(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        from truthbot.verify.adapters.gemini import GeminiAdapter
        # Force-reset cached redirect cache to avoid cross-test pollution.
        GeminiAdapter._url_cache = None
        return GeminiAdapter()

    def _fake_response(self, verdict_text: str, urls: list[str]) -> MagicMock:
        response = MagicMock()
        response.text = verdict_text
        candidate = MagicMock()
        gm = MagicMock()
        gm.web_search_queries = ["test query"]
        chunks = []
        for u in urls:
            chunk = MagicMock()
            chunk.web = MagicMock(uri=u)
            chunks.append(chunk)
        gm.grounding_chunks = chunks
        candidate.grounding_metadata = gm
        candidate.content = MagicMock(parts=[MagicMock(text=verdict_text)])
        response.candidates = [candidate]
        response.usage_metadata = MagicMock(
            prompt_token_count=200,
            candidates_token_count=100,
            cached_content_token_count=0,
        )
        return response

    def _install_fake(self, response, monkeypatch):
        from truthbot.verify.adapters import gemini as gemini_mod

        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = response
        fake_client.caches.create.side_effect = Exception("no cache in tests")

        fake_genai = MagicMock()
        fake_genai.Client.return_value = fake_client
        fake_types = MagicMock()

        # google.genai.types.{Tool,GoogleSearch,GenerateContentConfig,...}
        for attr in (
            "Tool",
            "GoogleSearch",
            "GenerateContentConfig",
            "CreateCachedContentConfig",
            "ThinkingConfig",
        ):
            setattr(fake_types, attr, MagicMock())

        import sys
        # Monkeypatch the imported google.genai modules used inside call_multi.
        monkeypatch.setitem(sys.modules, "google", MagicMock(genai=fake_genai))
        monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
        monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    def test_exact_match_kept(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        claims = [_claim(1)]
        retrieved = ["https://www.bls.gov/cpi.htm"]
        body = _verdict_json(claims, [["https://www.bls.gov/cpi.htm"]])
        self._install_fake(self._fake_response(body, retrieved), monkeypatch)

        verdicts = adapter.call_multi(
            claims, {claims[0].id: []}, inject_evidence=False
        )
        v = verdicts[0]
        assert v.web_sources == ["https://www.bls.gov/cpi.htm"]
        assert v.stripped_source_count == 0

    def test_fabricated_stripped(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        claims = [_claim(1)]
        retrieved = ["https://www.bls.gov/cpi.htm"]
        body = _verdict_json(claims, [["https://halluc.example/g"]])
        self._install_fake(self._fake_response(body, retrieved), monkeypatch)

        verdicts = adapter.call_multi(
            claims, {claims[0].id: []}, inject_evidence=False
        )
        v = verdicts[0]
        assert v.web_sources == []
        assert v.stripped_source_count == 1
        assert v.model_reported_sources == ["https://halluc.example/g"]

    def test_mixed_intersection(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        claims = [_claim(1)]
        retrieved = ["https://www.bls.gov/cpi.htm"]
        body = _verdict_json(
            claims, [["https://www.bls.gov/cpi.htm", "https://halluc.example/h"]]
        )
        self._install_fake(self._fake_response(body, retrieved), monkeypatch)
        verdicts = adapter.call_multi(
            claims, {claims[0].id: []}, inject_evidence=False
        )
        v = verdicts[0]
        assert v.web_sources == ["https://www.bls.gov/cpi.htm"]
        assert v.stripped_source_count == 1

    def test_omitted_falls_back_to_index_zero(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        claims = [_claim(1), _claim(2)]
        retrieved = ["https://www.bls.gov/cpi.htm"]
        body = _verdict_json(claims, [None, None])
        self._install_fake(self._fake_response(body, retrieved), monkeypatch)
        verdicts = adapter.call_multi(
            claims, {c.id: [] for c in claims}, inject_evidence=False
        )
        assert verdicts[0].web_sources == ["https://www.bls.gov/cpi.htm"]
        assert verdicts[1].web_sources == []


# ── Grok ──────────────────────────────────────────────────────────────────


class TestGrokMultiVerdictGrounding:
    """Layer 1d: GrokAdapter.call_multi (xAI Responses API via fake client)."""

    def _adapter(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        from truthbot.verify.adapters.grok import GrokAdapter
        return GrokAdapter()

    def _patch_client(self, adapter, verdict_text: str, urls: list[str], monkeypatch):
        # GrokAdapter's _call_with_search returns a tuple
        # (verdict_text, urls, tool_count, usage). Patch it to bypass HTTP.
        def fake_call_with_search(
            client, user_msg, max_output_tokens=None, max_tool_calls=None
        ):
            return verdict_text, list(urls), len(urls), MagicMock(
                input_tokens=100, output_tokens=50,
                prompt_tokens=100, completion_tokens=50,
            )

        monkeypatch.setattr(adapter, "_call_with_search", fake_call_with_search)

    def test_exact_match_kept(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        claims = [_claim(1)]
        retrieved = ["https://www.bls.gov/cpi.htm"]
        body = _verdict_json(claims, [["https://www.bls.gov/cpi.htm"]])
        self._patch_client(adapter, body, retrieved, monkeypatch)

        verdicts = adapter.call_multi(
            claims, {claims[0].id: []}, inject_evidence=False
        )
        v = verdicts[0]
        assert v.web_sources == ["https://www.bls.gov/cpi.htm"]
        assert v.stripped_source_count == 0

    def test_fabricated_stripped(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        claims = [_claim(1)]
        retrieved = ["https://www.bls.gov/cpi.htm"]
        body = _verdict_json(claims, [["https://halluc.example/k"]])
        self._patch_client(adapter, body, retrieved, monkeypatch)
        verdicts = adapter.call_multi(
            claims, {claims[0].id: []}, inject_evidence=False
        )
        v = verdicts[0]
        assert v.web_sources == []
        assert v.stripped_source_count == 1

    def test_mixed_intersection(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        claims = [_claim(1)]
        retrieved = ["https://www.bls.gov/cpi.htm", "https://cbo.gov/y"]
        body = _verdict_json(
            claims, [["https://www.bls.gov/cpi.htm", "https://halluc/x", "https://cbo.gov/y"]]
        )
        self._patch_client(adapter, body, retrieved, monkeypatch)
        verdicts = adapter.call_multi(
            claims, {claims[0].id: []}, inject_evidence=False
        )
        v = verdicts[0]
        assert set(v.web_sources) == {"https://www.bls.gov/cpi.htm", "https://cbo.gov/y"}
        assert v.stripped_source_count == 1

    def test_omitted_falls_back_to_index_zero(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        claims = [_claim(1), _claim(2)]
        retrieved = ["https://www.bls.gov/cpi.htm"]
        body = _verdict_json(claims, [None, None])
        self._patch_client(adapter, body, retrieved, monkeypatch)
        verdicts = adapter.call_multi(
            claims, {c.id: [] for c in claims}, inject_evidence=False
        )
        assert verdicts[0].web_sources == ["https://www.bls.gov/cpi.htm"]
        assert verdicts[1].web_sources == []
