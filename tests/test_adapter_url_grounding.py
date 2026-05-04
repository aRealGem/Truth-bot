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


# ── Universal trust-when-fired fallback (2026-05-01) ─────────────────────────
#
# Direct coverage for the fallback now living inside apply_url_grounding so
# every adapter benefits — not just OpenAI multi-claim batch through
# build_multi_verdicts. The arm-C empirical run (4d6b204a) showed Gemini
# stayed at 100% strip after the build_multi_verdicts-only fix because
# Gemini's batch + live parsers call apply_url_grounding directly. Pushing
# the fallback into apply_url_grounding fixes that universally. See
# metrics/adapter_interpretability/strip_audit_2026-05.md.


from truthbot.verify.adapters.base import apply_url_grounding  # noqa: E402


class TestApplyUrlGroundingTrustWhenFired:
    def test_fallback_fires_when_tool_count_positive_and_extraction_empty(self):
        """Tool fired (count > 0) but extractor returned empty list — bypass
        intersection, trust model. This is the path that fixes Gemini's 100%
        strip rate (redirect resolver returns empty after vertexaisearch
        cookie failure)."""
        raw = {
            "label": "True",
            "confidence": "High",
            "explanation": "x",
            "web_sources": [
                "https://www.bls.gov/news.release/cpi.htm",
                "https://www.bls.gov/news.release/archives/cpi_12182025.pdf",
            ],
        }
        ws, mrs, stripped = apply_url_grounding(raw, [], tool_call_count=3)
        assert ws == [
            "https://www.bls.gov/news.release/cpi.htm",
            "https://www.bls.gov/news.release/archives/cpi_12182025.pdf",
        ]
        assert mrs == ws  # MRS mirrors WS so audit trail still records the model emission
        assert stripped == 0

    def test_fallback_does_not_fire_when_tool_count_zero(self):
        """Strict intersection still strips when tools never fired —
        anti-fabrication unchanged for runs where the model declined to
        search. Default tool_call_count=0 (legacy callers) preserves the
        prior strict-strip semantics."""
        raw = {
            "label": "True",
            "confidence": "High",
            "explanation": "x",
            "web_sources": ["https://example.com/cited-without-search"],
        }
        ws, mrs, stripped = apply_url_grounding(raw, [], tool_call_count=0)
        assert ws == []
        assert mrs == ["https://example.com/cited-without-search"]
        assert stripped == 1

    def test_fallback_does_not_fire_when_extraction_non_empty(self):
        """When the harness captured at least one URL, full intersection
        runs — the fallback is exclusively for the harness-empty case.
        Preserves anti-fabrication when capture worked (xAI / Anthropic
        baseline)."""
        raw = {
            "label": "True",
            "confidence": "High",
            "explanation": "x",
            "web_sources": [
                "https://www.bls.gov/cpi.htm",
                "https://halluc.example/fabricated",
            ],
        }
        ws, mrs, stripped = apply_url_grounding(
            raw, ["https://www.bls.gov/cpi.htm"], tool_call_count=2
        )
        assert ws == ["https://www.bls.gov/cpi.htm"]
        assert "https://halluc.example/fabricated" in mrs
        assert stripped == 1

    def test_fallback_does_not_fire_when_model_omitted_web_sources(self):
        """If the model omitted web_sources entirely, the existing
        legacy backfill path (returns tool_retrieved as ws, empty mrs)
        wins — there's nothing to trust on the model side."""
        raw = {"label": "Unverifiable", "confidence": "Low", "explanation": "x"}
        ws, mrs, stripped = apply_url_grounding(raw, [], tool_call_count=3)
        assert ws == []
        assert mrs == []
        assert stripped == 0

    def test_default_tool_call_count_param_preserves_legacy_behavior(self):
        """Default ``tool_call_count=0`` keeps callers that don't pass the
        kwarg on the strict-intersection path. Backward-compat regression
        guard for any external/legacy caller of apply_url_grounding."""
        raw = {
            "label": "True",
            "confidence": "High",
            "explanation": "x",
            "web_sources": ["https://www.bls.gov/cpi.htm"],
        }
        ws, mrs, stripped = apply_url_grounding(raw, [])  # no tool_call_count
        assert ws == []
        assert stripped == 1

    def test_explicit_empty_web_sources_array_treated_as_no_citations(self):
        """``web_sources: []`` (model said "nothing relevant") is NOT the
        same as omitted web_sources. Strict semantics: no URLs to keep,
        no URLs to strip — fallback has nothing to act on."""
        raw = {
            "label": "Unverifiable",
            "confidence": "Low",
            "explanation": "x",
            "web_sources": [],
        }
        ws, mrs, stripped = apply_url_grounding(raw, [], tool_call_count=3)
        assert ws == []
        assert mrs == []
        assert stripped == 0


class TestApplyUrlGroundingStripNoKeepDiagnostic:
    """Diagnostic WARNING fires when strip-everything happens — disambiguates
    the two post-fallback paths that produce 100% strip rates: model didn't
    search (tool_count=0), or model searched but emitted near-miss URLs the
    harness didn't capture (tool_count>0, tool_retrieved non-empty, no
    overlap). Behavior change: none. Log line: one per claim that hits the
    case. Used to drive the next arm-D-style probe to clean attribution."""

    def test_warning_fires_on_case_A_no_tool_call(self, caplog):
        """Case (A): model emitted URLs without invoking search. Strict
        anti-fabrication strip is correct; WARNING records ``tool_count=0``
        so the operator sees this case and knows to fix it prompt-side."""
        import logging
        raw = {
            "label": "True",
            "confidence": "High",
            "explanation": "x",
            "web_sources": ["https://example.gov/no-search-cited"],
        }
        with caplog.at_level(logging.WARNING, logger="truthbot.verify.adapters.base"):
            ws, mrs, stripped = apply_url_grounding(
                raw, [], tool_call_count=0
            )
        assert ws == [] and stripped == 1
        matches = [r for r in caplog.records if "strip-no-keep" in r.message]
        assert len(matches) == 1
        msg = matches[0].message
        assert "tool_count=0" in msg
        assert "retrieved=0" in msg
        assert "reported=1" in msg
        assert "example.gov/no-search-cited" in msg

    def test_warning_fires_on_case_B_near_miss(self, caplog):
        """Case (B): model invoked search and the harness captured URLs,
        but the model's emitted citations don't match (.htm vs .pdf
        same-release pattern). WARNING records both samples so the
        operator can eyeball whether to fuzzy-match."""
        import logging
        raw = {
            "label": "True",
            "confidence": "High",
            "explanation": "x",
            "web_sources": [
                "https://www.bls.gov/news.release/archives/cpi_01132026.htm",
            ],
        }
        with caplog.at_level(logging.WARNING, logger="truthbot.verify.adapters.base"):
            ws, mrs, stripped = apply_url_grounding(
                raw,
                ["https://www.bls.gov/news.release/archives/cpi_12182025.pdf"],
                tool_call_count=2,
            )
        assert ws == [] and stripped == 1
        matches = [r for r in caplog.records if "strip-no-keep" in r.message]
        assert len(matches) == 1
        msg = matches[0].message
        assert "tool_count=2" in msg
        assert "retrieved=1" in msg
        assert "cpi_01132026.htm" in msg  # sample reported
        assert "cpi_12182025.pdf" in msg  # sample retrieved

    def test_warning_does_not_fire_on_partial_keep(self, caplog):
        """Partial strips (some kept, some stripped) are less alarming —
        harness is mostly working. The diagnostic only fires for the
        strip-everything case to keep log noise down on healthy runs."""
        import logging
        kept_url = "https://example.gov/real"
        raw = {
            "label": "True",
            "confidence": "High",
            "explanation": "x",
            "web_sources": [kept_url, "https://example.com/halluc"],
        }
        with caplog.at_level(logging.WARNING, logger="truthbot.verify.adapters.base"):
            ws, mrs, stripped = apply_url_grounding(
                raw, [kept_url], tool_call_count=2
            )
        assert ws == [kept_url] and stripped == 1
        assert not [r for r in caplog.records if "strip-no-keep" in r.message]

    def test_warning_does_not_fire_when_fallback_path_takes_over(self, caplog):
        """Trust-when-fired path returns before the diagnostic check —
        the model's URLs were trusted, so there's no strip to log."""
        import logging
        raw = {
            "label": "True",
            "confidence": "High",
            "explanation": "x",
            "web_sources": ["https://www.bls.gov/cpi.htm"],
        }
        with caplog.at_level(logging.WARNING, logger="truthbot.verify.adapters.base"):
            ws, mrs, stripped = apply_url_grounding(
                raw, [], tool_call_count=3
            )
        assert ws == ["https://www.bls.gov/cpi.htm"] and stripped == 0
        assert not [r for r in caplog.records if "strip-no-keep" in r.message]

    def test_warning_does_not_fire_when_model_emitted_nothing(self, caplog):
        """``web_sources: []`` or omitted — no URLs to strip, no
        diagnostic line. Healthy "model said nothing relevant" path."""
        import logging
        raw = {
            "label": "Unverifiable",
            "confidence": "Low",
            "explanation": "x",
            "web_sources": [],
        }
        with caplog.at_level(logging.WARNING, logger="truthbot.verify.adapters.base"):
            apply_url_grounding(raw, [], tool_call_count=3)
        assert not [r for r in caplog.records if "strip-no-keep" in r.message]


class TestApplyUrlGroundingTrackingParamStripping:
    """Arm-E (run 5d78f4df) revealed OpenAI's web_search consistently
    appends ``?utm_source=openai`` to retrieved URLs while the model
    cites the canonical form. Under literal comparison, every such URL
    flagged as a strip even though the model + tool retrieved the same
    page. ``_normalize_url_for_compare`` now strips tracking/attribution
    query params before comparison; values on real (non-tracking) query
    params are preserved."""

    def test_utm_source_openai_decoration_does_not_break_match(self):
        from truthbot.verify.adapters.base import _normalize_url_for_compare
        a = _normalize_url_for_compare("https://www.whitehouse.gov/freedom250/")
        b = _normalize_url_for_compare(
            "https://www.whitehouse.gov/freedom250/?utm_source=openai"
        )
        assert a == b
        assert a  # non-empty

    def test_full_utm_quintet_dropped(self):
        from truthbot.verify.adapters.base import _normalize_url_for_compare
        plain = _normalize_url_for_compare("https://example.com/article")
        decorated = _normalize_url_for_compare(
            "https://example.com/article"
            "?utm_source=openai&utm_medium=email&utm_campaign=truth"
            "&utm_content=cta&utm_term=fact"
        )
        assert plain == decorated

    def test_click_id_trackers_dropped(self):
        from truthbot.verify.adapters.base import _normalize_url_for_compare
        plain = _normalize_url_for_compare("https://news.example/2026/cpi")
        decorated = _normalize_url_for_compare(
            "https://news.example/2026/cpi?gclid=AB123&fbclid=XY456&msclkid=Z9"
        )
        assert plain == decorated

    def test_real_query_params_preserved_distinguish_urls(self):
        """Non-tracking query params (e.g. resource IDs, page numbers)
        must STILL distinguish two URLs — anti-fabrication intersection
        depends on it. Locks in that the stripper is param-name-keyed,
        not blanket."""
        from truthbot.verify.adapters.base import _normalize_url_for_compare
        a = _normalize_url_for_compare("https://eia.gov/todayinenergy/detail.php?id=55099")
        b = _normalize_url_for_compare("https://eia.gov/todayinenergy/detail.php?id=65184")
        assert a != b
        # And both stay non-empty.
        assert a and b

    def test_mixed_tracking_and_real_params_keeps_only_real(self):
        from truthbot.verify.adapters.base import _normalize_url_for_compare
        a = _normalize_url_for_compare("https://example.com/x?id=42")
        b = _normalize_url_for_compare(
            "https://example.com/x?id=42&utm_source=openai&fbclid=ABC"
        )
        assert a == b

    def test_param_name_match_is_case_insensitive(self):
        """Real-world URLs sometimes have UTM_SOURCE in caps. Match on
        param name should be case-insensitive."""
        from truthbot.verify.adapters.base import _normalize_url_for_compare
        plain = _normalize_url_for_compare("https://example.com/y")
        decorated = _normalize_url_for_compare(
            "https://example.com/y?UTM_SOURCE=openai&Fbclid=ABC"
        )
        assert plain == decorated

    def test_ground_truth_intersection_now_keeps_utm_decorated_pair(self):
        """End-to-end: model emits canonical URL, tool retrieves
        utm-decorated form (the OpenAI pattern from arm-E). Intersection
        keeps the model URL; strip count is 0. Pre-fix: 1 stripped."""
        from truthbot.verify.adapters.base import ground_truth_web_sources
        kept, stripped = ground_truth_web_sources(
            ["https://www.whitehouse.gov/freedom250/"],
            ["https://www.whitehouse.gov/freedom250/?utm_source=openai"],
        )
        assert kept == ["https://www.whitehouse.gov/freedom250/"]
        assert stripped == 0


class TestGeminiResolveModelReportedRedirects:
    """Arm-E showed Gemini emits raw vertexaisearch redirect URLs in its
    own ``web_sources`` JSON. The harness already resolves these on the
    tool-retrieved side; without symmetric resolution on the model side,
    the intersection can never overlap (resolved host vs raw redirect
    token are different strings). Helper resolves in-place; unresolvable
    redirects are dropped (they're opaque + session-cookied — useless
    citations regardless)."""

    @staticmethod
    def _build_adapter(monkeypatch):
        """Construct a GeminiAdapter without invoking the real client."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        from truthbot.verify.adapters.gemini import GeminiAdapter
        return GeminiAdapter()

    def test_resolves_redirect_to_underlying_url(self, monkeypatch):
        """The canonical case: model emits a redirect URL, resolver
        succeeds, redirect is replaced by the resolved URL in raw['web_sources']."""
        adapter = self._build_adapter(monkeypatch)

        from truthbot.verify.adapters import gemini as gemini_mod
        def fake_resolve(url, *, cache=None, timeout=5.0):
            # Resolve only the test redirect; pass others through.
            if "grounding-api-redirect/RESOLVES" in url:
                return "https://www.bls.gov/news.release/cpi.htm"
            return None
        monkeypatch.setattr(gemini_mod, "resolve_gemini_redirect", fake_resolve)

        raw = {
            "web_sources": [
                "https://vertexaisearch.cloud.google.com/grounding-api-redirect/RESOLVES",
                "https://www.bls.gov/cpi.htm",  # already canonical, pass-through
            ]
        }
        adapter._resolve_model_reported_redirects(raw)
        assert raw["web_sources"] == [
            "https://www.bls.gov/news.release/cpi.htm",
            "https://www.bls.gov/cpi.htm",
        ]

    def test_drops_unresolvable_redirect(self, monkeypatch):
        """Redirects that can't be resolved (session-cookie failure,
        timeout, etc.) are dropped — they're opaque tokens unusable as
        citations and would only inflate the model_reported_sources
        list with junk."""
        adapter = self._build_adapter(monkeypatch)

        from truthbot.verify.adapters import gemini as gemini_mod
        monkeypatch.setattr(
            gemini_mod, "resolve_gemini_redirect",
            lambda url, *, cache=None, timeout=5.0: None,
        )

        raw = {
            "web_sources": [
                "https://vertexaisearch.cloud.google.com/grounding-api-redirect/UNRESOLVABLE",
                "https://www.bls.gov/cpi.htm",
            ]
        }
        adapter._resolve_model_reported_redirects(raw)
        # The redirect is dropped; the canonical URL passes through.
        assert raw["web_sources"] == ["https://www.bls.gov/cpi.htm"]

    def test_no_op_when_web_sources_missing(self, monkeypatch):
        adapter = self._build_adapter(monkeypatch)
        raw = {"label": "Unverifiable"}  # no web_sources key
        adapter._resolve_model_reported_redirects(raw)
        assert "web_sources" not in raw

    def test_no_op_when_no_redirects_present(self, monkeypatch):
        adapter = self._build_adapter(monkeypatch)
        # If no redirects, the resolver shouldn't even be called.
        from truthbot.verify.adapters import gemini as gemini_mod
        called = {"n": 0}

        def spy(url, *, cache=None, timeout=5.0):
            called["n"] += 1
            return url

        monkeypatch.setattr(gemini_mod, "resolve_gemini_redirect", spy)
        raw = {
            "web_sources": [
                "https://www.bls.gov/cpi.htm",
                "https://apnews.com/article/foo",
            ]
        }
        adapter._resolve_model_reported_redirects(raw)
        assert called["n"] == 0
        assert raw["web_sources"] == [
            "https://www.bls.gov/cpi.htm",
            "https://apnews.com/article/foo",
        ]

    def test_skips_non_string_entries(self, monkeypatch):
        adapter = self._build_adapter(monkeypatch)
        raw = {"web_sources": ["https://x.com", None, 42, "https://y.com"]}
        adapter._resolve_model_reported_redirects(raw)
        assert raw["web_sources"] == ["https://x.com", "https://y.com"]
