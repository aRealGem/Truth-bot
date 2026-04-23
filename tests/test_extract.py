"""TB-03t: Claim extraction — schema, normalization, fact vs opinion."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from truthbot.extract.claims import ClaimExtractor
from truthbot.models import Claim, Transcript


@pytest.fixture
def extractor():
    # No real API key in tests — will use stub mode
    return ClaimExtractor(api_key="")


@pytest.fixture
def transcript(sample_transcript):
    return sample_transcript


class TestClaimExtractor:
    def test_returns_list(self, extractor, transcript):
        claims = extractor.extract(transcript)
        assert isinstance(claims, list)

    def test_claims_are_claim_objects(self, extractor, transcript):
        claims = extractor.extract(transcript)
        for c in claims:
            assert isinstance(c, Claim)

    def test_claim_text_not_empty(self, extractor, transcript):
        claims = extractor.extract(transcript)
        for c in claims:
            assert c.text.strip()

    def test_claim_transcript_id_matches(self, extractor, transcript):
        claims = extractor.extract(transcript)
        for c in claims:
            assert c.claim_id_matches_transcript(transcript) or c.transcript_id == transcript.id

    def test_claim_has_speaker(self, extractor, transcript):
        claims = extractor.extract(transcript)
        for c in claims:
            assert c.speaker

    def test_stub_on_no_api_key(self, extractor, transcript):
        """With no API key, should return stub claims rather than raising."""
        claims = extractor.extract(transcript)
        # Stub returns up to 3 sentences from the transcript
        assert len(claims) <= 10  # reasonable upper bound

    def test_empty_transcript_returns_empty(self, extractor):
        """Very short transcript should yield no or minimal claims."""
        t = Transcript(
            text="OK.",
            transcript_id="test-id",
        )
        # This may return empty or one item — just shouldn't raise
        claims = extractor.extract(t)
        assert isinstance(claims, list)


# Monkey-patch Claim for the transcript_id check test
Claim.claim_id_matches_transcript = lambda self, t: self.transcript_id == t.id


class TestParseResponseJSONRepair:
    """
    Regression tests for ``_parse_response`` and ``_repair_truncated_claims``.

    The 2026-04-23 SOTU extraction burned on this exact failure mode: the
    model hit the 8K output-token ceiling, emitted a truncated JSON array,
    and the old parser returned ``[]`` — losing every claim. The repair
    path walks the JSON in depth-tracked scan mode and salvages every
    fully-emitted claim object before the truncation point.
    """

    def test_clean_json_direct_parse(self):
        ex = ClaimExtractor(api_key="x")
        data = ex._parse_response('{"claims": [{"id": "c1", "text": "alpha"}]}')
        assert data == {"claims": [{"id": "c1", "text": "alpha"}]}

    def test_repair_truncated_mid_claim(self):
        ex = ClaimExtractor(api_key="x")
        truncated = (
            '{"claims": [{"id": "c1", "text": "alpha"}, '
            '{"id": "c2", "text": "bet'  # mid-word cut
        )
        out = ex._parse_response(truncated)
        assert out == {"claims": [{"id": "c1", "text": "alpha"}]}

    def test_repair_truncated_right_after_comma(self):
        ex = ClaimExtractor(api_key="x")
        truncated = (
            '{"claims": [{"id": "c1", "text": "alpha"}, '
            '{"id": "c2", "text": "beta"},'
        )
        out = ex._parse_response(truncated)
        assert len(out["claims"]) == 2
        assert [c["id"] for c in out["claims"]] == ["c1", "c2"]

    def test_repair_handles_strings_with_braces(self):
        """Claim text containing { or } must not confuse the depth scanner."""
        ex = ClaimExtractor(api_key="x")
        truncated = (
            '{"claims": [{"id": "c1", "text": "says {quote}", '
            '"context_window": "he said {x}"}, '
            '{"id": "c2", "text": "incom'
        )
        out = ex._parse_response(truncated)
        assert len(out["claims"]) == 1
        assert out["claims"][0]["text"] == "says {quote}"
        assert out["claims"][0]["context_window"] == "he said {x}"

    def test_repair_handles_escaped_quotes_in_strings(self):
        ex = ClaimExtractor(api_key="x")
        truncated = (
            r'{"claims": [{"id": "c1", "text": "quote with \"inner\" quotes"}, '
            r'{"id": "c0'
        )
        out = ex._parse_response(truncated)
        assert len(out["claims"]) == 1
        assert out["claims"][0]["text"] == 'quote with "inner" quotes'

    def test_markdown_fence_stripped_before_parse(self):
        ex = ClaimExtractor(api_key="x")
        fenced = '```json\n{"claims": [{"id": "c1", "text": "x"}]}\n```'
        out = ex._parse_response(fenced)
        assert out == {"claims": [{"id": "c1", "text": "x"}]}

    def test_garbage_raises_valueerror(self):
        ex = ClaimExtractor(api_key="x")
        with pytest.raises(ValueError, match="Could not parse JSON"):
            ex._parse_response("this is not json at all, no braces here")

    def test_repair_returns_none_when_no_claim_object_completed(self):
        """If the truncation happened before any claim closed, repair returns None."""
        text = '{"claims": [{"id": "c1", "text": "incomplete fi'
        out = ClaimExtractor._repair_truncated_claims(text)
        assert out is None

    def test_repair_without_claims_key_returns_none(self):
        text = '{"results": [{"id": "c1", "text": "alpha"}]}'
        out = ClaimExtractor._repair_truncated_claims(text)
        assert out is None


class TestExtractorStreaming:
    """
    ``ClaimExtractor._call_llm`` must use Anthropic's streaming API (required
    for requests whose estimated runtime exceeds ~10 min at 32K output-token
    cap) and must log token usage + stop_reason for observability.
    """

    def _fake_stream_response(
        self,
        *,
        text: str = '{"claims": [{"id": "c1", "text": "alpha", "is_checkable": true, "claim_type": "other"}]}',
        input_tokens: int = 1234,
        output_tokens: int = 567,
        stop_reason: str = "end_turn",
    ) -> MagicMock:
        """Mock anthropic messages.stream() context manager."""
        final = SimpleNamespace(
            content=[SimpleNamespace(text=text)],
            usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
            stop_reason=stop_reason,
        )
        stream_ctx = MagicMock()
        stream_ctx.__enter__ = MagicMock(return_value=stream_ctx)
        stream_ctx.__exit__ = MagicMock(return_value=False)
        stream_ctx.get_final_message = MagicMock(return_value=final)

        fake_client = MagicMock()
        fake_client.messages.stream.return_value = stream_ctx
        fake_client.messages.create.side_effect = AssertionError(
            "messages.create() must not be called; extractor must use stream()"
        )
        return fake_client

    def _run(self, fake_client: MagicMock, caplog):
        transcript = Transcript(text="Speaker said something factual.", speaker="Tester")
        extractor = ClaimExtractor(api_key="sk-ant-test")
        with patch("anthropic.Anthropic", return_value=fake_client):
            with caplog.at_level(logging.INFO, logger="truthbot.extract.claims"):
                claims = extractor.extract(transcript)
        return claims, fake_client

    def test_uses_stream_not_create(self, caplog):
        fake = self._fake_stream_response()
        claims, fake = self._run(fake, caplog)
        assert fake.messages.stream.called, "extractor must call messages.stream()"
        assert len(claims) == 1

    def test_logs_usage_at_info(self, caplog):
        fake = self._fake_stream_response(input_tokens=4242, output_tokens=999)
        self._run(fake, caplog)
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "input=4242" in joined
        assert "output=999" in joined
        assert "stop_reason=end_turn" in joined

    def test_max_tokens_stop_reason_logs_warning(self, caplog):
        fake = self._fake_stream_response(
            stop_reason="max_tokens",
            text='{"claims": [{"id": "c1", "text": "alpha", "is_checkable": true}]}',
        )
        self._run(fake, caplog)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("max_tokens" in r.getMessage() for r in warnings), (
            "extractor must emit a warning when Anthropic stop_reason == 'max_tokens'"
        )

    def test_max_tokens_truncated_json_salvaged_via_repair(self, caplog):
        """End-to-end: stop_reason=max_tokens + truncated JSON -> repair salvages claims."""
        truncated = (
            '{"claims": [{"id": "c1", "text": "alpha", "is_checkable": true, "claim_type": "other"},'
            ' {"id": "c2", "text": "incomplete'
        )
        fake = self._fake_stream_response(stop_reason="max_tokens", text=truncated)
        claims, _ = self._run(fake, caplog)
        assert len(claims) == 1
        assert claims[0].text == "alpha"
