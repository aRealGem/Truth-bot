"""TB-02t: Transcript ingestion — formats, metadata, edge cases."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from truthbot.ingest.transcript import TranscriptIngester, IngestResult
from truthbot.models import Transcript


@pytest.fixture
def ingester():
    return TranscriptIngester()


class TestIngestText:
    def test_basic_text(self, ingester):
        result = ingester.ingest_text("The unemployment rate fell to 3.4%.", speaker="POTUS")
        assert isinstance(result, IngestResult)
        assert isinstance(result.transcript, Transcript)
        assert result.transcript.speaker == "POTUS"
        assert "unemployment" in result.transcript.text

    def test_normalizes_whitespace(self, ingester):
        text = "  Hello.\n\n\n\nWorld.  "
        result = ingester.ingest_text(text, speaker="Test")
        assert "   " not in result.transcript.text
        # Should not have triple newlines
        assert "\n\n\n" not in result.transcript.text

    def test_word_count(self, ingester):
        text = "One two three four five."
        result = ingester.ingest_text(text)
        assert result.transcript.word_count == 5

    def test_explicit_date_preserved(self, ingester):
        d = datetime(2025, 1, 20)
        result = ingester.ingest_text("Some speech.", date=d)
        assert result.transcript.date == d

    def test_date_auto_detected(self, ingester):
        text = "Address delivered on January 20, 2025. The economy is strong."
        result = ingester.ingest_text(text)
        assert result.transcript.date is not None
        assert result.transcript.date.year == 2025

    def test_venue_preserved(self, ingester):
        result = ingester.ingest_text("Speech text.", venue="State of the Union")
        assert result.transcript.venue == "State of the Union"

    def test_empty_text_raises(self, ingester):
        with pytest.raises(ValueError):
            ingester.ingest_text("   ")

    def test_metadata_attached(self, ingester):
        result = ingester.ingest_text("Some text.", metadata={"source": "C-SPAN"})
        assert result.transcript.metadata["source"] == "C-SPAN"

    def test_source_url(self, ingester):
        result = ingester.ingest_text("Text.", source_url="https://example.com/speech")
        assert result.transcript.source_url == "https://example.com/speech"

    def test_srt_stripped(self, ingester):
        srt = """1
00:00:01,000 --> 00:00:04,000
Unemployment is at a record low.

2
00:00:04,500 --> 00:00:08,000
We have created millions of jobs.
"""
        result = ingester.ingest_text(srt)
        assert "00:00:01" not in result.transcript.text
        assert "unemployment" in result.transcript.text.lower()
        assert any("Detected" in w for w in result.warnings)


class TestIngestFile:
    def test_reads_txt_file(self, ingester, tmp_path):
        p = tmp_path / "speech.txt"
        p.write_text("We built a wall. It is beautiful.")
        result = ingester.ingest_file(p, speaker="Speaker")
        assert "wall" in result.transcript.text

    def test_missing_file_raises(self, ingester, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingester.ingest_file(tmp_path / "nonexistent.txt")

    def test_directory_raises(self, ingester, tmp_path):
        with pytest.raises(ValueError):
            ingester.ingest_file(tmp_path)


class TestIngestDispatch:
    def test_dispatches_text(self, ingester):
        result = ingester.ingest("Plain text about the economy.")
        assert result.transcript.text

    def test_dispatches_file(self, ingester, tmp_path):
        p = tmp_path / "speech.txt"
        p.write_text("A fine speech about jobs.")
        result = ingester.ingest(str(p))
        assert "jobs" in result.transcript.text
