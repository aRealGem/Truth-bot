"""TB-08t: Web publisher — HTML validity, SSE format, completeness."""

from __future__ import annotations

import re

import pytest

from truthbot.publish.web import WebPublisher


@pytest.fixture
def publisher(tmp_dir):
    return WebPublisher(output_dir=tmp_dir, base_url="https://example.com")


class TestWebPublisher:
    def test_generate_html_returns_string(self, publisher, sample_report):
        html = publisher.generate_html(sample_report)
        assert isinstance(html, str)
        assert len(html) > 100

    def test_html_has_doctype(self, publisher, sample_report):
        html = publisher.generate_html(sample_report)
        assert html.strip().startswith("<!DOCTYPE html>")

    def test_html_has_speaker(self, publisher, sample_report):
        html = publisher.generate_html(sample_report)
        assert "Test Politician" in html

    def test_html_has_verdict_labels(self, publisher, sample_report):
        html = publisher.generate_html(sample_report)
        assert "Mostly True" in html

    def test_html_has_og_tags(self, publisher, sample_report):
        html = publisher.generate_html(sample_report)
        assert 'property="og:title"' in html
        assert 'property="og:description"' in html

    def test_html_has_claim_text(self, publisher, sample_report):
        html = publisher.generate_html(sample_report)
        assert "50-year low" in html

    def test_write_report_creates_file(self, publisher, sample_report, tmp_dir):
        path = publisher.write_report(sample_report)
        assert path.exists()
        assert path.suffix == ".html"
        content = path.read_text()
        assert "truth-bot" in content

    def test_sse_event_format(self, publisher):
        event = publisher.sse_event("claim_verified", {"label": "True", "count": 1})
        assert event.startswith("event: claim_verified\n")
        assert "data:" in event
        assert event.endswith("\n\n")
        import json
        data_line = [l for l in event.split("\n") if l.startswith("data:")][0]
        data = json.loads(data_line[len("data:"):])
        assert data["label"] == "True"

    def test_html_escaping(self, publisher):
        from truthbot.publish.web import WebPublisher
        escaped = WebPublisher._escape('<script>alert("xss")</script>')
        assert "<script>" not in escaped
        assert "&lt;script&gt;" in escaped
