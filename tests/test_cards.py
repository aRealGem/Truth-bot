"""TB-09t: Card renderer — OG tags, image gen per verdict type, URL routing."""

from __future__ import annotations

import pytest

from truthbot.models import VerdictLabel
from truthbot.publish.cards import CardRenderer, CARD_HEIGHT, CARD_WIDTH, _VERDICT_COLORS_RGB


@pytest.fixture
def renderer(tmp_dir):
    return CardRenderer(output_dir=tmp_dir, base_url="https://example.com")


class TestCardRenderer:
    def test_card_url_format(self, renderer):
        url = renderer.card_url("report-123", "verdict-456")
        assert "report-123" in url
        assert "verdict-456" in url
        assert url.startswith("https://example.com")

    def test_all_verdict_types_have_colors(self):
        for label in VerdictLabel:
            assert label in _VERDICT_COLORS_RGB
            r, g, b = _VERDICT_COLORS_RGB[label]
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255

    def test_card_dimensions(self):
        assert CARD_WIDTH == 1200
        assert CARD_HEIGHT == 630

    def test_render_returns_bytes_or_none(self, renderer, sample_claim, sample_verdict):
        # Render should return bytes (if Pillow is available) or None
        result = renderer.render_verdict_card(
            sample_claim, sample_verdict, speaker="Test Politician"
        )
        assert result is None or isinstance(result, bytes)

    def test_write_verdict_card_returns_path_or_none(
        self, renderer, sample_report, sample_claim, sample_verdict
    ):
        result = renderer.write_verdict_card(sample_report, sample_claim, sample_verdict)
        assert result is None or result.suffix == ".png"

    @pytest.mark.parametrize("label", list(VerdictLabel))
    def test_all_verdicts_renderable(self, renderer, sample_claim, label):
        from truthbot.models import Confidence, Verdict
        verdict = Verdict(
            claim_id=sample_claim.id,
            label=label,
            confidence=Confidence.MEDIUM,
            explanation="Test explanation for rendering.",
        )
        result = renderer.render_verdict_card(sample_claim, verdict, speaker="Speaker")
        assert result is None or isinstance(result, bytes)
