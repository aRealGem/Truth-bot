"""
Verdict card renderer — OG image generation.

Generates Open Graph card images for each verdict, suitable for
sharing on social media. Each card shows:
  - Speaker name
  - Claim text (truncated)
  - Verdict label with color coding
  - Confidence level
  - truth-bot branding

Uses Pillow for image generation. Falls back to returning None if Pillow
is not available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from truthbot.models import Claim, Report, Verdict, VerdictLabel

logger = logging.getLogger(__name__)

# OG card dimensions (Twitter / OpenGraph standard)
CARD_WIDTH = 1200
CARD_HEIGHT = 630

_VERDICT_COLORS_RGB = {
    VerdictLabel.TRUE: (46, 125, 50),
    VerdictLabel.MOSTLY_TRUE: (85, 139, 47),
    VerdictLabel.MISLEADING: (245, 124, 0),
    VerdictLabel.EXAGGERATED: (239, 108, 0),
    VerdictLabel.FALSE: (198, 40, 40),
    VerdictLabel.UNVERIFIABLE: (84, 110, 122),
}


class CardRenderer:
    """
    Render verdict cards as PNG images for social sharing.

    Parameters
    ----------
    output_dir:
        Directory where card images are written.
    base_url:
        Base URL used to construct card URLs in the report.
    """

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        base_url: str = "https://example.com",
    ) -> None:
        self._output_dir = Path(output_dir) if output_dir else None
        self.base_url = base_url

    def render_verdict_card(
        self,
        claim: Claim,
        verdict: Verdict,
        speaker: str = "",
    ) -> Optional[bytes]:
        """
        Render a PNG image card for a single verdict.

        Parameters
        ----------
        claim:
            The claim this verdict is for.
        verdict:
            The verdict to display.
        speaker:
            Speaker name for the card header.

        Returns
        -------
        Optional[bytes]
            PNG bytes, or None if Pillow is unavailable.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            logger.warning("Pillow not available — card generation skipped.")
            return None

        color = _VERDICT_COLORS_RGB.get(verdict.label, (100, 100, 100))
        img = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)

        # Color bar on left
        draw.rectangle([0, 0, 12, CARD_HEIGHT], fill=color)

        # Verdict label block
        draw.rectangle([60, 40, 400, 110], fill=color)
        draw.text((80, 55), verdict.label.value.upper(), fill=(255, 255, 255))

        # Speaker
        if speaker:
            draw.text((60, 130), speaker, fill=(80, 80, 80))

        # Claim text (word-wrapped)
        claim_text = claim.text[:200]
        draw.text((60, 180), claim_text, fill=(30, 30, 30))

        # Confidence
        draw.text((60, 540), f"Confidence: {verdict.confidence.value}", fill=(120, 120, 120))

        # Branding
        draw.text((CARD_WIDTH - 200, CARD_HEIGHT - 40), "truth-bot", fill=(180, 180, 180))

        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def write_verdict_card(
        self,
        report: Report,
        claim: Claim,
        verdict: Verdict,
    ) -> Optional[Path]:
        """
        Render and write a verdict card PNG to disk.

        Parameters
        ----------
        report:
            The parent report (for directory structure and speaker name).
        claim:
            The claim to render.
        verdict:
            The verdict for the claim.

        Returns
        -------
        Optional[Path]
            Path to the written PNG, or None if rendering failed.
        """
        png = self.render_verdict_card(
            claim=claim,
            verdict=verdict,
            speaker=report.transcript.speaker,
        )
        if png is None:
            return None

        if self._output_dir is None:
            from truthbot.config import settings
            self._output_dir = settings.report_dir

        out = self._output_dir / f"report_{report.id}" / "cards"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{verdict.id}.png"
        path.write_bytes(png)
        logger.debug("Card written to %s", path)
        return path

    def card_url(self, report_id: str, verdict_id: str) -> str:
        """Return the expected public URL for a verdict card."""
        return f"{self.base_url}/reports/{report_id}/cards/{verdict_id}.png"
