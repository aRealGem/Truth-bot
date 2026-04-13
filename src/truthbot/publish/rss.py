"""
RSS/Atom feed output.

Generates a standard RSS 2.0 / Atom feed of fact-check reports.
Each report becomes a feed entry with:
  - Title: "Fact Check: {speaker} — {date}"
  - Summary: verdict counts
  - Full content: all verdicts with labels and explanations
  - Categories: unique claim categories
  - GUID: report ID
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from truthbot.models import Report, VerdictLabel
from truthbot.scoring.rubric import VERDICT_SCORES

logger = logging.getLogger(__name__)


class RSSPublisher:
    """
    Generate RSS 2.0 feeds from fact-check reports.

    Parameters
    ----------
    feed_title:
        Title of the RSS feed.
    feed_link:
        Canonical URL of the feed/website.
    feed_description:
        Short description of the feed.
    output_dir:
        Directory where feed XML files are written.
    """

    def __init__(
        self,
        feed_title: str = "truth-bot Fact Checks",
        feed_link: str = "https://example.com/fact-checks",
        feed_description: str = "Automated political rhetoric fact-checks",
        output_dir: Optional[str | Path] = None,
    ) -> None:
        self.feed_title = feed_title
        self.feed_link = feed_link
        self.feed_description = feed_description
        self._output_dir = Path(output_dir) if output_dir else None

    def generate_feed(self, reports: list[Report]) -> str:
        """
        Generate an RSS 2.0 XML document from a list of reports.

        Parameters
        ----------
        reports:
            Reports to include as feed entries, newest first.

        Returns
        -------
        str
            Complete RSS 2.0 XML string.
        """
        rss = ET.Element("rss", version="2.0")
        rss.set("xmlns:atom", "http://www.w3.org/2005/Atom")
        rss.set("xmlns:dc", "http://purl.org/dc/elements/1.1/")

        channel = ET.SubElement(rss, "channel")

        ET.SubElement(channel, "title").text = self.feed_title
        ET.SubElement(channel, "link").text = self.feed_link
        ET.SubElement(channel, "description").text = self.feed_description
        ET.SubElement(channel, "language").text = "en-us"
        ET.SubElement(channel, "generator").text = "truth-bot"
        ET.SubElement(channel, "lastBuildDate").text = self._rfc822(datetime.now(timezone.utc))

        atom_link = ET.SubElement(channel, "atom:link")
        atom_link.set("href", self.feed_link + "/feed.xml")
        atom_link.set("rel", "self")
        atom_link.set("type", "application/rss+xml")

        # Sort newest first
        sorted_reports = sorted(
            reports,
            key=lambda r: r.created_at,
            reverse=True,
        )

        for report in sorted_reports:
            self._add_item(channel, report)

        ET.indent(rss, space="  ")
        xml_str = ET.tostring(rss, encoding="unicode", xml_declaration=False)
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str

    def write_feed(self, reports: list[Report], filename: str = "feed.xml") -> Path:
        """
        Generate and write the RSS feed to disk.

        Parameters
        ----------
        reports:
            Reports to include.
        filename:
            Output filename (default: feed.xml).

        Returns
        -------
        Path
            Path to the written file.
        """
        if self._output_dir is None:
            from truthbot.config import settings
            self._output_dir = settings.report_dir

        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / filename

        xml = self.generate_feed(reports)
        path.write_text(xml, encoding="utf-8")
        logger.info("RSS feed written to %s (%d entries)", path, len(reports))
        return path

    def generate_entry(self, report: Report) -> str:
        """
        Generate a single RSS item XML fragment for one report.

        Parameters
        ----------
        report:
            The report to convert to an RSS item.

        Returns
        -------
        str
            XML string of the <item> element.
        """
        rss = ET.Element("rss", version="2.0")
        channel = ET.SubElement(rss, "channel")
        self._add_item(channel, report)
        ET.indent(rss, space="  ")
        xml = ET.tostring(list(channel)[0], encoding="unicode")
        return xml

    # ── Private helpers ───────────────────────────────────────────────────────

    def _add_item(self, channel: ET.Element, report: Report) -> None:
        """Append a <item> element to the channel."""
        item = ET.SubElement(channel, "item")

        date_str = (
            report.transcript.date.strftime("%Y-%m-%d")
            if report.transcript.date
            else "Unknown date"
        )
        title = f"Fact Check: {report.transcript.speaker} — {date_str}"
        ET.SubElement(item, "title").text = title

        link = report.report_url or f"{self.feed_link}/reports/{report.id}"
        ET.SubElement(item, "link").text = link
        ET.SubElement(item, "guid", isPermaLink="false").text = report.id

        pub_date = report.published_at or report.created_at
        ET.SubElement(item, "pubDate").text = self._rfc822(pub_date)

        # Description: verdict summary counts
        summary = report.verdict_summary
        desc_lines = [f"<strong>Fact Check: {report.transcript.speaker}</strong>", ""]
        for label, count in summary.items():
            if count > 0:
                desc_lines.append(f"• {label}: {count}")
        desc_lines.append(f"<br><a href='{link}'>Read full report</a>")
        ET.SubElement(item, "description").text = "\n".join(desc_lines)

        # Categories from claim categories
        categories = set()
        for claim in report.claims:
            if claim.category:
                categories.add(claim.category.title())
        for cat in sorted(categories):
            ET.SubElement(item, "category").text = cat

        # Full content with verdicts
        content = self._build_content(report)
        content_el = ET.SubElement(item, "dc:description")
        content_el.text = content

    def _build_content(self, report: Report) -> str:
        """Build full report text for the RSS item content."""
        lines = [f"Fact-check report for {report.transcript.speaker}"]
        if report.transcript.date:
            lines.append(f"Date: {report.transcript.date.strftime('%B %d, %Y')}")
        if report.transcript.venue:
            lines.append(f"Venue: {report.transcript.venue}")
        lines.append("")

        for verdict in report.verdicts:
            claim = next((c for c in report.claims if c.id == verdict.claim_id), None)
            claim_text = claim.text if claim else "(claim not found)"
            lines.append(f"[{verdict.label.value}] {claim_text}")
            lines.append(f"  Confidence: {verdict.confidence.value}")
            lines.append(f"  {verdict.explanation}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _rfc822(dt: datetime) -> str:
        """Format a datetime as RFC 822 (RSS date format)."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%a, %d %b %Y %H:%M:%S %z")
