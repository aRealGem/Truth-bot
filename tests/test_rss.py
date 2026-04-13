"""TB-11t: RSS publisher — XML validity, entry completeness."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from truthbot.publish.rss import RSSPublisher


@pytest.fixture
def publisher(tmp_dir):
    return RSSPublisher(
        feed_title="Test Feed",
        feed_link="https://example.com",
        feed_description="Test description",
        output_dir=tmp_dir,
    )


class TestRSSPublisher:
    def test_generate_feed_returns_xml(self, publisher, sample_report):
        xml = publisher.generate_feed([sample_report])
        assert xml.startswith("<?xml")
        assert "<rss" in xml

    def test_feed_has_item(self, publisher, sample_report):
        xml = publisher.generate_feed([sample_report])
        tree = ET.fromstring(xml)
        items = tree.findall("./channel/item")
        assert len(items) == 1

    def test_item_has_guid_and_link(self, publisher, sample_report):
        xml = publisher.generate_feed([sample_report])
        tree = ET.fromstring(xml)
        item = tree.find("./channel/item")
        guid = item.find("guid").text
        link = item.find("link").text
        assert guid == sample_report.id
        assert link

    def test_write_feed_creates_file(self, publisher, sample_report):
        path = publisher.write_feed([sample_report])
        assert path.exists()
        contents = path.read_text()
        assert "Test Feed" in contents

    def test_generate_entry_returns_item_fragment(self, publisher, sample_report):
        xml = publisher.generate_entry(sample_report)
        assert xml.startswith("<item>")
        assert "Speaker" not in xml  # ensures placeholder not literal

    def test_verdict_summary_in_description(self, publisher, sample_report):
        xml = publisher.generate_feed([sample_report])
        assert "Mostly True" in xml
