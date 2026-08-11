"""Atom feed rendering + validation (remediation v2, item 1.5).

The feed is rendered from the reports index — one <entry> per published
report — replacing the old static template that shipped a verbatim
[SITE_URL] placeholder, a hand-typed phantom entry, and a frozen <updated>
stamp. ``consistency.check_feed`` is the build-time guard.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from truthbot.publish.consistency import check_feed
from truthbot.publish.site import _render_feed

_NS = {"a": "http://www.w3.org/2005/Atom"}
_SITE_URL = "https://truthbot.example.org/site"


def _reports() -> list[dict]:
    return [
        {
            "id": "r1",
            "date": "2026-02-24",
            "speaker": "Donald Trump",
            "claim_count": 3,
            "verdict_distribution_strict": {"True": 1, "Falsey": 1, "False": 1},
            "url": "reports/2026-02-24-donald-trump-abc123.html",
            "generated_at": "2026-08-01T12:00:00Z",
        },
        {
            "id": "r2",
            "date": "2014-01-28",
            "speaker": 'Barack "no drama" Obama & guests',   # escaping fodder
            "claim_count": 1,
            "verdict_distribution_strict": {"True": 1},
            "url": "reports/2014-01-28-barack-obama-def456.html",
            "generated_at": "2026-07-30T09:30:00Z",
        },
    ]


def _write_site(tmp_path: Path, reports: list[dict]) -> Path:
    for r in reports:
        page = tmp_path / r["url"]
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text("<html></html>", encoding="utf-8")
    (tmp_path / "feed.xml").write_text(
        _render_feed(reports, _SITE_URL), encoding="utf-8")
    return tmp_path


def test_feed_has_one_entry_per_report_with_derived_fields() -> None:
    xml = _render_feed(_reports(), _SITE_URL)
    root = ET.fromstring(xml)
    entries = root.findall("a:entry", _NS)
    assert len(entries) == 2

    e1 = entries[0]
    assert e1.findtext("a:title", namespaces=_NS) == "Donald Trump — February 24, 2026"
    link = e1.find("a:link", _NS)
    assert link.get("href") == (
        f"{_SITE_URL}/reports/2026-02-24-donald-trump-abc123.html")
    assert e1.findtext("a:id", namespaces=_NS) == (
        "urn:truth-bot:report:2026-02-24-donald-trump-abc123")
    assert e1.findtext("a:published", namespaces=_NS) == "2026-02-24T00:00:00Z"
    assert e1.findtext("a:updated", namespaces=_NS) == "2026-08-01T12:00:00Z"
    # Summary derives from aggregation.family_verdict on the strict dist:
    # 1 of 3 decided true → "33% True".
    summary = e1.findtext("a:summary", namespaces=_NS)
    assert "3 claims checked" in summary
    assert "33% True" in summary
    assert "1 of 3 decided claims rated True" in summary

    # Feed-level <updated> is the max entry <updated>.
    assert root.findtext("a:updated", namespaces=_NS) == "2026-08-01T12:00:00Z"


def test_feed_escapes_text_and_has_no_phantom_or_placeholder() -> None:
    xml = _render_feed(_reports(), _SITE_URL)
    assert "[SITE_URL]" not in xml
    # The old template's hand-typed phantom entry must never resurface.
    assert "2026-03-04-donald-trump-165937" not in xml
    # Speaker with quotes/ampersand round-trips through XML parsing.
    root = ET.fromstring(xml)   # would raise on bad escaping
    titles = [e.findtext("a:title", namespaces=_NS)
              for e in root.findall("a:entry", _NS)]
    assert 'Barack "no drama" Obama & guests — January 28, 2014' in titles
    assert "&amp;" in xml and "& guests" not in xml


def test_check_feed_passes_on_fresh_render(tmp_path) -> None:
    reports = _reports()
    _write_site(tmp_path, reports)
    assert check_feed(tmp_path, reports) == []


def test_check_feed_catches_placeholder_and_missing_page(tmp_path) -> None:
    reports = _reports()
    _write_site(tmp_path, reports)
    # Missing linked page.
    (tmp_path / reports[0]["url"]).unlink()
    violations = check_feed(tmp_path, reports)
    assert any("missing under site root" in v for v in violations)
    # Legacy placeholder text.
    feed = tmp_path / "feed.xml"
    feed.write_text(feed.read_text(encoding="utf-8")
                    .replace(_SITE_URL, "[SITE_URL]"), encoding="utf-8")
    violations = check_feed(tmp_path, reports)
    assert any("[SITE_URL]" in v for v in violations)


def test_check_feed_catches_count_id_and_updated_drift(tmp_path) -> None:
    reports = _reports()
    _write_site(tmp_path, reports)
    # Entry count drift: index gains a report the feed doesn't know.
    extra = dict(reports[0], id="r3",
                 url="reports/2026-01-01-someone-zzz999.html")
    (tmp_path / extra["url"]).write_text("<html></html>", encoding="utf-8")
    assert any("entries" in v for v in check_feed(tmp_path, reports + [extra]))
    # Duplicate ids + stale feed <updated>.
    dup = [reports[0], dict(reports[0])]
    (tmp_path / "feed.xml").write_text(
        _render_feed(dup, _SITE_URL), encoding="utf-8")
    assert any("duplicate entry ids" in v for v in check_feed(tmp_path, dup))
    stale = (tmp_path / "feed.xml").read_text(encoding="utf-8").replace(
        "<updated>2026-08-01T12:00:00Z</updated>",
        "<updated>2020-01-01T00:00:00Z</updated>", 1)
    (tmp_path / "feed.xml").write_text(stale, encoding="utf-8")
    violations = check_feed(tmp_path, _reports())
    assert any("max entry" in v for v in violations)
