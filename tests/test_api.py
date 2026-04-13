"""TB-12t: JSON API — schema validation, pagination, filtering."""

from __future__ import annotations

import pytest

from truthbot.publish.api import ReportAPI


@pytest.fixture
def api(sample_report):
    api = ReportAPI([sample_report])
    return api


class TestReportAPI:
    def test_get_report(self, api, sample_report):
        result = api.get_report(sample_report.id)
        assert result is sample_report

    def test_get_report_missing(self, api):
        assert api.get_report("missing") is None

    def test_add_report(self, api, sample_report):
        api.add_report(sample_report)
        assert api.get_report(sample_report.id) is sample_report

    def test_list_reports_default(self, api):
        listing = api.list_reports()
        assert listing["total"] >= 1
        assert len(listing["results"]) >= 1

    def test_list_reports_pagination(self, api, sample_report):
        listing = api.list_reports(page=1, page_size=1)
        assert listing["page"] == 1
        assert listing["page_size"] == 1
        assert len(listing["results"]) <= 1

    def test_list_reports_filter(self, api, sample_report):
        listing = api.list_reports(speaker="Test Politician")
        assert listing["results"]
        listing_none = api.list_reports(speaker="Nobody")
        assert listing_none["results"] == []

    def test_serialize_report(self, api, sample_report):
        data = api.serialize_report(sample_report)
        assert data["id"] == sample_report.id
        assert "claims" in data
        assert data["claims"][0]["verdict"]["label"] == sample_report.verdicts[0].label.value

    def test_build_app_requires_fastapi(self, api, monkeypatch):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")
        app = api.build_app()
        assert app.title == "truth-bot API"
