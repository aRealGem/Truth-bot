"""
JSON API for embeds.

Exposes fact-check reports as a JSON API, suitable for embedding
fact-check widgets on external sites or powering a frontend dashboard.

Endpoints (stub — requires FastAPI + uvicorn to run):
  GET  /api/reports              — paginated list of reports
  GET  /api/reports/{id}         — single report
  GET  /api/reports/{id}/verdicts — verdicts for a report
  GET  /api/claims/{id}          — single claim
  GET  /health                   — liveness check
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from truthbot.models import Report

logger = logging.getLogger(__name__)


class ReportAPI:
    """
    In-memory report store + JSON serialization helpers.

    In production, replace the in-memory store with a real database.
    The FastAPI app can be constructed via `build_app()`.

    Parameters
    ----------
    reports:
        Optional initial list of reports to seed the store.
    """

    def __init__(self, reports: Optional[list[Report]] = None) -> None:
        self._store: dict[str, Report] = {}
        for r in reports or []:
            self._store[r.id] = r

    def add_report(self, report: Report) -> None:
        """Add a report to the store."""
        self._store[report.id] = report

    def get_report(self, report_id: str) -> Optional[Report]:
        """Retrieve a report by ID."""
        return self._store.get(report_id)

    def list_reports(
        self,
        page: int = 1,
        page_size: int = 20,
        speaker: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Return a paginated list of reports.

        Parameters
        ----------
        page:
            1-indexed page number.
        page_size:
            Items per page (max 100).
        speaker:
            Optional speaker name filter (case-insensitive substring match).

        Returns
        -------
        dict
            Pagination envelope: { total, page, page_size, results: [...] }
        """
        page_size = min(page_size, 100)
        items = sorted(self._store.values(), key=lambda r: r.created_at, reverse=True)

        if speaker:
            items = [r for r in items if speaker.lower() in r.transcript.speaker.lower()]

        total = len(items)
        start = (page - 1) * page_size
        end = start + page_size

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "results": [self._serialize_report_summary(r) for r in items[start:end]],
        }

    def serialize_report(self, report: Report) -> dict[str, Any]:
        """
        Serialize a full report to a JSON-compatible dict.

        Parameters
        ----------
        report:
            The report to serialize.

        Returns
        -------
        dict
            Full report representation.
        """
        return {
            "id": report.id,
            "speaker": report.transcript.speaker,
            "date": report.transcript.date.isoformat() if report.transcript.date else None,
            "venue": report.transcript.venue,
            "created_at": report.created_at.isoformat(),
            "published_at": report.published_at.isoformat() if report.published_at else None,
            "report_url": report.report_url,
            "bluesky_thread_url": report.bluesky_thread_url,
            "rss_feed_url": report.rss_feed_url,
            "total_claims": report.total_claims,
            "checkable_claims": report.checkable_claims,
            "verdict_summary": report.verdict_summary,
            "claims": [self._serialize_claim(c, report) for c in report.claims],
        }

    def build_app(self):
        """
        Build and return a FastAPI application.

        Returns a FastAPI app with all API routes registered.
        Requires: pip install fastapi uvicorn

        Usage::

            app = ReportAPI().build_app()
            # uvicorn truthbot.publish.api:app --reload
        """
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.middleware.cors import CORSMiddleware
        except ImportError:
            raise ImportError("FastAPI is required: pip install fastapi uvicorn")

        app = FastAPI(
            title="truth-bot API",
            description="Fact-check report JSON API",
            version="0.1.0",
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET"],
            allow_headers=["*"],
        )

        api = self  # capture for closures

        @app.get("/health")
        def health():
            return {"status": "ok", "reports": len(api._store)}

        @app.get("/api/reports")
        def list_reports(page: int = 1, page_size: int = 20, speaker: Optional[str] = None):
            return api.list_reports(page=page, page_size=page_size, speaker=speaker)

        @app.get("/api/reports/{report_id}")
        def get_report(report_id: str):
            report = api.get_report(report_id)
            if not report:
                raise HTTPException(status_code=404, detail="Report not found")
            return api.serialize_report(report)

        @app.get("/api/reports/{report_id}/verdicts")
        def get_verdicts(report_id: str):
            report = api.get_report(report_id)
            if not report:
                raise HTTPException(status_code=404, detail="Report not found")
            return [v.model_dump() for v in report.verdicts]

        return app

    # ── Private helpers ───────────────────────────────────────────────────────

    def _serialize_report_summary(self, report: Report) -> dict[str, Any]:
        """Lightweight report summary for list endpoints."""
        return {
            "id": report.id,
            "speaker": report.transcript.speaker,
            "date": report.transcript.date.isoformat() if report.transcript.date else None,
            "created_at": report.created_at.isoformat(),
            "total_claims": report.total_claims,
            "verdict_summary": report.verdict_summary,
            "report_url": report.report_url,
        }

    def _serialize_claim(self, claim, report: Report) -> dict[str, Any]:
        """Serialize a claim with its verdict."""
        verdict = report.verdict_for(claim.id)
        return {
            "id": claim.id,
            "text": claim.text,
            "category": claim.category,
            "is_checkable": claim.is_checkable,
            "verdict": {
                "label": verdict.label.value,
                "confidence": verdict.confidence.value,
                "explanation": verdict.explanation,
            } if verdict else None,
        }
