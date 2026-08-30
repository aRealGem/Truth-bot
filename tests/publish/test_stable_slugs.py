"""Stable report slugs (remediation v2 item DC-3') + the SITE_URL setting.

A report that knows its speech identity must land on the same URL every
re-render — the slug suffix derives from ``speech_id``, not the per-run UUID.
Legacy callers (no speech_id) keep the UUID-suffixed slug so nothing existing
moves until the Phase-2 regeneration.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime

from truthbot.config import settings
from truthbot.publish.site import SiteReport, _site_url


def _report(speech_id: str = "", report_id: str | None = None) -> SiteReport:
    return SiteReport(
        report_id=report_id or str(uuid.uuid4()),
        speaker="Barack Obama",
        role="President",
        date=datetime(2014, 1, 28),
        venue="U.S. Capitol",
        transcript_source_url="",
        bundles=[],
        speech_id=speech_id,
    )


def test_same_speech_id_yields_identical_slug_across_constructions() -> None:
    a = _report(speech_id="obama_2014")
    b = _report(speech_id="obama_2014")  # fresh run — different report_id
    assert a.report_id != b.report_id
    assert a.report_slug == b.report_slug
    expected = hashlib.sha1(b"obama_2014").hexdigest()[:6]
    assert a.report_slug == f"2014-01-28-barack-obama-{expected}"


def test_different_speech_ids_yield_different_slugs() -> None:
    assert _report(speech_id="obama_2014").report_slug != \
        _report(speech_id="clinton_1998").report_slug


def test_legacy_reports_without_speech_id_keep_uuid_suffix() -> None:
    rid = "aabbccdd-0000-1111-2222-333344445555"
    r = _report(report_id=rid)
    assert r.report_slug == "2014-01-28-barack-obama-aabbcc"


def test_site_url_default_and_env_override(monkeypatch) -> None:
    monkeypatch.delenv("TRUTHBOT_SITE_URL", raising=False)
    default = "https://arealgem.github.io/Truth-bot/site-pca"
    assert settings.site_url == default
    assert _site_url() == default
    monkeypatch.setenv("TRUTHBOT_SITE_URL", "https://truthbot.example.org/")
    assert settings.site_url == "https://truthbot.example.org"
    assert _site_url() == "https://truthbot.example.org"
