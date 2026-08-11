"""Redirect stubs for dead report URLs (DC-3', remediation v2).

Stable speech_id-derived slugs mean every re-render of a speech lands on the
same URL — but every slug shipped BEFORE the rotation is a dead link. The
data/report_aliases.json ledger maps each dead filename (recovered from git
history, plus the committed slugs that rotate once at the next regen) to its
stable target; ``SitePublisher.publish`` emits a meta-refresh + canonical
stub at each old filename whenever the target exists in the render.

Stubs therefore appear only at regeneration time — the committed site-pca/
tree is untouched until then.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from truthbot.models import VerdictLabel
from truthbot.publish.site import (
    SitePublisher,
    _load_report_aliases,
    _render_report_alias_stub,
)
from tests.test_site_render_aggregates import _make_bundle, _make_site_report

_REPO = Path(__file__).resolve().parents[2]

#: The five stable report slugs (sha1(speech_id)[:6] suffix).
_STABLE = {
    "trump_2026":   "2026-02-24-donald-trump-583aca.html",
    "biden_2022":   "2022-03-01-joe-biden-d359c0.html",
    "obama_2014":   "2014-01-28-barack-obama-bc9c9f.html",
    "gwbush_2006":  "2006-01-31-george-w-bush-82f462.html",
    "clinton_1998": "1998-01-27-bill-clinton-54f0ca.html",
}


def _obama_report():
    sr = _make_site_report([_make_bundle(
        VerdictLabel.TRUE, coarse_lenient="True", coarse_strict="True")])
    sr.speaker = sr.source_of_claims = "Barack Obama"
    sr.date = datetime(2014, 1, 28, tzinfo=timezone.utc)
    sr.speech_id = "obama_2014"
    return sr


def test_stable_slug_hashes_match_ledger_targets() -> None:
    for speech_id, fname in _STABLE.items():
        short = hashlib.sha1(speech_id.encode()).hexdigest()[:6]
        assert fname.endswith(f"-{short}.html"), (speech_id, fname)


def test_stub_emitted_with_canonical_when_target_exists(tmp_path: Path) -> None:
    old = "2014-01-28-barack-obama-d2489f.html"
    new = _STABLE["obama_2014"]
    pub = SitePublisher(site_root=tmp_path, report_aliases={old: new})
    pub.publish(_obama_report())

    assert (tmp_path / "reports" / new).exists()   # the real page
    stub = (tmp_path / "reports" / old).read_text(encoding="utf-8")
    assert f'http-equiv="refresh" content="0; url=./{new}"' in stub
    assert f'rel="canonical"' in stub and f"/reports/{new}" in stub
    assert "This report was re-adjudicated and republished." in stub


def test_stub_not_emitted_when_target_absent(tmp_path: Path) -> None:
    old = "2026-02-24-donald-trump-0570f7.html"
    new = _STABLE["trump_2026"]                    # trump is NOT published here
    pub = SitePublisher(site_root=tmp_path, report_aliases={old: new})
    pub.publish(_obama_report())
    assert not (tmp_path / "reports" / old).exists()


def test_self_alias_never_overwrites_the_real_page(tmp_path: Path) -> None:
    new = _STABLE["obama_2014"]
    pub = SitePublisher(site_root=tmp_path, report_aliases={new: new})
    pub.publish(_obama_report())
    page = (tmp_path / "reports" / new).read_text(encoding="utf-8")
    assert "re-adjudicated" not in page            # still the real report


def test_default_publisher_loads_repo_ledger_and_stubs_dead_obama_urls(
        tmp_path: Path) -> None:
    """No explicit aliases → the repo ledger applies: publishing the obama
    speech into a fresh root emits stubs for its dead slug AND the rotating
    committed slug; triage aliases stay un-stubbed because no triage page
    exists in this render."""
    pub = SitePublisher(site_root=tmp_path)        # default ledger
    pub.publish(_obama_report())
    reports = tmp_path / "reports"
    for old in ("2014-01-28-barack-obama-d2489f.html",
                "2014-01-28-barack-obama-4a245a.html"):
        assert "re-adjudicated" in (reports / old).read_text(encoding="utf-8")
    # No triage page in this render → the triage alias emits nothing.
    assert not (reports / "2014-01-28-barack-obama-4a245a-triage.html").exists()
    # Other speeches' aliases untouched (their targets are absent).
    assert not (reports / "2026-02-24-donald-trump-0570f7.html").exists()


def test_repo_ledger_is_valid_and_complete() -> None:
    """Schema + shape of data/report_aliases.json: every key/value is a
    reports/ filename, every target is one of the five stable slugs (or its
    -triage variant), no self-aliases, and the known dead slugs are all
    present alongside the five rotating committed slugs."""
    path = _REPO / "data" / "report_aliases.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["schema"] == "truthbot-report-aliases v1"
    aliases = doc["aliases"]
    assert aliases == _load_report_aliases(path)

    stable_targets = set(_STABLE.values()) | {
        f.replace(".html", "-triage.html") for f in _STABLE.values()}
    for old, new in aliases.items():
        assert old.endswith(".html") and new.endswith(".html")
        assert "/" not in old and "/" not in new
        assert old != new, f"self-alias in ledger: {old}"
        assert new in stable_targets, f"non-stable target: {old} -> {new}"

    # Dead report slugs recovered from git history + the 5 committed ones.
    expected_reports = (
        [f"2026-02-24-donald-trump-{h}.html"
         for h in ("0570f7", "514413", "5a611f", "ceff24", "30c0c4",
                   "e4fc1b", "0c33d1", "c460cf", "80ac5b")]
        + [f"2022-03-01-joe-biden-{h}.html"
           for h in ("384848", "87e372", "b4e623", "5f7ef2", "4d31d6",
                     "8c252c", "1fd3e2", "90050c", "aa8c46")]
        + ["2014-01-28-barack-obama-d2489f.html",
           "2014-01-28-barack-obama-4a245a.html",
           "2006-01-31-george-w-bush-ba3fbe.html",
           "1998-01-27-bill-clinton-93221f.html"]
    )
    for fname in expected_reports:
        assert fname in aliases, f"missing alias for {fname}"


def test_stub_render_matches_model_insights_pattern() -> None:
    html = _render_report_alias_stub("2014-01-28-barack-obama-bc9c9f.html")
    assert 'http-equiv="refresh"' in html
    assert 'rel="canonical"' in html
    assert "This report was re-adjudicated and republished." in html
