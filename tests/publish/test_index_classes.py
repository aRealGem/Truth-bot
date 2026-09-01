"""Index grouping by occasion class + the per-class stats strip (P-senate 3d).

Covers three things:

1. INDEX GROUPING -- ``_render_index`` groups report cards under section
   heads by ``report_class()``, in ``report_class_order()`` order (classes
   outside that list sort after it alphabetically; UNCLASSIFIED always sorts
   last). Card order WITHIN a section is untouched -- the grouping only picks
   a bucket, it never re-sorts.
2. PER-CLASS STRIP -- ``_class_stats_strip`` renders five figures (reports,
   claims checked, decided, true-leaning share of decided, n visible) scoped
   to ONE class. No figure anywhere compares one class to another.
3. LABEL RENAME -- "Leaders Reviewed" -> "Speakers Reviewed", both call
   sites, and the top-level index program-stats strip stays counts-only
   (its one existing percentage, Model Consensus, is untouched -- not new).

The hardest bar (see module docstring on ``test_presidential_report_pages``)
is that grouping the index must not perturb a REPORT PAGE'S own render path
at all: the five presidential report pages must come out byte-identical to
before this change, except for the literal label swap. That is checked by
loading the pre-change ``site.py`` (from git HEAD) as a second, independent
module living alongside the real one, and diffing actual rendered output --
not by trusting the diff we think we made.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from truthbot.models import (Claim, Confidence, ConsensusVerdict,
                             ModelVerdict, VerdictBundle, VerdictLabel)
from truthbot.publish import site
from truthbot.publish.site import (UNCLASSIFIED, SiteReport, _class_stats_strip,
                                   _render_index, _render_report, report_class,
                                   report_class_label, report_class_order)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SITE_PATH = _REPO_ROOT / "src" / "truthbot" / "publish" / "site.py"


# ── Fixture builders ──────────────────────────────────────────────────────────


def _report(speech_id, speaker, url, claim_count=3, dist_strict=None,
            include_speech_id=True):
    """A minimal reports.json-shaped dict -- the same shape _render_index
    actually receives, per _report_meta()."""
    row = {
        "id": speech_id or speaker,
        "speaker": speaker,
        "url": url,
        "date": "2026-01-01",
        "venue": "",
        "claim_count": claim_count,
        "verdict_distribution": {},
        "verdict_distribution_strict": dist_strict or {},
    }
    if include_speech_id:
        row["speech_id"] = speech_id
    return row


def _bundle(label: VerdictLabel, claim_id: str) -> VerdictBundle:
    claim = Claim(id=claim_id, transcript_id="t", text=f"Claim {claim_id}.",
                  speaker="Speaker", context="ctx", category="economy",
                  is_checkable=True)
    mvs = [
        ModelVerdict(adapter_name=f"adapter-{i}", model_id=f"model-{i}",
                     claim_id=claim.id, label=label, confidence=Confidence.HIGH,
                     explanation="r")
        for i in range(3)
    ]
    consensus = ConsensusVerdict(
        claim_id=claim.id, model_verdicts=mvs, consensus_label=label,
        consensus_verdict=label.value, confidence=Confidence.HIGH,
        agreement=True, consensus_strength="strong", explanation="x",
    )
    return VerdictBundle(claim=claim, speaker="Speaker", date_str="2026-01-01",
                         model_verdicts=mvs, consensus=consensus)


def _bundles():
    # Shared across old/new renders (see test below) so claim ids and
    # timestamps can never be a source of spurious diff.
    return [
        _bundle(VerdictLabel.TRUE, "c-true"),
        _bundle(VerdictLabel.FALSE, "c-false"),
        _bundle(VerdictLabel.UNVERIFIABLE, "c-unverifiable"),
    ]


def _site_report_kwargs(speaker: str, speech_id: str) -> dict:
    return dict(
        report_id="00000000-0000-0000-0000-000000000000",
        speaker=speaker,
        role="President",
        date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        venue="Test Venue",
        transcript_source_url="",
        source_of_claims=speaker,
        source_of_claims_professional_public_title="President",
        event="Test Event",
        channel="",
        generated_at=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        speech_id=speech_id,
    )


# ── 1. Index grouping ──────────────────────────────────────────────────────────


def test_sections_appear_in_authored_order_presidential_before_senate(monkeypatch):
    monkeypatch.setattr(site, "_report_classes", lambda: {
        "classes": {"sen1": "senate_floor", "pres1": "presidential_address"},
        "labels": {"senate_floor": "Senate floor speeches",
                   "presidential_address": "Presidential addresses"},
        "labels_inline": {}, "order": ["presidential_address", "senate_floor"],
    })
    # Senate listed FIRST in the input -- section order must still come from
    # report_class_order(), never from input order.
    reports = [_report("sen1", "Senator A", "reports/a.html"),
               _report("pres1", "President B", "reports/b.html")]
    html = _render_index(reports, {"total_claims": 0, "total_leaders": 0,
                                   "avg_consensus": 0})
    assert html.index("Presidential addresses") < html.index("Senate floor speeches")


def test_class_outside_order_sorts_after_alphabetically(monkeypatch):
    monkeypatch.setattr(site, "_report_classes", lambda: {
        "classes": {"z1": "zeta_class", "p1": "presidential_address",
                    "a1": "alpha_class"},
        "labels": {"presidential_address": "Presidential addresses",
                   "zeta_class": "Zeta class", "alpha_class": "Alpha class"},
        "labels_inline": {}, "order": ["presidential_address"],
    })
    reports = [_report("z1", "Z", "reports/z.html"),
               _report("p1", "P", "reports/p.html"),
               _report("a1", "A", "reports/al.html")]
    html = _render_index(reports, {"total_claims": 0, "total_leaders": 0,
                                   "avg_consensus": 0})
    pres_i, alpha_i, zeta_i = (html.index(s) for s in
                               ("Presidential addresses", "Alpha class", "Zeta class"))
    assert pres_i < alpha_i < zeta_i


def test_unclassified_report_still_appears_under_its_own_section_last(monkeypatch):
    monkeypatch.setattr(site, "_report_classes", lambda: {
        "classes": {"known1": "presidential_address", "known2": "some_other"},
        "labels": {"presidential_address": "Presidential addresses",
                   "some_other": "Some other"},
        "labels_inline": {}, "order": ["presidential_address", "some_other"],
    })
    reports = [_report("known1", "Known Speaker", "reports/known.html"),
               _report("known2", "Other Speaker", "reports/other.html"),
               _report("mystery1", "Mystery Speaker", "reports/mystery.html")]
    html = _render_index(reports, {"total_claims": 0, "total_leaders": 0,
                                   "avg_consensus": 0})
    # Never dropped -- the card is still there.
    assert "Mystery Speaker" in html
    assert "Unclassified" in html
    pres_i = html.index("Presidential addresses")
    other_i = html.index("Some other")
    unclass_i = html.index("Unclassified")
    assert pres_i < other_i < unclass_i


def test_legacy_report_missing_speech_id_field_is_unclassified_not_dropped(monkeypatch):
    """Today's real reports.json entries predate this field entirely (no
    speech_id key at all, not just an empty one) -- must fail closed to
    UNCLASSIFIED-and-visible, never KeyError or silent omission."""
    monkeypatch.setattr(site, "_report_classes", lambda: {
        "classes": {}, "labels": {}, "labels_inline": {}, "order": [],
    })
    legacy = _report(None, "Legacy Speaker", "reports/legacy.html",
                     include_speech_id=False)
    assert "speech_id" not in legacy
    html = _render_index([legacy], {"total_claims": 0, "total_leaders": 0,
                                    "avg_consensus": 0})
    assert "Legacy Speaker" in html
    assert "Unclassified" in html


def test_card_order_within_a_section_is_preserved_not_resorted(monkeypatch):
    monkeypatch.setattr(site, "_report_classes", lambda: {
        "classes": {"s1": "senate_floor", "s2": "senate_floor", "s3": "senate_floor"},
        "labels": {"senate_floor": "Senate floor speeches"},
        "labels_inline": {}, "order": ["senate_floor"],
    })
    # Deliberately not alphabetical and not id-ordered.
    reports = [_report("s3", "Zeta", "reports/z.html"),
               _report("s1", "Alpha", "reports/a.html"),
               _report("s2", "Mid", "reports/m.html")]
    html = _render_index(reports, {"total_claims": 0, "total_leaders": 0,
                                   "avg_consensus": 0})
    assert html.index("Zeta") < html.index("Alpha") < html.index("Mid")


def test_section_with_no_reports_renders_nothing(monkeypatch):
    monkeypatch.setattr(site, "_report_classes", lambda: {
        "classes": {"p1": "presidential_address"},
        "labels": {"presidential_address": "Presidential addresses",
                   "senate_floor": "Senate floor speeches"},
        "labels_inline": {}, "order": ["presidential_address", "senate_floor"],
    })
    reports = [_report("p1", "P", "reports/p.html")]
    html = _render_index(reports, {"total_claims": 0, "total_leaders": 0,
                                   "avg_consensus": 0})
    assert "Presidential addresses" in html
    assert "Senate floor speeches" not in html


def test_real_registry_groups_the_authored_corpus_correctly():
    """No monkeypatch -- uses the real data/report_classes.json, which today
    authors 5 presidential + 4 senate speech_ids (see test_report_classes.py
    PRESIDENTIAL / SENATE lists)."""
    presidential_ids = ["clinton_1998", "gwbush_2006", "obama_2014",
                        "biden_2022", "trump_2026"]
    senate_ids = ["budd_2025-04-02", "cruz_2026-06-24", "tillis_2025-01-23",
                  "warren_2025-04-29"]
    reports = ([_report(sid, sid, f"reports/{sid}.html") for sid in senate_ids]
              + [_report(sid, sid, f"reports/{sid}.html") for sid in presidential_ids])
    html = _render_index(reports, {"total_claims": 0, "total_leaders": 0,
                                   "avg_consensus": 0})
    assert html.index("Presidential addresses") < html.index("Senate floor speeches")
    for sid in presidential_ids + senate_ids:
        assert sid in html


# ── 2. Per-class strip ─────────────────────────────────────────────────────────


def test_class_stats_strip_figures_are_scoped_and_correct():
    group = [
        _report("p1", "A", "reports/a.html",
                dist_strict={"True": 3, "Truthy": 0, "Unverifiable": 1,
                            "Falsey": 0, "False": 1}),
        _report("p2", "B", "reports/b.html",
                dist_strict={"True": 1, "Truthy": 1, "Unverifiable": 0,
                            "Falsey": 1, "False": 0}),
    ]
    html = _class_stats_strip(group)
    # reports=2, claims_checked=5+3=8, decided=4+3=7, true=3+2=5 -> 71%
    assert '<span class="cs-num">2</span><span class="cs-lbl">reports</span>' in html
    assert ('<span class="cs-num">8</span><span class="cs-lbl">claims checked</span>'
            in html)
    assert '<span class="cs-num">7</span><span class="cs-lbl">decided</span>' in html
    assert ('<span class="cs-num">71%</span>'
            '<span class="cs-lbl">true-leaning share of decided</span>' in html)
    assert '<span class="cs-num">7</span><span class="cs-lbl">n visible</span>' in html


def test_class_stats_strip_handles_zero_decided_without_crashing():
    group = [_report("p1", "A", "reports/a.html",
                     dist_strict={"Unverifiable": 3})]
    html = _class_stats_strip(group)
    assert '<span class="cs-num">0</span><span class="cs-lbl">decided</span>' in html
    assert ('<span class="cs-num">&mdash;</span>'
            '<span class="cs-lbl">true-leaning share of decided</span>' in html
            or '<span class="cs-num">—</span>'
               '<span class="cs-lbl">true-leaning share of decided</span>' in html)


def test_strip_is_scoped_per_class_not_pooled_across_classes(monkeypatch):
    """Presidential is 100% true-leaning of decided, Senate is 0%. If the
    strips were pooled they would both show a blended 50% -- they must not."""
    monkeypatch.setattr(site, "_report_classes", lambda: {
        "classes": {"p1": "presidential_address", "s1": "senate_floor"},
        "labels": {"presidential_address": "Presidential addresses",
                   "senate_floor": "Senate floor speeches"},
        "labels_inline": {}, "order": ["presidential_address", "senate_floor"],
    })
    reports = [
        _report("p1", "Pres", "reports/p.html",
                dist_strict={"True": 5, "Truthy": 0, "Unverifiable": 0,
                            "Falsey": 0, "False": 0}),
        _report("s1", "Sen", "reports/s.html",
                dist_strict={"True": 0, "Truthy": 0, "Unverifiable": 0,
                            "Falsey": 0, "False": 5}),
    ]
    html = _render_index(reports, {"total_claims": 10, "total_leaders": 2,
                                   "avg_consensus": 0.5})
    assert '<span class="cs-num">100%</span>' in html
    assert '<span class="cs-num">0%</span>' in html
    assert '<span class="cs-num">50%</span>' not in html


def test_no_cross_class_comparison_language_anywhere_on_the_index(monkeypatch):
    monkeypatch.setattr(site, "_report_classes", lambda: {
        "classes": {"p1": "presidential_address", "s1": "senate_floor"},
        "labels": {"presidential_address": "Presidential addresses",
                   "senate_floor": "Senate floor speeches"},
        "labels_inline": {}, "order": ["presidential_address", "senate_floor"],
    })
    reports = [
        _report("p1", "Pres", "reports/p.html", dist_strict={"True": 4, "False": 1}),
        _report("s1", "Sen", "reports/s.html", dist_strict={"True": 1, "False": 4}),
    ]
    html = _render_index(reports, {"total_claims": 10, "total_leaders": 2,
                                   "avg_consensus": 0.5})
    lowered = html.lower()
    for phrase in (" vs ", " vs.", "compared to", "compared with",
                   "higher than", "lower than", "outperform", "relative to",
                   "than the presidential", "than the senate",
                   "than presidential", "than senate"):
        assert phrase not in lowered, f"found forbidden comparison phrase {phrase!r}"


# ── 3. Label rename ─────────────────────────────────────────────────────────────


def test_leaders_reviewed_string_is_gone_speakers_reviewed_appears_twice():
    source = _SITE_PATH.read_text(encoding="utf-8")
    assert "Leaders Reviewed" not in source
    assert source.count("Speakers Reviewed") == 2


def test_index_program_stats_uses_speakers_reviewed_label():
    html = _render_index([], {"total_claims": 0, "total_leaders": 0,
                              "avg_consensus": 0})
    assert "Speakers Reviewed" in html
    assert "Leaders Reviewed" not in html


def test_top_level_program_stats_strip_introduces_no_new_percentage():
    """Counts only, except the ALREADY-existing Model Consensus % -- that one
    predates this change and is not something 3d introduced."""
    html = _render_index([], {"total_claims": 5, "total_leaders": 2,
                              "avg_consensus": 0.42})
    stats_start = html.index('<div class="stats">')
    how_start = html.index('<div class="how-strip">')
    top_level_segment = html[stats_start:how_start]
    assert top_level_segment.count("%") == 1  # Model Consensus, unchanged


# ── 4. The hard bar: report pages must render byte-identical ───────────────────


def _load_module_from_source(source: str, sibling_dir: Path, label: str):
    """Load ``source`` as a standalone module file placed ALONGSIDE the real
    site.py (not in a tmp dir), so its ``Path(__file__).resolve().parents[3]``
    / ``.parent`` lookups (data/report_classes.json, assets/icons/, ...)
    resolve exactly the way they do in production. Cleaned up by the caller."""
    tmp_path = sibling_dir / f"_snapshot_{label}_{uuid.uuid4().hex[:8]}.py"
    tmp_path.write_text(source, encoding="utf-8")
    mod_name = f"truthbot_publish_site_{label}_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(mod_name, tmp_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        tmp_path.unlink(missing_ok=True)
        raise
    return module, tmp_path


@pytest.fixture(scope="module")
def pre_rename_site():
    """The site.py module as it existed at git HEAD, i.e. before this
    session's uncommitted 3d edits -- loaded as an independent module so we
    can render with it directly and diff actual bytes against the current
    code, rather than trusting our own account of what changed.

    Skips (rather than fails) once HEAD no longer contains "Leaders
    Reviewed" -- i.e. once this work has been committed past the rename, at
    which point there is no historical revision left in HEAD to diff
    against, and test_leaders_reviewed_string_is_gone_speakers_reviewed_...
    above is the durable regression guard instead.
    """
    try:
        proc = subprocess.run(
            ["git", "show", "HEAD:src/truthbot/publish/site.py"],
            cwd=_REPO_ROOT, capture_output=True, text=True, timeout=20, check=True,
        )
    except Exception as exc:
        pytest.skip(f"could not read HEAD's site.py via git: {exc}")
    old_source = proc.stdout
    if "Leaders Reviewed" not in old_source:
        pytest.skip("HEAD no longer contains the pre-rename label -- this "
                    "worktree has been committed past 3d; the byte-identical "
                    "diff needs a pre-rename revision to compare against.")
    module, tmp_path = _load_module_from_source(old_source, _SITE_PATH.parent, "orig")
    try:
        yield module
    finally:
        sys.modules.pop(module.__name__, None)
        tmp_path.unlink(missing_ok=True)
        for pyc in tmp_path.parent.glob("__pycache__/" + tmp_path.stem + ".*.pyc"):
            pyc.unlink(missing_ok=True)


PRESIDENTIAL_FIXTURES = [
    ("clinton_1998", "Bill Clinton"),
    ("gwbush_2006", "George W. Bush"),
    ("obama_2014", "Barack Obama"),
    ("biden_2022", "Joe Biden"),
    ("trump_2026", "Donald Trump"),
]


@pytest.mark.parametrize("speech_id,speaker", PRESIDENTIAL_FIXTURES)
def test_presidential_report_pages_byte_identical_except_label_rename(
        pre_rename_site, speech_id, speaker):
    """THE hard bar (task 3d acceptance criterion): grouping the INDEX must
    not perturb a REPORT PAGE'S own render path at all. Renders the same
    SiteReport (same shared bundle objects, so claim ids/timestamps can never
    be a spurious source of diff) through the pre-change module and the
    current module, and requires the two HTML strings to be identical after
    reversing exactly the one intended edit."""
    bundles = _bundles()  # shared -- identical claim ids/content both sides
    kwargs = _site_report_kwargs(speaker, speech_id)

    old_report = pre_rename_site.SiteReport(bundles=bundles, **kwargs)
    new_report = SiteReport(bundles=bundles, **kwargs)

    old_html = pre_rename_site._render_report(old_report)
    new_html = _render_report(new_report)

    assert "Leaders Reviewed" in old_html
    assert "Leaders Reviewed" not in new_html
    assert "Speakers Reviewed" in new_html
    assert old_html.replace("Leaders Reviewed", "Speakers Reviewed") == new_html
