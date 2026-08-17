"""Corrections-machinery tests (P67.6 / PR-3, remediation T1.5).

Pins: schema validation fails loudly; corrections apply to artifact rows
in-memory with old-verdict guard; the note threads bridge → provenance →
claim-card strip; the Corrections page renders entries and an honest empty
state; every page footer links the page.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from truthbot.publish.corrections import (
    CorrectionsError,
    apply_to_artifact,
    load_corrections,
    note_for,
)
from truthbot.publish.site import _render_corrections

ENTRY = {
    "sid": "biden_2022:0115",
    "speech_id": "biden_2022",
    "old_verdict": "FALSE",
    "new_verdict": "TRUE",
    "reason": "Panel confused quarterly annualized GDP with the 5.7% annual figure.",
    "date": "2026-07-21",
    "source": "agreed-verdict-audit-2026-07-21",
}


def _write(tmp_path: Path, doc: dict) -> Path:
    p = tmp_path / "corrections.json"
    p.write_text(json.dumps(doc))
    return p


def test_load_corrections_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_corrections(tmp_path / "nope.json") == []


def test_load_corrections_validates_schema_and_entries(tmp_path: Path) -> None:
    good = _write(tmp_path, {"schema": "truthbot-corrections v1", "entries": [ENTRY]})
    assert load_corrections(good) == [ENTRY]

    with pytest.raises(CorrectionsError, match="schema"):
        load_corrections(_write(tmp_path, {"schema": "v0", "entries": []}))
    with pytest.raises(CorrectionsError, match="missing"):
        load_corrections(_write(tmp_path, {
            "schema": "truthbot-corrections v1",
            "entries": [{**ENTRY, "reason": ""}]}))
    with pytest.raises(CorrectionsError, match="same verdict"):
        load_corrections(_write(tmp_path, {
            "schema": "truthbot-corrections v1",
            "entries": [{**ENTRY, "new_verdict": "FALSE"}]}))
    with pytest.raises(CorrectionsError, match="duplicate"):
        load_corrections(_write(tmp_path, {
            "schema": "truthbot-corrections v1", "entries": [ENTRY, ENTRY]}))


def _artifact() -> dict:
    return {
        "meta": {"speech_id": "biden_2022", "date": "2022-03-01"},
        "rows": [
            {"sid": "biden_2022:0115", "verdict": "FALSE", "status": "resolved",
             "reasoning": "r", "votes": {"FALSE": 2}, "escalated": False},
            {"sid": "biden_2022:0244", "verdict": "FALSE", "status": "resolved",
             "reasoning": "r", "votes": {"FALSE": 2}, "escalated": False},
        ],
        "claims": [],
    }


def test_apply_to_artifact_rewrites_row_and_stamps_note() -> None:
    artifact = _artifact()
    assert apply_to_artifact(artifact, [ENTRY]) == 1
    row = artifact["rows"][0]
    assert row["verdict"] == "TRUE"
    assert row["corrected"]["old"] == "FALSE"
    assert "Corrected FALSE → TRUE (2026-07-21)" in row["corrected"]["note"]
    # untouched sibling
    assert artifact["rows"][1].get("corrected") is None


def test_apply_guards_old_verdict_mismatch_and_unknown_sid() -> None:
    artifact = _artifact()
    with pytest.raises(CorrectionsError, match="expects old verdict"):
        apply_to_artifact(artifact, [{**ENTRY, "old_verdict": "MISLEADING",
                                      "new_verdict": "TRUE"}])
    with pytest.raises(CorrectionsError, match="absent"):
        apply_to_artifact(_artifact(), [{**ENTRY, "sid": "biden_2022:9999"}])


def test_apply_ignores_other_speeches() -> None:
    artifact = _artifact()
    assert apply_to_artifact(artifact, [{**ENTRY, "speech_id": "trump_2026",
                                         "sid": "trump_2026:0001"}]) == 0


def test_correction_note_threads_into_provenance_and_renders_via_one_path() -> None:
    from truthbot.verdict.bridge import _build_provenance
    from truthbot.models import VerdictProvenance

    row = {"votes": {"FALSE": 2}, "split": False, "escalated": False,
           "corrected": {"note": note_for(ENTRY)}}
    prov = _build_provenance(row, {"layer_a": {"label": "check-worthy"}})
    assert "Corrected FALSE → TRUE" in prov.correction_note

    from truthbot.publish.site import (_correction_provenance_html,
                                       _pca_provenance_strip)
    from tests.test_site_render_aggregates import _make_bundle
    from truthbot.models import VerdictLabel

    # F14: the note is emitted by the ONE shared helper, NOT by the provenance
    # chain strip — so it survives when the strip renders nothing.
    full = VerdictProvenance(layer_a_label="check-worthy", panel_votes={"True": 2},
                             correction_note=note_for(ENTRY))
    html = _correction_provenance_html(full)
    assert "pca-correction" in html and "Corrected FALSE" in html
    assert 'href="../corrections.html"' in html
    # The chain strip no longer carries the note (single path, no drift).
    b = _make_bundle(VerdictLabel.TRUE, coarse_lenient="True", coarse_strict="True")
    b.consensus.provenance = full
    assert "pca-correction" not in _pca_provenance_strip(b)

    # The gated/minimal case: an EMPTY provenance chain still renders the note.
    gated = VerdictProvenance(correction_note=note_for(ENTRY))
    assert _pca_provenance_strip(
        _make_bundle(VerdictLabel.UNVERIFIABLE)) is not None  # smoke
    assert "Corrected FALSE" in _correction_provenance_html(gated)


def test_corrections_page_renders_entries_and_empty_state() -> None:
    html = _render_corrections([ENTRY])
    assert "biden_2022:0115" in html
    assert "FALSE" in html and "TRUE" in html
    assert "2026-07-21" in html
    assert "never" in html  # "never applied silently" policy sentence

    empty = _render_corrections([])
    assert "No corrections have been issued" in empty


def test_footers_link_corrections_page(tmp_path: Path) -> None:
    from tests.test_site_render_aggregates import _make_bundle, _make_site_report
    from truthbot.models import VerdictLabel
    from truthbot.publish.site import SitePublisher

    pub = SitePublisher(site_root=tmp_path, corrections=[])
    sr = _make_site_report([_make_bundle(
        VerdictLabel.TRUE, coarse_lenient="True", coarse_strict="True")])
    report_path = pub.publish(sr)
    assert 'href="./corrections.html"' in (tmp_path / "index.html").read_text()
    assert 'href="../corrections.html"' in report_path.read_text()
    assert "No corrections have been issued" in (tmp_path / "corrections.html").read_text()
    assert 'corrections.html" rel="related"' in (tmp_path / "feed.xml").read_text()


def test_corrections_page_renders_editorial_notes() -> None:
    html = _render_corrections([ENTRY], notes=[
        {"date": "2026-07-21", "text": "Headline moved from X to Y."}])
    assert "corrections-note" in html
    assert "Headline moved from X to Y." in html


def test_report_banner_derives_from_corrected_bundles(tmp_path: Path) -> None:
    from tests.test_site_render_aggregates import _make_bundle, _make_site_report
    from truthbot.models import VerdictLabel, VerdictProvenance
    from truthbot.publish.site import _render_report

    corrected = _make_bundle(VerdictLabel.TRUE, coarse_lenient="True",
                             coarse_strict="True")
    corrected.consensus.provenance = VerdictProvenance(
        layer_a_label="check-worthy", panel_votes={"True": 2},
        correction_note=note_for(ENTRY))
    plain = _make_bundle(VerdictLabel.FALSE, coarse_lenient="False",
                         coarse_strict="False")
    html = _render_report(_make_site_report([corrected, plain]))
    assert "report-correction-banner" in html
    assert "1 verdict on this report was revised on 2026-07-21" in html

    # no corrections -> no banner
    html2 = _render_report(_make_site_report([plain]))
    assert "report-correction-banner" not in html2


def test_repo_corrections_file_is_valid_and_matches_net_ledger() -> None:
    """The committed corrections must load cleanly and every entry must trace to
    the DC-6' net ledger (F6) with a matching verdict move.

    The 2026-07-21 agreed-verdict audit that used to back this ledger describes a
    run that has since been re-adjudicated from scratch; the live ledger is now
    the LIVE-vs-STAGED net across the five recorded hops, and its provenance is
    ``metrics/remediation_v2/dc6_net_ledger.json`` (whose own gate cross-checks
    every net verdict against the publishing head)."""
    import pytest
    repo = Path(__file__).resolve().parents[2]
    cpath = repo / "data" / "corrections.json"
    npath = repo / "metrics" / "remediation_v2" / "dc6_net_ledger.json"
    if not (cpath.exists() and npath.exists()):
        pytest.skip("corrections/net-ledger artifacts not in this checkout")
    entries = load_corrections(cpath)
    net = {e["sid"]: e for e in json.loads(npath.read_text())["entries"]}
    # the changelog is exactly the net ledger's ledger-eligible set, no more.
    assert {e["sid"] for e in entries} == set(net)
    # The verdict move is pinned for entries that CLAIM DC-6' provenance. A
    # later correction wave supersedes an entry in place — the ledger permits
    # one entry per sid, and the displaced one is archived, never deleted — so
    # its move legitimately differs from the DC-6' record, which describes the
    # wave being superseded and cannot anticipate the next one. Scoped by
    # source rather than relaxed: an entry still claiming DC-6' provenance must
    # still match DC-6' exactly.
    for e in entries:
        rec = net.get(e["sid"])
        assert rec is not None, e["sid"]
        if not str(e.get("source", "")).startswith("dc6-"):
            continue
        assert rec["old_verdict"] == e["old_verdict"]
        assert rec["new_verdict"] == e["new_verdict"]


def test_a_superseded_correction_survives_and_the_history_joins_up() -> None:
    """A correction may only be replaced if its predecessor stays readable.

    One entry per sid means a re-correction REPLACES rather than appends. That
    is only honest if the displaced entry is still on disk AND the two join up:
    otherwise the page shows the latest move and quietly loses that the claim
    had been corrected before — which is the exact thing this ledger exists to
    prevent.

    Scoped to archives that declare ``kind == "entry-supersede"``. The older
    archives are whole-ledger snapshots taken when the corpus was
    re-adjudicated FROM SCRATCH (the DC-6' clean-slate reset), so their entries
    describe a different run and legitimately do not continue into the live
    ones — ``trump_2026:0057`` goes TRUE in the 2026-08-06 snapshot and
    MISLEADING live, and that is not a broken chain, it is a different corpus.
    """
    import pytest
    repo = Path(__file__).resolve().parents[2]
    cpath = repo / "data" / "corrections.json"
    if not cpath.exists():
        pytest.skip("corrections not in this checkout")
    live = {e["sid"]: e for e in load_corrections(cpath)}
    archived: dict[str, list] = {}
    for arch in sorted((repo / "data").glob("corrections-archive-*.json")):
        doc = json.loads(arch.read_text())
        if doc.get("kind") != "entry-supersede":
            continue
        for e in doc.get("entries") or []:
            archived.setdefault(e["sid"], []).append(e)

    joined = 0
    for sid, e in live.items():
        prior = archived.get(sid) or []
        if not prior:
            continue
        last = prior[-1]
        assert e["old_verdict"].upper() == last["new_verdict"].upper(), (
            f"{sid}: the live correction starts at {e['old_verdict']} but its "
            f"archived predecessor ended at {last['new_verdict']} — the "
            "published history would not join up")
        joined += 1
    assert joined, "no superseded correction found — this would pass vacuously"


# ── Empty-state vs ledger display (remediation v2, 1.11) ─────────────────────
#
# Root cause of the shipped bug: rerender_pca_site.py --corrections skip
# passed corrections=None to SitePublisher while still passing notes, so
# corrections.html rendered the 2026-07-21 audit note AND "No corrections
# have been issued" at once. The flag now governs apply_to_artifact only;
# the page always renders the full ledger, and the empty state appears
# ONLY when the ledger itself is empty.


def test_corrections_page_with_entries_and_notes_has_no_empty_state() -> None:
    html = _render_corrections([ENTRY], notes=[
        {"date": "2026-07-21", "text": "Audit note."}])
    assert "No corrections have been issued" not in html
    assert "corrections-table" in html
    assert "Audit note." in html


def test_corrections_page_truly_empty_ledger_renders_empty_state() -> None:
    html = _render_corrections([], notes=[])
    assert "No corrections have been issued" in html
    assert "corrections-table" not in html


def test_check_site_flags_page_with_both_table_and_empty_state(tmp_path: Path) -> None:
    """The both-present state is impossible via _render_corrections; the
    lint catches a caller that stitches the page together inconsistently."""
    from tests.test_site_render_aggregates import _make_bundle, _make_site_report
    from truthbot.models import VerdictLabel
    from truthbot.publish.consistency import check_site
    from truthbot.publish.site import SitePublisher

    pub = SitePublisher(site_root=tmp_path, corrections=[ENTRY])
    pub.publish(_make_site_report([_make_bundle(
        VerdictLabel.TRUE, coarse_lenient="True", coarse_strict="True")]))
    page = tmp_path / "corrections.html"
    assert "corrections-table" in page.read_text()
    violations = check_site(tmp_path)
    assert not any("corrections.html" in v for v in violations)

    # Tamper: append the empty-state sentence next to the rendered table.
    page.write_text(page.read_text().replace(
        "</table>",
        "</table><p class=\"dim\">No corrections have been issued for the "
        "currently published reports.</p>"))
    violations = check_site(tmp_path)
    assert any("corrections.html" in v and "BOTH" in v for v in violations)


def test_rerender_script_passes_full_ledger_for_display_on_skip() -> None:
    """The script wires the FULL ledger + resolution-state changes into
    SitePublisher for display, and (F12) passes the mode to render_artifact so
    'skip' annotates strips in place while 'apply' rewrites verdicts. Default is
    'skip'. Pin the wiring at source level (running main() needs artifacts)."""
    import re
    from pathlib import Path as _P

    src = (_P(__file__).resolve().parents[2] /
           "scripts" / "rerender_pca_site.py").read_text(encoding="utf-8")
    # Publisher receives the full ledger AND the F9 resolution-state changes.
    assert re.search(r"SitePublisher\(\s*site_root=args\.site_root,\s*"
                     r"corrections=corrections", src), \
        "publisher must receive the full ledger for display"
    assert "resolution_changes=resolution" in src
    # render_artifact receives the ledger + the mode + the resolution set.
    assert re.search(
        r"render_artifact\(p, publisher, args\.role, corrections=corrections",
        src)
    assert "mode=args.corrections" in src and "resolution=resolution" in src
    # F12: default is skip (annotate, do not rewrite).
    assert re.search(r'--corrections".*?default="skip"', src, re.S)
