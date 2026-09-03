"""A run that leaves head resolution must leave the site (FR-0901-10).

data/reports.json accumulates across renders, so before this the index and
publishing_heads() could disagree and the site went on publishing a withdrawn
report -- which is exactly what happened to the vetoed Cruz speech.
"""
import json
import pytest

from truthbot.publish.site import SitePublisher

GEN = "g1"


def _corpus(tmp_path, held_b):
    runs = tmp_path / "runs"
    runs.mkdir()
    meta = {"a": {"speech_id": "sp_a", "generation": GEN},
            "b": {"speech_id": "sp_b", "generation": GEN}}
    if held_b:
        meta["b"]["held"] = "vetoed, test"
    (runs / "methodology_manifest.json").write_text(
        json.dumps({"current_generation": GEN, "runs": meta}), encoding="utf-8")
    for rid, sid in (("a", "sp_a"), ("b", "sp_b")):
        (runs / f"{rid}.json").write_text(
            json.dumps({"meta": {"speech_id": sid}, "evidence": {"x:1": []}}),
            encoding="utf-8")
    return runs


def _site(tmp_path):
    root = tmp_path / "site"
    (root / "data").mkdir(parents=True)
    (root / "reports").mkdir()
    (root / "claims").mkdir()
    reports = [{"id": "RA", "speech_id": "sp_a", "url": "reports/a.html", "date": "2020-01-01"},
               {"id": "RB", "speech_id": "sp_b", "url": "reports/b.html", "date": "2021-01-01"}]
    claims = [{"id": "CA1", "report_id": "RA"}, {"id": "CB1", "report_id": "RB"},
              {"id": "CB2", "report_id": "RB"}]
    (root / "data" / "reports.json").write_text(json.dumps(reports), encoding="utf-8")
    (root / "data" / "claims.json").write_text(json.dumps(claims), encoding="utf-8")
    for n in ("a", "b"):
        (root / "reports" / f"{n}.html").write_text(f"<p>{n}</p>", encoding="utf-8")
        (root / "reports" / f"{n}-triage.html").write_text(f"<p>{n} triage</p>", encoding="utf-8")
    for c in ("CA1", "CB1", "CB2"):
        (root / "claims" / f"{c}.html").write_text(f"<p>{c}</p>", encoding="utf-8")
    return root, reports, claims


def test_a_held_run_leaves_every_surface(tmp_path):
    runs = _corpus(tmp_path, held_b=True)
    root, reports, claims = _site(tmp_path)
    pub = SitePublisher(site_root=root, runs_dir=runs)

    kept_r, kept_c = pub._prune_unpublished(list(reports), list(claims))

    # 1. reports.json row
    assert [r["speech_id"] for r in kept_r] == ["sp_a"]
    # 2. claims.json rows
    assert [c["id"] for c in kept_c] == ["CA1"]
    # 3. report page + 4. triage page
    assert not (root / "reports" / "b.html").exists()
    assert not (root / "reports" / "b-triage.html").exists()
    # 5. claim pages
    assert not (root / "claims" / "CB1.html").exists()
    assert not (root / "claims" / "CB2.html").exists()
    # 6. the feed renders from the pruned index, so the entry goes with it
    pub._write_feed(kept_r)
    feed = (root / "feed.xml").read_text(encoding="utf-8")
    assert "reports/b.html" not in feed
    assert "reports/a.html" in feed


def test_the_unheld_run_is_byte_identical(tmp_path):
    runs = _corpus(tmp_path, held_b=True)
    root, reports, claims = _site(tmp_path)
    before = {p.name: p.read_bytes() for p in
              [root / "reports" / "a.html", root / "reports" / "a-triage.html",
               root / "claims" / "CA1.html"]}

    SitePublisher(site_root=root, runs_dir=runs)._prune_unpublished(
        list(reports), list(claims))

    for name, blob in before.items():
        hit = next(p for p in root.rglob(name))
        assert hit.read_bytes() == blob, f"{name} was modified by the prune"


def test_nothing_is_pruned_when_no_run_is_held(tmp_path):
    runs = _corpus(tmp_path, held_b=False)
    root, reports, claims = _site(tmp_path)
    kept_r, kept_c = SitePublisher(
        site_root=root, runs_dir=runs)._prune_unpublished(list(reports), list(claims))
    assert len(kept_r) == 2 and len(kept_c) == 3
    assert (root / "reports" / "b.html").exists()


def test_a_row_with_no_speech_id_is_left_alone(tmp_path):
    """Legacy rows predate the field; unknown must not mean delete."""
    runs = _corpus(tmp_path, held_b=True)
    root, reports, claims = _site(tmp_path)
    legacy = [{"id": "RL", "url": "reports/l.html", "date": "2019-01-01"}]
    kept_r, _ = SitePublisher(site_root=root, runs_dir=runs)._prune_unpublished(
        list(reports) + legacy, list(claims))
    assert "RL" in [r["id"] for r in kept_r]


def test_a_broken_lineage_prunes_nothing(tmp_path):
    """publishing_heads() raising is not a licence to delete the site."""
    runs = _corpus(tmp_path, held_b=False)
    (runs / "methodology_manifest.json").write_text(
        json.dumps({"runs": {}}), encoding="utf-8")   # no current_generation
    root, reports, claims = _site(tmp_path)
    kept_r, kept_c = SitePublisher(
        site_root=root, runs_dir=runs)._prune_unpublished(list(reports), list(claims))
    assert len(kept_r) == 2 and len(kept_c) == 3
