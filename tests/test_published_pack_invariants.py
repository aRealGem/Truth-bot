"""Published-data invariant lints over stored pca_runs artifacts
(remediation v2, 1.4).

The methodology manifest pins every artifact to the generation it was produced
under. Current-generation runs must satisfy the current invariants (S5 cap,
era, zero fact-check URLs); older-generation runs are permanently legacy —
this is what blocks re-publishing them as-is.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from truthbot.publish.consistency import check_run_artifacts

REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO / "metrics" / "pca_runs"
MANIFEST = RUNS_DIR / "methodology_manifest.json"

pytestmark = pytest.mark.skipif(
    not RUNS_DIR.is_dir(), reason="metrics/pca_runs not present in this checkout")


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_exists_and_declares_current_generation():
    m = _manifest()
    assert m["schema"] == "truthbot-methodology-manifest v1"
    assert m["current_generation"] in m["generations"]
    for run_id, row in m["runs"].items():
        assert row["generation"] in m["generations"], (
            f"{run_id}: unknown generation {row['generation']!r}")
        assert isinstance(row["published"], bool)
        assert row["speech_id"]


def test_current_generation_runs_satisfy_invariants():
    """(i) per-sid S5 cap, (ii) zero era violations, (iii) zero fact-check
    URLs — asserted over every current-generation run's stored artifact."""
    violations = check_run_artifacts(REPO)
    assert violations == []


@pytest.mark.parametrize("run_prefix,why", [
    ("28965cdf", "obama_2014 pre-s5-cap: political tier without the <=3 cap "
                 "+ rescue-leg era violations"),
    ("23939712", "trump_2026 pre-s5-tiering: no political tier existed"),
    ("7208bbbb", "biden_2022 pre-s5-tiering: no political tier existed"),
])
def test_pre_s5_published_runs_are_permanently_legacy(run_prefix, why):
    """DIRECT REGRESSION: the three published pre-S5 artifacts must never be
    labeled current_generation — that label is what would let them re-publish
    as-is, and they carry exactly the defects the generation gate exists for."""
    m = _manifest()
    rows = {rid: row for rid, row in m["runs"].items()
            if rid.startswith(run_prefix)}
    assert rows, f"run {run_prefix} missing from the manifest"
    for rid, row in rows.items():
        assert row["generation"] != m["current_generation"], (
            f"{rid} must not be current_generation: {why}")
        assert row["published"] is True, (
            f"{rid}: this regression pins the PUBLISHED pre-S5 artifacts")


def test_manifest_is_complete_over_the_runs_directory():
    """Every artifact on disk has a manifest row (and every row an artifact,
    which check_run_artifacts asserts) — a new run cannot dodge the generation
    gate by simply not being classified."""
    m = _manifest()
    on_disk = {p.stem for p in RUNS_DIR.glob("*.json")
               if p.name != MANIFEST.name}
    missing = on_disk - set(m["runs"])
    assert not missing, f"artifacts without a manifest row: {sorted(missing)}"


def test_absent_artifact_is_skipped_not_a_violation(tmp_path):
    """A manifest row whose artifact file is not in the checkout must not read
    as an integrity failure — CI clones carry the manifest without the large
    (partly untracked) run artifacts. Regression: this failed CI on 2026-08-03
    while passing locally, where every artifact happens to exist."""
    runs = tmp_path / "metrics" / "pca_runs"
    runs.mkdir(parents=True)
    (runs / "methodology_manifest.json").write_text(json.dumps({
        "schema": "truthbot-methodology-manifest v1",
        "current_generation": "v2.3-role-axis-s5cap",
        "generations": {"v2.3-role-axis-s5cap": "current"},
        "runs": {"deadbeef-0000-0000-0000-000000000000": {
            "speech_id": "gwbush_2006", "generation": "v2.3-role-axis-s5cap",
            "published": False}},
    }))
    assert check_run_artifacts(tmp_path) == []


def test_present_artifact_still_fails_on_violation(tmp_path):
    """The gate keeps its teeth: a PRESENT current-generation artifact that
    breaks an invariant (here: a fact-check URL in the pack) still fails."""
    runs = tmp_path / "metrics" / "pca_runs"
    runs.mkdir(parents=True)
    run_id = "deadbeef-0000-0000-0000-000000000001"
    (runs / "methodology_manifest.json").write_text(json.dumps({
        "schema": "truthbot-methodology-manifest v1",
        "current_generation": "v2.3-role-axis-s5cap",
        "generations": {"v2.3-role-axis-s5cap": "current"},
        "runs": {run_id: {"speech_id": "gwbush_2006",
                          "generation": "v2.3-role-axis-s5cap",
                          "published": False}},
    }))
    (runs / f"{run_id}.json").write_text(json.dumps({
        "run_id": run_id, "meta": {"speech_id": "gwbush_2006"},
        "claims": [], "rows": [],
        "evidence": {"gwbush_2006:0001": [{
            "source_url": "https://www.politifact.com/factchecks/x/",
            "source_tier": "Established", "published_at": "2006-01-30",
            "snippet": "x"}]},
    }))
    assert any("factcheck" in v.lower() or "fact-check" in v.lower()
               for v in check_run_artifacts(tmp_path))
