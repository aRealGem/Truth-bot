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
