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

from truthbot.publish.consistency import (UNFIT_STANCE_NULL_RATE,
                                          check_publish_gate,
                                          check_run_artifacts,
                                          check_run_fitness, is_fit_to_gate,
                                          run_fitness_report)

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


# ── Fitness to gate (remediation v2 Phase A, A1) ─────────────────────────────
#
# Separate from the invariant lint above ON PURPOSE. An invariant violation
# means a pack broke a rule; unfit-to-gate means the evidence was never scored,
# so the T2.4 forced-Unverifiable gate is measuring retrieval silence. Mixing
# them would make `check_run_artifacts(...) == []` vacuous, which is why the
# fitness conditions come back in their own list and bite at publish time.


def _artifact(evidence: dict) -> dict:
    return {"meta": {"speech_id": "gwbush_2006"}, "claims": [], "rows": [],
            "evidence": evidence}


def _items(n: int, *, scored: bool, nulls: int) -> list[dict]:
    """n evidence dicts; ``nulls`` of them carry no stance."""
    return [{"source_url": f"https://x/{i}", "source_tier": "Established",
             "relevance_score": 0.87 if scored else 0.5,
             "supports_claim": None if i < nulls else True}
            for i in range(n)]


def test_is_fit_to_gate_accepts_a_scored_run_with_few_nulls():
    fit, reason = is_fit_to_gate(_artifact({"s:1": _items(20, scored=True, nulls=2)}))
    assert fit, reason
    assert "20 items relevance-scored" in reason


def test_is_fit_to_gate_rejects_a_run_whose_relevance_is_entirely_default():
    """Condition (a): not one item was ever seen by the relevance layer. This
    is the state of EVERY stored run — the v2 pack path has no scoring step."""
    fit, reason = is_fit_to_gate(_artifact({"s:1": _items(20, scored=False, nulls=0)}))
    assert not fit
    assert "relevance is entirely default" in reason


def test_is_fit_to_gate_rejects_an_excess_stance_null_rate():
    """Condition (b), independent of (a): relevance scored, but a quarter of
    the pack carries no stance, so a quarter of it cannot credit the quota."""
    fit, reason = is_fit_to_gate(_artifact({"s:1": _items(20, scored=True, nulls=5)}))
    assert not fit
    assert "stance-null rate 25.0%" in reason


def test_is_fit_to_gate_boundary_is_the_named_threshold():
    n = 20
    at = int(UNFIT_STANCE_NULL_RATE * n)          # exactly at the ceiling
    assert is_fit_to_gate(_artifact({"s:1": _items(n, scored=True, nulls=at)}))[0]
    assert not is_fit_to_gate(
        _artifact({"s:1": _items(n, scored=True, nulls=at + 1)}))[0]


def test_is_fit_to_gate_fails_closed_on_a_run_with_no_evidence():
    assert not is_fit_to_gate({"meta": {}, "rows": []})[0]
    assert not is_fit_to_gate(_artifact({}))[0]


@pytest.mark.skipif(not (REPO / "metrics" / "pca_runs").is_dir(),
                    reason="metrics/pca_runs not present")
def test_every_stored_run_artifact_is_currently_unfit_to_gate():
    """THE RETROACTIVE RECORD. Not one stored run — published or rebuilt — has
    a single relevance-scored evidence item, because score_evidence is
    unreachable from build_evidence_pack_v2. Every gate-forced Unverifiable on
    the published site was decided by a quota the scoring layer never fed.

    If this test ever fails it means a run got genuinely scored: update it,
    do not delete it."""
    rows = run_fitness_report(REPO)
    assert rows, "no artifacts present to report on"
    assert all(r["relevance_scored"] == 0 for r in rows)
    assert all(r["fit_to_gate"] is False for r in rows)


@pytest.mark.skipif(not (REPO / "metrics" / "pca_runs").is_dir(),
                    reason="metrics/pca_runs not present")
def test_fitness_is_reported_separately_and_never_as_an_invariant_violation():
    """Both halves at once: the fitness report is non-empty (every run is
    unfit) AND check_run_artifacts still returns []. The existing suite's
    `violations == []` keeps meaning what it always meant."""
    assert check_run_fitness(REPO)
    assert check_run_artifacts(REPO) == []


def test_publish_gate_refuses_an_unfit_run_and_passes_a_fit_one():
    unfit = check_publish_gate(_artifact({"s:1": _items(20, scored=False, nulls=0)}))
    assert len(unfit) == 1
    assert "unfit-to-gate, refusing to publish" in unfit[0]
    assert "gwbush_2006" in unfit[0]
    assert check_publish_gate(_artifact({"s:1": _items(20, scored=True, nulls=1)})) == []


def test_rerender_refuses_to_publish_an_unfit_artifact(tmp_path):
    """The gate has teeth where it matters: the publish path itself. The check
    runs before any rendering work, so a None publisher is never reached."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rerender_pca_site", REPO / "scripts" / "rerender_pca_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    path = tmp_path / "run.json"
    path.write_text(json.dumps(_artifact({"s:1": _items(20, scored=False, nulls=0)})))
    with pytest.raises(SystemExit) as exc:
        mod.render_artifact(path, publisher=None, role="President")
    assert "PUBLISH GATE FAILED" in str(exc.value)
    assert "unfit-to-gate" in str(exc.value)


def test_rerender_allows_an_unfit_artifact_only_for_a_staged_review_render(tmp_path):
    """--allow-unfit-gate (require_fit=False) is the DC-6 review surface, not a
    publish: it proceeds past the gate but says so out loud."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rerender_pca_site", REPO / "scripts" / "rerender_pca_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    path = tmp_path / "run.json"
    path.write_text(json.dumps(_artifact({"s:1": _items(20, scored=False, nulls=0)})))
    # Passes the gate, then fails later for an unrelated reason (no publisher)
    # — what matters is that SystemExit from the gate is NOT what stops it.
    with pytest.raises(Exception) as exc:
        mod.render_artifact(path, publisher=None, role="President",
                            require_fit=False)
    assert "PUBLISH GATE FAILED" not in str(exc.value)
