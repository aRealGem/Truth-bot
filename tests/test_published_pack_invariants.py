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

from truthbot.publish.consistency import (RUN_COHORT_GLOSS, RUN_COHORT_ORDER,
                                          UNFIT_STANCE_NULL_RATE,
                                          check_publish_gate,
                                          check_run_artifacts,
                                          check_run_fitness,
                                          fitness_composition, fitness_finding,
                                          is_fit_to_gate, run_cohort,
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
def test_only_the_score_propagated_heads_are_fit_to_gate():
    """THE RETROACTIVE RECORD. Every gate-forced Unverifiable on the published
    site was decided by a quota the scoring layer never fed: score_evidence was
    unreachable from build_evidence_pack_v2, so no stored run carried a single
    relevance-scored evidence item.

    UPDATED THREE TIMES, following this test's own standing instruction to
    update rather than delete once a run gets genuinely scored:

      * the adjudication wave (2026-08-09) scored the 29 packs it
        re-adjudicated and nothing else — 29 repaired packs could not pull a
        corpus-wide stance-null rate under the 15% ceiling, so every run stayed
        unfit;
      * the score propagation (2026-08-10, scripts/propagate_rescores.py)
        merged the B1a+B2 sidecars — the scores the wave had only OVERLAID at
        runtime — into a new publishing head per speech. Those heads are the
        first artifacts whose STORED evidence is the evidence their verdicts
        were actually reached on.
      * the Senate floor speeches (2026-09-01, FR-0901-02) are the first runs
        scored NATIVELY -- their evidence was scored by the pipeline that
        produced them, not repaired after the fact by a wave or a propagation.
        They carry meta.scoring = "native" (authored at registration), which is
        a third legitimate provenance for scored evidence rather than a stray.
        Two of the four still come back unfit: budd_2025-04-02 at 47.6% and
        tillis_2025-01-23 at 31.8% stance-null. The ceiling does not move to
        accommodate a speech, so both are HELD (FR-0901-04) and leave head
        resolution; budd is additionally relabelled pre-s5-cap, because its
        :0012 pack stores 4 POLITICAL items against a <=3 cap and so does not
        satisfy the methodology its generation names.

    Both halves of the record are held here:

      * scoring is CONFINED to the wave artifacts and the propagated heads. A
        scored item anywhere else means something re-scored a run outside those
        two passes, which nothing is supposed to do;
      * fitness is confined to the propagated heads, and not even all of them.
        trump_2026 comes back at 21.0% stance-null — the scorer itself read 309
        of its 1,472 items as stance-free — so its head is UNFIT and stays
        unfit. The ceiling does not move to accommodate a speech.
    """
    rows = run_fitness_report(REPO)
    assert rows, "no artifacts present to report on"

    runs_dir = REPO / "metrics" / "pca_runs"

    def _meta(run_id: str) -> dict:
        path = runs_dir / f"{run_id}.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text("utf-8")).get("meta") or {}

    scored = [r for r in rows if r["relevance_scored"]]
    assert scored, "no scored run at all — the repaired artifacts are missing"
    def _provenance(run_id: str) -> bool:
        m = _meta(run_id)
        return bool(m.get("wave") or m.get("score_propagation")
                    or m.get("scoring") == "native")

    stray = [r["run_id"] for r in scored if not _provenance(r["run_id"])]
    assert not stray, (
        f"scored evidence outside the wave/propagation/native: {stray}")

    assert {r["speech_id"] for r in rows if r["fit_to_gate"]} == {
        "gwbush_2006", "clinton_1998", "obama_2014", "biden_2022",
        "cruz_2026-06-24", "warren_2025-04-29"}
    for row in rows:
        if row["fit_to_gate"]:
            assert _provenance(row["run_id"]), (
                f"{row['run_id']}: fit to gate without a scoring provenance — "
                "something moved the bar instead of the evidence")


@pytest.mark.skipif(not (REPO / "metrics" / "pca_runs").is_dir(),
                    reason="metrics/pca_runs not present")
def test_fitness_is_reported_separately_and_never_as_an_invariant_violation():
    """Both halves at once: the fitness report is non-empty (every run is
    unfit) AND check_run_artifacts still returns []. The existing suite's
    `violations == []` keeps meaning what it always meant."""
    assert check_run_fitness(REPO)
    assert check_run_artifacts(REPO) == []


# ── the denominator that has to travel with the number ───────────────────────
#
# "Every stored run is unfit to gate" is stated over 17 artifacts on a site
# that publishes 5 reports. A bare 17 beside a 5-report corpus is unreadable —
# it looks either like triple-counting or like a much larger failure than it
# is. So the composition (5 published + 5 rebuilt + 7 superseded) is computed,
# not typed, and it is asserted everywhere the number appears.

def test_run_cohort_splits_published_rebuilt_and_superseded():
    current = "v2.3-role-axis-s5cap"
    # Published is published whatever vintage it ran on — the live corpus is
    # deliberately not all one generation.
    assert run_cohort({"published": True, "generation": "pre-s5-tiering"},
                      current) == "published"
    assert run_cohort({"published": False, "generation": current},
                      current) == "rebuilt"
    assert run_cohort({"published": False, "generation": "pre-s5-cap"},
                      current) == "superseded"


def test_fitness_composition_spells_out_the_denominator():
    rows = [{"cohort": "published"}] * 2 + [{"cohort": "superseded"}]
    line = fitness_composition(rows)
    assert line == ("3 stored run artifacts = 2 published (live on the site) "
                    "+ 1 superseded (retained per archive-never-delete)")
    # A cohort with no members is omitted, never printed as "0 rebuilt".
    assert "rebuilt" not in line


def test_every_cohort_has_a_gloss_and_a_place_in_the_order():
    assert set(RUN_COHORT_ORDER) == set(RUN_COHORT_GLOSS)


def test_fitness_report_rows_carry_their_cohort():
    rows = run_fitness_report(REPO)
    assert all(r["cohort"] in RUN_COHORT_GLOSS for r in rows)
    assert {r["cohort"] for r in rows} == set(RUN_COHORT_ORDER)


def test_the_lint_states_the_tally_with_its_denominator_and_names_cohorts():
    """The bare 17 is the misreading risk, so the lint's own output carries the
    composition on line one and a cohort on every run line.

    The numerator is no longer the whole corpus — the score-propagated heads
    are fit — so it is READ OFF the report rather than assumed equal to the
    denominator. A tally that could only be written as "N of N" would have to
    be rewritten every time a run got repaired, which is precisely when a lint
    must not need editing.
    """
    lines = check_run_fitness(REPO)
    rows = run_fitness_report(REPO)
    unfit = [r for r in rows if not r["fit_to_gate"]]
    assert unfit and len(unfit) < len(rows)
    assert lines[0] == (f"{len(unfit)} of {len(rows)} stored run artifacts "
                        f"unfit to gate — {fitness_composition(rows)}")
    for line in lines[1:]:
        assert any(f", {c}" in line for c in RUN_COHORT_ORDER)


def test_the_finding_text_carries_the_composition_too():
    finding = fitness_finding(run_fitness_report(REPO))
    assert fitness_composition(run_fitness_report(REPO)) in finding
    assert "unfit to gate" in finding


def test_committed_a1_report_is_what_the_generator_emits():
    """The report artifact is generated, not hand-maintained — which is how the
    finding text and the cohort split stay in one place. Regenerate with
    scripts/emit_a1_fitness_report.py if this fails."""
    import importlib.util

    path = REPO / "metrics" / "remediation_v2" / "a1_fitness_report.json"
    if not path.exists():
        pytest.skip("A1 fitness report not generated in this tree")
    spec = importlib.util.spec_from_file_location(
        "emit_a1_fitness_report", REPO / "scripts" / "emit_a1_fitness_report.py")
    emit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(emit)

    committed = json.loads(path.read_text("utf-8"))
    fresh = emit.build_report(REPO, generated=committed["generated"])

    # Compare only the runs whose artifact FILE is in this checkout. The report
    # is a record of every stored artifact on the machine that generated it; a
    # fresh clone (CI) carries only the tracked ones, so a whole-document
    # equality would fail on data availability rather than on drift — the same
    # confusion that broke CI on 2026-08-03. Rows we can regenerate must match
    # exactly; rows we cannot see are not evidence of anything.
    runs_dir = REPO / "metrics" / "pca_runs"
    fresh_rows = {r["run_id"]: r for r in fresh["runs"]}
    for row in committed["runs"]:
        rid = row["run_id"]
        if not (runs_dir / f"{rid}.json").exists():
            continue
        assert rid in fresh_rows, f"{rid}: artifact present but not regenerated"
        assert row == fresh_rows[rid], f"{rid}: committed row != regenerated"
    assert {r["run_id"] for r in fresh["runs"]} <= {r["run_id"] for r in committed["runs"]}, \
        "generator saw an artifact the committed report does not record — regenerate"
    # And the finding a human reads names the denominator, not a bare count.
    assert fitness_composition(committed["runs"]) in committed["finding"]
    assert set(committed["cohorts"]) == set(RUN_COHORT_ORDER)


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


# ── A4: speech dates must resolve, or nothing publishes ──────────────────────


def _manifest_only(tmp_path, speech_id: str) -> Path:
    runs = tmp_path / "metrics" / "pca_runs"
    runs.mkdir(parents=True)
    (runs / "methodology_manifest.json").write_text(json.dumps({
        "schema": "truthbot-methodology-manifest v1",
        "current_generation": "v2.3-role-axis-s5cap",
        "generations": {"v2.3-role-axis-s5cap": "current",
                        "pre-s5-tiering": "legacy"},
        "runs": {"deadbeef-0000-0000-0000-000000000002": {
            "speech_id": speech_id, "generation": "v2.3-role-axis-s5cap",
            "published": False}},
    }))
    return tmp_path


def test_unknown_speech_id_is_a_publish_violation(tmp_path):
    """An unregistered speech disables era gating wholesale — era_lint has no
    date to compare against — so it fails CLOSED before publication."""
    violations = check_run_artifacts(_manifest_only(tmp_path, "reagan_1984"))
    assert any("reagan_1984" in v and "no utterance date" in v
               for v in violations)


def test_a_statically_pinned_speech_satisfies_the_date_check(tmp_path):
    assert check_run_artifacts(_manifest_only(tmp_path, "clinton_1998")) == []


def test_a_runner_registered_speech_also_satisfies_the_date_check(tmp_path):
    """"Statically pinned OR runner-registered" — a transcript outside the
    pinned corpus publishes only after register_speech_date() ran for it."""
    from datetime import date as _date

    from truthbot.verdict import speech_context

    root = _manifest_only(tmp_path, "carter_1979")
    assert check_run_artifacts(root)                      # unpinned → violation
    speech_context.register_speech_date("carter_1979", _date(1979, 1, 23))
    try:
        assert check_run_artifacts(root) == []
    finally:
        speech_context.SPEECH_DATE.pop("carter_1979", None)


def test_the_date_check_covers_legacy_generations_too(tmp_path):
    """The generation gate skips old runs for the S5/era/factcheck invariants;
    the speech-date check must NOT be skippable that way — an unpinned speech
    is a publish-path defect regardless of when the run was produced."""
    runs = tmp_path / "metrics" / "pca_runs"
    runs.mkdir(parents=True)
    (runs / "methodology_manifest.json").write_text(json.dumps({
        "schema": "truthbot-methodology-manifest v1",
        "current_generation": "v2.3-role-axis-s5cap",
        "generations": {"v2.3-role-axis-s5cap": "current",
                        "pre-s5-tiering": "legacy"},
        "runs": {"deadbeef-0000-0000-0000-000000000003": {
            "speech_id": "reagan_1984", "generation": "pre-s5-tiering",
            "published": True}},
    }))
    assert any("reagan_1984" in v for v in check_run_artifacts(tmp_path))


@pytest.mark.skipif(not (REPO / "metrics" / "pca_runs").is_dir(),
                    reason="metrics/pca_runs not present")
def test_every_stored_run_speech_resolves_to_a_date():
    speeches = {row["speech_id"] for row in _manifest()["runs"].values()}
    from truthbot.verdict.speech_context import speech_date_for
    for s in speeches:
        assert speech_date_for(f"{s}:0001") is not None, s
