"""The $0 score-propagation merge: sidecar scores → published-head artifacts.

B1a and B2 bought stance and relevance for all 4,344 stored evidence items, but
parked them in sidecars; the adjudication wave overlaid them at runtime, so the
verdicts were reached on scored evidence while the artifacts that would be
PUBLISHED still carried the unscored originals. This module tests the merge that
closes that gap.

What is worth testing here is not that a score lands — it is everything that
would make a merge look successful while being wrong: a row that quietly found
no home, an item nobody covered being counted as covered, a verdict moving, or
the parent artifact being edited instead of a child being written.

No model or API calls, and nothing here writes into metrics/pca_runs.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO / "metrics" / "pca_runs"


@pytest.fixture(scope="module")
def pr():
    spec = importlib.util.spec_from_file_location(
        "propagate_rescores", REPO / "scripts" / "propagate_rescores.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ev(url: str, **kw) -> dict:
    d = {"claim_id": "s:1", "source_name": "Src", "source_url": url,
         "source_tier": "Government", "snippet": "…",
         "retrieved_at": "2026-08-01T00:00:00"}
    d.update(kw)
    return d


def _row(url: str, stance, rel: float, **kw) -> dict:
    d = {"source_url": url, "supports_claim": stance, "relevance_score": rel}
    d.update(kw)
    return d


# ── 1. the merge lands, and lands the RIGHT vintage ──────────────────────────

def test_scores_land_on_the_matching_item(pr):
    evidence = {"s:1": [_ev("https://a.gov/x"), _ev("https://b.org/y")]}
    scored, tel = pr.propagate_evidence(evidence, {"s:1": [
        _row("https://a.gov/x", True, 0.9),
        _row("https://b.org/y", False, 0.7, one_line_why="claim says 3.2M; table shows 2.9M"),
    ]})
    a, b = scored["s:1"]
    assert (a["supports_claim"], a["relevance_score"]) == (True, 0.9)
    assert (b["supports_claim"], b["relevance_score"]) == (False, 0.7)
    assert b["one_line_why"].startswith("claim says")
    assert tel["matched"] == tel["items"] == 2


def test_the_join_normalizes_the_url_the_way_consolidate_dedups(pr):
    """Trailing slash and case are not a miss — the join key is deliberately the
    same normalization ``consolidate`` dedups on, so a pack can never hold two
    items that collide here."""
    evidence = {"s:1": [_ev("https://A.GOV/x/")]}
    scored, tel = pr.propagate_evidence(
        evidence, {"s:1": [_row("https://a.gov/x", True, 0.9)]})
    assert scored["s:1"][0]["supports_claim"] is True
    assert tel["sidecar_unmatched"] == [] and tel["artifact_unscored"] == []


def test_b2_overrides_b1a_per_sid_not_per_item(pr):
    """The merge order the wave used. B2 re-scored a targeted subset into its
    own file; a sid B2 touched takes B2's rows ENTIRELY, a sid it did not keeps
    B1a's — merging per item would let a B1a row survive inside a pack B2
    rewrote, which is a vintage the panel never saw."""
    from regate_from_rescore import merge_sidecars

    b1a = {"speech_id": "sp", "sids": {"s:1": [_row("https://a.gov/x", True, 0.9)],
                                       "s:2": [_row("https://c.gov/z", True, 0.8)]},
           "spend_usd": 0.4, "pass_label": "b1a"}
    b2 = {"speech_id": "sp", "sids": {"s:1": [_row("https://a.gov/x", False, 0.2)]},
          "spend_usd": 0.1, "pass_label": "b2"}
    merged = merge_sidecars(b1a, b2)
    assert merged["sids"]["s:1"][0]["supports_claim"] is False   # B2 wins
    assert merged["sids"]["s:2"][0]["supports_claim"] is True    # B1a survives
    assert merged["spend_usd"] == pytest.approx(0.5)             # both were paid

    scored, _ = pr.propagate_evidence(
        {"s:1": [_ev("https://a.gov/x")], "s:2": [_ev("https://c.gov/z")]},
        merged["sids"])
    assert scored["s:1"][0]["relevance_score"] == 0.2
    assert scored["s:2"][0]["relevance_score"] == 0.8


# ── 2. nothing is dropped silently, in EITHER direction ──────────────────────

def test_a_sidecar_row_with_no_home_is_reported_not_swallowed(pr):
    _, tel = pr.propagate_evidence(
        {"s:1": [_ev("https://a.gov/x")]},
        {"s:1": [_row("https://a.gov/x", True, 0.9),
                 _row("https://gone.gov/q", True, 0.9)]})
    assert [(r["sid"], r["why"]) for r in tel["sidecar_unmatched"]] == \
        [("s:1", "url not in pack")]
    assert tel["sidecar_unmatched"][0]["source_url"] == "https://gone.gov/q"


def test_a_whole_sid_the_head_no_longer_carries_is_reported(pr):
    _, tel = pr.propagate_evidence(
        {"s:1": [_ev("https://a.gov/x")]},
        {"s:1": [_row("https://a.gov/x", True, 0.9)],
         "s:9": [_row("https://d.gov/w", True, 0.9)]})
    assert tel["sids_missing_from_head"] == ["s:9"]
    assert [r["why"] for r in tel["sidecar_unmatched"]] == ["sid not in head"]


def test_an_item_no_sidecar_covers_keeps_its_values_and_is_counted(pr):
    """Evidence the wave and escape runs added AFTER the re-score. Inventing a
    score for it would be worse than counting it, so it keeps what it had."""
    evidence = {"s:1": [_ev("https://a.gov/x"),
                        _ev("https://new.gov/late", supports_claim=True,
                            relevance_score=0.5)]}
    scored, tel = pr.propagate_evidence(
        evidence, {"s:1": [_row("https://a.gov/x", False, 0.9)]})
    late = scored["s:1"][1]
    assert (late["supports_claim"], late["relevance_score"]) == (True, 0.5)
    assert [(r["sid"], r["why"]) for r in tel["artifact_unscored"]] == \
        [("s:1", "item not in sidecar")]
    assert tel["matched"] == 1 and tel["items"] == 2


def test_a_pack_the_rescore_never_reached_is_reported_whole(pr):
    _, tel = pr.propagate_evidence(
        {"s:1": [_ev("https://a.gov/x")], "s:7": [_ev("https://e.gov/v")]},
        {"s:1": [_row("https://a.gov/x", True, 0.9)]})
    assert tel["sids_missing_from_sidecar"] == ["s:7"]
    assert [r["why"] for r in tel["artifact_unscored"]] == ["sid never re-scored"]


def test_the_source_evidence_map_is_never_mutated(pr):
    """The head is the parent of what we write. Overlaying into it in place
    would edit a prior artifact in memory and, one ``write`` later, on disk."""
    evidence = {"s:1": [_ev("https://a.gov/x")]}
    before = copy.deepcopy(evidence)
    pr.propagate_evidence(evidence, {"s:1": [_row("https://a.gov/x", True, 0.9)]})
    assert evidence == before


# ── 3. verdict invariance ────────────────────────────────────────────────────

def _head(**meta) -> dict:
    m = {"speech_id": "gwbush_2006", "speaker": "G", "date": "2006-01-31",
         "pipeline_generation": "v2.3-role-axis-s5cap"}
    m.update(meta)
    return {"run_id": "parent-run", "meta": m,
            "claims": [{"sid": "s:1", "text": "c"}],
            "rows": [{"sid": "s:1", "verdict": "TRUE", "reasoning": "r"},
                     {"sid": "s:2", "verdict": "UNVERIFIABLE", "reasoning": "r"}],
            "characterization": [], "roster": {},
            "evidence": {"s:1": [_ev("https://a.gov/x")]}}


def test_the_new_artifact_carries_the_same_sids_and_the_same_verdicts(pr):
    head = _head()
    scored, tel = pr.propagate_evidence(head["evidence"],
                                        {"s:1": [_row("https://a.gov/x", True, 0.9)]})
    new = pr.build_artifact(head, scored, speech="gwbush_2006", sidecars=[],
                            telemetry=tel)
    assert pr.verdict_map(new) == pr.verdict_map(head)
    assert pr.verdict_map(new) == {"s:1": "TRUE", "s:2": "UNVERIFIABLE"}
    assert new["rows"] == head["rows"] and new["claims"] == head["claims"]


def test_a_merge_that_moved_a_verdict_refuses_to_write(pr, tmp_path, monkeypatch):
    """The assertion has teeth: propagate_speech raises BEFORE anything reaches
    disk if the verdict map moved."""
    head = _head()
    path = tmp_path / "head.json"
    path.write_text(json.dumps(head), encoding="utf-8")

    real_build = pr.build_artifact

    def _sabotage(h, ev, **kw):
        out = real_build(h, ev, **kw)
        out["rows"] = [dict(r, verdict="FALSE") for r in out["rows"]]
        return out

    monkeypatch.setattr(pr, "merged_sidecar", lambda s, **k: {"sids": {}})
    monkeypatch.setattr(pr, "sidecar_vintages", lambda s, **k: [])
    monkeypatch.setattr(pr, "build_artifact", _sabotage)
    with pytest.raises(SystemExit) as exc:
        pr.propagate_speech("gwbush_2006", path)
    assert "VERDICTS MOVED" in str(exc.value)
    assert list(tmp_path.glob("*.json")) == [path]


# ── 4. lineage, generation and the meta note ─────────────────────────────────

def test_lineage_points_at_the_head_and_the_generation_does_not_move(pr):
    head = _head()
    new = pr.build_artifact(head, head["evidence"], speech="gwbush_2006",
                            sidecars=[], telemetry={"packs": 1, "items": 1,
                                                    "matched": 1,
                                                    "sidecar_unmatched": [],
                                                    "artifact_unscored": []})
    assert new["meta"]["rebuild_of"] == "parent-run"
    assert new["run_id"] != head["run_id"]
    # Generation UNCHANGED: this merge attaches scores bought for these very
    # items; it does not change how a pack was built, retrieved or gated.
    assert new["meta"]["pipeline_generation"] == head["meta"]["pipeline_generation"]
    assert new["meta"]["cost_usd"] == 0.0


def test_the_meta_note_names_the_sidecars_and_their_vintages(pr):
    sidecars = [{"pass": "b1a", "path": "metrics/remediation_v2/rescored_x.json",
                 "model": "claude-haiku", "generated": "2026-08-08T11:32:00",
                 "scored_against_run": "parent-run", "sids": 90,
                 "spend_usd": 0.44}]
    new = pr.build_artifact(_head(), {}, speech="gwbush_2006", sidecars=sidecars,
                            telemetry={"packs": 0, "items": 0, "matched": 0,
                                       "sidecar_unmatched": [],
                                       "artifact_unscored": []})
    note = new["meta"]["score_propagation"]
    assert note["sidecars"] == sidecars
    assert "B1a first, B2 overriding per SID" in note["merge_order"]
    assert "source_url" in note["join"]
    assert "never re-retrieved" in note["retrieval"]
    assert note["verdicts"].startswith("unchanged")


def test_manifest_update_is_purely_additive(pr, tmp_path):
    from phase3_rebuild import update_manifest

    p = tmp_path / "methodology_manifest.json"
    before = {"schema": "truthbot-methodology-manifest v1",
              "current_generation": "v2.3-role-axis-s5cap",
              "runs": {"old-run": {"speech_id": "gwbush_2006",
                                   "generation": "v2.3-role-axis-s5cap",
                                   "published": True}}}
    p.write_text(json.dumps(before), encoding="utf-8")
    update_manifest("new-run", "gwbush_2006", manifest_path=p)
    after = json.loads(p.read_text(encoding="utf-8"))
    assert after["runs"]["old-run"] == before["runs"]["old-run"]
    assert after["runs"]["new-run"] == {"speech_id": "gwbush_2006",
                                        "generation": "v2.3-role-axis-s5cap",
                                        "published": False}


# ── 5. the shipped tree ──────────────────────────────────────────────────────

SPEECHES = ("gwbush_2006", "clinton_1998", "obama_2014", "biden_2022",
            "trump_2026")


@pytest.fixture(scope="module")
def heads(pr):
    """``pr`` first: loading it is what puts ``scripts/`` on ``sys.path``."""
    from rerender_pca_site import publishing_heads
    return publishing_heads(RUNS_DIR)


def test_every_publishing_head_is_a_score_propagation_artifact(pr, heads):
    for speech in SPEECHES:
        path = heads.get(speech)
        if path is None:
            pytest.skip(f"{speech}: no artifact in this checkout")
        meta = json.loads(path.read_text(encoding="utf-8"))["meta"]
        assert meta.get("score_propagation"), (
            f"{speech}: the head a publish would render does not carry the "
            "merged scores — its evidence is not what was adjudicated")
        assert meta["score_propagation"]["sidecars"], f"{speech}: no vintages"


def test_the_head_scores_its_evidence_and_its_parent_still_does_not(pr, heads):
    """The two halves of "prior artifacts are untouched": the child carries the
    scores, and the parent it derives from is byte-for-byte the unscored
    artifact it always was. If the merge had edited in place, the parent's rate
    would have moved with the child's."""
    from truthbot.verdict.consolidator import scoring_telemetry_from_artifact

    for speech in SPEECHES:
        path = heads.get(speech)
        if path is None:
            pytest.skip(f"{speech}: no artifact in this checkout")
        child = json.loads(path.read_text(encoding="utf-8"))
        parent_path = RUNS_DIR / f"{child['meta']['rebuild_of']}.json"
        if not parent_path.exists():
            continue
        parent = json.loads(parent_path.read_text(encoding="utf-8"))

        ct = scoring_telemetry_from_artifact(child["evidence"])
        pt = scoring_telemetry_from_artifact(parent["evidence"])
        assert ct["items"] == pt["items"], f"{speech}: item count moved"
        assert ct["scored_rate"] > 0.9, f"{speech}: child barely scored"
        assert pt["scored_rate"] < 0.1, (
            f"{speech}: the PARENT is scored too — the merge mutated a prior "
            "artifact instead of writing a new one")
        assert ct["stance_null_rate"] < pt["stance_null_rate"]
        # …and the merge moved provenance only.
        assert pr.verdict_map(child) == pr.verdict_map(parent), \
            f"{speech}: a verdict moved between parent and child"


def test_four_of_the_five_heads_now_pass_the_publish_gate(heads):
    """The measured state, asserted rather than described. trump_2026 is the
    one speech the honest merge does NOT rescue: 309 of its 1,472 items came
    back stance-null from the scorer itself, 21.0% against a 15% ceiling. The
    ceiling is not moved to accommodate it — is_fit_to_gate still returns False,
    and this test fails if someone quietly lowers the bar.

    F13: trump publishes anyway, but through an owner-ratified, registry-keyed
    exception (D-B) that DISCLOSES it is over the line — not by lowering the
    ceiling and not via a CLI bypass. The exception is data; every other speech
    keeps hard enforcement, so a second unfit speech would still refuse.
    """
    from truthbot.publish.consistency import (
        STANCE_NULL_GATE_EXCEPTIONS, check_publish_gate, is_fit_to_gate,
        publish_gate_notice)

    fit = {}
    for speech in SPEECHES:
        path = heads.get(speech)
        if path is None:
            pytest.skip(f"{speech}: no artifact in this checkout")
        doc = json.loads(path.read_text(encoding="utf-8"))
        fit[speech] = is_fit_to_gate(doc)[0]
        excepted = speech in STANCE_NULL_GATE_EXCEPTIONS
        # publishable = fit OR excepted; the notice appears iff excepted-unfit.
        assert (check_publish_gate(doc) == []) is (fit[speech] or excepted)
        assert bool(publish_gate_notice(doc)) is (excepted and not fit[speech])

    # The ceiling is untouched: trump is still measured UNFIT.
    assert fit == {"gwbush_2006": True, "clinton_1998": True,
                   "obama_2014": True, "biden_2022": True,
                   "trump_2026": False}
    # …and it is the ONLY speech carrying an exception.
    assert set(STANCE_NULL_GATE_EXCEPTIONS) == {"trump_2026"}


def test_default_invocation_renders_end_to_end(tmp_path, monkeypatch):
    """F12: the DEFAULT invocation (no flags — ``--corrections skip`` and NO
    unfit bypass) renders the whole corpus end-to-end on a fresh clone. This is
    the regression that the publish PATH works, not just its parts: the ledger
    completeness gate passes, trump publishes under the keyed D-B exception, the
    strips are annotated, and the resolution-state section reaches the page.
    """
    import importlib.util

    if not any((RUNS_DIR).glob("*.json")):
        pytest.skip("no pca_runs artifacts in this checkout")
    spec = importlib.util.spec_from_file_location(
        "rerender_pca_site", REPO / "scripts" / "rerender_pca_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    site = tmp_path / "site"
    monkeypatch.setattr(
        "sys.argv", ["rerender_pca_site.py", "--site-root", str(site)])
    mod.main()  # default flags; raises SystemExit on any gate failure

    corr = (site / "corrections.html").read_text(encoding="utf-8")
    assert "Resolution-state changes" in corr           # F9 section
    assert corr.count("vt-split") == 12                  # the 12 net-visible
    assert list(site.glob("reports/*donald-trump*.html"))  # trump published

    # F14 count gate: EVERY stamped correction note reaches the page — including
    # the gated-UNVERIFIABLE claims whose provenance strip is empty. The number
    # rendered on the claim permalinks (one page per claim, no aliases) must equal
    # the number stamped (entries + resolution-state changes); a template that
    # drops the note fails HERE, in CI, not in review.
    ledger = json.loads((REPO / "data" / "corrections.json").read_text("utf-8"))
    stamped = len(ledger["entries"]) + len(ledger.get("resolution_state_changes", []))
    rendered = sum(p.read_text(encoding="utf-8").count('class="pca-correction"')
                   for p in (site / "claims").glob("*.html"))
    assert rendered == stamped, (
        f"{rendered} correction notes rendered on claim pages but {stamped} were "
        "stamped — a template is dropping the note")


def test_a_second_unfit_speech_still_refuses_despite_the_trump_exception():
    """F13 guard: the D-B exception is keyed to trump_2026 alone. A different
    speech at the same unfitness gets no free pass — the gate refuses it and
    emits no exception notice."""
    from truthbot.publish.consistency import check_publish_gate, publish_gate_notice

    unfit = {"meta": {"speech_id": "gwbush_2006"},
             "evidence": {"gwbush_2006:9001": [
                 {"source_url": "https://a.gov/x", "relevance_score": 0.7,
                  "supports_claim": None} for _ in range(20)]}}
    assert check_publish_gate(unfit) != []          # refuses
    assert publish_gate_notice(unfit) == ""         # no exception notice


# ── 6. the head is data, not a timestamp (F4) ────────────────────────────────

def test_head_resolution_is_deterministic_not_mtime(tmp_path):
    """The head is the leaf of the ``rebuild_of`` DAG at the current generation,
    resolved from data that travels with the repo — NOT the newest mtime. On a
    fresh clone every file carries the checkout time, so the old mtime rule was
    undefined and this suite was non-deterministic. This builds a tiny run tree,
    stamps the ROOT as newest (what would have won the mtime rule) and the real
    head as oldest, and asserts the leaf is chosen anyway — then repeats with the
    uniform mtimes a clone actually produces.
    """
    import os

    from truthbot.publish.heads import publishing_heads

    def _write(rid, rebuild_of, gen="genX", mtime=None):
        doc = {"meta": {"speech_id": "s1", "rebuild_of": rebuild_of,
                        "generation": gen}, "evidence": {}, "rows": [],
               "claims": []}
        p = tmp_path / f"{rid}.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        if mtime is not None:
            os.utime(p, (mtime, mtime))
        return p

    _write("root", None, mtime=3000)          # newest — would win "newest wins"
    _write("mid", "root", mtime=2000)
    leaf = _write("leaf", "mid", mtime=1000)   # oldest — but the real head
    # A superseded, off-generation root of the same speech must not resurface.
    _write("stale", None, gen="old-gen", mtime=9000)
    (tmp_path / "methodology_manifest.json").write_text(json.dumps({
        "current_generation": "genX",
        "runs": {"root": {"generation": "genX"}, "mid": {"generation": "genX"},
                 "leaf": {"generation": "genX"},
                 "stale": {"generation": "old-gen"}},
    }), encoding="utf-8")

    assert publishing_heads(tmp_path) == {"s1": leaf}

    for p in tmp_path.glob("*.json"):          # the fresh-clone condition
        os.utime(p, (5000, 5000))
    assert publishing_heads(tmp_path) == {"s1": leaf}


def test_a_forked_lineage_is_a_build_fault_not_a_coin_flip(tmp_path):
    """Two leaves for one speech at the current generation is ambiguous, and the
    resolver refuses rather than picking one — the failure the mtime rule hid by
    silently choosing whichever file was touched last."""
    from truthbot.publish.heads import publishing_heads

    for rid, ro in (("root", None), ("a", "root"), ("b", "root")):
        (tmp_path / f"{rid}.json").write_text(json.dumps(
            {"meta": {"speech_id": "s1", "rebuild_of": ro, "generation": "genX"},
             "evidence": {}, "rows": [], "claims": []}), encoding="utf-8")
    (tmp_path / "methodology_manifest.json").write_text(json.dumps({
        "current_generation": "genX",
        "runs": {"root": {"generation": "genX"}, "a": {"generation": "genX"},
                 "b": {"generation": "genX"}},
    }), encoding="utf-8")

    with pytest.raises(SystemExit, match="exactly one head"):
        publishing_heads(tmp_path)
