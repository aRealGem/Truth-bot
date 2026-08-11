"""DC-6 review package (scripts/dc6_package.py) — offline, $0.

The package is what a human reads before deciding whether to publish the
Phase-3 rebuild, so the arithmetic in it has to be trustworthy in a way a
prose summary never is. These tests pin the four things that could silently
lie:

* the aggregation math (synthetic diffs with known answers);
* the corrections entries — validated against the REAL loader in
  ``truthbot.publish.corrections``, not a re-implementation of its rules;
* the badge diff's KEY — proving that keying on claim id is vacuous (ids are
  minted per render) while keying on (speaker, text) matches;
* the clean-slate ledger reset — nothing that was in the old ledger may
  disappear without being in the archive.

No model or API calls anywhere.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from truthbot.publish.corrections import SCHEMA, CorrectionsError, load_corrections, load_notes

REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "dc6_package", REPO / "scripts" / "dc6_package.py")
dc6 = importlib.util.module_from_spec(_SPEC)
sys.modules["dc6_package"] = dc6
_SPEC.loader.exec_module(dc6)      # must import clean with no key present


# ── synthetic fixtures ───────────────────────────────────────────────────────

def _diff(speech, *, per_sid, old_tally, new_tally,
          old_run="old00000", new_run="new00000"):
    """A verdict diff shaped exactly like phase3_rebuild.build_verdict_diff."""
    counts = {"unchanged": 0, "decided_to_decided_changed": 0,
              "newly_gated": 0, "newly_decided": 0, "split_changes": 0,
              "other": 0}
    for entry in per_sid:
        counts[entry["category"]] += 1
    return {
        "speech_id": speech,
        "rebuild_of": old_run,
        "new_run_id": new_run,
        "old_tally": old_tally,
        "new_tally": new_tally,
        "n_compared": len(per_sid),
        "counts": counts,
        "gate_forced_new": sum(1 for e in per_sid
                               if e["new"] == "gated-UNVERIFIABLE"),
        "per_sid": per_sid,
    }


def _entry(sid, old, new, category, text="a claim"):
    return {"sid": sid, "old": old, "new": new, "category": category,
            "text": text}


@pytest.fixture
def synthetic_diffs():
    """Two speeches, hand-countable: 5 + 4 claims, 4 + 3 changed."""
    clinton = _diff(
        "clinton_1998",
        per_sid=[
            _entry("clinton_1998:0001", "TRUE", "TRUE", "unchanged"),
            _entry("clinton_1998:0002", "TRUE", "MISLEADING",
                   "decided_to_decided_changed"),
            _entry("clinton_1998:0003", "TRUE", "gated-UNVERIFIABLE",
                   "newly_gated"),
            _entry("clinton_1998:0004", "gated-UNVERIFIABLE", "FALSE",
                   "newly_decided"),
            _entry("clinton_1998:0005", "TRUE", "Models split",
                   "split_changes"),
        ],
        old_tally={"TRUE": 4, "MISLEADING": 0, "UNVERIFIABLE": 1},
        new_tally={"TRUE": 1, "MISLEADING": 1, "FALSE": 1,
                   "UNVERIFIABLE": 1, "Models split": 1})
    trump = _diff(
        "trump_2026",
        per_sid=[
            _entry("trump_2026:0001", "FALSE", "FALSE", "unchanged"),
            _entry("trump_2026:0002", "FALSE", "TRUE",
                   "decided_to_decided_changed"),
            _entry("trump_2026:0003", "MISLEADING", "gated-UNVERIFIABLE",
                   "newly_gated"),
            _entry("trump_2026:0004", "UNVERIFIABLE", "gated-UNVERIFIABLE",
                   "newly_gated"),
        ],
        old_tally={"FALSE": 2, "MISLEADING": 1, "UNVERIFIABLE": 1},
        new_tally={"FALSE": 1, "TRUE": 1, "UNVERIFIABLE": 2})
    return [clinton, trump]


# ── aggregation math ─────────────────────────────────────────────────────────

def test_aggregate_totals_are_the_sum_of_the_per_speech_counts(synthetic_diffs):
    agg = dc6.aggregate(synthetic_diffs)
    corpus = agg["corpus"]
    assert corpus["claims"] == 9
    assert corpus["unchanged"] == 2
    assert corpus["decided_to_decided_changed"] == 2
    assert corpus["newly_gated"] == 3
    assert corpus["newly_decided"] == 1
    assert corpus["split_changes"] == 1
    assert corpus["changed_total"] == 7
    # the invariant that makes the headline honest
    assert corpus["unchanged"] + corpus["changed_total"] == corpus["claims"]


def test_aggregate_per_speech_changed_total_excludes_unchanged(synthetic_diffs):
    agg = dc6.aggregate(synthetic_diffs)
    clinton = agg["per_speech"]["clinton_1998"]
    assert clinton["claims"] == 5
    assert clinton["unchanged"] == 1
    assert clinton["changed_total"] == 4
    assert clinton["old_run_id"] == "old00000"
    assert clinton["new_run_id"] == "new00000"


def test_distribution_maps_gated_uv_onto_the_published_unverifiable_badge():
    """The site publishes one word for both; the table must not invent a
    sixth bucket the reader has never seen on the site."""
    dist = dc6._dist_from_tally({"TRUE": 3, "gated-UNVERIFIABLE": 2,
                                 "UNVERIFIABLE": 1, "Models split": 1})
    assert dist["Unverifiable"] == 3
    assert dist["True"] == 3
    assert dist["Models split"] == 1
    assert set(dist) == set(dc6.DISPLAY_ORDER)


def test_decided_rate_excludes_unverifiable_and_split():
    rate = dc6.decided_rate({"True": 6, "Mostly True": 0, "Misleading": 1,
                             "False": 1, "Unverifiable": 1, "Models split": 1})
    assert rate == {"decided": 8, "total": 10, "rate": 0.8}


def test_decided_rate_of_an_empty_distribution_is_zero_not_a_crash():
    assert dc6.decided_rate({})["rate"] == 0.0


def test_distributions_reports_old_and_new_side_by_side(synthetic_diffs):
    dist = dc6.distributions(synthetic_diffs)
    clinton = dist["per_speech"]["clinton_1998"]
    assert clinton["old_decided"] == {"decided": 4, "total": 5, "rate": 0.8}
    assert clinton["new_decided"] == {"decided": 3, "total": 5, "rate": 0.6}
    assert clinton["decided_rate_delta"] == pytest.approx(-0.2)
    assert clinton["denominator_mismatch"] is False
    corpus = dist["corpus"]
    assert corpus["old_decided"]["total"] == 9
    assert corpus["new_decided"]["total"] == 9


def test_distributions_flags_a_changed_denominator():
    """A rebuild that adjudicated fewer claims than the published run must not
    have its decided-rate compared as if the denominators matched."""
    diff = _diff("gwbush_2006",
                 per_sid=[_entry("gwbush_2006:0001", "TRUE", "TRUE",
                                 "unchanged")],
                 old_tally={"TRUE": 2},
                 new_tally={"TRUE": 1})
    dist = dc6.distributions([diff])
    assert dist["per_speech"]["gwbush_2006"]["denominator_mismatch"] is True


def test_era_parity_spread_direction(synthetic_diffs):
    parity = dc6.distributions(synthetic_diffs)["era_parity"]
    # clinton 0.8 → 0.6, trump 0.75 → 0.5: spread 0.05 → 0.10, i.e. WIDER
    assert parity["old_spread"]["spread"] == pytest.approx(0.05)
    assert parity["new_spread"]["spread"] == pytest.approx(0.10)
    assert parity["narrowed"] is False


# ── anecdote-adjusted parity (A10) ───────────────────────────────────────────
#
# An Unverifiable on a personal anecdote is the expected outcome, not a gate
# failure, so a decided-rate that counts it as one partly measures how many
# guests a speech thanked. These pin the adjustment AND its honesty: both bases
# reported, and unjoinable claims counted out loud rather than assumed.


def _run_artifact(speech, types: dict[str, str]) -> dict:
    """A run artifact carrying layer_a.claim_type for the given sids."""
    return {"meta": {"speech_id": speech},
            "rows": [{"sid": sid} for sid in types],
            "claims": [{"sid": sid, "text": f"text for {sid}",
                        "layer_a": {"claim_type": t} if t else {}}
                       for sid, t in types.items()]}


def _anecdote_diffs(tmp_path):
    """clinton: 4 claims, no anecdotes. trump: 4 claims, 2 anecdotes, and both
    anecdotes came back Unverifiable — the exact shape the adjustment is for."""
    (tmp_path / "cli00000.json").write_text(json.dumps(_run_artifact(
        "clinton_1998", {f"clinton_1998:000{i}": "statistical"
                         for i in range(1, 5)})), encoding="utf-8")
    (tmp_path / "tru00000.json").write_text(json.dumps(_run_artifact(
        "trump_2026", {"trump_2026:0001": "statistical",
                       "trump_2026:0002": "statistical",
                       "trump_2026:0003": "personal-anecdote",
                       "trump_2026:0004": "personal-anecdote"})),
        encoding="utf-8")
    clinton = _diff(
        "clinton_1998", new_run="cli00000",
        per_sid=[_entry("clinton_1998:0001", "TRUE", "TRUE", "unchanged"),
                 _entry("clinton_1998:0002", "TRUE", "TRUE", "unchanged"),
                 _entry("clinton_1998:0003", "TRUE", "TRUE", "unchanged"),
                 _entry("clinton_1998:0004", "TRUE", "UNVERIFIABLE",
                        "newly_gated")],
        old_tally={"TRUE": 4}, new_tally={"TRUE": 3, "UNVERIFIABLE": 1})
    trump = _diff(
        "trump_2026", new_run="tru00000",
        per_sid=[_entry("trump_2026:0001", "TRUE", "TRUE", "unchanged"),
                 _entry("trump_2026:0002", "TRUE", "TRUE", "unchanged"),
                 _entry("trump_2026:0003", "TRUE", "UNVERIFIABLE",
                        "newly_gated"),
                 _entry("trump_2026:0004", "TRUE", "UNVERIFIABLE",
                        "newly_gated")],
        old_tally={"TRUE": 4}, new_tally={"TRUE": 2, "UNVERIFIABLE": 2})
    return [clinton, trump]


def test_anecdote_adjustment_changes_the_denominator_not_the_numerator(tmp_path):
    ap = dc6.anecdote_parity(_anecdote_diffs(tmp_path), runs_dir=tmp_path,
                             site_root=tmp_path)
    trump = ap["per_speech"]["trump_2026"]
    assert trump["anecdotes"] == 2
    assert trump["anecdotes_abstained_new"] == 2
    # Raw: 2 of 4 decided. Adjusted: the 2 anecdotes leave the denominator,
    # and both were abstentions, so the 2 decided claims are now 2 of 2.
    assert trump["new_raw"] == {"decided": 2, "total": 4, "rate": 0.5}
    assert trump["new_adjusted"] == {"decided": 2, "total": 2, "rate": 1.0}


def test_a_speech_with_no_anecdotes_is_untouched_by_the_adjustment(tmp_path):
    ap = dc6.anecdote_parity(_anecdote_diffs(tmp_path), runs_dir=tmp_path,
                             site_root=tmp_path)
    clinton = ap["per_speech"]["clinton_1998"]
    assert clinton["anecdotes"] == 0
    assert clinton["new_raw"] == clinton["new_adjusted"]


def test_both_spreads_are_reported_and_can_disagree(tmp_path):
    """The finding A10 exists to expose: raw says the gap widened, adjusted
    says it closed. Reporting only one would have decided the question by
    choosing a denominator."""
    ap = dc6.anecdote_parity(_anecdote_diffs(tmp_path), runs_dir=tmp_path,
                             site_root=tmp_path)
    # raw new: clinton 75%, trump 50% → 25 pts apart.
    assert ap["spread"]["new_raw"]["spread"] == pytest.approx(0.25)
    # adjusted new: clinton 75%, trump 100% → 25 pts, but the other way round.
    assert ap["spread"]["new_adjusted"]["spread"] == pytest.approx(0.25)
    assert ap["spread"]["new_raw"]["min_speech"] == "trump_2026"
    assert ap["spread"]["new_adjusted"]["min_speech"] == "clinton_1998"
    assert ap["spread"]["raw_narrowed"] is False        # 0 pts → 25 pts
    assert ap["spread"]["adjusted_narrowed"] is False


def test_corpus_totals_are_the_sum_of_the_per_speech_populations(tmp_path):
    ap = dc6.anecdote_parity(_anecdote_diffs(tmp_path), runs_dir=tmp_path,
                             site_root=tmp_path)
    assert ap["corpus"]["new_raw"]["total"] == 8
    assert ap["corpus"]["new_adjusted"]["total"] == 6   # the 2 anecdotes drop
    assert ap["corpus"]["new_raw"]["decided"] == ap["corpus"][
        "new_adjusted"]["decided"] == 5


def test_missing_claim_type_joins_from_the_published_claims_json(tmp_path):
    """The artifact does not carry the provenance for one sid; the published
    claims.json does, keyed on (speaker, normalised claim text)."""
    (tmp_path / "tru00000.json").write_text(json.dumps({
        "meta": {"speech_id": "trump_2026"},
        "rows": [{"sid": "trump_2026:0001"}],
        "claims": [{"sid": "trump_2026:0001",
                    "text": "  A  guest's  story.  "}],   # no layer_a
    }), encoding="utf-8")
    site = tmp_path / "site"
    (site / "data").mkdir(parents=True)
    (site / "data" / "reports.json").write_text(json.dumps(
        [{"id": "r1", "speaker": "Donald Trump"}]), encoding="utf-8")
    (site / "data" / "claims.json").write_text(json.dumps(
        [{"report_id": "r1", "claim_text": "A guest's story.",
          "provenance": {"layer_a_claim_type": "personal-anecdote"}}]),
        encoding="utf-8")

    diff = _diff("trump_2026", new_run="tru00000",
                 per_sid=[_entry("trump_2026:0001", "TRUE", "UNVERIFIABLE",
                                 "newly_gated")],
                 old_tally={"TRUE": 1}, new_tally={"UNVERIFIABLE": 1})
    ap = dc6.anecdote_parity([diff], runs_dir=tmp_path, site_root=site)
    assert ap["join"] == {"from_artifact": 0, "from_claims_json": 1,
                          "unresolved": 0, "unresolved_sids": [],
                          "site_root": str(site), "site_index_size": 1}
    assert ap["per_speech"]["trump_2026"]["anecdotes"] == 1


def test_an_unjoinable_claim_is_reported_not_silently_non_anecdote(tmp_path):
    """Counting an unclassified claim as "not an anecdote" moves the adjusted
    rate in a known direction, so the count has to be visible."""
    (tmp_path / "tru00000.json").write_text(json.dumps({
        "meta": {"speech_id": "trump_2026"},
        "rows": [{"sid": "trump_2026:0009"}],
        "claims": [{"sid": "trump_2026:0009", "text": "Unmatchable text."}],
    }), encoding="utf-8")
    diff = _diff("trump_2026", new_run="tru00000",
                 per_sid=[_entry("trump_2026:0009", "TRUE", "TRUE",
                                 "unchanged")],
                 old_tally={"TRUE": 1}, new_tally={"TRUE": 1})
    ap = dc6.anecdote_parity([diff], runs_dir=tmp_path, site_root=tmp_path)
    assert ap["join"]["unresolved"] == 1
    assert ap["join"]["unresolved_sids"] == ["trump_2026:0009"]
    # Still counted — as a non-anecdote — but the flag above says so.
    assert ap["per_speech"]["trump_2026"]["new_adjusted"]["total"] == 1


def test_markdown_carries_both_bases_and_the_join_provenance(tmp_path):
    ap = dc6.anecdote_parity(_anecdote_diffs(tmp_path), runs_dir=tmp_path,
                             site_root=tmp_path)
    md = dc6.render_markdown({
        "aggregate": dc6.aggregate(_anecdote_diffs(tmp_path)),
        "distributions": dc6.distributions(_anecdote_diffs(tmp_path)),
        "anecdote_parity": ap, "coverage": [], "changed_claims": [],
        "spend": dc6.spend_table(_anecdote_diffs(tmp_path)),
        "flags": [], "generation": dc6.GENERATION,
        "generated": dc6.REBUILD_DATE,
        "corrections": {"changed_total": 0, "ledger_eligible": 0,
                        "not_ledger_representable": 0, "archive_path": "x.json",
                        "archived_entries": 0, "archived_notes": 0,
                        "proposed_entries": 0},
    })
    assert "### Anecdote-adjusted parity" in md
    assert "Raw spread" in md and "Anecdote-adjusted spread" in md
    assert "Provenance of the anecdote flag" in md
    assert "personal-anecdote" in md


# ── the committed regeneration ───────────────────────────────────────────────


def test_committed_review_carries_the_anecdote_parity_section():
    """A10 output is committed, not just computable — the review file a human
    reads has to contain both bases."""
    review_path = _PKG / "dc6_review.json"
    if not review_path.exists():
        pytest.skip("DC-6 package not generated in this tree")
    ap = json.loads(review_path.read_text("utf-8"))["anecdote_parity"]
    assert ap["anecdote_claim_type"] == "personal-anecdote"
    assert set(ap["per_speech"]) == set(dc6.SPEECH_ORDER)
    for basis in ("old_raw", "new_raw", "old_adjusted", "new_adjusted"):
        assert 0.0 < ap["corpus"][basis]["rate"] <= 1.0
        assert ap["spread"][basis]["spread"] >= 0.0
    md = (_PKG / "dc6_review.md").read_text("utf-8")
    assert "### Anecdote-adjusted parity" in md
    # The pre-A10 findings survive the regeneration untouched.
    assert "**Era parity**" in md
    assert "## 5. Every changed claim" in md


# ── spend-basis disclosure (DC-B1 revision 4 carry-forward) ──────────────────
#
# The corpus spend total is TWO kinds of number added together (a proxy receipt
# and a list-rate estimate), and the two resumed runs lost part of their
# off-proxy leg outright. That was known when the DC-B1 estimate excluded them
# from the per-claim rate basis — an exclusion that is only honest if the
# reason travels with the total into the DC-6' final ledger instead of being
# quietly inherited. These tests are what stops it disappearing in a future
# regeneration.

def _disclosure_says_everything(text: str) -> None:
    assert "MIXED COST BASIS" in text
    assert "LEDGER-TRUE" in text and "ESTIMATE" in text
    for speech in dc6.RESUMED_SPEECHES:
        assert speech in text
    assert "LOWER BOUND" in text
    assert "append_chunk_journal" in text     # the named mechanism, not a vibe


def test_spend_table_carries_the_mixed_basis_and_undercount_disclosure(
        synthetic_diffs):
    spend = dc6.spend_table(synthetic_diffs)
    _disclosure_says_everything(spend["basis_disclosure"])
    assert spend["cost_basis"]["disclosure"] == spend["basis_disclosure"]
    assert spend["cost_basis"]["mixed"] is True
    assert spend["cost_basis"]["offproxy_is_lower_bound"] is True
    assert spend["cost_basis"]["resumed_speeches"] == list(dc6.RESUMED_SPEECHES)


def test_rendered_spend_section_states_the_disclosure(tmp_path):
    """It has to be in the prose a human actually reads, not only in the JSON
    nobody opens."""
    md = dc6.render_markdown({
        "aggregate": dc6.aggregate(_anecdote_diffs(tmp_path)),
        "distributions": dc6.distributions(_anecdote_diffs(tmp_path)),
        "anecdote_parity": dc6.anecdote_parity(
            _anecdote_diffs(tmp_path), runs_dir=tmp_path, site_root=tmp_path),
        "coverage": [], "changed_claims": [],
        "spend": dc6.spend_table(_anecdote_diffs(tmp_path)),
        "flags": [], "generation": dc6.GENERATION,
        "generated": dc6.REBUILD_DATE,
        "corrections": {"changed_total": 0, "ledger_eligible": 0,
                        "not_ledger_representable": 0, "archive_path": "x.json",
                        "archived_entries": 0, "archived_notes": 0,
                        "proposed_entries": 0},
    })
    section = md.split("## 6. Spend + provenance", 1)[1]
    assert "**Cost basis — read this before quoting any number below.**" in section
    _disclosure_says_everything(section)


def test_committed_review_carries_the_spend_basis_disclosure():
    """And it is in the COMMITTED package, so a regeneration that drops it
    fails here rather than shipping a total that reads like a receipt."""
    review_path = _PKG / "dc6_review.json"
    if not review_path.exists():
        pytest.skip("DC-6 package not generated in this tree")
    spend = json.loads(review_path.read_text("utf-8"))["spend"]
    _disclosure_says_everything(spend["basis_disclosure"])
    _disclosure_says_everything((_PKG / "dc6_review.md").read_text("utf-8"))


# ── changed-claim listing ────────────────────────────────────────────────────

def test_changed_claims_orders_the_most_consequential_class_first(
        synthetic_diffs, tmp_path):
    changes = dc6.changed_claims(synthetic_diffs, runs_dir=tmp_path)
    assert len(changes) == 7
    classes = [c["change_class"] for c in changes]
    assert classes.index("decided_to_decided_changed") == 0
    assert classes.index("newly_gated") > classes.index(
        "decided_to_decided_changed")
    assert classes.index("newly_decided") > classes.index("newly_gated")
    assert classes.index("split_changes") > classes.index("newly_decided")
    assert "unchanged" not in classes


def test_changed_claims_truncates_text_and_carries_the_new_rationale(tmp_path):
    (tmp_path / "new00000.json").write_text(json.dumps({
        "meta": {"speech_id": "clinton_1998"},
        "rows": [{"sid": "clinton_1998:0002", "reasoning": "R" * 400}],
        "claims": [{"sid": "clinton_1998:0002", "text": "T" * 400}],
    }), encoding="utf-8")
    diff = _diff("clinton_1998",
                 per_sid=[_entry("clinton_1998:0002", "TRUE", "MISLEADING",
                                 "decided_to_decided_changed")],
                 old_tally={"TRUE": 1}, new_tally={"MISLEADING": 1})
    change = dc6.changed_claims([diff], runs_dir=tmp_path)[0]
    assert len(change["claim_text"]) <= 140
    assert len(change["rationale"]) <= 200
    assert change["rationale"].startswith("RRR")
    assert change["old_verdict"] == "True" and change["new_verdict"] == "Misleading"


# ── corrections entries ──────────────────────────────────────────────────────

def _write_ledger(path: Path, entries: list[dict], notes=None) -> Path:
    path.write_text(json.dumps({"schema": SCHEMA, "notes": notes or [],
                                "entries": entries}), encoding="utf-8")
    return path


def test_generated_entries_load_through_the_real_corrections_loader(
        synthetic_diffs, tmp_path):
    changes = dc6.changed_claims(synthetic_diffs, runs_dir=tmp_path)
    doc = dc6.correction_entries(changes)
    ledger = _write_ledger(tmp_path / "corrections.json", doc["entries"])
    loaded = load_corrections(ledger)
    assert loaded == doc["entries"]
    for entry in loaded:
        # ``claim_text`` joined the entry with the adjudication wave (S-8): an
        # entry that says only "TRUE → MISLEADING" sends the reader somewhere
        # else to find out WHAT moved. The loader passes unrecognised keys
        # through untouched, so the ledger contract is unchanged — which is
        # exactly what the round-trip above is holding.
        assert set(entry) == {"sid", "speech_id", "claim_text", "old_verdict",
                              "new_verdict", "reason", "date", "source"}
        assert entry["date"] == dc6.REBUILD_DATE
        assert len(entry["date"].split("-")) == 3      # YYYY-MM-DD like the old ones


def test_every_changed_claim_is_accounted_for_exactly_once(
        synthetic_diffs, tmp_path):
    changes = dc6.changed_claims(synthetic_diffs, runs_dir=tmp_path)
    doc = dc6.correction_entries(changes)
    assert doc["changed_total"] == len(changes)
    assert doc["ledger_eligible"] + doc["not_ledger_representable"] == len(changes)
    sids = ([e["sid"] for e in doc["entries"]]
            + [e["sid"] for e in doc["non_ledger_changes"]])
    assert sorted(sids) == sorted(c["sid"] for c in changes)
    assert len(sids) == len(set(sids))


def test_changes_the_ledger_vocabulary_cannot_express_are_set_aside_not_dropped(
        synthetic_diffs, tmp_path):
    """"Models split" has no ledger label, and panel-UV → gate-forced-UV
    publishes the same badge; both are reported, neither is invented into a
    correction the loader would reject."""
    changes = dc6.changed_claims(synthetic_diffs, runs_dir=tmp_path)
    doc = dc6.correction_entries(changes)
    excluded = {e["sid"]: e["excluded_because"] for e in doc["non_ledger_changes"]}
    assert "clinton_1998:0005" in excluded          # TRUE → Models split
    assert "vocabulary" in excluded["clinton_1998:0005"]
    assert "trump_2026:0004" in excluded            # UNVERIFIABLE → gated-UV
    assert "provenance-only" in excluded["trump_2026:0004"]
    for entry in doc["non_ledger_changes"]:
        assert entry["reason"]                      # still explained


def test_a_correction_entry_that_would_be_a_no_op_is_never_emitted(tmp_path):
    """load_corrections rejects old == new; the builder must not hand it one."""
    diff = _diff("trump_2026",
                 per_sid=[_entry("trump_2026:0009", "UNVERIFIABLE",
                                 "gated-UNVERIFIABLE", "newly_gated")],
                 old_tally={"UNVERIFIABLE": 1},
                 new_tally={"UNVERIFIABLE": 1})
    doc = dc6.correction_entries(dc6.changed_claims([diff], runs_dir=tmp_path))
    assert doc["entries"] == []
    assert doc["not_ledger_representable"] == 1
    # and prove the loader would indeed have rejected the naive entry
    bad = _write_ledger(tmp_path / "bad.json", [{
        "sid": "trump_2026:0009", "speech_id": "trump_2026",
        "old_verdict": "UNVERIFIABLE", "new_verdict": "UNVERIFIABLE",
        "reason": "r", "date": dc6.REBUILD_DATE, "source": "s"}])
    with pytest.raises(CorrectionsError):
        load_corrections(bad)


def test_reason_cites_the_dry_run_dispositions_when_they_exist(tmp_path):
    diff = _diff("obama_2014",
                 per_sid=[_entry("obama_2014:0221", "TRUE", "MISLEADING",
                                 "decided_to_decided_changed")],
                 old_tally={"TRUE": 1}, new_tally={"MISLEADING": 1})
    changes = dc6.changed_claims([diff], runs_dir=tmp_path)
    doc = dc6.correction_entries(
        changes, {"obama_2014:0221": {"era-violation": 5, "s5-capped": 2}})
    reason = doc["entries"][0]["reason"]
    assert "Phase-2 dry run" in reason                  # labelled as a projection
    assert "7 evidence item(s)" in reason
    assert "5 era-invalid" in reason and "2 over-cap political" in reason


def test_reason_falls_back_to_the_approved_generic_line(tmp_path):
    diff = _diff("obama_2014",
                 per_sid=[_entry("obama_2014:0999", "TRUE", "FALSE", "other")],
                 old_tally={"TRUE": 1}, new_tally={"FALSE": 1})
    changes = dc6.changed_claims([diff], runs_dir=tmp_path)
    reason = dc6.correction_entries(changes)["entries"][0]["reason"]
    assert reason.lower().startswith("re-adjudicated under the unified v2.3 pipeline")
    assert "corrected tier registry, era gate, and political-source cap" in reason


def test_reason_does_not_call_a_move_into_unverifiable_a_substantive_verdict(
        tmp_path):
    diff = _diff("obama_2014",
                 per_sid=[_entry("obama_2014:0218", "TRUE", "UNVERIFIABLE",
                                 "decided_to_decided_changed")],
                 old_tally={"TRUE": 1}, new_tally={"UNVERIFIABLE": 1})
    changes = dc6.changed_claims([diff], runs_dir=tmp_path)
    reason = dc6.correction_entries(changes)["entries"][0]["reason"]
    assert "panel itself returned Unverifiable" in reason
    assert "different substantive verdict" not in reason


# ── badge diff keying ────────────────────────────────────────────────────────

def _claim(cid, speaker, text, verdict):
    return {"id": cid, "speaker": speaker, "claim_text": text,
            "verdict": verdict}


def test_id_keying_would_be_vacuous_but_text_keying_matches():
    """Claim ids are minted per render (uuid4), so the two sites share none.
    An id-keyed diff would report every claim as removed-and-re-added; the
    text-keyed diff sees the corpus is the same and one verdict moved."""
    old = [_claim("old-1", "Bill Clinton", "Crime has dropped.", "True"),
           _claim("old-2", "Bill Clinton", "Welfare rolls are down.", "True")]
    new = [_claim("new-1", "Bill Clinton", "Crime has dropped.", "Misleading"),
           _claim("new-2", "Bill Clinton", "Welfare rolls are down.", "True")]

    id_matched = {c["id"] for c in old} & {c["id"] for c in new}
    assert id_matched == set()                    # the vacuity, demonstrated

    diff = dc6.badge_diff(old, new)
    assert diff["matched"] == 2
    assert diff["only_old"] == 0 and diff["only_new"] == 0
    assert diff["verdict_changes"] == 1
    assert diff["id_overlap"] == 0
    assert diff["id_keying_would_be_vacuous"] is True
    assert diff["changes"][0]["old_verdict"] == "True"
    assert diff["changes"][0]["new_verdict"] == "Misleading"


def test_badge_diff_normalises_whitespace_but_not_speaker():
    old = [_claim("a", "Bill Clinton", "Crime   has\ndropped.", "True")]
    new = [_claim("b", "Bill Clinton", "Crime has dropped.", "True")]
    assert dc6.badge_diff(old, new)["matched"] == 1
    # same text, different speaker → deliberately NOT the same claim
    other = [_claim("c", "Donald Trump", "Crime has dropped.", "True")]
    diff = dc6.badge_diff(old, other)
    assert diff["matched"] == 0
    assert diff["only_old"] == 1 and diff["only_new"] == 1


def test_badge_diff_surfaces_a_dropped_claim_rather_than_hiding_it():
    old = [_claim("a", "Donald Trump", "One.", "False"),
           _claim("b", "Donald Trump", "Two.", "True")]
    new = [_claim("c", "Donald Trump", "Two.", "True")]
    diff = dc6.badge_diff(old, new)
    assert diff["only_old"] == 1
    assert diff["only_old_claims"][0]["claim_text"] == "One."
    assert diff["verdict_changes"] == 0


def test_reconcile_accounts_for_badge_invisible_moves(synthetic_diffs, tmp_path):
    """The per-speech diffs count contract labels; the badge counts published
    words. The difference is exactly the provenance-only moves."""
    agg = dc6.aggregate(synthetic_diffs)
    entries = dc6.correction_entries(
        dc6.changed_claims(synthetic_diffs, runs_dir=tmp_path))
    # 7 changes, 1 of which (UNVERIFIABLE → gated-UV) is invisible on the badge
    badge = {"verdict_changes": 6}
    rec = dc6.reconcile(badge, agg, entries)
    assert rec["per_speech_changed_total"] == 7
    assert rec["badge_invisible_changes"] == 1
    assert rec["badge_invisible_sids"] == ["trump_2026:0004"]
    assert rec["expected_badge_changes"] == 6
    assert rec["agree"] is True


def test_reconcile_reports_a_disagreement_instead_of_absorbing_it(
        synthetic_diffs, tmp_path):
    agg = dc6.aggregate(synthetic_diffs)
    entries = dc6.correction_entries(
        dc6.changed_claims(synthetic_diffs, runs_dir=tmp_path))
    rec = dc6.reconcile({"verdict_changes": 4}, agg, entries)
    assert rec["agree"] is False
    assert rec["delta"] == -2


# ── clean-slate ledger reset ─────────────────────────────────────────────────

def test_proposed_ledger_loses_nothing_that_was_in_the_old_one():
    current = {
        "schema": SCHEMA,
        "notes": [{"date": "2026-07-21", "text": "old audit note"}],
        "entries": [
            {"sid": "biden_2022:0026", "speech_id": "biden_2022",
             "old_verdict": "FALSE", "new_verdict": "TRUE", "reason": "r1",
             "date": "2026-07-21", "source": "audit"},
            {"sid": "trump_2026:0690", "speech_id": "trump_2026",
             "old_verdict": "FALSE", "new_verdict": "UNVERIFIABLE",
             "reason": "r2", "date": "2026-07-21", "source": "audit"},
        ],
    }
    new_entries = [{"sid": "clinton_1998:0007", "speech_id": "clinton_1998",
                    "old_verdict": "TRUE", "new_verdict": "MISLEADING",
                    "reason": "r", "date": dc6.REBUILD_DATE,
                    "source": dc6.CORRECTION_SOURCE}]
    archive_name = "data/corrections-archive-2026-08-06.json"
    proposed = dc6.proposed_ledger(current, new_entries, archive_name)

    # the reset really is a reset: only the rebuild entries survive live
    assert proposed["entries"] == new_entries
    assert len(proposed["notes"]) == 1
    # …but every archived entry is still findable, and the ledger says where
    assert archive_name in proposed["notes"][0]["text"]
    archived_sids = {e["sid"] for e in current["entries"]}
    live_sids = {e["sid"] for e in proposed["entries"]}
    assert archived_sids - live_sids == archived_sids     # all moved to archive
    assert not archived_sids & live_sids                  # none duplicated live


def test_proposed_ledger_note_is_factual_about_counts():
    current = {"schema": SCHEMA, "notes": [{"date": "x", "text": "y"}],
               "entries": [{"sid": f"s:{i}", "speech_id": "s",
                            "old_verdict": "TRUE", "new_verdict": "FALSE",
                            "reason": "r", "date": "2026-07-21",
                            "source": "a"} for i in range(17)]}
    entries = [{"sid": "clinton_1998:0007", "speech_id": "clinton_1998",
                "old_verdict": "TRUE", "new_verdict": "MISLEADING",
                "reason": "r", "date": dc6.REBUILD_DATE, "source": "s"}]
    note = dc6.proposed_ledger(current, entries, "data/archive.json",
                               n_non_ledger=16)["notes"][0]
    assert note["date"] == dc6.REBUILD_DATE
    assert "17 correction entries" in note["text"]
    assert "1 claims are published" in note["text"]
    assert "16 claims changed in a way" in note["text"]
    assert dc6.GENERATION in note["text"]
    # no claim about which way any pre-2026-08-06 verdict moved
    assert "was wrong" not in note["text"].lower()


def test_proposed_ledger_round_trips_through_the_real_loader(tmp_path):
    current = {"schema": SCHEMA, "notes": [], "entries": []}
    entries = [{"sid": "clinton_1998:0007", "speech_id": "clinton_1998",
                "old_verdict": "TRUE", "new_verdict": "MISLEADING",
                "reason": "r", "date": dc6.REBUILD_DATE,
                "source": dc6.CORRECTION_SOURCE}]
    proposed = dc6.proposed_ledger(current, entries, "data/archive.json")
    path = tmp_path / "corrections.json"
    path.write_text(json.dumps(proposed), encoding="utf-8")
    assert load_corrections(path) == entries
    assert len(load_notes(path)) == 1


# ── the committed package ────────────────────────────────────────────────────

_PKG = REPO / "metrics" / "remediation_v2"


@pytest.mark.skipif(not (_PKG / "dc6_corrections_entries.json").exists(),
                    reason="DC-6 package not generated in this tree")
def test_committed_entries_are_loadable_and_complete(tmp_path):
    doc = json.loads((_PKG / "dc6_corrections_entries.json").read_text("utf-8"))
    ledger = _write_ledger(tmp_path / "c.json", doc["entries"])
    assert len(load_corrections(ledger)) == doc["ledger_eligible"]
    assert doc["ledger_eligible"] + doc["not_ledger_representable"] == \
        doc["changed_total"]
    review = json.loads((_PKG / "dc6_review.json").read_text("utf-8"))
    assert doc["changed_total"] == review["aggregate"]["corpus"]["changed_total"]


@pytest.mark.skipif(not (_PKG / "dc6_net_ledger.json").exists(),
                    reason="DC-6' net ledger not generated in this tree")
def test_committed_net_ledger_supersedes_and_archives_the_prior_ledger():
    """F6 supersede: the live ledger is exactly the DC-6' net ledger's eligible
    set, the prior ledger is archived (never deleted), and the framing prose is
    a draft-flagged note that cannot render as final."""
    net = json.loads((_PKG / "dc6_net_ledger.json").read_text("utf-8"))
    archive = REPO / "data" / "corrections-archive-2026-08-10.json"
    live = REPO / "data" / "corrections.json"
    doc = json.loads(live.read_text("utf-8"))
    assert archive.exists(), "the clean-slate reset must archive, never delete"
    assert load_corrections(archive)                      # archived copy loads clean
    current = load_corrections(live)
    # the changelog is exactly the net ledger's ledger-eligible entries.
    assert {e["sid"] for e in current} == {e["sid"] for e in net["entries"]}
    assert net["completeness_ok"] and not net["head_mismatches"]
    # F9: the resolution-state section is exactly the net-visible non-ledger set.
    assert ({e["sid"] for e in doc.get("resolution_state_changes", [])}
            == {e["sid"] for e in net["non_ledger_changes"]
                if not e.get("net_unchanged")})
    # F11: NOTHING ccagent-authored renders as final — EVERY note is draft.
    notes = doc["notes"]
    assert notes and all(n.get("draft") for n in notes)


@pytest.mark.skipif(not (_PKG / "dc6_review.json").exists(),
                    reason="DC-6 package not generated in this tree")
def test_committed_review_reconciles_badge_diff_with_the_per_speech_diffs():
    review = json.loads((_PKG / "dc6_review.json").read_text("utf-8"))
    if "reconciliation" not in review:
        pytest.skip("review generated without a staged render")
    assert review["reconciliation"]["agree"] is True
    assert review["badge_diff"]["id_keying_would_be_vacuous"] is True


@pytest.mark.skipif(not (_PKG / "dc6_review.json").exists(),
                    reason="DC-6 package not generated in this tree")
def test_committed_review_never_reports_a_published_site_render():
    """The DC-6 render is staged. If this ever points at site-pca/, the
    package stopped being a review and became a publish."""
    review = json.loads((_PKG / "dc6_review.json").read_text("utf-8"))
    root = (review.get("render") or {}).get("site_root", "")
    assert not root or not root.rstrip("/").endswith("site-pca")
