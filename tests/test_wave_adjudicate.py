"""The adjudication wave runner (scripts/wave_adjudicate.py) — offline, $0.

Nothing here touches a model, a proxy or the network. The wave's whole value
proposition is that it re-uses evidence already on disk, so the parts worth
holding are the ones a reviewer has to trust BEFORE any money moves:

  * the claim set is DERIVED (released ∪ named extras ∪ split extras, minus
    whatever the ratified rules now gate) rather than retyped — the 2026-08-09
    repricing exists because a hand-copied set was wrong once already;
  * the computed exhibit's admissibility rule is enforced at the wave's write
    point AND at the bridge, and a C-EVAL claim is refused at both;
  * a wave artifact replaces exactly the rows it adjudicated and no others.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
_SPEC = importlib.util.spec_from_file_location(
    "wave_adjudicate", REPO / "scripts" / "wave_adjudicate.py")
wa = importlib.util.module_from_spec(_SPEC)
sys.modules["wave_adjudicate"] = wa
_SPEC.loader.exec_module(wa)          # must import clean with no key present

from truthbot.publish import computed_exhibit as ce  # noqa: E402


# ── the claim set ────────────────────────────────────────────────────────────

def test_wave_set_is_released_plus_extras_minus_the_newly_gated():
    """The set is assembled, never retyped. A named extra the ratified rules
    now GATE is answered deterministically and for free, so it must fall out —
    that is the T-1 correction, and paying a panel for it would buy nothing."""
    flipset = {"released_sids": ["a:1", "a:2"],
               "newly_gated_sids": ["b:9"]}
    wave = wa.wave_set(flipset, named_extras=("b:9", "b:7"),
                       split_extras=("c:3",))
    assert wave["sids"] == ["a:1", "a:2", "b:7", "c:3"]
    assert wave["reason"]["a:1"] == "released"
    assert wave["reason"]["b:7"] == "named-extra"
    assert wave["reason"]["c:3"] == "models-split extra"
    assert "b:9" in wave["dropped"]


def test_wave_set_counts_an_extra_that_is_also_released_only_once():
    """Double-counting a released claim that is also a named extra would
    inflate both the set and the bill."""
    wave = wa.wave_set({"released_sids": ["a:1"], "newly_gated_sids": []},
                       named_extras=("a:1",), split_extras=())
    assert wave["sids"] == ["a:1"]
    assert wave["reason"]["a:1"] == "released"     # gate release wins


def test_wave_set_groups_by_speech():
    wave = wa.wave_set({"released_sids": ["a:2", "b:1", "a:1"],
                        "newly_gated_sids": []},
                       named_extras=(), split_extras=())
    assert wave["by_speech"] == {"a": ["a:1", "a:2"], "b": ["b:1"]}


# ── the computed exhibit ─────────────────────────────────────────────────────

EXHIBIT = {
    "series": "CPILFESL", "source": "ALFRED", "vintage_date": "2026-02-24",
    "inputs": {"2025-09-01": 330.418, "2025-12-01": 331.814},
    "formula": "(Dec/Sep)^4 - 1", "result": 0.01701,
}


def test_exhibit_is_refused_on_a_c_eval_claim():
    """The load-bearing constraint: arithmetic cannot settle an evaluative
    claim, so a C-EVAL shape gets no exhibit — and the refusal is RETURNED as
    a reason, never swallowed, so it reaches the run report."""
    ex, why = wa.exhibit_for("trump_2026:0031", EXHIBIT, "c-eval")
    assert ex is None
    assert "INADMISSIBLE" in why and "c-eval" in why


def test_exhibit_is_offered_only_to_the_claims_it_was_built_for():
    ex, why = wa.exhibit_for("trump_2026:0191", EXHIBIT, "c-count")
    assert ex is None and why == ""


def test_exhibit_attaches_on_an_admissible_shape():
    ex, why = wa.exhibit_for("trump_2026:0031", EXHIBIT, "c-count")
    assert ex is not None and why == ""
    assert ex["vintage_date"] == "2026-02-24"


def test_exhibit_context_shows_formula_inputs_and_vintage():
    """R-2 requires all three visible. The PANEL text is checked for the same
    three the page is, because a panel that cannot see the vintage is being
    asked to trust the number."""
    text = wa.exhibit_context(EXHIBIT)
    assert "(Dec/Sep)^4 - 1" in text
    assert "330.418" in text and "331.814" in text
    assert "2026-02-24" in text
    assert "1.701%" in text
    # It must present itself as evidence about the number, not as a verdict.
    assert "not a verdict" in text


# ── artifact assembly ────────────────────────────────────────────────────────

SOURCE_ART = {
    "run_id": "old-run",
    "meta": {"speaker": "X", "date": "2026-02-24", "speech_id": "trump_2026"},
    "claims": [{"sid": "s:1", "text": "one"}, {"sid": "s:2", "text": "two"}],
    "rows": [{"sid": "s:1", "verdict": "TRUE"},
             {"sid": "s:2", "verdict": "UNVERIFIABLE"}],
    "evidence": {"s:1": [], "s:2": []},
}


def test_merge_wave_rows_replaces_only_the_adjudicated_sids():
    """An artifact that restated rows the wave never looked at would be
    claiming work that was not done."""
    merged = wa.merge_wave_rows(SOURCE_ART, [{"sid": "s:2", "verdict": "FALSE"}])
    assert [r["sid"] for r in merged] == ["s:1", "s:2"]     # order preserved
    assert merged[0] == {"sid": "s:1", "verdict": "TRUE"}   # untouched
    assert merged[1]["verdict"] == "FALSE"


def test_write_wave_artifact_keeps_lineage_and_never_touches_the_source(tmp_path):
    path, payload = wa.write_wave_artifact(
        SOURCE_ART, wa.merge_wave_rows(SOURCE_ART, []), {},
        {"name": "prod", "seats": {}}, speech_id="trump_2026",
        wave_sids=["s:2"], reasons={"s:2": "released"},
        deferred_gated=["s:9"], rules={"utterance_record": True,
                                       "statistical_release": True},
        exhibits={}, out_dir=tmp_path, cost_usd=1.25)
    assert path.parent == tmp_path
    meta = payload["meta"]
    assert meta["rebuild_of"] == "old-run"
    assert meta["pipeline_generation"] == wa.PIPELINE_GENERATION
    assert meta["wave"]["sids_adjudicated"] == ["s:2"]
    assert meta["wave"]["retrieval"].startswith("none")
    # The claims the ratified rules gate OUTSIDE the wave are recorded, not
    # applied — silence there is how a deferred decision becomes a lost one.
    assert meta["wave"]["deferred_newly_gated"] == ["s:9"]
    assert payload["run_id"] != SOURCE_ART["run_id"]
    assert SOURCE_ART["rows"][0]["verdict"] == "TRUE"       # source untouched


def test_go_refusal_requires_a_budget():
    assert wa.go_refusal(None) and "budget" in wa.go_refusal(None).lower()
    assert wa.go_refusal(0) is not None
    assert wa.go_refusal(3.28) is None


# ── the audited escape (--extra-sids) ────────────────────────────────────────

def test_plain_sids_refuses_a_sid_outside_the_claim_set():
    """``--sids`` SLICES; it never extends. Silently intersecting — which is
    what it used to do — turns a typo, or a claim the flip set never released,
    into a quietly smaller run with no complaint."""
    msg = wa.sids_refusal(["a:1", "a:2"], ["a:1", "b:9"])
    assert msg and "b:9" in msg
    assert "--extra-sids" in msg          # it names the audited way in
    assert wa.sids_refusal(["a:1", "a:2"], ["a:1"]) is None


def test_escape_requires_a_reason():
    """An unexplained escape is indistinguishable from a widened wave."""
    msg = wa.escape_refusal(["a:9"], "   ", "r3")
    assert msg and "--reason" in msg
    assert wa.escape_refusal(["a:9"], "publish-blocking blank rationale",
                             "r3") is None


def test_escape_refuses_to_run_under_the_waves_own_tag():
    """The escape writes a report, a diff and a journal. Under the wave's tag
    those overwrite the wave's — prior artifacts are never mutated."""
    msg = wa.escape_refusal(["a:9"], "because", wa.DEFAULT_TAG)
    assert msg and "--tag" in msg and "OVERWRITE" in msg


def test_escape_refuses_a_malformed_sid_before_any_spend():
    msg = wa.escape_refusal(["not-a-sid"], "because", "r3")
    assert msg and "malformed" in msg


def test_reason_without_extra_sids_is_refused():
    assert wa.escape_refusal(None, "why", "wave") is not None
    assert wa.escape_refusal(None, "", "wave") is None


def test_escape_set_is_exactly_the_named_sids_and_does_not_widen_the_wave():
    """The escape is its OWN run over its OWN claims. ``wave_set`` keeps
    returning what it returned before — that is what "must not silently widen
    the wave set" means in code, asserted rather than promised."""
    flipset = {"released_sids": ["a:1", "a:2"], "newly_gated_sids": []}
    before = wa.wave_set(flipset, named_extras=(), split_extras=())["sids"]
    esc = wa.escape_set(["b:9", "b:9", "c:1"], "publish-blocking defect")
    after = wa.wave_set(flipset, named_extras=(), split_extras=())["sids"]

    assert esc["sids"] == ["b:9", "c:1"]              # de-duplicated
    assert esc["by_speech"] == {"b": ["b:9"], "c": ["c:1"]}
    assert all(r.startswith(wa.ESCAPE_REASON) and "publish-blocking defect" in r
               for r in esc["reason"].values())
    assert before == after == ["a:1", "a:2"]
    assert not set(esc["sids"]) & set(after)


def test_escape_provenance_records_the_reason_the_sids_and_the_untouched_wave():
    prov = wa.escape_provenance(["b:9"], "  blank rationale  ", "r3",
                                ["a:1", "a:2"])
    assert prov["reason"] == "blank rationale"
    assert prov["sids"] == ["b:9"]
    assert prov["tag"] == "r3"
    assert prov["wave_set_widened"] is False
    assert prov["wave_set_size"] == 2


def test_escape_output_paths_never_collide_with_the_waves():
    """Same guarantee as the tag refusal, checked at the paths themselves."""
    for speech in ("biden_2022", "trump_2026"):
        assert wa.diff_path(speech, "r3") != wa.diff_path(speech)
        assert wa.journal_path(speech, "r3") != wa.journal_path(speech)
    assert wa.report_path("r3") != wa.report_path()
    # and the default tag still resolves to the wave's own filenames
    assert wa.report_path().name == "wave_report.json"
    assert wa.diff_path("biden_2022").name == "wave_biden_2022_verdict_diff.json"
    assert wa.journal_path("biden_2022").name == "biden_2022_wave.jsonl"


def test_an_escaped_sid_is_admitted_by_the_sids_slice():
    """The two guards compose: what --extra-sids admits, --sids may slice."""
    esc = wa.escape_set(["b:9", "c:1"], "because")
    assert wa.sids_refusal(esc["sids"], ["b:9"]) is None
    assert wa.sids_refusal(esc["sids"], ["d:1"]) is not None


def test_escape_artifact_carries_its_provenance_and_inherits_source_meta(tmp_path):
    """An escape artifact has to say it was an escape, why, and keep the meta
    of the head it was built on — a repair that erases the previous repair's
    record is not an improvement."""
    source = dict(SOURCE_ART,
                  meta=dict(SOURCE_ART["meta"],
                            rulings={"date": "2026-08-10"},
                            wave={"date": "2026-08-09",
                                  "sids_adjudicated": ["s:1"]}))
    prov = wa.escape_provenance(["s:2"], "blank rationale", "r3", ["a:1"])
    _path, payload = wa.write_wave_artifact(
        source, wa.merge_wave_rows(source, []), {},
        {"name": "prod", "seats": {}}, speech_id="trump_2026",
        wave_sids=["s:2"], reasons={"s:2": "extra-sid escape: blank rationale"},
        deferred_gated=[], rules={"utterance_record": True,
                                  "statistical_release": True},
        exhibits={}, out_dir=tmp_path, escape=prov, inherit_meta=True,
        remediation="extra-sid escape (r3)")
    meta = payload["meta"]
    assert meta["escape_run"]["escape"]["reason"] == "blank rationale"
    assert meta["escape_run"]["sids_adjudicated"] == ["s:2"]
    assert meta["remediation"] == "extra-sid escape (r3)"
    # The wave's own block survives verbatim: this run is not that wave.
    assert meta["wave"] == {"date": "2026-08-09", "sids_adjudicated": ["s:1"]}
    assert meta["rulings"] == {"date": "2026-08-10"}
    assert meta["rebuild_of"] == "old-run"


def test_a_plain_wave_artifact_is_unchanged_by_the_escape_wiring(tmp_path):
    """Regression guard on the default path: no escape block, no inherited
    meta, and the run still records itself under ``wave``."""
    _path, payload = wa.write_wave_artifact(
        SOURCE_ART, wa.merge_wave_rows(SOURCE_ART, []), {},
        {"name": "prod", "seats": {}}, speech_id="trump_2026",
        wave_sids=["s:2"], reasons={"s:2": "released"}, deferred_gated=["s:9"],
        rules={"utterance_record": True, "statistical_release": True},
        exhibits={}, out_dir=tmp_path)
    meta = payload["meta"]
    assert "escape_run" not in meta
    assert meta["remediation"] == wa.WAVE_TAG
    assert meta["wave"]["date"] == wa.WAVE_DATE
    assert set(meta) == {"speaker", "date", "speech_id", "venue", "roster",
                         "n_sentences", "n_check_worthy", "cost_usd",
                         "rebuild_of", "pipeline_generation", "remediation",
                         "wave"}


def test_settled_delta_never_rounds_a_bill_down():
    """The proxy's spend counter is written asynchronously, so the first read
    can report $0 for a call that cost money. Two reads, keep the larger."""
    class _Lane:
        reads = iter([1.0, 1.0036])

        @staticmethod
        def proxy_key_spend():
            return next(_Lane.reads)

    assert wa.settled_delta(_Lane, 1.0, settle_s=0) == pytest.approx(0.0036)


# ── the bridge boundary ──────────────────────────────────────────────────────

def test_bridge_carries_a_row_exhibit_into_the_published_provenance():
    from truthbot.verdict import bridge as bridge_mod

    prov = bridge_mod._build_provenance(
        {"sid": "s:1", "computed_exhibit": EXHIBIT},
        {"layer_a": {"claim_shape": "c-count"}})
    assert prov.computed_exhibit["series"] == "CPILFESL"
    assert ce.exhibit_html(prov.computed_exhibit,
                           claim_shape="c-count") != ""


def test_bridge_drops_an_exhibit_on_a_c_eval_claim_without_raising():
    """Defense in depth: the wave refuses to attach one, and if a row ever
    carried one anyway the bridge drops it rather than publishing arithmetic
    under an evaluative judgment. Dropping, not raising — the renderer already
    refuses to draw it, so failing the publish would trade an identical page
    for an outage."""
    from truthbot.verdict import bridge as bridge_mod

    prov = bridge_mod._build_provenance(
        {"sid": "s:1", "computed_exhibit": EXHIBIT},
        {"layer_a": {"claim_shape": "c-eval"}})
    assert prov.computed_exhibit == {}


def test_bridge_leaves_provenance_untouched_when_there_is_no_exhibit():
    from truthbot.verdict import bridge as bridge_mod

    prov = bridge_mod._build_provenance({"sid": "s:1"}, {"layer_a": {}})
    assert prov.computed_exhibit == {}


# ── the ratified rules ───────────────────────────────────────────────────────

def test_both_ratified_rules_are_default_on():
    """D15 and D16(alpha) were ratified 2026-08-09 and switched on. The wave
    passes them explicitly, but if the ambient default ever moved back the
    wave's packs would stop matching the flip set that priced it."""
    assert wa.rules_default_state() == {"utterance_record": True,
                                        "statistical_release": True}


# ── the real flip set (skips on a checkout without it) ───────────────────────

@pytest.mark.skipif(not wa.FLIPSET_PATH.exists(),
                    reason="regate_flipset.json absent")
def test_the_real_wave_is_29_claims_and_drops_0343():
    import json

    wave = wa.wave_set(json.loads(wa.FLIPSET_PATH.read_text("utf-8")))
    assert len(wave["sids"]) == 29
    assert "trump_2026:0343" in wave["dropped"]
    assert "trump_2026:0462" in wave["sids"]      # the xfail this wave answers
    assert len(wave["sids"]) == len(set(wave["sids"]))
