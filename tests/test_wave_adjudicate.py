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
