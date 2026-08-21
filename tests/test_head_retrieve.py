"""The publishing-head retrieval runner (scripts/head_retrieve.py) — offline, $0.

Nothing here touches a model, a proxy or the network. This runner is the first
one that BUYS retrieval for hand-named claims off the head, so the parts worth
holding are the ones that decide whether money is well spent, and the ones that
decide what a funded run is allowed to change:

  * it cannot write a pca_runs artifact, because writing one would silently
    make the run THE PUBLISHING HEAD — the hazard is pinned here as a fact,
    not as a comment;
  * completeness is judged over the REQUESTED sids, since phase3's whole-speech
    guard can never be satisfied by a named subset;
  * the head carries no claim shapes, so shapes must come from the sidecar —
    registering the head's own claims would register zero and silently fall
    back to the legacy quota;
  * the cost report keeps the ledger-true, the estimated and the unpriced lanes
    apart, because a subscription call reported as $0.00 is a claim the lane is
    free.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
_SPEC = importlib.util.spec_from_file_location(
    "head_retrieve", REPO / "scripts" / "head_retrieve.py")
hr = importlib.util.module_from_spec(_SPEC)
sys.modules["head_retrieve"] = hr
_SPEC.loader.exec_module(hr)          # must import clean with no key present

import phase3_rebuild as p3           # noqa: E402
import wave_adjudicate as wa          # noqa: E402

PROBE = ["trump_2026:0659", "trump_2026:0090", "trump_2026:0405"]
_HEAD_PRESENT = (REPO / "metrics" / "pca_runs").exists()
needs_head = pytest.mark.skipif(not _HEAD_PRESENT,
                                reason="pca_runs artifacts not present")


def _head():
    return hr.head_source("trump_2026")


# ── artifact abstinence: the load-bearing property ───────────────────────────

@needs_head
def test_writing_an_artifact_here_would_move_the_publishing_head(tmp_path):
    """The reason this runner has no artifact writer, pinned as a fact.

    shipping_artifact selects the unique LEAF whose lineage reaches the rulings
    pass. An artifact with rebuild_of=<head> makes the head a child, so the new
    file becomes that leaf — i.e. the artifact the site renders. If this test
    ever fails, the hazard is gone and the abstinence can be revisited; until
    then, nothing in this module may learn to write one."""
    from reshape_rerun_0031 import shipping_artifact

    src, art = _head()
    head_id = art["run_id"]
    (tmp_path / f"{head_id}.json").write_text(src.read_text(encoding="utf-8"),
                                              encoding="utf-8")
    probe_id = "00000000-dead-beef-0000-000000000000"
    (tmp_path / f"{probe_id}.json").write_text(json.dumps({
        "run_id": probe_id,
        "meta": {"speech_id": "trump_2026", "rebuild_of": head_id},
        "claims": [], "rows": []}), encoding="utf-8")

    _p, chosen = shipping_artifact("trump_2026", runs_dir=tmp_path)
    assert chosen["run_id"] == probe_id, (
        "a probe artifact rebuilt off the head DID become the publishing head")


def test_the_module_has_no_artifact_writer():
    """Absence is the safety property. A branch can be flipped by a later edit;
    a symbol that was never imported cannot be called by accident."""
    for name in ("write_wave_artifact", "write_new_artifact", "update_manifest",
                 "successor_run_id", "merge_wave_rows", "merge_wave_evidence"):
        assert not hasattr(hr, name), f"{name} must not be reachable here"


def test_write_artifact_flag_refuses_and_explains(capsys):
    """The flag exists only so the argument arrives at the moment somebody
    reaches for it, rather than being rediscovered from a moved head."""
    rc = hr.main(["--speech", "trump_2026", "--sids", "trump_2026:0090",
                  "--tag", "t1", "--write-artifact"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "PUBLISHING HEAD" in out
    assert "never reads the manifest" in out


def test_reshape_rerun_is_not_imported_at_module_level():
    """reshape_rerun_0031 imports wave_adjudicate at import time, and that cycle
    is already broken by a lazy import. A third module-level edge revives it."""
    src = (REPO / "scripts" / "head_retrieve.py").read_text(encoding="utf-8")
    body = src.split('"""', 2)[-1]          # past the module docstring
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and not line.startswith(" "):
            assert "reshape_rerun_0031" not in stripped


# ── completeness is over the requested sids ──────────────────────────────────

def test_completeness_is_over_the_requested_sids_not_the_speech():
    rows = [{"sid": "a:1"}, {"sid": "a:2"}]
    complete, missing = hr.requested_complete(["a:1", "a:2", "a:3"], rows)
    assert not complete and missing == ["a:3"]


def test_all_requested_banked_is_complete_though_the_speech_is_not():
    """The case phase3's guard gets wrong: three claims of a 182-claim speech
    are bought, all three bank, and the run IS complete."""
    rows = [{"sid": s} for s in PROBE]
    complete, missing = hr.requested_complete(PROBE, rows)
    assert complete and missing == []


def test_phase3s_whole_speech_guard_would_never_fire_on_a_named_subset():
    """Why this runner does not copy phase3_rebuild's completeness guard."""
    full_sids = {f"trump_2026:{i:04d}" for i in range(182)}
    have_sids = set(PROBE)
    assert not (full_sids <= have_sids)
    assert hr.requested_complete(PROBE, [{"sid": s} for s in PROBE])[0]


# ── head sourcing is structural, not advisory ────────────────────────────────

def test_head_source_always_asks_for_the_head(monkeypatch):
    seen = {}

    def fake(speech, **kw):
        seen.update(speech=speech, kw=kw)
        return Path("x.json"), {"run_id": "r"}

    monkeypatch.setattr(wa, "source_artifact", fake)
    hr.head_source("trump_2026")
    assert seen["kw"] == {"head": True}


def test_no_cli_option_can_select_a_source_artifact():
    """The pre-wave artifact is unreachable, not discouraged."""
    import argparse

    parser = None
    for obj in vars(hr).values():
        if isinstance(obj, argparse.ArgumentParser):
            parser = obj
    # main() builds its parser locally, so inspect the source instead.
    src = (REPO / "scripts" / "head_retrieve.py").read_text(encoding="utf-8")
    for banned in ('"--source"', '"--artifact"', '"--phase3"', '"--rebuild"',
                   '"--pre-wave"'):
        assert banned not in src


def test_head_refusal_fires_on_the_pinned_phase3_rebuild_id():
    """Retrieving against the pre-wave artifact would discard every ruling that
    landed since — the footgun phase3_rebuild's --sids can only warn about."""
    pinned = p3.SPEECHES["trump_2026"]["run_id"]
    msg = hr.head_refusal("trump_2026", {"run_id": pinned, "meta": {}})
    assert msg and "pre-wave" in msg and "No spend attempted" in msg


def test_head_refusal_fires_on_an_artifact_with_no_lineage():
    msg = hr.head_refusal("trump_2026", {"run_id": "loose", "meta": {}})
    assert msg and "not the publishing head" in msg


@needs_head
def test_head_refusal_passes_the_real_head():
    _p, art = _head()
    assert hr.head_refusal("trump_2026", art) is None


# ── resume identity ──────────────────────────────────────────────────────────

def test_header_line_does_not_disturb_the_shared_journal_loader(tmp_path):
    """Our header shares a file with publish_pipeline's loader. It must
    contribute no rows and no cost, or every resume misreports its own spend."""
    from truthbot.verdict import publish_pipeline

    j = tmp_path / "j.jsonl"
    header = hr.journal_header({"run_id": "abc"}, "trump_2026", "t", ["a:1"])
    with j.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        fh.write(json.dumps({"chunk": 1, "rows": [{"sid": "a:1"}],
                             "evidence": {}, "cost_usd": 0.25}) + "\n")
    rows, _packs, cost, _roster = publish_pipeline.load_chunk_journal(j)
    assert rows == [{"sid": "a:1"}]
    assert cost == pytest.approx(0.25)


def test_resume_refuses_a_journal_written_against_a_different_head(tmp_path):
    """A verdict decided against the old head is a superseded verdict wearing
    the current run's tag."""
    j = tmp_path / "j.jsonl"
    j.write_text(json.dumps(hr.journal_header(
        {"run_id": "OLDHEAD"}, "trump_2026", "t", ["a:1"])) + "\n",
        encoding="utf-8")
    msg = hr.header_refusal(j, {"run_id": "NEWHEAD"}, "trump_2026")
    assert msg and "head moved" in msg


def test_resume_refuses_a_journal_written_by_another_script(tmp_path):
    j = tmp_path / "j.jsonl"
    j.write_text(json.dumps({"chunk": 1, "rows": []}) + "\n", encoding="utf-8")
    msg = hr.header_refusal(j, {"run_id": "r"}, "trump_2026")
    assert msg and "no" in msg and "header" in msg


def test_resume_accepts_its_own_journal(tmp_path):
    j = tmp_path / "j.jsonl"
    j.write_text(json.dumps(hr.journal_header(
        {"run_id": "SAME"}, "trump_2026", "t", ["a:1"])) + "\n",
        encoding="utf-8")
    assert hr.header_refusal(j, {"run_id": "SAME"}, "trump_2026") is None


def test_a_missing_journal_is_a_fresh_run(tmp_path):
    assert hr.header_refusal(tmp_path / "nope.jsonl", {"run_id": "r"},
                             "trump_2026") is None


# ── journal namespacing ──────────────────────────────────────────────────────

def test_journals_cannot_collide_with_the_wave_or_phase3(tmp_path):
    """The headret_ infix is hardcoded, so no --tag value can produce another
    lane's filename — and therefore cannot append to a banked journal."""
    for speech in p3.SPEECHES:
        chunk, packs = hr.journal_paths(speech, "wave")
        assert chunk != wa.journal_path(speech, "wave")
        assert chunk != p3.journal_paths(speech)[0]
        assert packs != p3.journal_paths(speech)[1]
        assert "headret" in chunk.name and "headret" in packs.name


@needs_head
def test_every_headret_journal_on_disk_was_written_by_this_runner():
    """The stable form of 'our journals never collide'. Asserting our names are
    ABSENT from disk was wrong: it goes false the moment the runner is used
    (it did, on the 2026-08-14 probe). The property that actually matters is
    that anything wearing our infix carries our schema header — i.e. we never
    adopted, or were adopted into, another lane's file."""
    for p in (REPO / "metrics" / "journals").glob(f"*_{hr.RUNNER_TAG_INFIX}_*.jsonl"):
        if p.name.endswith("_packs.jsonl"):
            continue          # packs journals are records, not headed streams
        assert hr.read_journal_header(p) is not None, (
            f"{p.name} wears the headret infix but has no "
            f"{hr.JOURNAL_SCHEMA} header")


def test_reserved_and_malformed_tags_are_refused():
    for bad in ("wave", "p3rebuild", "r3", "s5rescue"):
        assert hr.tag_refusal(bad)
    for bad in ("", "a/b", "a b", "a.b"):
        assert hr.tag_refusal(bad)
    assert hr.tag_refusal("d17dprobe0") is None


def test_report_path_is_prefixed_so_it_is_not_a_wave_report():
    assert "headret" in hr.report_path("d17dprobe0").name


# ── shapes: the head has none ────────────────────────────────────────────────

@needs_head
def test_the_head_carries_no_claim_shapes_for_the_probe_sids():
    """If this ever changes, shapes_for can be simplified — until then,
    trusting the artifact's own layer_a would register nothing."""
    _p, art = _head()
    by_sid = {c["sid"]: c for c in art["claims"]}
    for sid in PROBE:
        assert not (by_sid[sid].get("layer_a") or {}).get("claim_shape")


@needs_head
def test_shapes_are_resolved_from_the_sidecar():
    _p, art = _head()
    shapes, n_filled = hr.shapes_for(art, "trump_2026")
    assert shapes["trump_2026:0659"] == "c-third"
    assert shapes["trump_2026:0090"] == "c-third"
    assert shapes["trump_2026:0405"] == "c-eval"
    assert n_filled > 0


@needs_head
def test_registering_the_heads_claims_directly_registers_almost_nothing():
    """The trap, pinned. The head carries exactly ONE shape — the R-1 shape
    correction the rulings pass wrote onto trump_2026:0031 — and none for the
    other 181 claims. So trusting the artifact's own layer_a covers 1/182, and
    with --legacy-quota-ok the rest would silently run the LEGACY evidential-
    role quota, answering a different question than the rest of the corpus."""
    from truthbot.verdict import shape_registry

    _p, art = _head()
    try:
        n = shape_registry.register_claim_shapes(art["claims"])
        assert n == 1
        assert shape_registry.shape_for("trump_2026:0031") == "c-count"
        for sid in PROBE:
            assert not shape_registry.shape_for(sid)
    finally:
        shape_registry.clear()


def test_shape_refusal_is_scoped_to_the_selected_sids():
    assert p3.shape_refusal(3, 3, False) is None
    assert p3.shape_refusal(2, 3, False)
    assert p3.shape_refusal(0, 3, True) is None


# ── spend honesty ────────────────────────────────────────────────────────────

def _split(proxy=0.04, off=0.08, banked=0.0, r1=3):
    usage = {"R2": [{}] * 6, "R3": [{}]}
    return hr.spend_split(proxy, off, usage, r1, banked)


def test_offproxy_is_labelled_an_estimate_not_a_ledger_reading():
    s = _split()
    assert s["proxy"]["confidence"] == "ledger-true"
    assert "ESTIMATE" in s["offproxy"]["confidence"]


def test_r1_is_reported_unpriced_not_zero():
    """A subscription call reported as $0.00 is a claim the lane is free."""
    s = _split()
    assert s["r1_worker"]["usd"] is None
    assert "UNPRICED" in s["r1_worker"]["confidence"]
    assert s["r1_worker"]["calls"] == 3


def test_billed_total_is_the_pessimistic_number_the_breaker_enforces():
    s = _split(proxy=0.04, off=0.08, banked=0.01)
    assert s["billed_total_usd"] == pytest.approx(0.13)
    assert s["estimated_share"] == pytest.approx(0.08 / 0.13, rel=1e-3)


def test_the_lanes_are_never_fused_into_one_headline():
    s = _split()
    assert {"proxy", "offproxy", "r1_worker"} <= set(s)
    assert s["proxy"]["usd"] != s["billed_total_usd"]


def test_lane_projection_states_its_estimated_share_and_refuses_a_band():
    proj = hr.lane_projection(_split(), 3, hr.WEB_TIER1_LANE_N)
    assert proj["n_lane"] == 81
    assert proj["ledger_true_usd"] == pytest.approx(0.04 * 27)
    assert proj["estimated_usd"] == pytest.approx(0.08 * 27)
    assert "point projection" in proj["kind"]
    assert "band" in proj["kind"]


def test_phase_r_spend_is_banked_into_the_chunk_journal(tmp_path):
    """The hard cap must survive a resume.

    Phase R writes only to the PACKS journal, so load_chunk_journal returned
    banked_cost=0 and a resumed run restarted its ceiling from zero — $0.45 of
    retrieval followed by a resume would authorise another full --budget. The
    2026-08-14 probe hit exactly this: phase P reported $0.00 having inherited
    none of phase R's $0.256. Banking it as a rows-less chunk record makes the
    shared loader carry the cost without polluting rows or packs."""
    from truthbot.verdict import publish_pipeline

    j = tmp_path / "chunk.jsonl"
    publish_pipeline.append_chunk_journal(j, 0, [], {}, 0.2560)
    rows, packs, cost, _roster = publish_pipeline.load_chunk_journal(j)
    assert rows == [] and packs == {}
    assert cost == pytest.approx(0.2560)


def test_lane_projection_carries_banked_spend_from_a_prior_session():
    """A resumed run holds its retrieval cost in `banked`. Projecting only the
    live lanes priced the 81-claim lane at $0.00 on the real probe."""
    s = _split(proxy=0.0, off=0.0, banked=0.2560)
    proj = hr.lane_projection(s, 3, 81)
    assert proj["total_usd"] == pytest.approx(0.2560 * 27)
    assert proj["banked_usd"] == pytest.approx(0.2560 * 27)


def test_lane_projection_is_empty_when_nothing_banked():
    assert hr.lane_projection(_split(), 0, 81) == {}


def test_estimate_refuses_to_quote_a_band_for_this_lane(capsys):
    """S-12: the probe exists because the lane has no measured constant.
    Printing one from a constant measured on full-speech rebuilds would
    recreate the guess it is meant to replace."""
    text = hr.estimate_report("trump_2026", PROBE)
    assert "UNMEASURED" in text
    assert "wrong payload" in text
    assert "may not be quoted" in text


def test_estimate_names_the_unpriced_r1_lane():
    assert "Max subscription" in hr.estimate_report("trump_2026", PROBE)


# ── refusal composition ──────────────────────────────────────────────────────

def test_go_refusal_is_phase3s_so_the_r2_economy_guard_applies():
    """This runner RETRIEVES. wave_adjudicate's budget-only refusal would let
    the 2026-08-01 2.5x R2 overspend back in."""
    assert hr.p3.go_refusal is p3.go_refusal
    msg = p3.go_refusal({}, 0.5)
    assert msg and "TRUTHBOT_R2_MODEL" in msg
    assert p3.go_refusal({"TRUTHBOT_R2_MODEL": "gpt-5-mini"}, 0.5) is None


def test_budget_is_required_with_go():
    msg = p3.go_refusal({"TRUTHBOT_R2_MODEL": "gpt-5-mini"}, None)
    assert msg and "--budget" in msg


def test_reason_is_required_with_go_but_not_for_a_plan():
    assert hr.reason_refusal("", True)
    assert hr.reason_refusal("", False) is None
    assert hr.reason_refusal("pricing the lane", True) is None


def test_select_claims_is_phase3s():
    """One implementation of 'refuse an unknown sid rather than silently
    retrieving nothing'."""
    assert hr.p3.select_claims is p3.select_claims


def test_unknown_sid_exits_clean_rather_than_tracebacking(capsys):
    rc = hr.main(["--speech", "trump_2026", "--sids", "trump_2026:9999",
                  "--tag", "t1"])
    assert rc == 2
    assert "not in this speech's artifact" in capsys.readouterr().out


# ── anti-drift on the shared pricing instrument ──────────────────────────────

def test_offproxy_rates_are_the_same_object_as_phase3s():
    """The estimator the 81-claim lane is bought on must exist exactly once."""
    assert hr.p3.MODEL_RATES is p3.MODEL_RATES


def test_the_metered_retriever_helper_is_shared_not_copied():
    assert hasattr(p3, "metered_offproxy_retrievers")
    src = (REPO / "scripts" / "head_retrieve.py").read_text(encoding="utf-8")
    assert "class MeteredR2" not in src, "the pricing instrument was copied"
    assert "metered_offproxy_retrievers()" in src


def test_offproxy_estimator_prices_a_known_usage_payload():
    from truthbot import costs

    _primary, _retry, est, usage = p3.metered_offproxy_retrievers()
    assert est() == 0.0
    usage["R2"].append({"model": "gpt-5-mini",
                        "usage": {"input_tokens": 1000, "output_tokens": 500}})
    rin, rout = costs.rates("gpt-5-mini")
    assert est() == pytest.approx((1000 * rin + 500 * rout) / 1e6)


def test_an_unknown_model_is_priced_pessimistically_never_zero():
    """A response from a model we do not know must not contribute $0 — that
    biases the cap loose in the one direction that under-reports a bill."""
    _primary, _retry, est, usage = p3.metered_offproxy_retrievers()
    usage["R3"].append({"model": "no-such-model",
                        "usage": {"input_tokens": 1000, "output_tokens": 0}})
    assert est() > 0


def test_the_packs_journal_has_exactly_one_writer(tmp_path, monkeypatch):
    """Phase R journaling belongs to build_packs_phase, which also owns the
    resume path. Wiring make_pack_builder's packs_journal as well wrote every
    pack TWICE — harmless to a resume (load_packs_journal builds a dict) but it
    doubles the apparent retrieval work in the artifact a lane gets priced
    from. Caught in the real 2026-08-14 probe run."""
    from truthbot.verdict import proxy_lane
    from truthbot.verdict.evidence_pack import EvidencePack
    from truthbot.verdict.retrieval_phase import build_packs_phase

    # The breaker reads the ledger; this test is offline and $0.
    monkeypatch.setattr(proxy_lane, "proxy_key_spend", lambda *a, **k: 0.0)
    j = tmp_path / "packs.jsonl"

    def raw(sid, text, context):
        return EvidencePack(sid=sid, window=None, items=[], gate_code="")

    builder = p3.make_pack_builder(
        build_pack=raw, cap=1e9, start_spend=0.0,
        offproxy_est=lambda: 0.0, banked_cost=0.0,
        packs_journal=None)          # <- the fix under test
    build_packs_phase([{"sid": "a:1", "text": "t", "context": ""}],
                      builder, journal_path=j)
    lines = [ln for ln in j.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1, f"pack journaled {len(lines)}x, expected once"


def test_head_retrieve_does_not_double_wire_the_packs_journal():
    """The wiring itself, pinned at the call site."""
    src = (REPO / "scripts" / "head_retrieve.py").read_text(encoding="utf-8")
    assert "packs_journal=None)" in src
    assert "packs_journal=packs_journal)" not in src


def test_r1_call_counting_does_not_touch_retrievers_module():
    """ClaudeWorkerRetriever has no _post seam, so the instance method is
    wrapped. retry reuses the SAME instance, so one patch covers both rounds."""
    calls = []

    class FakeR1:
        def shortlist(self, *a, **k):
            calls.append(a)
            return []

    r1 = FakeR1()
    primary = (r1, object())
    counter = hr._count_r1(primary)
    r1.shortlist("claim")
    r1.shortlist("claim2")
    assert counter["n"] == 2
    assert len(calls) == 2
