"""Tests for the v2 PCA publish orchestrator (run_pca_verify + lane factory).

Fully offline: Layer A's A2 lane and the PCA adjudicate lane are injected fakes.
"""
from __future__ import annotations

from datetime import date

from truthbot.models import VerdictBundle, VerdictLabel
from truthbot.verdict import publish_pipeline as pp
from truthbot.verdict import speech_context
from truthbot.verdict.evidence_pack import EvidencePack, PackItem
from truthbot.models import SourceTier


# ── fakes ──────────────────────────────────────────────────────────────────

def _sentences(n):
    # statistical sentences A1 routes to A2 (not dropped as ceremonial)
    return [
        {"sid": f"sp:{i:04d}", "text": f"Metric {i} rose by {i} percent in 2026.",
         "context": f"|| Metric {i} rose by {i} percent in 2026. ||"}
        for i in range(n)
    ]


def _fake_classify_all_checkworthy(sents):
    return [{"sid": s["sid"], "label": "check-worthy",
             "text": s["text"], "context": s["context"]} for s in sents]


def _pack(sid):
    return EvidencePack(sid=sid, window=None, items=[
        PackItem(pack_id="E1", source_name="BLS", source_url="https://bls.gov/x",
                 tier=SourceTier.GOVERNMENT, snippet="snip",
                 retrieved_at="2026-01-01T00:00:00+00:00", sha256="x")])


def _pack_with_retrieval(sid, cost_usd):
    """EvidencePack is frozen, so retrieval telemetry goes in at construction —
    which is also how the v2 builder sets it."""
    return EvidencePack(
        sid=sid, window=None,
        items=[PackItem(pack_id="E1", source_name="BLS",
                        source_url="https://bls.gov/x",
                        tier=SourceTier.GOVERNMENT, snippet="snip",
                        retrieved_at="2026-01-01T00:00:00+00:00", sha256="x")],
        retrieval={"cost_usd": cost_usd, "calls": 2, "unpriced_calls": 0,
                   "by_adapter": {"openai": cost_usd}})


def _make_fake_adjudicate(calls):
    """Returns an adjudicate_fn that records each chunk and echoes resolved rows."""
    def fake_adjudicate(chunk):
        calls.append(list(chunk))
        rows = [{"sid": c["sid"], "status": "resolved", "verdict": "TRUE",
                 "confidence": 0.9, "citations": ["E1"], "reasoning": "ok",
                 "votes": {"TRUE": 3}} for c in chunk]
        packs = {c["sid"]: _pack(c["sid"]) for c in chunk}
        return rows, {"packs": packs, "cost_usd": 0.1}
    return fake_adjudicate


# ── run_pca_verify ─────────────────────────────────────────────────────────

def test_run_pca_verify_chunks_bridges_and_totals_cost():
    calls = []
    res = pp.run_pca_verify(
        _sentences(5),
        layer_a_fn=_fake_classify_all_checkworthy,
        adjudicate_fn=_make_fake_adjudicate(calls),
        chunk_size=2,
    )
    # 5 check-worthy → chunks of 2 → 3 adjudicate calls (2,2,1)
    assert res.n_sentences == 5
    assert res.n_check_worthy == 5
    assert res.n_chunks == 3
    assert [len(c) for c in calls] == [2, 2, 1]
    # one bundle per check-worthy claim, in order, all bridged to TRUE
    assert len(res.bundles) == 5
    assert all(isinstance(b, VerdictBundle) for b in res.bundles)
    assert [b.consensus.claim_id for b in res.bundles] == [f"sp:{i:04d}" for i in range(5)]
    assert res.bundles[0].consensus.consensus_label is VerdictLabel.TRUE
    # cited pack wired through to web_sources + evidence corpus
    assert res.bundles[0].model_verdicts[0].web_sources == ["https://bls.gov/x"]
    assert res.evidence["sp:0000"][0].source_tier is SourceTier.GOVERNMENT
    # cost summed across the 3 chunks (0.1 each)
    assert abs(res.cost_usd - 0.3) < 1e-9


def test_retrieval_cost_is_tracked_but_kept_out_of_cost_usd():
    """Retrieval spend must never enter ``cost_usd``.

    ``cost_usd`` is what the budget breaker averages to project the NEXT
    chunk's cost. Folding a one-off retrieval total into it would forecast a
    per-chunk cost no chunk will ever incur and halt runs early — so the two
    legs are reported side by side instead.
    """
    def adj_with_retrieval(chunk):
        rows = [{"sid": c["sid"], "status": "resolved", "verdict": "TRUE",
                 "confidence": 0.9, "citations": ["E1"], "reasoning": "ok",
                 "votes": {"TRUE": 3}} for c in chunk]
        packs = {c["sid"]: _pack_with_retrieval(c["sid"], 0.25) for c in chunk}
        return rows, {"packs": packs, "cost_usd": 0.1}

    res = pp.run_pca_verify(
        _sentences(3),
        layer_a_fn=_fake_classify_all_checkworthy,
        adjudicate_fn=adj_with_retrieval,
        chunk_size=1,
        banked_retrieval_cost_usd=1.0,
    )
    # Adjudication only — unchanged by the retrieval legs.
    assert abs(res.cost_usd - 0.3) < 1e-9
    # 1.0 banked from a prior run + 3 packs x 0.25 retrieved this run.
    assert abs(res.retrieval_cost_usd - 1.75) < 1e-9


def test_packs_without_retrieval_telemetry_report_zero_retrieval_cost():
    """v1 packs and journal-resumed packs carry no snapshot; that must not crash."""
    res = pp.run_pca_verify(
        _sentences(2),
        layer_a_fn=_fake_classify_all_checkworthy,
        adjudicate_fn=_make_fake_adjudicate([]),
        chunk_size=1,
    )
    assert res.retrieval_cost_usd == 0.0


def test_run_pca_verify_captures_roster_once():
    # The panel roster is identical across chunks; run_pca_verify captures the
    # first non-empty notes["roster"] and never overwrites it.
    def adj_with_roster(chunk):
        rows = [{"sid": c["sid"], "status": "resolved", "verdict": "TRUE",
                 "confidence": 0.9, "citations": [], "reasoning": "ok",
                 "votes": {"TRUE": 3}} for c in chunk]
        return rows, {"packs": {}, "cost_usd": 0.0,
                      "roster": {"name": "dev", "seats": {"proposer": ["mistral"]}}}

    res = pp.run_pca_verify(
        _sentences(3),
        layer_a_fn=_fake_classify_all_checkworthy,
        adjudicate_fn=adj_with_roster,
        chunk_size=1,
    )
    assert res.roster == {"name": "dev", "seats": {"proposer": ["mistral"]}}


def test_run_pca_verify_roster_defaults_none_when_absent():
    # Legacy/offline adjudicate that emits no roster note → result.roster stays None.
    res = pp.run_pca_verify(
        _sentences(2),
        layer_a_fn=_fake_classify_all_checkworthy,
        adjudicate_fn=_make_fake_adjudicate([]),
        chunk_size=1,
    )
    assert res.roster is None


def test_run_pca_verify_progress_callback():
    seen = []
    pp.run_pca_verify(
        _sentences(3),
        layer_a_fn=_fake_classify_all_checkworthy,
        adjudicate_fn=_make_fake_adjudicate([]),
        chunk_size=1,
        on_progress=lambda i, n, rows: seen.append((i, n, len(rows))),
    )
    assert seen == [(1, 3, 1), (2, 3, 1), (3, 3, 1)]


def test_run_pca_verify_empty_queue_skips_adjudicate():
    calls = []
    # classify sends everything to characterization → empty check-worthy queue
    def none_checkworthy(sents):
        return [{"sid": s["sid"], "label": "unimportant",
                 "text": s["text"], "context": s["context"]} for s in sents]

    res = pp.run_pca_verify(
        _sentences(3),
        layer_a_fn=none_checkworthy,
        adjudicate_fn=_make_fake_adjudicate(calls),
        chunk_size=2,
    )
    assert res.n_check_worthy == 0
    assert res.bundles == []
    assert res.n_chunks == 0
    assert calls == []  # adjudicate never called → no spend
    # characterization stream is preserved for the publisher
    assert len(res.characterization) == 3


# ── build_pca_lane_fns ─────────────────────────────────────────────────────

def test_lane_factory_folds_cost_and_toggles_crm114(monkeypatch):
    from truthbot.checkworthy import classifier
    from truthbot.verdict import adjudicator

    captured = {}

    class _Manifest:
        total_cost_usd = 0.42

    def fake_classify(hm, sentences, tier="cheap", on_parse_error="raise"):
        captured["a2_tier"] = tier
        captured["classify_hm"] = hm
        return [{"sid": s["sid"], "label": "check-worthy"} for s in sentences], _Manifest()

    def fake_adjudicate(hm, claims, **kwargs):
        captured["adj_kwargs"] = kwargs
        captured["adj_hm"] = hm
        rows = [{"sid": c["sid"], "status": "resolved", "verdict": "TRUE",
                 "confidence": 0.8, "citations": [], "reasoning": "", "votes": {"TRUE": 3}}
                for c in claims]
        return rows, _Manifest(), {"packs": {}, "open_book": False}

    monkeypatch.setattr(classifier, "classify", fake_classify)
    monkeypatch.setattr(adjudicator, "adjudicate", fake_adjudicate)

    hm_classify, hm_verdict = object(), object()
    # provider=None → closed-book → two_stage must be forced off
    layer_a_fn, adjudicate_fn = pp.build_pca_lane_fns(
        hm_classify, hm_verdict, provider=None, crm114=True, roster="dev", a2_tier="standard")

    a2_rows = layer_a_fn([{"sid": "sp:0000", "text": "t", "context": "c"}])
    assert a2_rows[0]["label"] == "check-worthy"
    assert captured["a2_tier"] == "standard"

    rows, notes = adjudicate_fn([{"sid": "sp:0000", "text": "t", "context": "c"}])
    assert rows[0]["verdict"] == "TRUE"
    assert notes["cost_usd"] == 0.42                 # manifest cost folded in
    assert captured["adj_kwargs"]["two_stage"] is False   # no provider → no CRM-114
    assert captured["adj_kwargs"]["roster"] == "dev"
    # PCA panel composition captured in notes: roster name + real "dev" seats.
    assert notes["roster"]["name"] == "dev"
    seats = notes["roster"]["seats"]
    assert seats["proposer"] == ["mistral"]
    assert seats["critic"] == ["dsv4-flash"]
    assert seats["arbiter"] == ["claude-haiku"]
    # each lane is routed to its own engine (classify=identity parser, verdict=parse_verdict)
    assert captured["classify_hm"] is hm_classify
    assert captured["adj_hm"] is hm_verdict


def test_lane_factory_paces_layer_a_in_batches(monkeypatch):
    from truthbot.checkworthy import classifier

    class _M:
        total_cost_usd = 0.0

    calls = []
    naps = []

    def fake_classify(hm, sentences, tier="cheap", on_parse_error="raise"):
        calls.append([s["sid"] for s in sentences])
        return [{"sid": s["sid"], "label": "check-worthy"} for s in sentences], _M()

    monkeypatch.setattr(classifier, "classify", fake_classify)
    layer_a_fn, _ = pp.build_pca_lane_fns(
        object(), object(), provider=None, layer_a_batch=2, layer_a_pause_s=0.5,
        sleep_fn=lambda s: naps.append(s))

    sents = [{"sid": f"s:{i}", "text": "t", "context": "c"} for i in range(5)]
    rows = layer_a_fn(sents)
    # 5 sentences, batch 2 → 3 classify calls (2,2,1), 2 inter-batch pauses
    assert [len(c) for c in calls] == [2, 2, 1]
    assert naps == [0.5, 0.5]                 # paused between batches, not after the last
    assert len(rows) == 5


def test_lane_factory_enables_crm114_with_provider(monkeypatch):
    from truthbot.verdict import adjudicator

    captured = {}

    class _Manifest:
        total_cost_usd = 0.0

    def fake_adjudicate(hm, claims, **kwargs):
        captured["adj_kwargs"] = kwargs
        return [], _Manifest(), {}

    monkeypatch.setattr(adjudicator, "adjudicate", fake_adjudicate)
    _, adjudicate_fn = pp.build_pca_lane_fns(
        object(), object(), provider=object(), crm114=True)
    adjudicate_fn([])
    assert captured["adj_kwargs"]["two_stage"] is True
    assert captured["adj_kwargs"]["evidence_provider"] is not None


# ── prepare_speech ─────────────────────────────────────────────────────────

def test_prepare_speech_segments_and_registers_date():
    sents = pp.prepare_speech("The deficit fell by 5 percent. Growth was strong.",
                              "custom_speech_2099", date(2099, 5, 1))
    assert sents[0]["sid"] == "custom_speech_2099:0000"
    # utterance date registered so temporal grounding resolves for this speech
    assert speech_context.speech_date_for("custom_speech_2099:0000") == date(2099, 5, 1)
