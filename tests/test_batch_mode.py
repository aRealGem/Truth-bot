"""Batch job descriptor persistence + submit/reconcile round-trip with mocked clients."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from truthbot.models import Claim, Confidence, ModelVerdict, VerdictLabel
from truthbot.verify.batch import (
    BatchDispatcher,
    parse_provider_results,
    read_batch_job,
    reconcile_run,
    sidecar_path,
)


# ── Legacy descriptor surface ─────────────────────────────────────────────────


def test_record_and_read_batch_job(tmp_path: Path) -> None:
    md = tmp_path / "metrics"
    d = BatchDispatcher(md)
    path = d.record_job(
        "run-abc",
        transcript_meta={"speaker": "X"},
        work_units=[{"claim_id": "1", "claim_text": "t"}],
    )
    assert path.exists()
    data = read_batch_job(md, "run-abc")
    assert data is not None
    assert data["run_id"] == "run-abc"
    assert data["status"] == "pending"
    assert len(data["work_units"]) == 1


def test_poll_missing(tmp_path: Path) -> None:
    assert BatchDispatcher(tmp_path / "m").poll("nope") == "missing"


def test_poll_pending(tmp_path: Path) -> None:
    md = tmp_path / "metrics"
    BatchDispatcher(md).record_job("r1", transcript_meta={}, work_units=[])
    assert BatchDispatcher(md).poll("r1") == "pending"


# ── Anthropic submit + reconcile round-trip with mocked SDK ──────────────────


class _FakeAnthropicAdapter:
    """Minimal adapter stand-in implementing the LLMAdapter batch contract."""

    adapter_name = "anthropic"
    model_id = "claude-opus-4-7"
    required_env_key = "ANTHROPIC_API_KEY"
    supports_batch = True

    def build_batch_payload(self, claim, evidence, *, inject_evidence=True):
        return {
            "model": self.model_id,
            "max_tokens": 2048,
            "system": [{"type": "text", "text": "sys"}],
            "messages": [{"role": "user", "content": claim.text}],
        }

    def parse_batch_response(self, raw, claim):
        # ``raw`` is a dict mimicking an Anthropic ``message`` envelope
        text_blocks = [b for b in raw.get("content", []) if b.get("type") == "text"]
        verdict_json = json.loads(text_blocks[0]["text"])
        usage = raw.get("usage", {})
        return ModelVerdict(
            adapter_name=self.adapter_name,
            model_id=raw.get("model", self.model_id),
            claim_id=claim.id,
            label=VerdictLabel(verdict_json["label"]),
            confidence=Confidence(verdict_json["confidence"]),
            explanation=verdict_json.get("explanation", ""),
            tier="frontier",
            synthesis_mode="batch",
            cached_input_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
        )


def _claim(text="Unemployment fell.") -> Claim:
    return Claim(transcript_id="t1", text=text, speaker="Test")


def test_submit_and_reconcile_round_trip_anthropic(tmp_path, monkeypatch) -> None:
    """End-to-end: submit → poll complete → fetch results → cache bundle."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("TRUTHBOT_METRICS_DIR", str(tmp_path / "metrics"))
    import truthbot.metrics.telemetry as tel

    tel._telemetry_instance = None

    adapter = _FakeAnthropicAdapter()
    claim_a = _claim("Claim A")
    claim_b = _claim("Claim B")
    claims_with_evidence = [(claim_a, []), (claim_b, [])]

    fake_batch = SimpleNamespace(id="msgbatch_test_123")
    counts_all_done = SimpleNamespace(
        succeeded=2, errored=0, canceled=0, expired=0, processing=0
    )
    retrieved = SimpleNamespace(
        processing_status="ended", request_counts=counts_all_done
    )

    def _mk_result(custom_id, label):
        return SimpleNamespace(
            custom_id=custom_id,
            result=SimpleNamespace(
                type="succeeded",
                message={
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "label": label,
                                    "confidence": "High",
                                    "explanation": f"batched verdict for {custom_id}",
                                    "web_sources": [],
                                }
                            ),
                        }
                    ],
                    "model": "claude-opus-4-7",
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                },
            ),
        )

    client_mock = MagicMock()
    client_mock.messages.batches.create.return_value = fake_batch
    client_mock.messages.batches.retrieve.return_value = retrieved
    # The submit uses ``_custom_id`` which is "<adapter>::<claim_id[:40]>"
    from truthbot.verify.batch import _custom_id

    cid_a = _custom_id(claim_a.id, "anthropic")
    cid_b = _custom_id(claim_b.id, "anthropic")
    client_mock.messages.batches.results.return_value = iter(
        [_mk_result(cid_a, "True"), _mk_result(cid_b, "False")]
    )

    metrics_dir = tmp_path / "metrics"
    dispatcher = BatchDispatcher(metrics_dir)

    with patch("anthropic.Anthropic", return_value=client_mock):
        descriptor_path = dispatcher.submit(
            "run-test",
            adapters=[adapter],
            claims_with_evidence=claims_with_evidence,
            transcript_meta={
                "speaker": "Test",
                "date": "2026-02-24",
                "triaged_claim_ids": [],
                "triaged_claims": [],
            },
            inject_evidence=False,
            sidecar_live_adapters=None,
        )
        assert descriptor_path.exists()

        # submit → should register an anthropic provider job
        desc = read_batch_job(metrics_dir, "run-test")
        assert desc["provider_jobs"]["anthropic"]["batch_id"] == "msgbatch_test_123"
        assert desc["status"] == "submitted"

        # poll → complete (retrieve returns ended)
        assert dispatcher.poll("run-test") == "complete"

        # reconcile → parses results, builds bundles, caches them
        class _FakeEngine:
            def __init__(self):
                self.bundles = {}

            def finalize_bundle(
                self, claim, speaker, date_str, model_verdicts, evidence_count=0
            ):
                from truthbot.models import ConsensusVerdict, VerdictBundle

                lbl = model_verdicts[0].label if model_verdicts else VerdictLabel.UNVERIFIABLE
                cons = ConsensusVerdict(
                    claim_id=claim.id,
                    model_verdicts=model_verdicts,
                    consensus_label=lbl,
                    confidence=Confidence.HIGH,
                    agreement=True,
                    consensus_strength="single",
                    explanation="fake",
                )
                b = VerdictBundle(
                    claim=claim,
                    speaker=speaker,
                    date_str=date_str,
                    model_verdicts=model_verdicts,
                    consensus=cons,
                    evidence_count=evidence_count,
                )
                self.bundles[claim.id] = b
                return b

            def maybe_resolve_early(self, claim, speaker="", date_str=""):
                return None, []

        engine = _FakeEngine()
        result = reconcile_run(
            metrics_dir,
            "run-test",
            adapters_by_name={"anthropic": adapter},
            engine=engine,
        )
        assert result["status"] == "complete"
        assert len(result["bundles"]) == 2
        labels = sorted(b.consensus.consensus_label.value for b in result["bundles"])
        assert labels == ["False", "True"]


def test_reconcile_pending_returns_early(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    metrics_dir = tmp_path / "metrics"
    dispatcher = BatchDispatcher(metrics_dir)

    counts_unfinished = SimpleNamespace(
        succeeded=1, errored=0, canceled=0, expired=0, processing=1
    )
    retrieved = SimpleNamespace(
        processing_status="in_progress", request_counts=counts_unfinished
    )
    fake_batch = SimpleNamespace(id="msgbatch_pending")
    client_mock = MagicMock()
    client_mock.messages.batches.create.return_value = fake_batch
    client_mock.messages.batches.retrieve.return_value = retrieved

    adapter = _FakeAnthropicAdapter()
    claim = _claim()
    with patch("anthropic.Anthropic", return_value=client_mock):
        dispatcher.submit(
            "run-pending",
            adapters=[adapter],
            claims_with_evidence=[(claim, [])],
            transcript_meta={"speaker": "T", "date": "2026-02-24"},
            inject_evidence=False,
        )
        result = reconcile_run(
            metrics_dir,
            "run-pending",
            adapters_by_name={"anthropic": adapter},
            engine=MagicMock(),
        )
    assert result["status"] == "pending"
    assert result["pending_providers"][0][0] == "anthropic"


def test_batch_cost_telemetry_has_real_batch_job_id(tmp_path, monkeypatch) -> None:
    """After reconcile, adapter_calls.jsonl rows must carry the real batch_job_id."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("TRUTHBOT_METRICS_DIR", str(tmp_path / "metrics"))
    import truthbot.metrics.telemetry as tel

    tel._telemetry_instance = None

    adapter = _FakeAnthropicAdapter()
    claim = _claim()

    fake_batch = SimpleNamespace(id="msgbatch_real_id")
    counts_all_done = SimpleNamespace(
        succeeded=1, errored=0, canceled=0, expired=0, processing=0
    )
    retrieved = SimpleNamespace(
        processing_status="ended", request_counts=counts_all_done
    )
    from truthbot.verify.batch import _custom_id

    result_row = SimpleNamespace(
        custom_id=_custom_id(claim.id, "anthropic"),
        result=SimpleNamespace(
            type="succeeded",
            message={
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "label": "True",
                                "confidence": "High",
                                "explanation": "ok",
                                "web_sources": [],
                            }
                        ),
                    }
                ],
                "model": "claude-opus-4-7",
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        ),
    )
    client_mock = MagicMock()
    client_mock.messages.batches.create.return_value = fake_batch
    client_mock.messages.batches.retrieve.return_value = retrieved
    client_mock.messages.batches.results.return_value = iter([result_row])

    metrics_dir = tmp_path / "metrics"
    dispatcher = BatchDispatcher(metrics_dir)

    engine = MagicMock()
    engine.finalize_bundle.return_value = MagicMock()
    engine.maybe_resolve_early.return_value = (None, [])

    with patch("anthropic.Anthropic", return_value=client_mock):
        dispatcher.submit(
            "run-tele",
            adapters=[adapter],
            claims_with_evidence=[(claim, [])],
            transcript_meta={"speaker": "T", "date": "2026-02-24"},
            inject_evidence=False,
        )
        reconcile_run(
            metrics_dir,
            "run-tele",
            adapters_by_name={"anthropic": adapter},
            engine=engine,
        )

    log_path = metrics_dir / "adapter_calls.jsonl"
    assert log_path.exists()
    rows = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tele_rows = [r for r in rows if r.get("run_id") == "run-tele"]
    assert tele_rows, "expected at least one telemetry row for the reconcile run"
    assert all(r["mode"] == "batch" for r in tele_rows)
    assert all(r["batch_job_id"] == "msgbatch_real_id" for r in tele_rows)


def test_parse_provider_results_unknown_custom_id_is_dropped() -> None:
    adapter = _FakeAnthropicAdapter()
    claim = _claim()
    rows = [{"custom_id": "unknown", "status": "succeeded", "message": {}}]
    out = parse_provider_results(
        "anthropic", rows, adapter, claim_by_id={claim.id: claim}, custom_id_to_claim={}
    )
    assert out == []


def test_parse_provider_results_errored_row_becomes_unverifiable() -> None:
    adapter = _FakeAnthropicAdapter()
    claim = _claim()
    custom_map = {"anthropic::abc": claim.id}
    rows = [{"custom_id": "anthropic::abc", "status": "errored", "error": "boom"}]
    out = parse_provider_results(
        "anthropic", rows, adapter, claim_by_id={claim.id: claim}, custom_id_to_claim=custom_map
    )
    assert len(out) == 1
    assert out[0].label == VerdictLabel.UNVERIFIABLE
    assert out[0].no_response is True


# ── Multi-claim round-trip (claims_per_request > 1) ──────────────────────────


class _FakeAnthropicMultiAdapter(_FakeAnthropicAdapter):
    """Adapter that supports both single and multi-claim payloads."""

    max_claims_per_request = 8

    def build_multi_batch_payload(
        self, claims, evidence_by_claim, *, inject_evidence=True, max_evidence_per_claim=5
    ):
        return {
            "model": self.model_id,
            "max_tokens": 1024 + 1024 * len(claims),
            "system": [{"type": "text", "text": "sys"}],
            "messages": [
                {
                    "role": "user",
                    "content": "\n".join(
                        f"claim_id={c.id}: {c.text}" for c in claims
                    ),
                }
            ],
        }

    def parse_multi_batch_response(self, raw, claims, *, batch_call_id=""):
        from truthbot.verify.adapters.base import (
            build_multi_verdicts,
            parse_multi_claim_json,
        )

        text_blocks = [b for b in raw.get("content", []) if b.get("type") == "text"]
        text = text_blocks[0]["text"] if text_blocks else ""
        try:
            raw_by_claim = parse_multi_claim_json(text, claims)
        except json.JSONDecodeError:
            raw_by_claim = {}
        usage = raw.get("usage", {})
        return build_multi_verdicts(
            claims,
            raw_by_claim,
            adapter_name=self.adapter_name,
            model_id=raw.get("model", self.model_id),
            call_usage={
                "input_tokens": int(usage.get("input_tokens", 0) or 0),
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
                "cached_input_tokens": int(
                    usage.get("cache_read_input_tokens", 0) or 0
                ),
            },
            batch_call_id=batch_call_id,
        )


def test_submit_and_reconcile_multi_claim_round_trip(tmp_path, monkeypatch) -> None:
    """Submit 9 claims with claims_per_request=5 → expect 2 chunks (5 + 4)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("TRUTHBOT_METRICS_DIR", str(tmp_path / "metrics"))
    import truthbot.metrics.telemetry as tel

    tel._telemetry_instance = None

    adapter = _FakeAnthropicMultiAdapter()
    claims = [_claim(f"Claim #{i}") for i in range(9)]
    claims_with_evidence = [(c, []) for c in claims]

    captured_requests: dict[str, list] = {}

    fake_batch = SimpleNamespace(id="msgbatch_multi_rt")
    counts_all_done = SimpleNamespace(
        succeeded=2, errored=0, canceled=0, expired=0, processing=0
    )
    retrieved = SimpleNamespace(
        processing_status="ended", request_counts=counts_all_done
    )

    def _mk_multi_result(custom_id, chunk_claims):
        verdicts_json = json.dumps(
            [
                {
                    "claim_id": c.id,
                    "label": "True" if i % 2 == 0 else "False",
                    "confidence": "High",
                    "explanation": f"{c.id} verdict",
                }
                for i, c in enumerate(chunk_claims)
            ]
        )
        return SimpleNamespace(
            custom_id=custom_id,
            result=SimpleNamespace(
                type="succeeded",
                message={
                    "content": [{"type": "text", "text": verdicts_json}],
                    "model": "claude-opus-4-7",
                    "usage": {
                        "input_tokens": 2000,
                        "output_tokens": 500,
                        "cache_read_input_tokens": 128,
                    },
                },
            ),
        )

    client_mock = MagicMock()
    client_mock.messages.batches.create.return_value = fake_batch
    client_mock.messages.batches.retrieve.return_value = retrieved

    def _capture_create(*args, **kwargs):
        captured_requests["requests"] = kwargs.get("requests") or (args[0] if args else [])
        return fake_batch

    client_mock.messages.batches.create.side_effect = _capture_create

    metrics_dir = tmp_path / "metrics"
    dispatcher = BatchDispatcher(metrics_dir)

    with patch("anthropic.Anthropic", return_value=client_mock):
        dispatcher.submit(
            "run-multi",
            adapters=[adapter],
            claims_with_evidence=claims_with_evidence,
            transcript_meta={
                "speaker": "Test",
                "date": "2026-02-24",
                "triaged_claim_ids": [],
                "triaged_claims": [],
            },
            inject_evidence=False,
            claims_per_request=5,
        )

        # Exactly two requests: chunk of 5 + chunk of 4 (not 9 single-claim requests).
        reqs = captured_requests["requests"]
        assert len(reqs) == 2, f"expected 2 chunked requests, got {len(reqs)}"

        desc = read_batch_job(metrics_dir, "run-multi")
        provider_entry = desc["provider_jobs"]["anthropic"]
        assert provider_entry["chunk_size"] == 5
        assert provider_entry.get("custom_id_to_claim") in (None, {})
        multi_map = provider_entry["custom_id_to_claims"]
        assert len(multi_map) == 2
        chunk_sizes = sorted(len(v) for v in multi_map.values())
        assert chunk_sizes == [4, 5]

        # Now arrange results to return one row per chunk.
        multi_custom_ids = list(multi_map.keys())
        chunk_by_cid: dict[str, list] = {
            cid: [c for c in claims if c.id in multi_map[cid]]
            for cid in multi_custom_ids
        }
        client_mock.messages.batches.results.return_value = iter(
            [_mk_multi_result(cid, chunk_by_cid[cid]) for cid in multi_custom_ids]
        )

        class _FakeEngine:
            def finalize_bundle(
                self, claim, speaker, date_str, model_verdicts, evidence_count=0
            ):
                from truthbot.models import ConsensusVerdict, VerdictBundle

                lbl = (
                    model_verdicts[0].label
                    if model_verdicts
                    else VerdictLabel.UNVERIFIABLE
                )
                cons = ConsensusVerdict(
                    claim_id=claim.id,
                    model_verdicts=model_verdicts,
                    consensus_label=lbl,
                    confidence=Confidence.HIGH,
                    agreement=True,
                    consensus_strength="single",
                    explanation="fake",
                )
                return VerdictBundle(
                    claim=claim,
                    speaker=speaker,
                    date_str=date_str,
                    model_verdicts=model_verdicts,
                    consensus=cons,
                    evidence_count=evidence_count,
                )

            def maybe_resolve_early(self, claim, speaker="", date_str=""):
                return None, []

        result = reconcile_run(
            metrics_dir,
            "run-multi",
            adapters_by_name={"anthropic": adapter},
            engine=_FakeEngine(),
        )

    assert result["status"] == "complete"
    # 9 bundles out — every claim accounted for.
    assert len(result["bundles"]) == 9

    # Telemetry: 9 rows (one per verdict); claim_count reflects chunk sizes.
    log_path = metrics_dir / "adapter_calls.jsonl"
    rows = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    multi_rows = [r for r in rows if r.get("run_id") == "run-multi"]
    assert len(multi_rows) == 9
    # One chunk carries cache_read_input_tokens on the index-0 verdict only.
    cached_rows = [r for r in multi_rows if r.get("cache_read_input_tokens", 0) > 0]
    # Exactly one index-0 per chunk (2 chunks → 2 rows with cached tokens).
    assert len(cached_rows) == 2
    claim_counts = sorted({r["claim_count"] for r in multi_rows})
    assert claim_counts == [4, 5]

    # Usage attribution: each chunk's input/output tokens are recorded on the
    # index-0 verdict row only, so costs are billed once per batched API call
    # (not N-times). Siblings must carry zero tokens / zero cost.
    index_zero_rows = [r for r in multi_rows if r.get("batch_call_index") == 0]
    assert len(index_zero_rows) == 2
    for r in index_zero_rows:
        assert r["input_tokens"] == 2000, r
        assert r["output_tokens"] == 500, r
        assert r["cache_read_input_tokens"] == 128, r
        assert r["estimated_cost_usd"] > 0, r
    sibling_rows = [r for r in multi_rows if r.get("batch_call_index", 0) > 0]
    assert len(sibling_rows) == 7
    for r in sibling_rows:
        assert r["input_tokens"] == 0, r
        assert r["output_tokens"] == 0, r
        assert r["cache_read_input_tokens"] == 0, r
        assert r["estimated_cost_usd"] == 0, r


def test_submit_multi_claim_falls_back_to_single_when_adapter_cap_is_one(
    tmp_path, monkeypatch
) -> None:
    """If the adapter's max_claims_per_request is 1, multi mode degrades to singles."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    class _SingleOnlyAdapter(_FakeAnthropicAdapter):
        max_claims_per_request = 1

    adapter = _SingleOnlyAdapter()
    claims = [_claim(f"c{i}") for i in range(3)]
    claims_with_evidence = [(c, []) for c in claims]

    captured: dict[str, list] = {}
    fake_batch = SimpleNamespace(id="msgbatch_clamp")
    client_mock = MagicMock()

    def _capture_create(*args, **kwargs):
        captured["requests"] = kwargs.get("requests") or (args[0] if args else [])
        return fake_batch

    client_mock.messages.batches.create.side_effect = _capture_create

    metrics_dir = tmp_path / "metrics"
    dispatcher = BatchDispatcher(metrics_dir)
    with patch("anthropic.Anthropic", return_value=client_mock):
        dispatcher.submit(
            "run-clamp",
            adapters=[adapter],
            claims_with_evidence=claims_with_evidence,
            transcript_meta={"speaker": "T", "date": "2026-02-24"},
            inject_evidence=False,
            claims_per_request=8,
        )

    # Adapter cap of 1 means 3 single-claim requests, not 1 big chunk.
    assert len(captured["requests"]) == 3
    desc = read_batch_job(metrics_dir, "run-clamp")
    entry = desc["provider_jobs"]["anthropic"]
    assert entry["chunk_size"] == 1
    assert entry.get("custom_id_to_claims") in (None, {})
    assert len(entry["custom_id_to_claim"]) == 3


def test_sidecar_roundtrip(tmp_path) -> None:
    """Sidecar writer/reader must preserve ModelVerdict JSON."""
    from truthbot.verify.batch import _append_sidecar, load_sidecar

    p = tmp_path / "metrics" / "batch_sidecar" / "run-s.jsonl"
    mv = ModelVerdict(
        adapter_name="xai",
        model_id="grok-4",
        claim_id="c-1",
        label=VerdictLabel.TRUE,
        confidence=Confidence.HIGH,
        explanation="grok sidecar",
    )
    _append_sidecar(p, mv)
    loaded = load_sidecar(p)
    assert len(loaded) == 1
    assert loaded[0].adapter_name == "xai"
    assert loaded[0].label == VerdictLabel.TRUE
