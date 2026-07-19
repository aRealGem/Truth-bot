"""PCA replay-artifact persistence + offline re-bridge round-trip.

A live PCA publish is ~1hr of sequential proxy calls. If a render change needs a
re-publish, we must NOT need another live run — so `_persist_pca_run` writes the raw
adjudication rows + claim dicts (with Layer A provenance) to disk, and re-bridging
that artifact offline must reproduce the bundles, provenance and all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from truthbot.models import Evidence, SourceTier
from truthbot.pipeline import _persist_pca_run
from truthbot.verdict import bridge


@dataclass
class _FakeResult:
    """Stand-in for PcaVerifyResult carrying just the persisted fields."""
    rows: list = field(default_factory=list)
    claims: list = field(default_factory=list)
    characterization: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    roster: dict | None = None


def _resolved_row(sid, verdict, votes, **extra):
    row = {
        "sid": sid, "status": "resolved", "verdict": verdict,
        "confidence": 0.9, "citations": [], "reasoning": "because",
        "votes": votes, "split": False, "escalated": False,
    }
    row.update(extra)
    return row


def _claim(sid, source):
    return {"sid": sid, "text": f"Claim {sid}.",
            "layer_a": {"label": "check-worthy", "source": source}}


class TestPersistPcaRun:
    def test_writes_artifact_with_rows_claims_and_meta(self, tmp_path):
        result = _FakeResult(
            rows=[_resolved_row("s:0", "TRUE", {"TRUE": 2, "FALSE": 1})],
            claims=[_claim("s:0", "A2")],
            characterization=[{"sid": "s:1", "label": "non-check-worthy"}],
        )
        path = _persist_pca_run(
            "run-1", result,
            meta={"speaker": "Tester", "speech_id": "spx"},
            metrics_dir=tmp_path,
        )
        assert path == tmp_path / "pca_runs" / "run-1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["run_id"] == "run-1"
        assert payload["meta"]["speaker"] == "Tester"
        assert len(payload["rows"]) == 1
        assert payload["claims"][0]["layer_a"]["source"] == "A2"
        assert payload["characterization"][0]["label"] == "non-check-worthy"

    def test_offline_rebridge_round_trip_reproduces_bundles_and_provenance(self, tmp_path):
        rows = [
            _resolved_row("s:0", "FALSE", {"MISLEADING": 2, "FALSE": 1},
                          crm114={"stage1": "MISLEADING", "final": "FALSE"}),
            {"sid": "s:1", "status": "disagreement", "verdict": None,
             "confidence": None, "citations": [], "reasoning": "",
             "votes": {"TRUE": 1, "FALSE": 1}, "split": True, "escalated": True},
        ]
        claims = [_claim("s:0", "A2"), _claim("s:1", "A1")]
        result = _FakeResult(rows=rows, claims=claims)

        path = _persist_pca_run("run-rt", result, meta={}, metrics_dir=tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        # Re-bridge straight from the persisted artifact — no LLM, no packs.
        out = bridge.bridge(payload["rows"], payload["claims"])
        assert [b.consensus.claim_id for b in out.bundles] == ["s:0", "s:1"]

        resolved = out.bundles[0].consensus
        assert resolved.consensus_verdict == "False"
        assert resolved.provenance.panel_votes == {"Misleading": 2, "False": 1}
        assert resolved.provenance.layer_a_source == "A2"
        assert resolved.provenance.crm114_final == "FALSE"

        split = out.bundles[1].consensus
        assert split.consensus_verdict == "Models split"
        assert split.provenance.panel_split is True
        assert split.provenance.panel_votes == {"True": 1, "False": 1}

    def test_persists_evidence_pack(self, tmp_path):
        result = _FakeResult(
            rows=[_resolved_row("s:0", "TRUE", {"TRUE": 3})],
            claims=[_claim("s:0", "A2")],
            evidence={
                "s:0": [
                    Evidence(
                        claim_id="s:0", source_name="BLS",
                        source_url="https://bls.gov/a",
                        source_tier=SourceTier.GOVERNMENT, snippet="gov snippet",
                    ),
                    Evidence(
                        claim_id="s:0", source_name="AP",
                        source_url="https://ap.org/b",
                        source_tier=SourceTier.WIRE, snippet="wire snippet",
                    ),
                ]
            },
        )
        path = _persist_pca_run("run-ev", result, meta={}, metrics_dir=tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "evidence" in payload
        evs = payload["evidence"]["s:0"]
        assert len(evs) == 2
        assert [e["source_url"] for e in evs] == [
            "https://bls.gov/a", "https://ap.org/b"]
        assert evs[0]["source_name"] == "BLS"
        assert evs[0]["source_tier"] == "Government"

    def test_persists_panel_roster(self, tmp_path):
        # The PCA panel composition (which model fills each seat) is a per-run
        # fact and must survive into the replay artifact for offline re-render.
        result = _FakeResult(
            rows=[_resolved_row("s:0", "TRUE", {"TRUE": 3})],
            claims=[_claim("s:0", "A2")],
            roster={"name": "dev", "seats": {
                "proposer": ["mistral"], "critic": ["dsv4-flash"],
                "arbiter": ["claude-haiku"]}},
        )
        path = _persist_pca_run("run-roster", result, meta={}, metrics_dir=tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["roster"]["name"] == "dev"
        assert payload["roster"]["seats"]["proposer"] == ["mistral"]
        assert payload["roster"]["seats"]["arbiter"] == ["claude-haiku"]

    def test_persist_roster_absent_is_null(self, tmp_path):
        # Legacy-clean: a result with no roster persists a null (not a crash).
        path = _persist_pca_run("run-noroster", _FakeResult(), meta={}, metrics_dir=tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["roster"] is None

    def test_persistence_failure_does_not_raise(self, tmp_path, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("pathlib.Path.mkdir", boom)
        path = _persist_pca_run("run-err", _FakeResult(), meta={}, metrics_dir=tmp_path)
        assert isinstance(path, Path)
