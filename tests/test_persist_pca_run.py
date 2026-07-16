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

from truthbot.pipeline import _persist_pca_run
from truthbot.verdict import bridge


@dataclass
class _FakeResult:
    """Stand-in for PcaVerifyResult carrying just the persisted fields."""
    rows: list = field(default_factory=list)
    claims: list = field(default_factory=list)
    characterization: list = field(default_factory=list)


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

    def test_persistence_failure_does_not_raise(self, tmp_path, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("pathlib.Path.mkdir", boom)
        path = _persist_pca_run("run-err", _FakeResult(), meta={}, metrics_dir=tmp_path)
        assert isinstance(path, Path)
