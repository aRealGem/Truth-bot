"""Regression test for extracted-claim persistence.

The 2026-04-23 SOTU run burned 31 minutes of live-triage + Gemini errors and
the extracted claims were only stored in the batch-job descriptor, which
itself was only written *after* the provider batch submission succeeded. When
submit failed mid-flight, every extracted claim was lost. Re-running a SOTU
that retains all 108 claims requires a disk artifact written the instant
``ClaimExtractor.extract()`` returns.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from truthbot.models import Claim
from truthbot.pipeline import _persist_extracted_claims


def _mk_claim(text: str, transcript_id: str = "t1", **kwargs) -> Claim:
    return Claim(transcript_id=transcript_id, text=text, speaker="Tester", **kwargs)


class TestPersistExtractedClaims:
    def test_writes_jsonl_with_one_line_per_claim(self, tmp_path):
        claims = [_mk_claim("alpha"), _mk_claim("beta"), _mk_claim("gamma")]
        path = _persist_extracted_claims(
            "run-abc", claims, metrics_dir=tmp_path
        )
        assert path == tmp_path / "extractions" / "run-abc.jsonl"
        assert path.exists()

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        parsed = [json.loads(line) for line in lines]
        assert [p["text"] for p in parsed] == ["alpha", "beta", "gamma"]
        # Required claim-schema fields survive the round-trip
        for p in parsed:
            assert "id" in p
            assert "transcript_id" in p
            assert "speaker" in p
            assert "text" in p
            assert "is_checkable" in p

    def test_round_trip_via_pydantic(self, tmp_path):
        """Each persisted row parses back into a Claim without data loss."""
        original = [_mk_claim("alpha"), _mk_claim("beta", category="economy")]
        path = _persist_extracted_claims("run-rt", original, metrics_dir=tmp_path)
        restored = [
            Claim.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        assert [c.text for c in restored] == [c.text for c in original]
        assert [c.id for c in restored] == [c.id for c in original]
        assert restored[1].category == "economy"

    def test_empty_claim_list_writes_empty_file(self, tmp_path):
        path = _persist_extracted_claims("run-empty", [], metrics_dir=tmp_path)
        assert path.exists()
        assert path.read_text(encoding="utf-8") == ""

    def test_creates_nested_extractions_dir(self, tmp_path):
        """metrics/ root may exist but extractions/ may not yet."""
        metrics_root = tmp_path / "metrics"
        assert not (metrics_root / "extractions").exists()
        _persist_extracted_claims("run-mk", [_mk_claim("x")], metrics_dir=metrics_root)
        assert (metrics_root / "extractions").is_dir()
        assert (metrics_root / "extractions" / "run-mk.jsonl").is_file()

    def test_persistence_failure_does_not_raise(self, tmp_path, monkeypatch):
        """A filesystem error must not crash the pipeline post-extraction."""

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("pathlib.Path.mkdir", boom)

        path = _persist_extracted_claims(
            "run-err", [_mk_claim("x")], metrics_dir=tmp_path
        )
        assert isinstance(path, Path)
