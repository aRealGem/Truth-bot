"""Integration TB-PIPELINE: End-to-end pipeline with mocked externals."""

from __future__ import annotations

import pytest

from truthbot.pipeline import Pipeline


class DummyExtractor:
    def extract(self, transcript):
        from truthbot.models import Claim
        return [
            Claim(
                transcript_id=transcript.id,
                text="Unemployment is at a 50-year low.",
                speaker=transcript.speaker,
                context="Unemployment is at a 50-year low.",
                category="economy",
                is_checkable=True,
            )
        ]


class DummyEngine:
    def verify(self, claim):
        from truthbot.models import Confidence, Evidence, SourceTier, Verdict, VerdictLabel
        evidence = [
            Evidence(
                claim_id=claim.id,
                source_name="BLS",
                source_url="https://bls.gov",
                source_tier=SourceTier.GOVERNMENT,
                snippet="Data snippet",
                supports_claim=True,
            )
        ]
        verdict = Verdict(
            claim_id=claim.id,
            label=VerdictLabel.TRUE,
            confidence=Confidence.HIGH,
            explanation="BLS data confirms the claim.",
        )
        return evidence, verdict

    def verify_many(self, claims):
        return [
            (claim, *self.verify(claim))  # type: ignore[misc]
            for claim in claims
        ]


@pytest.fixture
def pipeline(tmp_dir, monkeypatch):
    pipeline = Pipeline(output_dir=tmp_dir, base_url="https://example.com", post_bluesky=False)
    pipeline.extractor = DummyExtractor()
    pipeline.engine = DummyEngine()
    # Cache: use temporary dir to avoid cross-test interference
    from truthbot.cache.claims import ClaimCache
    pipeline.cache = ClaimCache(cache_dir=tmp_dir)
    return pipeline


class TestPipeline:
    def test_run_creates_report(self, pipeline):
        report = pipeline.run("Unemployment is at a 50-year low.", speaker="Tester")
        assert report.transcript.speaker == "Tester"
        assert report.claims
        assert report.verdicts
        assert report.report_url

    def test_run_writes_html(self, pipeline, tmp_dir):
        report = pipeline.run("Jobs boom.", speaker="Tester")
        html_path = tmp_dir / f"{report.id}.html"
        assert html_path.exists()

    def test_cache_hit_skips_engine(self, pipeline, monkeypatch):
        # First run populates cache
        pipeline.run("Inflation is down.", speaker="Tester")
        # Monkeypatch engine to ensure cache is used on second run
        pipeline.engine.verify = lambda claim: (_ for _ in ()).throw(RuntimeError("Should not run"))
        report = pipeline.run("Inflation is down.", speaker="Tester")
        assert report.verdicts

    def test_cli_entrypoint_help(self, tmp_path, monkeypatch, capsys):
        import truthbot.pipeline as pipeline_module

        # Simulate running CLI with missing args → prints help and exits
        with pytest.raises(SystemExit):
            pipeline_module.main()

        captured = capsys.readouterr()
        assert "usage" in captured.err or "usage" in captured.out
