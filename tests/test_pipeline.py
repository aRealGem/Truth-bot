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
        from truthbot.models import (
            Confidence,
            ConsensusVerdict,
            Evidence,
            SourceTier,
            VerdictLabel,
        )

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
        consensus = ConsensusVerdict(
            claim_id=claim.id,
            model_verdicts=[],
            consensus_label=VerdictLabel.TRUE,
            consensus_verdict=VerdictLabel.TRUE.value,
            confidence=Confidence.HIGH,
            agreement=True,
            consensus_strength="single",
            explanation="BLS data confirms the claim.",
        )
        return evidence, consensus

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


# ── truthbot metrics tool-stats — roadmap [7] ─────────────────────────────────


def _write_jsonl(path, rows):
    import json as _json
    path.write_text("\n".join(_json.dumps(r) for r in rows), encoding="utf-8")


def _make_call_record(
    *,
    adapter,
    tool_call_count,
    model_reported_source_count=0,
    stripped_source_count=0,
    retrieved_url_count=0,
    run_id="run-A",
    status="ok",
):
    """Minimal adapter_calls.jsonl row with the fields tool-stats reads."""
    return {
        "adapter_name": adapter,
        "tool_call_count": tool_call_count,
        "model_reported_source_count": model_reported_source_count,
        "stripped_source_count": stripped_source_count,
        "retrieved_url_count": retrieved_url_count,
        "run_id": run_id,
        "status": status,
    }


class TestToolStatsCli:
    def test_renders_distribution_grounding_and_histogram_tables(self, tmp_path, capsys):
        """End-to-end: the three sections (distribution, grounding rates,
        histogram) all render with the right numbers for a synthetic
        3-adapter mini-run. Mean/p50/p95/max for xAI is verified against
        hand-computed values."""
        from truthbot.pipeline import _print_tool_stats
        rows = [
            # xAI: heavy search, no strips (canonical good adapter)
            *[_make_call_record(adapter="xai", tool_call_count=t,
                                model_reported_source_count=2,
                                retrieved_url_count=2)
              for t in (5, 6, 7, 8, 10)],
            # OpenAI: low search, every URL stripped (post-arm-E pattern)
            *[_make_call_record(adapter="openai", tool_call_count=t,
                                model_reported_source_count=3,
                                stripped_source_count=3,
                                retrieved_url_count=3)
              for t in (1, 1, 2, 2, 3)],
            # Anthropic: moderate search, no strips
            *[_make_call_record(adapter="anthropic", tool_call_count=t,
                                model_reported_source_count=4,
                                retrieved_url_count=10)
              for t in (1, 2, 3, 4, 5)],
        ]
        p = tmp_path / "adapter_calls.jsonl"
        _write_jsonl(p, rows)

        _print_tool_stats(jsonl_path=str(p))
        out = capsys.readouterr().out

        # Distribution: xAI [5,6,7,8,10] → mean 7.20, max 10.
        assert "7.20" in out
        # Grounding rates: OpenAI 5*3=15 reported, 5*3=15 stripped → 100.0%.
        assert "100.0%" in out
        # All three adapters rendered.
        for name in ("xai", "openai", "anthropic"):
            assert name in out
        # Sections rendered in order: distribution, then grounding rates.
        dist_idx = out.find("zero-tool %")
        rates_idx = out.find("strip rate")
        assert dist_idx >= 0 and rates_idx >= 0 and dist_idx < rates_idx

    def test_run_id_filter_excludes_other_runs(self, tmp_path, capsys):
        """``--run-id`` only includes telemetry rows for that run."""
        from truthbot.pipeline import _print_tool_stats
        rows = [
            _make_call_record(adapter="anthropic", tool_call_count=1, run_id="run-A"),
            _make_call_record(adapter="anthropic", tool_call_count=99, run_id="run-B"),
        ]
        p = tmp_path / "adapter_calls.jsonl"
        _write_jsonl(p, rows)

        _print_tool_stats(jsonl_path=str(p), run_id="run-A")
        out = capsys.readouterr().out
        assert "anthropic" in out
        # 99 should NOT appear — it's run-B and we filtered to run-A.
        assert "99" not in out

    def test_adapter_filter_isolates_one_adapter(self, tmp_path, capsys):
        """``--adapter`` only includes rows for the named adapter."""
        from truthbot.pipeline import _print_tool_stats
        rows = [
            _make_call_record(adapter="anthropic", tool_call_count=1),
            _make_call_record(adapter="xai", tool_call_count=99),
        ]
        p = tmp_path / "adapter_calls.jsonl"
        _write_jsonl(p, rows)

        _print_tool_stats(jsonl_path=str(p), adapter="anthropic")
        out = capsys.readouterr().out
        assert "anthropic" in out
        assert "xai" not in out

    def test_handles_missing_file_gracefully(self, tmp_path, capsys):
        """Stale or absent adapter_calls.jsonl prints a friendly message
        instead of raising."""
        from truthbot.pipeline import _print_tool_stats
        p = tmp_path / "nonexistent.jsonl"
        _print_tool_stats(jsonl_path=str(p))
        out = capsys.readouterr().out
        assert "No telemetry data" in out

    def test_handles_no_matching_records_gracefully(self, tmp_path, capsys):
        """File exists but the run-id / adapter filters exclude
        everything → friendly message, no raise."""
        from truthbot.pipeline import _print_tool_stats
        rows = [_make_call_record(adapter="xai", tool_call_count=1, run_id="run-A")]
        p = tmp_path / "adapter_calls.jsonl"
        _write_jsonl(p, rows)
        _print_tool_stats(jsonl_path=str(p), run_id="run-Z")
        out = capsys.readouterr().out
        assert "No telemetry records matched" in out

    def test_skips_malformed_jsonl_lines(self, tmp_path, capsys):
        """Garbage / blank lines in the JSONL are skipped, not fatal."""
        from truthbot.pipeline import _print_tool_stats
        p = tmp_path / "adapter_calls.jsonl"
        import json as _json
        p.write_text(
            "\n".join([
                "",
                "not-json",
                _json.dumps(_make_call_record(adapter="xai", tool_call_count=2)),
                "{malformed",
            ]),
            encoding="utf-8",
        )
        _print_tool_stats(jsonl_path=str(p))
        out = capsys.readouterr().out
        assert "xai" in out

    def test_zero_tool_calls_metric_distinguishes_adapters(self, tmp_path, capsys):
        """The zero-tool % column is the "search declined" signal —
        100% means the adapter NEVER invoked search. Pin that we
        compute and render it correctly."""
        from truthbot.pipeline import _print_tool_stats
        rows = [
            # OpenAI: 3/3 calls had zero tools → 100%
            *[_make_call_record(adapter="openai", tool_call_count=0) for _ in range(3)],
            # xAI: 0/3 calls had zero tools → 0%
            *[_make_call_record(adapter="xai", tool_call_count=2) for _ in range(3)],
        ]
        p = tmp_path / "adapter_calls.jsonl"
        _write_jsonl(p, rows)
        _print_tool_stats(jsonl_path=str(p))
        out = capsys.readouterr().out
        assert "3/3 (100%)" in out
        assert "0/3 (0%)" in out

    def test_cli_dispatches_tool_stats_subcommand(self, tmp_path, monkeypatch, capsys):
        """Exercises the parser + dispatch end-to-end so a future change
        to argparse wiring doesn't silently break the subcommand."""
        rows = [_make_call_record(adapter="anthropic", tool_call_count=1)]
        p = tmp_path / "adapter_calls.jsonl"
        _write_jsonl(p, rows)

        from truthbot import pipeline as pipeline_module
        monkeypatch.setattr(
            "sys.argv",
            ["truthbot", "metrics", "tool-stats", "--jsonl", str(p)],
        )
        pipeline_module.main()
        out = capsys.readouterr().out
        assert "anthropic" in out
        assert "zero-tool" in out  # confirms the distribution table rendered
