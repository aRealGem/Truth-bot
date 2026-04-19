"""
End-to-end pipeline orchestrator.

Ties together all modules in sequence:
  Ingest → Extract Claims → Verify (Evidence + Verdicts) → Score → Publish

Can be run from the CLI via `truthbot` or called programmatically.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from truthbot.models import Report, Verdict

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Orchestrates the full fact-checking pipeline.

    Parameters
    ----------
    connectors:
        Optional list of source connectors (overrides defaults).
    cache_dir:
        Optional cache directory path.
    output_dir:
        Optional output directory for HTML/RSS reports.
    base_url:
        Base URL for published report links.
    post_bluesky:
        Whether to post results to Bluesky.
    dry_run:
        If True, skip all external API calls and use stubs throughout.
    """

    def __init__(
        self,
        connectors=None,
        cache_dir: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
        base_url: str = "https://example.com",
        post_bluesky: bool = False,
        dry_run: bool = False,
    ) -> None:
        self._connectors = connectors
        self._cache_dir = cache_dir
        self._output_dir = output_dir
        self._base_url = base_url
        self._post_bluesky = post_bluesky
        self._dry_run = dry_run

        self._setup_components()

    def _setup_components(self) -> None:
        """Instantiate all pipeline components."""
        from truthbot.cache.claims import ClaimCache
        from truthbot.extract.claims import ClaimExtractor
        from truthbot.ingest.transcript import TranscriptIngester
        from truthbot.publish.api import ReportAPI
        from truthbot.publish.bluesky import BlueskyPublisher
        from truthbot.publish.cards import CardRenderer
        from truthbot.publish.rss import RSSPublisher
        from truthbot.publish.web import WebPublisher
        from truthbot.scoring.rubric import ScoringRubric
        from truthbot.verify.engine import VerificationEngine

        self.ingester = TranscriptIngester()
        self.extractor = ClaimExtractor()
        self.engine = VerificationEngine(connectors=self._connectors)
        self.rubric = ScoringRubric()
        self.cache = ClaimCache(cache_dir=self._cache_dir)
        self.web = WebPublisher(output_dir=self._output_dir, base_url=self._base_url)
        self.rss = RSSPublisher(output_dir=self._output_dir)
        self.cards = CardRenderer(output_dir=self._output_dir, base_url=self._base_url)
        self.bluesky = BlueskyPublisher()
        self.api = ReportAPI()

    def run(
        self,
        source: str | Path,
        speaker: str = "Unknown",
        date: Optional[datetime] = None,
        venue: Optional[str] = None,
    ) -> Report:
        """
        Run the full pipeline on a transcript source.

        Parameters
        ----------
        source:
            URL, file path, or raw transcript text.
        speaker:
            Speaker name or title.
        date:
            Speech date.
        venue:
            Venue or event name.

        Returns
        -------
        Report
            The completed fact-check report.
        """
        logger.info("Pipeline starting for speaker: %s", speaker)

        # 1. Ingest
        ingest_result = self.ingester.ingest(source, speaker=speaker, date=date, venue=venue)
        for w in ingest_result.warnings:
            logger.warning("Ingest warning: %s", w)
        transcript = ingest_result.transcript
        logger.info("Ingested transcript: %d words", transcript.word_count)

        # 2. Extract claims
        claims = self.extractor.extract(transcript)
        logger.info("Extracted %d claims", len(claims))

        # 3. Verify each claim (with cache check)
        all_evidence = []
        all_verdicts = []

        for claim in claims:
            if not claim.is_checkable:
                logger.debug("Skipping non-checkable claim: %s", claim.text[:60])
                continue

            # Check cache first
            cached = self.cache.get(claim.text)
            if cached:
                from truthbot.models import Confidence, VerdictLabel
                verdict = Verdict(
                    claim_id=claim.id,
                    label=VerdictLabel(cached.verdict_label),
                    confidence=Confidence(cached.confidence),
                    explanation=cached.explanation + " [from cache]",
                )
                all_verdicts.append(verdict)
                continue

            # Verify
            evidence, consensus = self.engine.verify(claim)
            all_evidence.extend(evidence)
            # Build backward-compat Verdict from consensus
            verdict = Verdict(
                claim_id=claim.id,
                label=consensus.consensus_label,
                confidence=consensus.confidence,
                explanation=consensus.explanation,
                model_id="consensus",
                evidence_ids=[e.id for e in evidence],
            )
            all_verdicts.append(verdict)

            # Cache the result
            self.cache.put(
                claim_text=claim.text,
                verdict_label=verdict.label.value,
                confidence=verdict.confidence.value,
                explanation=verdict.explanation,
                evidence_urls=[e.source_url for e in evidence],
            )

        # 4. Build report
        report = Report(
            transcript=transcript,
            claims=claims,
            evidence=all_evidence,
            verdicts=all_verdicts,
        )

        # 5. Publish
        self._publish(report)
        logger.info("Pipeline complete. Report ID: %s", report.id)
        return report

    def _publish(self, report: Report) -> None:
        """Publish a completed report to all configured outputs."""
        # HTML
        try:
            html_path = self.web.write_report(report)
            report.report_url = f"{self._base_url}/reports/{report.id}.html"
        except Exception as exc:
            logger.error("HTML publish failed: %s", exc)

        # RSS
        try:
            self.rss.write_feed([report])
        except Exception as exc:
            logger.error("RSS publish failed: %s", exc)

        # Bluesky
        if self._post_bluesky:
            try:
                url = self.bluesky.post_report(report)
                if url:
                    report.bluesky_thread_url = url
            except Exception as exc:
                logger.error("Bluesky publish failed: %s", exc)

        # Register with API
        self.api.add_report(report)

        from datetime import timezone
        report.published_at = datetime.now(timezone.utc)


def _print_metrics_summary(jsonl_path: str | None = None) -> None:
    """Read adapter_calls.jsonl and print a per-adapter summary table."""
    import json
    from collections import defaultdict
    from pathlib import Path

    if jsonl_path:
        path = Path(jsonl_path)
    else:
        from truthbot.config import settings
        path = settings.metrics_dir / "adapter_calls.jsonl"

    if not path.exists():
        print("No telemetry data found.")
        return

    # Aggregate per adapter
    stats: dict[str, dict] = defaultdict(lambda: {
        "calls": 0, "ok": 0, "errors": 0,
        "input_tokens": 0, "output_tokens": 0,
        "tool_calls": 0, "urls": 0,
        "total_cost": 0.0, "total_ms": 0,
    })

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = rec.get("adapter_name", "unknown")
            s = stats[name]
            s["calls"] += 1
            if rec.get("status") == "ok":
                s["ok"] += 1
            else:
                s["errors"] += 1
            s["input_tokens"] += rec.get("input_tokens", 0)
            s["output_tokens"] += rec.get("output_tokens", 0)
            s["tool_calls"] += rec.get("tool_call_count", 0)
            s["urls"] += rec.get("retrieved_url_count", 0)
            s["total_cost"] += rec.get("estimated_cost_usd", 0.0)
            s["total_ms"] += rec.get("wall_clock_ms", 0)

    if not stats:
        print("No telemetry records found in file.")
        return

    print(f"\n{'Adapter':<14} {'Calls':>6} {'OK':>6} {'Errors':>7} "
          f"{'In Tok':>8} {'Out Tok':>8} {'Tools':>6} {'URLs':>6} "
          f"{'Cost $':>9} {'Avg ms':>8}")
    print("-" * 90)

    for name, s in sorted(stats.items()):
        avg_ms = s["total_ms"] // max(s["calls"], 1)
        print(
            f"{name:<14} {s['calls']:>6} {s['ok']:>6} {s['errors']:>7} "
            f"{s['input_tokens']:>8} {s['output_tokens']:>8} {s['tool_calls']:>6} {s['urls']:>6} "
            f"{s['total_cost']:>9.6f} {avg_ms:>8}"
        )

    total_cost = sum(s["total_cost"] for s in stats.values())
    total_calls = sum(s["calls"] for s in stats.values())
    print("-" * 90)
    print(f"{'TOTAL':<14} {total_calls:>6}{'':>24}{'':>8}{'':>8}{'':>6}{'':>6} {total_cost:>9.6f}")



def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="truth-bot: automated political rhetoric fact-checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--transcript",
        "-t",
        help="Path to transcript file, URL, or raw text (use '-' for stdin)",
    )
    parser.add_argument("--speaker", "-s", default="Unknown", help="Speaker name")
    parser.add_argument("--date", "-d", help="Speech date (YYYY-MM-DD)")
    parser.add_argument("--venue", "-v", help="Venue or event name")
    parser.add_argument("--output-dir", "-o", help="Output directory for reports")
    parser.add_argument("--base-url", default="https://example.com", help="Base URL for links")
    parser.add_argument("--post-bluesky", action="store_true", help="Post results to Bluesky")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no external calls)")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")

    subparsers = parser.add_subparsers(dest="subcommand")

    # metrics subcommand
    metrics_parser = subparsers.add_parser("metrics", help="Metrics and telemetry commands")
    metrics_sub = metrics_parser.add_subparsers(dest="metrics_cmd")
    summary_parser = metrics_sub.add_parser("summary", help="Print per-adapter summary table")
    summary_parser.add_argument(
        "--jsonl",
        help="Path to adapter_calls.jsonl (default: settings.metrics_dir/adapter_calls.jsonl)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle metrics subcommand
    if getattr(args, "subcommand", None) == "metrics":
        if getattr(args, "metrics_cmd", None) == "summary":
            _print_metrics_summary(getattr(args, "jsonl", None))
            return
        else:
            parser.print_help()
            return

    # Read from stdin if requested
    source = args.transcript or ""
    if source == "-":
        source = sys.stdin.read()
    elif not source:
        parser.print_help()
        sys.exit(1)

    date = None
    if args.date:
        date = datetime.strptime(args.date, "%Y-%m-%d")

    pipeline = Pipeline(
        output_dir=args.output_dir,
        base_url=args.base_url,
        post_bluesky=args.post_bluesky,
        dry_run=args.dry_run,
    )

    report = pipeline.run(
        source=source,
        speaker=args.speaker,
        date=date,
        venue=args.venue,
    )

    # Print summary to stdout
    print(f"\n✓ Fact-check complete")
    print(f"  Report ID : {report.id}")
    print(f"  Speaker   : {report.transcript.speaker}")
    print(f"  Claims    : {report.total_claims} total, {report.checkable_claims} checkable")
    print(f"  Verdicts  :")
    for label, count in report.verdict_summary.items():
        if count > 0:
            print(f"    {label:20s} {count}")
    if report.report_url:
        print(f"  Report    : {report.report_url}")
    if report.bluesky_thread_url:
        print(f"  Bluesky   : {report.bluesky_thread_url}")


if __name__ == "__main__":
    main()
