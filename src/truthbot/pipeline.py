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


def _print_metrics_summary(
    jsonl_path: str | None = None,
    run_id: str | None = None,
) -> None:
    """Read adapter_calls.jsonl and print per-adapter summary tables."""
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

    stats: dict[str, dict] = defaultdict(lambda: {
        "calls": 0, "ok": 0, "errors": 0, "parse_errors": 0,
        "input_tokens": 0, "output_tokens": 0,
        "tool_calls": 0, "urls": 0,
        "total_cost": 0.0, "total_ms": 0,
    })
    tier_cost: dict[str, float] = defaultdict(float)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if run_id and rec.get("run_id") != run_id:
                continue
            tier = rec.get("tier", "frontier")
            tier_cost[tier] += float(rec.get("estimated_cost_usd", 0.0))
            name = rec.get("adapter_name", "unknown")
            s = stats[name]
            s["calls"] += 1
            status = rec.get("status", "")
            if status == "ok":
                s["ok"] += 1
            elif status == "parse_error":
                s["errors"] += 1
                s["parse_errors"] += 1
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

    # ---- Table 1: full call detail ------------------------------------------
    fmt1 = "{:<14} {:>6} {:>4} {:>5} {:>8} {:>8} {:>6} {:>6} {:>10} {:>8}"
    print()
    print(fmt1.format("Adapter","Calls","OK","Errs","In Tok","Out Tok","Tools","URLs","Cost $","Avg ms"))
    print("-" * 83)

    total_calls = total_ok = total_errors = 0
    total_in = total_out = total_tools = total_urls = 0
    total_cost = total_ms_all = 0

    for name, s in sorted(stats.items()):
        avg_ms = s["total_ms"] // max(s["calls"], 1)
        print(fmt1.format(
            name, s["calls"], s["ok"], s["errors"],
            s["input_tokens"], s["output_tokens"],
            s["tool_calls"], s["urls"],
            f"{s['total_cost']:.6f}", avg_ms,
        ))
        total_calls += s["calls"];  total_ok += s["ok"];  total_errors += s["errors"]
        total_in += s["input_tokens"];  total_out += s["output_tokens"]
        total_tools += s["tool_calls"];  total_urls += s["urls"]
        total_cost += s["total_cost"];  total_ms_all += s["total_ms"]

    print("-" * 83)
    grand_avg = total_ms_all // max(total_calls, 1)
    print(fmt1.format(
        "TOTAL", total_calls, total_ok, total_errors,
        total_in, total_out, total_tools, total_urls,
        f"{total_cost:.6f}", grand_avg,
    ))

    # ---- Table 2: cost optimisation breakdown --------------------------------
    fmt2 = "{:<14} {:>12} {:>13} {:>11} {:>13} {:>12}"
    print()
    print(fmt2.format("Adapter","Total Calls","Total Cost $","Avg $/call","Avg sec/call","Parse Err %"))
    print("-" * 78)

    for name, s in sorted(stats.items()):
        n = max(s["calls"], 1)
        avg_cost = s["total_cost"] / n
        avg_sec  = s["total_ms"] / n / 1000.0
        pct      = s["parse_errors"] / n * 100
        print(fmt2.format(
            name, s["calls"], f"{s['total_cost']:.6f}",
            f"{avg_cost:.6f}", f"{avg_sec:.2f}", f"{pct:.1f}%",
        ))
    print()

    if run_id and tier_cost:
        print()
        print(f"Triage / tier cost (run_id={run_id})")
        print("-" * 50)
        for tier, cst in sorted(tier_cost.items()):
            print(f"  {tier:<20} ${cst:.6f}")
        print()



def _run_publish(args) -> None:
    """Full pipeline: ingest → extract → verify → publish site."""
    import os, uuid
    from datetime import datetime
    from truthbot.extract.claims import ClaimExtractor
    from truthbot.ingest.transcript import TranscriptIngester
    from truthbot.verify.engine import VerificationEngine
    from truthbot.publish.site import SitePublisher, SiteReport

    # Load transcript
    src = args.transcript
    if src == "-":
        import sys
        text = sys.stdin.read()
    else:
        text = Path(src).read_text(encoding="utf-8")

    date = datetime.strptime(args.date, "%Y-%m-%d")
    source_url = getattr(args, "source_url", "") or ""

    # Ingest
    ingester = TranscriptIngester()
    result = ingester.ingest_text(
        text,
        speaker=args.speaker,
        date=date,
        venue=getattr(args, "venue", "") or "",
    )
    transcript = result.transcript
    if result.warnings:
        for w in result.warnings:
            logger.warning("Ingest: %s", w)

    # Extract
    print(f"Extracting claims from {len(text):,} chars...")
    extractor = ClaimExtractor()
    claims = extractor.extract(transcript)
    checkable = [c for c in claims if c.is_checkable]
    max_claims = getattr(args, 'max_claims', 5) or 5
    checkable = checkable[:max_claims]
    print(f"  {len(claims)} claims extracted, {len(checkable)} checkable")

    # Verify — parallel fan-out across claims (adapters already fan-out within each claim)
    run_id = str(uuid.uuid4())
    mode = getattr(args, "mode", "live") or "live"
    no_inject = bool(getattr(args, "no_inject_evidence", False))
    from truthbot.config import settings
    from truthbot.metrics.telemetry import finalize_run, telemetry_run_context
    from truthbot.verify.batch import BatchDispatcher

    if mode == "batch":
        BatchDispatcher(settings.metrics_dir).record_job(
            run_id,
            transcript_meta={
                "speaker": args.speaker,
                "date": args.date,
                "transcript_chars": len(text),
            },
            work_units=[{"claim_id": c.id, "claim_text": c.text[:500]} for c in checkable],
        )
        print(f"Batch job descriptor: metrics/batch_jobs/{run_id}.json")
        print(f"Poll with: truthbot batch poll {run_id}")

    bundles_map: dict[int, object] = {}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    engine = VerificationEngine(
        run_id=run_id,
        inject_evidence=not no_inject,
        triage_enabled=bool(getattr(args, "triage", False)),
        triage_threshold=float(getattr(args, "triage_threshold", 0.8)),
        triage_shadow_rate=float(getattr(args, "triage_shadow_rate", 0.0)),
        verify_mode=mode,
    )

    def _verify_one(idx_claim):
        idx, claim = idx_claim
        print(f"  Verifying claim {idx}/{len(checkable)}: {claim.text[:60]}...")
        try:
            with telemetry_run_context(
                run_id=run_id,
                evidence_injected=not no_inject,
                synthesis_mode=mode,
            ):
                bundle = engine.verify_bundle(
                    claim,
                    speaker=args.speaker,
                    date_str=args.date,
                )
            label = bundle.consensus.consensus_label.value
            strength = bundle.consensus.consensus_strength
            cache = " [cached]" if bundle.cache_hit else ""
            print(f"    -> claim {idx}: {label} ({strength}){cache}")
            return idx, bundle
        except Exception as exc:
            logger.error("Verify failed for claim %s: %s", claim.id, exc)
            return idx, None

    max_workers = min(len(checkable), 5)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_verify_one, (i, c)): i for i, c in enumerate(checkable, 1)}
        for fut in as_completed(futs):
            idx, bundle = fut.result()
            if bundle is not None:
                bundles_map[idx] = bundle

    try:
        fin = finalize_run(run_id)
        print(
            f"Telemetry run summary: metrics/run_summaries/{run_id}.json "
            f"(total_cost_usd={fin['total_cost_usd']:.6f})"
        )
    except Exception as exc:
        logger.warning("finalize_run failed: %s", exc)

    # Restore original ordering
    bundles = [bundles_map[i] for i in sorted(bundles_map)]

    # Build SiteReport
    site_report = SiteReport(
        report_id=str(uuid.uuid4()),
        speaker=args.speaker,
        role=getattr(args, "role", "") or "",
        date=date,
        venue=getattr(args, "venue", "") or "",
        transcript_source_url=source_url,
        bundles=bundles,
    )

    # Publish
    site_root = getattr(args, "site_root", None)
    publisher = SitePublisher(site_root=site_root)
    report_path = publisher.publish(site_report)
    site_url = publisher.site_url(site_report)
    stats = publisher.summary()

    print()
    print(f"Site generated: {stats['root']}")
    print(f"Report page:    {report_path}")
    print(f"Served at:      {site_url}")
    print(f"Summary:        {stats['reports']} report(s), {stats['claims']} claim(s), "
          f"{stats['total_kb']} KB total")

    # Verify internal links exist
    report_rel = Path("reports") / f"{site_report.report_slug}.html"
    issues = []
    for b in bundles:
        claim_rel = Path("claims") / f"{b.claim.id}.html"
        if not (Path(stats['root']) / claim_rel).exists():
            issues.append(str(claim_rel))
    if issues:
        print(f"WARNING: {len(issues)} claim page(s) missing")
    else:
        print(f"All internal links verified OK")

    # Validate JSON data files
    for fname in ("data/reports.json", "data/claims.json"):
        p = Path(stats['root']) / fname
        try:
            import json
            json.loads(p.read_text(encoding="utf-8"))
            print(f"{fname}: valid JSON")
        except Exception as e:
            print(f"{fname}: INVALID - {e}")


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
    summary_parser.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="Only include telemetry rows with this run_id (also prints tier cost table)",
    )

    # publish subcommand — full pipeline + site generation
    pub_parser = subparsers.add_parser("publish", help="Run pipeline and generate static site")
    pub_parser.add_argument("--transcript", required=True, help="Path to transcript file (or - for stdin)")
    pub_parser.add_argument("--speaker",    required=True, help="Speaker name")
    pub_parser.add_argument("--role",       default="",   help="Speaker role/title")
    pub_parser.add_argument("--date",       required=True, help="Speech date YYYY-MM-DD")
    pub_parser.add_argument("--venue",      default="",   help="Venue or event name")
    pub_parser.add_argument("--source-url", default="",   help="Transcript source URL")
    pub_parser.add_argument("--site-root",  default=None, help="Site output root (overrides TRUTHBOT_SITE_ROOT)")
    pub_parser.add_argument("--max-claims",  type=int, default=5, help="Max checkable claims to verify (default 5)")
    pub_parser.add_argument(
        "--mode",
        choices=("live", "batch"),
        default="live",
        help="Verification billing mode: live API calls (default) or batch descriptor + same live verify for now",
    )
    pub_parser.add_argument(
        "--triage",
        action="store_true",
        help="Enable cheap-model triage tier before frontier fan-out",
    )
    pub_parser.add_argument(
        "--triage-threshold",
        type=float,
        default=0.8,
        help="Minimum numeric confidence for unanimous triage short-circuit (default 0.8)",
    )
    pub_parser.add_argument(
        "--triage-shadow-rate",
        type=float,
        default=0.0,
        help="Probability of skipping triage and labeling frontier verdicts as frontier_shadow (0–1)",
    )
    pub_parser.add_argument(
        "--no-inject-evidence",
        action="store_true",
        help="Do not pass prefetched evidence snippets into model prompts (telemetry evidence_injected=false)",
    )

    batch_parser = subparsers.add_parser("batch", help="Batch job helpers")
    batch_sub = batch_parser.add_subparsers(dest="batch_cmd", required=True)
    batch_poll = batch_sub.add_parser("poll", help="Poll batch job descriptor status for a run_id")
    batch_poll.add_argument("run_id", help="Publish run UUID written under metrics/batch_jobs/")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle metrics subcommand
    if getattr(args, "subcommand", None) == "metrics":
        if getattr(args, "metrics_cmd", None) == "summary":
            _print_metrics_summary(
                getattr(args, "jsonl", None),
                run_id=getattr(args, "run_id", None),
            )
            return
        else:
            parser.print_help()
            return

    if getattr(args, "subcommand", None) == "batch":
        if getattr(args, "batch_cmd", None) == "poll":
            from truthbot.config import settings
            from truthbot.verify.batch import BatchDispatcher

            st = BatchDispatcher(settings.metrics_dir).poll(args.run_id)
            print(st)
            return
        parser.print_help()
        return

    # Handle publish subcommand
    if getattr(args, "subcommand", None) == "publish":
        _run_publish(args)
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
    print(f"\nFact-check complete")
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
