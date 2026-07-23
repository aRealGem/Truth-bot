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


def _print_tool_stats(
    jsonl_path: str | None = None,
    run_id: str | None = None,
    adapter: str | None = None,
) -> None:
    """Print per-adapter tool-call distribution + URL-grounding rates.

    Reads ``metrics/adapter_calls.jsonl`` and produces three tables for
    post-mortem / arm-X probe analysis:

    1. **Tool-call distribution** — per-adapter mean / p50 / p95 / max of
       ``tool_call_count``, plus the share of records with zero tool
       calls (the "search declined" signal — high for adapters that
       answered from training data; low for adapters that grounded
       every call).
    2. **URL-grounding rates** — per-adapter totals of
       ``model_reported_source_count`` and ``stripped_source_count``,
       with the strip-rate ratio. Mirrors the ``fabrication`` block
       in ``run_summary.json`` but reproducible from telemetry alone
       and easier to slice by ``--run-id`` / ``--adapter``.
    3. **Tool-call histogram** — record counts at each tool-call bin
       (0, 1, 2, 3, 4+). Useful for spotting bi-modal patterns where
       half the calls grounded heavily and half not at all.

    Filter knobs:
      * ``--run-id`` — only include records for one ``run_id``
      * ``--adapter`` — only one adapter (anthropic / openai / gemini / xai)
    """
    import json
    from collections import defaultdict
    from pathlib import Path

    if jsonl_path:
        path = Path(jsonl_path)
    else:
        from truthbot.config import settings
        path = settings.metrics_dir / "adapter_calls.jsonl"

    if not path.exists():
        print(f"No telemetry data found at {path}.")
        return

    # tool_counts[adapter] = list of per-record tool_call_count values
    tool_counts: dict[str, list[int]] = defaultdict(list)
    reported: dict[str, int] = defaultdict(int)
    stripped: dict[str, int] = defaultdict(int)
    retrieved: dict[str, int] = defaultdict(int)

    rows_seen = 0
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
            name = rec.get("adapter_name", "unknown")
            if adapter and name != adapter:
                continue
            rows_seen += 1
            tool_counts[name].append(int(rec.get("tool_call_count", 0) or 0))
            reported[name] += int(rec.get("model_reported_source_count", 0) or 0)
            stripped[name] += int(rec.get("stripped_source_count", 0) or 0)
            retrieved[name] += int(rec.get("retrieved_url_count", 0) or 0)

    if rows_seen == 0:
        print("No telemetry records matched the filters.")
        return

    def _percentile(values: list[int], pct: float) -> float:
        """Linear-interpolation percentile. ``values`` need not be sorted."""
        if not values:
            return 0.0
        s = sorted(values)
        if len(s) == 1:
            return float(s[0])
        k = (len(s) - 1) * pct
        lo = int(k)
        hi = min(lo + 1, len(s) - 1)
        frac = k - lo
        return s[lo] * (1 - frac) + s[hi] * frac

    # ── Table 1: tool-call distribution ─────────────────────────────────────
    fmt1 = "{:<12} {:>4} {:>6} {:>6} {:>6} {:>5} {:>14}"
    print()
    print(fmt1.format(
        "Adapter", "n", "mean", "p50", "p95", "max", "zero-tool %"
    ))
    print("-" * 60)
    total_records = 0
    for name in sorted(tool_counts):
        counts = tool_counts[name]
        n = len(counts)
        total_records += n
        mean = sum(counts) / n if n else 0.0
        p50 = _percentile(counts, 0.50)
        p95 = _percentile(counts, 0.95)
        mx = max(counts) if counts else 0
        zero = sum(1 for c in counts if c == 0)
        zero_pct = zero / n * 100 if n else 0.0
        print(fmt1.format(
            name, n, f"{mean:.2f}", f"{p50:.1f}",
            f"{p95:.1f}", mx, f"{zero}/{n} ({zero_pct:.0f}%)",
        ))
    print("-" * 60)
    print(fmt1.format("TOTAL", total_records, "", "", "", "", ""))

    # ── Table 2: URL-grounding rates ────────────────────────────────────────
    fmt2 = "{:<12} {:>10} {:>9} {:>11} {:>13}"
    print()
    print(fmt2.format(
        "Adapter", "reported", "stripped", "strip rate", "tool URLs"
    ))
    print("-" * 60)
    for name in sorted(tool_counts):
        r = reported[name]
        s = stripped[name]
        rate = (s / r * 100) if r else 0.0
        rate_str = f"{rate:.1f}%" if r else "—"
        print(fmt2.format(name, r, s, rate_str, retrieved[name]))

    # ── Table 3: tool-call histogram ────────────────────────────────────────
    fmt3 = "{:<12} {:>4} {:>4} {:>4} {:>4} {:>5}"
    print()
    print(fmt3.format("Adapter", "0", "1", "2", "3", "4+"))
    print("-" * 40)
    for name in sorted(tool_counts):
        bins = [0, 0, 0, 0, 0]  # 0, 1, 2, 3, 4+
        for c in tool_counts[name]:
            bins[min(c, 4)] += 1
        print(fmt3.format(name, *bins))
    print()


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


def resolve_inject_evidence(
    evidence_source: str,
    *,
    no_inject_flag: bool = False,
    inject_flag: bool = False,
) -> bool:
    """
    Decide whether prompts should include pre-gathered evidence snippets.

    The default tracks ``evidence_source`` so telemetry stays honest: when no
    provider is fetching snippets, we don't pretend to inject them. Explicit
    CLI flags override the default; ``--no-inject-evidence`` wins over
    ``--inject-evidence`` if both are set.
    """
    if no_inject_flag:
        return False
    if inject_flag:
        return True
    return evidence_source.strip().lower() != "none"


def _routes_to_batch(adapter, settings) -> bool:
    """Phase 3a — honor ``TRUTHBOT_OPENAI_LIVE`` to flip OpenAI to sidecar.

    The default static routing (``supports_batch=True`` → batch API,
    ``False`` → sidecar live) is preserved. The single override is
    OpenAI: when ``settings.openai_live_mode`` is truthy, route OpenAI
    through the sidecar so verdicts complete in seconds instead of the
    3–24h batch SLA we hit on the ``ed7be4ad-…`` SOTU run.

    Promoted to module scope so the routing decision is testable in
    isolation — see ``tests/test_openai_live_routing.py``.
    """
    if not getattr(adapter, "supports_batch", False):
        return False
    if (
        getattr(adapter, "adapter_name", "") == "openai"
        and getattr(settings, "openai_live_mode", False)
    ):
        return False
    return True


def _run_publish_batch_submit(
    *,
    args,
    engine,
    claims: list,
    checkable: list,
    run_id: str,
    inject_evidence: bool,
    source_url: str,
    max_claims: int,
    settings,
) -> None:
    """Batch-mode submit: triage live, submit batch, run Grok sidecar, print poll cmd, exit."""
    from truthbot.verify.batch import BatchDispatcher

    triaged_bundles: list = []
    triaged_claims_json: list = []
    claims_with_evidence: list = []

    speaker = args.speaker
    date_str = args.date

    for claim in checkable:
        bundle, evidence = engine.maybe_resolve_early(
            claim, speaker=speaker, date_str=date_str
        )
        if bundle is not None:
            triaged_bundles.append(bundle)
            triaged_claims_json.append(claim.model_dump(mode="json"))
            print(f"  early-resolve: {claim.text[:60]}... -> cached / triage")
        else:
            claims_with_evidence.append((claim, evidence))

    batch_adapters = [a for a in engine.adapters if _routes_to_batch(a, settings)]
    sidecar_adapters = [
        a for a in engine.adapters if not _routes_to_batch(a, settings)
    ]

    requested_cpr = getattr(args, "claims_per_request", None)
    if requested_cpr is None:
        requested_cpr = settings.claims_per_request
    requested_cpr = max(1, int(requested_cpr))

    effective_chunk_size_by_adapter = {
        a.adapter_name: min(
            requested_cpr, max(1, int(getattr(a, "max_claims_per_request", 1)))
        )
        for a in batch_adapters
    }

    transcript_meta = {
        "speaker": speaker,
        "role": getattr(args, "role", "") or "",
        "date": date_str,
        "venue": getattr(args, "venue", "") or "",
        "source_url": source_url,
        "site_root": getattr(args, "site_root", None),
        "total_claims_extracted": len(claims),
        "total_claims_checkable": sum(1 for c in claims if c.is_checkable),
        "claims_verified_target": len(checkable),
        "claims_batched": len(claims_with_evidence),
        "claims_triaged_auto": len(triaged_bundles),
        "triaged_claim_ids": [b.claim.id for b in triaged_bundles],
        "triaged_claims": triaged_claims_json,
        "max_claims_cap": max_claims,
        "triage_enabled": bool(getattr(args, "triage", False)),
        "adapters_batch": [a.adapter_name for a in batch_adapters],
        "adapters_sidecar": [a.adapter_name for a in sidecar_adapters],
        "claims_per_request_requested": requested_cpr,
        "claims_per_request_effective": effective_chunk_size_by_adapter,
    }

    disp = BatchDispatcher(settings.metrics_dir)
    if claims_with_evidence and batch_adapters:
        descriptor_path = disp.submit(
            run_id,
            adapters=batch_adapters,
            claims_with_evidence=claims_with_evidence,
            transcript_meta=transcript_meta,
            inject_evidence=inject_evidence,
            sidecar_live_adapters=sidecar_adapters,
            claims_per_request=requested_cpr,
            max_evidence_per_claim_in_batch=settings.max_evidence_per_claim_in_batch,
        )
    else:
        descriptor_path = disp.record_job(
            run_id,
            transcript_meta=transcript_meta,
            work_units=[],
            provider_hints={},
        )

    max_effective = max(effective_chunk_size_by_adapter.values(), default=1)
    savings_pct = (1 - 1 / max_effective) * 100 if max_effective > 1 else 0.0

    print()
    print("=" * 72)
    print(f"Batch submitted. run_id = {run_id}")
    print(f"  Descriptor:          {descriptor_path}")
    print(
        f"  Claims extracted:    {transcript_meta['total_claims_extracted']}"
    )
    print(
        f"  Claims checkable:    {transcript_meta['total_claims_checkable']}"
        f"  (cap: {max_claims})"
    )
    print(
        f"  Claims triaged live: {transcript_meta['claims_triaged_auto']}"
    )
    print(
        f"  Claims to batch:     {transcript_meta['claims_batched']}"
    )
    print(f"  Batch providers:     {transcript_meta['adapters_batch']}")
    print(f"  Sidecar providers:   {transcript_meta['adapters_sidecar']}")
    print(
        f"  Claims per request:  {requested_cpr} requested; "
        f"effective {effective_chunk_size_by_adapter or '{}'}"
        + (
            f"  (~{savings_pct:.0f}% per-call overhead amortized)"
            if max_effective > 1
            else ""
        )
    )
    print()
    print(f"Poll status:         truthbot batch poll {run_id}")
    print(f"Reconcile+publish:   truthbot batch reconcile {run_id}")
    print("=" * 72)


def _run_urls_check(
    *,
    sidecar_path: str,
    cache_path: str,
    timeout: float,
    max_workers: int,
) -> None:
    """Phase 3b: HEAD-check every unique URL from a sidecar file.

    Prints a classification summary (ok / bot-blocked / dead-4xx / etc.)
    and persists results to the shared cache. Operates read-only on
    verdict data — does not mutate sidecar or verdict caches.
    """
    import json
    from collections import Counter
    from pathlib import Path as _Path

    from truthbot.verify.url_validation import UrlCache, check_urls_bulk

    sidecar = _Path(sidecar_path)
    if not sidecar.exists():
        print(f"sidecar not found: {sidecar}")
        sys.exit(2)

    urls: set[str] = set()
    with sidecar.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for u in row.get("web_sources") or []:
                if isinstance(u, str) and u:
                    urls.add(u)

    print(f"unique URLs in {sidecar.name}: {len(urls)}")
    if not urls:
        return

    cache = UrlCache.load(_Path(cache_path))
    print(f"cache pre-load: {len(cache.entries)} entries")

    results = check_urls_bulk(
        urls, timeout=timeout, max_workers=max_workers, cache=cache
    )
    cache.save(_Path(cache_path))

    classes = Counter(r.failure_class for r in results.values())
    likely_real = sum(1 for r in results.values() if r.likely_real)

    print()
    print("classification:")
    for cls, n in classes.most_common():
        print(f"  {cls:15s} {n}")
    print()
    pct = 100 * likely_real / max(1, len(results))
    print(f"likely real (ok + bot-blocked): {likely_real}/{len(results)} ({pct:.0f}%)")

    suspect = [
        r for r in results.values()
        if r.failure_class in ("dead-4xx", "malformed", "dns", "cert-error")
    ]
    if suspect:
        print()
        print("suspect URLs (likely hallucinated or dead):")
        for r in sorted(suspect, key=lambda x: (x.failure_class, x.url)):
            print(f"  [{r.failure_class}] {r.url}")


def _run_urls_filter_sidecar(
    *,
    sidecar_path: str,
    out_path: Optional[str],
    cache_path: str,
    timeout: float,
    max_workers: int,
) -> None:
    """Layer 3: write a *cleaned* sidecar with broken URLs stripped.

    The output JSONL adds three new fields per row:
        * ``verified_sources``   — reachable.
        * ``unverified_sources`` — bot-blocked or transient (likely real).
        * ``broken_sources``     — dead-4xx / malformed / dns / cert-error.

    ``web_sources`` in the output is rewritten to ``verified + unverified``
    so the existing publish-layer code continues to work without
    rendering known-broken URLs. ``model_reported_sources`` (the raw
    pre-intersection list from Layer 1) is left untouched as the audit
    trail.
    """
    from pathlib import Path as _Path

    from truthbot.verify.url_validation import UrlCache, filter_sidecar

    in_p = _Path(sidecar_path)
    if not in_p.exists():
        print(f"sidecar not found: {in_p}")
        sys.exit(2)

    if out_path:
        out_p = _Path(out_path)
    else:
        # Default lives alongside the input with a `.cleaned.jsonl`
        # suffix so it's obvious which file the publish pipeline should
        # consume.
        stem = in_p.name
        if stem.endswith(".jsonl"):
            stem = stem[: -len(".jsonl")]
        out_p = in_p.with_name(f"{stem}.cleaned.jsonl")

    cache_p = _Path(cache_path)
    cache = UrlCache.load(cache_p)
    pre = len(cache.entries)

    stats = filter_sidecar(
        in_p,
        out_p,
        cache=cache,
        timeout=timeout,
        max_workers=max_workers,
    )
    cache.save(cache_p)

    print(f"input : {in_p}")
    print(f"output: {out_p}")
    print(f"cache : {cache_p} ({pre} -> {len(cache.entries)} entries)")
    print()
    print(f"rows processed   : {stats['rows']}")
    print(f"verified URLs    : {stats['verified']}")
    print(f"unverified URLs  : {stats['unverified']} (likely real, bot-blocked/transient)")
    print(f"broken URLs      : {stats['broken']} (stripped from web_sources)")


def _run_publish_batch_reconcile(
    run_id: str,
    *,
    site_root: Optional[str] = None,
    validate_urls: bool = False,
) -> None:
    """Poll → parse → merge → cache → publish for a previously submitted batch run."""
    import uuid as _uuid
    from datetime import datetime

    from truthbot.config import settings
    from truthbot.metrics.telemetry import finalize_run
    from truthbot.publish.site import SitePublisher, SiteReport
    from truthbot.verify.adapters.base import AdapterUnavailable
    from truthbot.verify.batch import reconcile_run
    from truthbot.verify.engine import VerificationEngine

    adapters_by_name: dict = {}
    for cls_import in (
        ("truthbot.verify.adapters.anthropic", "AnthropicAdapter"),
        ("truthbot.verify.adapters.openai", "OpenAIAdapter"),
        ("truthbot.verify.adapters.gemini", "GeminiAdapter"),
        ("truthbot.verify.adapters.grok", "GrokAdapter"),
    ):
        mod_name, cls_name = cls_import
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            adapters_by_name[cls.adapter_name] = cls()
        except AdapterUnavailable as exc:
            logger.info("reconcile: skipping %s (%s)", cls_name, exc)
        except Exception as exc:
            logger.warning("reconcile: failed to build %s: %s", cls_name, exc)

    engine = VerificationEngine(run_id=run_id, verify_mode="batch")

    result = reconcile_run(
        settings.metrics_dir,
        run_id,
        adapters_by_name=adapters_by_name,
        engine=engine,
        validate_urls=validate_urls,
    )

    status = result["status"]
    if status == "missing":
        print(f"No batch descriptor for run_id={run_id}")
        sys.exit(1)
    if status == "pending":
        print(f"Batch run_id={run_id} still pending:")
        for provider, st in result.get("pending_providers", []):
            print(
                f"  - {provider}: {st.get('raw_status', st.get('status', '?'))}"
                f" ({st.get('done', 0)}/{st.get('total', 0)})"
            )
        sys.exit(2)

    meta = result["transcript_meta"]
    bundles = result["bundles"]
    triaged = result.get("triaged_bundles", [])
    all_bundles = triaged + bundles

    try:
        fin = finalize_run(run_id)
        print(
            f"Telemetry run summary: metrics/run_summaries/{run_id}.json "
            f"(total_cost_usd={fin['total_cost_usd']:.6f})"
        )
    except Exception as exc:
        logger.warning("finalize_run failed: %s", exc)

    date_val = None
    if meta.get("date"):
        try:
            date_val = datetime.strptime(meta["date"], "%Y-%m-%d")
        except Exception:
            pass

    site_report = SiteReport(
        report_id=str(_uuid.uuid4()),
        speaker=meta.get("speaker", ""),
        role=meta.get("role", ""),
        date=date_val,
        venue=meta.get("venue", ""),
        transcript_source_url=meta.get("source_url", ""),
        bundles=all_bundles,
    )

    effective_site_root = site_root or meta.get("site_root")
    publisher = SitePublisher(site_root=effective_site_root)
    report_path = publisher.publish(site_report)
    site_url = publisher.site_url(site_report)
    stats = publisher.summary()

    print()
    print(f"Reconciled run_id={run_id}:")
    print(f"  Total claims extracted: {meta.get('total_claims_extracted', '?')}")
    print(f"  Triaged (cached/live):  {len(triaged)}")
    print(f"  Batched (reconciled):   {len(bundles)}")
    print(f"Site generated: {stats['root']}")
    print(f"Report page:    {report_path}")
    print(f"Served at:      {site_url}")


def _persist_extracted_claims(
    run_id: str,
    claims: list,
    metrics_dir: str | Path = "metrics",
) -> Path:
    """Write extracted claims to ``metrics/extractions/<run_id>.jsonl``.

    One claim per line, each a JSON dump of ``claim.model_dump()``. Called
    immediately after extraction so that if batch submit/reconcile later
    fails, we can re-drive the pipeline without re-running Claude Sonnet.

    Returns the written path. Never raises on filesystem or JSON issues —
    extraction has already succeeded; a persistence failure should not
    abort the run. Caller sees it via the log warning.
    """
    import json as _json

    try:
        extractions_dir = Path(metrics_dir) / "extractions"
        extractions_dir.mkdir(parents=True, exist_ok=True)
        path = extractions_dir / f"{run_id}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for claim in claims:
                f.write(_json.dumps(claim.model_dump(), default=str) + "\n")
        logger.info(
            "Persisted %d extracted claims to %s",
            len(claims),
            path,
        )
        return path
    except Exception as exc:
        logger.warning("Failed to persist extracted claims: %s", exc)
        return Path(metrics_dir) / "extractions" / f"{run_id}.jsonl"


def _persist_pca_run(
    run_id: str,
    result,
    *,
    meta: dict,
    metrics_dir: str | Path = "metrics",
) -> Path:
    """Write a PCA run's replay artifact to ``metrics/pca_runs/<run_id>.json``.

    Holds ``{meta, claims, rows, characterization}`` — the raw adjudication rows
    plus the claim dicts (with Layer A provenance) that were fed to the bridge. This
    is the minimum needed to re-bridge and re-publish OFFLINE, with no LLM spend: the
    live PCA run is ~1hr of proxy calls, so a persisted row set is the difference
    between a free re-render and a full re-run (mirrors ``_persist_extracted_claims``).

    Best-effort: verification has already succeeded, so a persistence failure logs a
    warning rather than aborting the publish. Returns the (intended) path.
    """
    import json as _json

    pca_runs_dir = Path(metrics_dir) / "pca_runs"
    path = pca_runs_dir / f"{run_id}.json"
    try:
        pca_runs_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "meta": meta,
            "claims": list(getattr(result, "claims", []) or []),
            "rows": list(getattr(result, "rows", []) or []),
            "characterization": list(getattr(result, "characterization", []) or []),
            "roster": getattr(result, "roster", None),
            "evidence": {
                sid: [
                    ev.model_dump(mode="json") if hasattr(ev, "model_dump") else dict(ev)
                    for ev in evs
                ]
                for sid, evs in (getattr(result, "evidence", {}) or {}).items()
            },
        }
        path.write_text(_json.dumps(payload, default=str, ensure_ascii=False), encoding="utf-8")
        logger.info(
            "Persisted PCA replay artifact (%d rows) to %s",
            len(payload["rows"]),
            path,
        )
    except Exception as exc:
        logger.warning("Failed to persist PCA replay artifact: %s", exc)
    return path


def _preflight_key_sanity() -> None:
    """Validate any set API keys before any spend.

    Only reports keys that are **set** but have an obviously-broken shape
    (truncated, whitespace-wrapped, trailing ``>``, wrong prefix). Missing
    keys are silently allowed — the adapter layer's ``AdapterUnavailable``
    path handles those. Raises ``SystemExit`` with a human-readable report
    when any set key fails the shape check, so the operator catches the
    typo before extraction burns the first Claude Sonnet call.
    """
    import os

    from truthbot.verify.adapters.key_sanity import validate_api_key

    provider_env_vars = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "xai": "XAI_API_KEY",
    }
    failures: list[str] = []
    for provider, env_var in provider_env_vars.items():
        value = os.environ.get(env_var)
        if not value:
            continue
        result = validate_api_key(provider, value)
        if not result.ok:
            failures.append(f"  {env_var} ({provider}): {result.reason}")
    if failures:
        raise SystemExit(
            "API key preflight failed — refusing to start:\n"
            + "\n".join(failures)
        )


def _default_speech_id(speaker: str, date) -> str:
    """Fallback sid prefix from speaker + year when --speech-id is omitted.

    The prefix only needs to be stable and registrable — ``prepare_speech``
    registers it against the CLI ``--date`` so temporal grounding resolves. For the
    pinned SOTU fixtures pass ``--speech-id trump_2026`` / ``biden_2022`` to reuse
    the registered utterance dates exactly."""
    import re

    slug = re.sub(r"[^a-z0-9]+", "_", (speaker or "speech").lower()).strip("_") or "speech"
    return f"{slug}_{date.year}"


def _build_open_book_provider():
    """Layer C evidence provider: Brave + FactCheck connectors (both keyed on
    BRAVE_API_KEY), time-scoped per claim inside adjudicate, with the
    cheap-model relevance middle step (query generation + relevance/supports
    scoring) when the proxy key is present. Returns None (→ closed-book) when
    no Brave key is set, so the run degrades loudly rather than silently
    faking evidence. Mirrors the eval driver's provider builder."""
    import os

    if not os.environ.get("BRAVE_API_KEY"):
        return None
    from truthbot.verify.evidence_provider import build_evidence_provider
    from truthbot.verify.relevance import build_relevance_provider
    from truthbot.verify.sources.brave import BraveSearchConnector
    from truthbot.verify.sources.factcheck import FactCheckConnector

    brave = BraveSearchConnector(max_results=5)
    factcheck = FactCheckConnector(max_results=3)
    refined = build_relevance_provider(brave, [factcheck])
    if refined is not None:
        return refined
    # No proxy key for the cheap scorer — legacy tier-ranked retrieval.
    return build_evidence_provider(source="connectors", connectors=[brave, factcheck])


def _build_v2_pack_builder():
    """shared_pack_v2 (P67.9): bind the R1/R2/R3 retriever trio into the
    ``pack_builder`` hook (trio shortlists → deterministic consolidator →
    T2.4 quality gate). Fails LOUD when a lane is missing — a dead retriever
    key would otherwise yield silent empty shortlists and gate every claim
    Unverifiable, which is a broken run, not a verdict."""
    import os
    import shutil

    missing = []
    if shutil.which("claude") is None:
        missing.append("claude CLI (R1 opus worker)")
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY (R2 gpt browsing)")
    if not os.environ.get("XAI_API_KEY"):
        missing.append("XAI_API_KEY (R3 grok search)")
    if missing:
        print("BLOCKED (--evidence-mode v2): missing " + "; ".join(missing)
              + ". No spend attempted.")
        sys.exit(1)
    from truthbot.verdict.evidence_pack_v2 import build_evidence_pack_v2
    from truthbot.verify.retrievers import (ClaudeWorkerRetriever,
                                            GrokSearchRetriever,
                                            OpenAIBrowsingRetriever)

    trio = (ClaudeWorkerRetriever(), OpenAIBrowsingRetriever(),
            GrokSearchRetriever())

    def pack_builder(sid: str, text: str, context: str):
        return build_evidence_pack_v2(sid, text, trio, context=context)

    return pack_builder


def _publish_bundles(args, bundles: list, date, source_url: str,
                     characterization: Optional[list] = None,
                     panel_roster: Optional[dict] = None) -> None:
    """Shared publish tail: bundles → SiteReport → static site (+ link/JSON checks).

    Used by both the legacy and the v2 (PCA) verify paths so they emit byte-identical
    site structure from the same ``VerdictBundle`` list."""
    import json
    import uuid

    from truthbot.publish.site import SitePublisher, SiteReport

    site_report = SiteReport(
        report_id=str(uuid.uuid4()),
        speaker=args.speaker,
        role=getattr(args, "role", "") or "",
        date=date,
        venue=getattr(args, "venue", "") or "",
        transcript_source_url=source_url,
        bundles=bundles,
        characterization=list(characterization or []),
        panel_roster=dict(panel_roster or {}),
    )
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

    issues = []
    for b in bundles:
        claim_rel = Path("claims") / f"{b.claim.id}.html"
        if not (Path(stats['root']) / claim_rel).exists():
            issues.append(str(claim_rel))
    if issues:
        print(f"WARNING: {len(issues)} claim page(s) missing")
    else:
        print("All internal links verified OK")

    for fname in ("data/reports.json", "data/claims.json"):
        p = Path(stats['root']) / fname
        try:
            json.loads(p.read_text(encoding="utf-8"))
            print(f"{fname}: valid JSON")
        except Exception as e:
            print(f"{fname}: INVALID - {e}")


def _run_publish_pca(args) -> None:
    """v2 publish path: segment → Layer A → chunked PCA adjudicate → bridge → site.

    Uses the HydraMind PCA stack over the truth-bot proxy lane instead of the legacy
    per-provider ``VerificationEngine``. Open-book (Brave/FactCheck + CRM-114) when
    BRAVE_API_KEY is set; degrades loudly to closed-book otherwise. Costs real proxy
    spend — this is the live path PR-D exercises end-to-end."""
    from collections import Counter
    from datetime import datetime

    from truthbot.verdict import adjudicator, proxy_lane, publish_pipeline

    if not proxy_lane.key_present():
        print(proxy_lane.BLOCKED_MSG)
        sys.exit(1)

    src = args.transcript
    if src == "-":
        text = sys.stdin.read()
    else:
        text = Path(src).read_text(encoding="utf-8")
    date = datetime.strptime(args.date, "%Y-%m-%d")
    source_url = getattr(args, "source_url", "") or ""

    speech_id = getattr(args, "speech_id", None) or _default_speech_id(args.speaker, date)
    sentences = publish_pipeline.prepare_speech(text, speech_id, date.date())
    print(f"Segmented {len(sentences)} sentence(s) (speech_id={speech_id}, "
          f"utterance={date.date().isoformat()})")

    evidence_mode = getattr(args, "evidence_mode", "v1") or "v1"
    provider = None
    pack_builder = None
    if evidence_mode == "v2":
        pack_builder = _build_v2_pack_builder()   # fails LOUD on a missing lane
    else:
        provider = _build_open_book_provider()
        if provider is None:
            print("WARNING: BRAVE_API_KEY not set — running CLOSED-BOOK "
                  "(no evidence, CRM-114 disabled).")

    crm114 = not bool(getattr(args, "no_crm114", False))
    open_book = provider is not None or pack_builder is not None
    # Two engines: Layer A classify needs the raw/identity parser (parse_a2 reads the
    # {"label", …} JSON itself); the verdict panel + CRM-114 need parse_verdict.
    hm_classify = proxy_lane.build_hydramind()
    hm_verdict = proxy_lane.build_hydramind(response_parser=adjudicator.parse_verdict)
    layer_a_fn, adjudicate_fn = publish_pipeline.build_pca_lane_fns(
        hm_classify, hm_verdict, provider, pack_builder=pack_builder,
        crm114=crm114,
        roster=getattr(args, "roster", "dev") or "dev",
        a2_tier=getattr(args, "a2_tier", "cheap") or "cheap",
    )
    chunk_size = int(getattr(args, "chunk_size", 6) or 6)

    def _on_progress(i: int, n: int, rows: list) -> None:
        dist = Counter(r.get("verdict") or r.get("status") for r in rows)
        print(f"  adjudicate chunk {i}/{n}: {dict(dist)}")

    mode_label = ("open-book+crm114" if open_book and crm114
                  else ("open-book" if open_book else "closed-book"))
    print(f"Verifying via PCA (roster={getattr(args, 'roster', 'dev')}, "
          f"chunk_size={chunk_size}, evidence={evidence_mode}, "
          f"mode={mode_label})...")
    # P67.3: chunk journal + resume + preflight budget probe. The journal
    # defaults ON (metrics/journals/<speech_id>.jsonl): a mid-run failure keeps
    # every completed chunk, and re-running with --resume re-spends only on
    # sids that never finished.
    from truthbot.config import settings as _cfg
    journal_path = getattr(args, "journal", None) or (
        Path(_cfg.metrics_dir) / "journals" / f"{speech_id}.jsonl")
    resume_rows: list = []
    resume_packs: dict = {}
    if getattr(args, "resume", False):
        resume_rows, resume_packs, prior_cost, _ = \
            publish_pipeline.load_chunk_journal(journal_path)
        if resume_rows:
            print(f"  resume: {len(resume_rows)} journaled rows from "
                  f"{journal_path} (banked spend ${prior_cost:.2f})")
    budget_check = None
    budget_cap = float(getattr(args, "budget_cap", 0) or 0)
    if budget_cap:
        from truthbot.verdict.proxy_lane import proxy_key_spend
        _start_spend = proxy_key_spend()

        def budget_check() -> float:
            return budget_cap - (proxy_key_spend() - _start_spend)

    result = publish_pipeline.run_pca_verify(
        sentences,
        layer_a_fn=layer_a_fn,
        adjudicate_fn=adjudicate_fn,
        chunk_size=chunk_size,
        on_progress=_on_progress,
        resume_rows=resume_rows,
        resume_packs=resume_packs,
        journal_path=journal_path,
        budget_check=budget_check,
    )
    print(f"Layer A: {result.n_check_worthy}/{result.n_sentences} check-worthy; "
          f"adjudicated in {result.n_chunks} chunk(s); spend ${result.cost_usd:.4f}")

    # Persist the raw rows + claims BEFORE publishing so a re-render never needs
    # another live run. run_id ties the artifact to this speech/run.
    import uuid
    from truthbot.config import settings as _settings
    run_id = str(uuid.uuid4())
    _persist_pca_run(
        run_id,
        result,
        meta={
            "speaker": args.speaker,
            "date": args.date,
            "speech_id": speech_id,
            "venue": getattr(args, "venue", "") or "",
            "roster": getattr(args, "roster", "dev") or "dev",
            "n_sentences": result.n_sentences,
            "n_check_worthy": result.n_check_worthy,
            "cost_usd": result.cost_usd,
        },
        metrics_dir=_settings.metrics_dir,
    )

    _publish_bundles(args, result.bundles, date, source_url,
                     characterization=getattr(result, "characterization", None),
                     panel_roster=getattr(result, "roster", None))


def _run_publish(args) -> None:
    """Full pipeline: ingest → extract → verify → publish site."""
    import os, uuid
    from datetime import datetime
    from truthbot.config import settings
    from truthbot.extract.claims import ClaimExtractor
    from truthbot.ingest.transcript import TranscriptIngester
    from truthbot.verify.engine import VerificationEngine
    from truthbot.publish.site import SitePublisher, SiteReport

    # v2 (HydraMind PCA) verify path is the default; --engine legacy selects the
    # per-provider VerificationEngine. Batch mode is legacy-only, so a pca+batch
    # request falls back to the legacy batch path rather than silently ignoring --mode.
    engine = getattr(args, "engine", "pca")
    if engine == "pca" and getattr(args, "mode", "live") == "batch":
        print("NOTE: --mode batch is legacy-only; using --engine legacy for this run.")
        engine = "legacy"
    if engine == "pca":
        _run_publish_pca(args)
        return

    _preflight_key_sanity()

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
    # Persist claims to disk BEFORE any slicing or verification work. If the
    # batch submit step later fails, this jsonl is the only record of what
    # Claude Sonnet actually produced — resubmitting then costs nothing, not
    # another $0.50-ish extraction.
    run_id = str(uuid.uuid4())
    _persist_extracted_claims(run_id, claims, metrics_dir=settings.metrics_dir)
    all_checkable = [c for c in claims if c.is_checkable]
    # ``--max-claims 0`` means "no cap / verify every checkable claim". We always
    # preserve the full extracted/checkable counts in logs and telemetry even if
    # the verification slice is smaller.
    raw_cap = getattr(args, "max_claims", 0)
    max_claims = int(raw_cap) if raw_cap is not None else 0
    if max_claims and max_claims > 0:
        checkable = all_checkable[:max_claims]
    else:
        checkable = all_checkable
    print(
        f"  {len(claims)} claims extracted total, {len(all_checkable)} checkable, "
        f"{len(checkable)} selected for verification"
        + (f" (cap={max_claims})" if max_claims > 0 else " (no cap)")
    )

    # Verify — parallel fan-out across claims (adapters already fan-out within each claim).
    # ``run_id`` was already minted up-top so the extractions/<run_id>.jsonl
    # side-file matches whatever telemetry + batch descriptors use.
    mode = getattr(args, "mode", "live") or "live"
    from truthbot.metrics.telemetry import finalize_run, telemetry_run_context
    from truthbot.verify.batch import BatchDispatcher

    inject_evidence = resolve_inject_evidence(
        settings.evidence_source,
        no_inject_flag=bool(getattr(args, "no_inject_evidence", False)),
        inject_flag=bool(getattr(args, "inject_evidence", False)),
    )
    print(
        f"  evidence_source={settings.evidence_source} inject_evidence={inject_evidence}"
    )

    engine = VerificationEngine(
        run_id=run_id,
        inject_evidence=inject_evidence,
        triage_enabled=bool(getattr(args, "triage", False)),
        triage_threshold=float(getattr(args, "triage_threshold", 0.8)),
        triage_shadow_rate=float(getattr(args, "triage_shadow_rate", 0.0)),
        verify_mode=mode,
    )

    if mode == "batch":
        _run_publish_batch_submit(
            args=args,
            engine=engine,
            claims=claims,
            checkable=checkable,
            run_id=run_id,
            inject_evidence=inject_evidence,
            source_url=source_url,
            max_claims=max_claims,
            settings=settings,
        )
        return

    bundles_map: dict[int, object] = {}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Phase E — live multi-claim fan-out: when the operator requested
    # multi-claim bundling (``TRUTHBOT_CLAIMS_PER_REQUEST > 1``) AND at least
    # one active adapter has ``max_claims_per_request > 1``, route the live
    # path through ``verify_bundles_batch`` so Grok/Gemini can fold
    # SYNTHESIS_SYSTEM over N claims per API call. Adapters that don't
    # override ``call_multi`` (Anthropic, OpenAI today) inherit the default
    # loop → byte-identical to the per-claim path.
    claims_per_request = getattr(args, "claims_per_request", None)
    if claims_per_request is None:
        claims_per_request = settings.claims_per_request
    claims_per_request = max(1, int(claims_per_request))
    any_multi_capable = any(
        int(getattr(a, "max_claims_per_request", 1)) > 1 for a in engine.adapters
    )
    use_multi_fanout = (
        claims_per_request > 1 and any_multi_capable and len(checkable) > 1
    )

    if use_multi_fanout:
        print(
            f"  Live multi-claim fan-out: claims_per_request={claims_per_request}, "
            f"{len(checkable)} claim(s)"
        )
        try:
            with telemetry_run_context(
                run_id=run_id,
                evidence_injected=inject_evidence,
                synthesis_mode=mode,
            ):
                bundles_list = engine.verify_bundles_batch(
                    checkable,
                    speaker=args.speaker,
                    date_str=args.date,
                )
            # Preserve positional index → bundle map for downstream ordering.
            by_claim_id = {b.claim.id: b for b in bundles_list}
            for idx, claim in enumerate(checkable, 1):
                bundle = by_claim_id.get(claim.id)
                if bundle is None:
                    logger.warning(
                        "verify_bundles_batch dropped claim %s", claim.id
                    )
                    continue
                label = bundle.consensus.consensus_label.value
                strength = bundle.consensus.consensus_strength
                cache = " [cached]" if bundle.cache_hit else ""
                print(f"    -> claim {idx}: {label} ({strength}){cache}")
                bundles_map[idx] = bundle
        except Exception as exc:
            logger.error("verify_bundles_batch failed: %s", exc)
    else:
        def _verify_one(idx_claim):
            idx, claim = idx_claim
            print(f"  Verifying claim {idx}/{len(checkable)}: {claim.text[:60]}...")
            try:
                with telemetry_run_context(
                    run_id=run_id,
                    evidence_injected=inject_evidence,
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
            futs = {
                pool.submit(_verify_one, (i, c)): i
                for i, c in enumerate(checkable, 1)
            }
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

    tool_stats_parser = metrics_sub.add_parser(
        "tool-stats",
        help=(
            "Per-adapter tool-call distribution + URL-grounding rates "
            "(post-mortem helper for arm-X probes)"
        ),
    )
    tool_stats_parser.add_argument(
        "--jsonl",
        help="Path to adapter_calls.jsonl (default: settings.metrics_dir/adapter_calls.jsonl)",
    )
    tool_stats_parser.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="Only include telemetry rows with this run_id",
    )
    tool_stats_parser.add_argument(
        "--adapter",
        default=None,
        help="Only include rows from this adapter (anthropic / openai / gemini / xai)",
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
    pub_parser.add_argument(
        "--max-claims",
        type=int,
        default=0,
        help=(
            "Max checkable claims to verify. 0 (default) = no cap, verify every "
            "checkable claim the extractor returns. The total-extracted and "
            "total-checkable counts are always preserved in logs/telemetry."
        ),
    )
    pub_parser.add_argument(
        "--mode",
        choices=("live", "batch"),
        default="live",
        help="Verification billing mode: live API calls (default) or batch descriptor + same live verify for now",
    )
    pub_parser.add_argument(
        "--engine",
        choices=("legacy", "pca"),
        default="pca",
        help=(
            "Verification engine. 'pca' (default) = the v2 HydraMind stack (segment → "
            "Layer A check-worthy → open-book PCA + CRM-114 → bridge), over the "
            "truth-bot proxy lane; ignores --mode/--triage/--claims-per-request. "
            "'legacy' = the per-provider VerificationEngine (required for --mode batch)."
        ),
    )
    pub_parser.add_argument(
        "--speech-id",
        dest="speech_id",
        default=None,
        help=(
            "sid prefix for the PCA path (temporal-grounding key). Use the pinned "
            "fixture slug (trump_2026 / biden_2022) to reuse its registered utterance "
            "date; omitted → derived from speaker+year and registered against --date."
        ),
    )
    pub_parser.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=6,
        help="PCA path: check-worthy claims per adjudicate call (proxy rate-limit control; default 6).",
    )
    pub_parser.add_argument(
        "--evidence-mode",
        dest="evidence_mode",
        choices=("v1", "v2"),
        default="v1",
        help=(
            "Retrieval stack (T2.7). 'v1' (default) = Brave/FactCheck connectors "
            "(shared_pack_v1, the ablation baseline). 'v2' = R1/R2/R3 retriever "
            "trio → deterministic consolidator with the T2.4 quality gate "
            "(shared_pack_v2, the Phase 3 rerun stack); needs the claude CLI, "
            "OPENAI_API_KEY and XAI_API_KEY."
        ),
    )
    pub_parser.add_argument(
        "--journal",
        dest="journal",
        default=None,
        help="P67.3: chunk-journal JSONL path (default metrics/journals/<speech_id>.jsonl).",
    )
    pub_parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="P67.3: load the journal and skip already-adjudicated sids (banked spend).",
    )
    pub_parser.add_argument(
        "--budget-cap",
        dest="budget_cap",
        type=float,
        default=0.0,
        help="P67.3: halt BEFORE a chunk when proxy-key headroom under this cap "
             "falls below the projected chunk cost (0 = no probe).",
    )
    pub_parser.add_argument(
        "--roster",
        default="dev",
        help="PCA path: HydraMind roster (default 'dev' = cheap tiers).",
    )
    pub_parser.add_argument(
        "--a2-tier",
        dest="a2_tier",
        default="cheap",
        help="PCA path: Layer A A2 classifier tier (cheap=haiku / standard=sonnet / frontier=opus).",
    )
    pub_parser.add_argument(
        "--no-crm114",
        dest="no_crm114",
        action="store_true",
        help="PCA path: disable the CRM-114 FALSE-vs-MISLEADING stage-2 discriminator (open-book only).",
    )
    pub_parser.add_argument(
        "--claims-per-request",
        dest="claims_per_request",
        type=int,
        default=None,
        help=(
            "Bundle this many atomic claims into a single LLM request "
            "(per provider, clamped by adapter.max_claims_per_request). "
            "Default reads TRUTHBOT_CLAIMS_PER_REQUEST (1 = no batching). "
            "Currently honored in --mode batch only; --mode live ignores."
        ),
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
        help=(
            "Force-skip evidence injection even when TRUTHBOT_EVIDENCE_SOURCE is set "
            "(telemetry evidence_injected=false). Wins over --inject-evidence."
        ),
    )
    pub_parser.add_argument(
        "--inject-evidence",
        action="store_true",
        help=(
            "Force-enable evidence injection even when TRUTHBOT_EVIDENCE_SOURCE=none "
            "(default tracks the source: none -> off, connectors/datahoover -> on)."
        ),
    )

    batch_parser = subparsers.add_parser("batch", help="Batch job helpers")
    batch_sub = batch_parser.add_subparsers(dest="batch_cmd", required=True)
    batch_poll = batch_sub.add_parser(
        "poll", help="Poll batch job status for a run_id (no reconcile/publish)"
    )
    batch_poll.add_argument("run_id", help="Publish run UUID written under metrics/batch_jobs/")
    batch_reconcile = batch_sub.add_parser(
        "reconcile",
        help="Poll → fetch results → merge sidecar → cache bundles → publish site",
    )
    batch_reconcile.add_argument("run_id", help="Publish run UUID written under metrics/batch_jobs/")
    batch_reconcile.add_argument(
        "--site-root",
        default=None,
        help="Override site output root (else reads from descriptor or TRUTHBOT_SITE_ROOT)",
    )
    batch_reconcile.add_argument(
        "--validate-urls",
        action="store_true",
        help=(
            "HEAD-check every URL across all merged verdicts and populate "
            "ModelVerdict.url_classifications so the publish layer renders "
            "the three trust tiers (verified / unverified / broken). Uses "
            "the metrics/url_cache.jsonl cache so re-checks are nearly "
            "free. Recommended for the final published report."
        ),
    )

    # Phase 3b: URL reachability subcommand.
    urls_parser = subparsers.add_parser(
        "urls", help="URL reachability checks over a run's web_sources"
    )
    urls_sub = urls_parser.add_subparsers(dest="urls_cmd", required=True)
    urls_check = urls_sub.add_parser(
        "check",
        help="HEAD-check every unique URL in a sidecar and print classification stats",
    )
    urls_check.add_argument(
        "sidecar",
        help="Path to metrics/batch_sidecar/<run_id>.jsonl (one row per verdict)",
    )
    urls_check.add_argument(
        "--cache",
        default="metrics/url_cache.jsonl",
        help="Path to the URL reachability cache (default: metrics/url_cache.jsonl)",
    )
    urls_check.add_argument("--timeout", type=float, default=5.0)
    urls_check.add_argument("--max-workers", type=int, default=8)

    urls_filter = urls_sub.add_parser(
        "filter-sidecar",
        help=(
            "HEAD-check every URL in a sidecar and write a cleaned sidecar "
            "with URLs partitioned into verified / unverified / broken "
            "(Layer 3 of the anti-hallucination defense-in-depth)."
        ),
    )
    urls_filter.add_argument(
        "sidecar",
        help="Path to metrics/batch_sidecar/<run_id>.jsonl (input).",
    )
    urls_filter.add_argument(
        "--out",
        default=None,
        help=(
            "Output path for the cleaned sidecar. Defaults to "
            "<sidecar-stem>.cleaned.jsonl alongside the input."
        ),
    )
    urls_filter.add_argument(
        "--cache",
        default="metrics/url_cache.jsonl",
        help="Path to the URL reachability cache (default: metrics/url_cache.jsonl).",
    )
    urls_filter.add_argument("--timeout", type=float, default=5.0)
    urls_filter.add_argument("--max-workers", type=int, default=8)

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
        elif getattr(args, "metrics_cmd", None) == "tool-stats":
            _print_tool_stats(
                getattr(args, "jsonl", None),
                run_id=getattr(args, "run_id", None),
                adapter=getattr(args, "adapter", None),
            )
            return
        else:
            parser.print_help()
            return

    if getattr(args, "subcommand", None) == "urls":
        if getattr(args, "urls_cmd", None) == "check":
            _run_urls_check(
                sidecar_path=args.sidecar,
                cache_path=args.cache,
                timeout=args.timeout,
                max_workers=args.max_workers,
            )
            return
        if getattr(args, "urls_cmd", None) == "filter-sidecar":
            _run_urls_filter_sidecar(
                sidecar_path=args.sidecar,
                out_path=args.out,
                cache_path=args.cache,
                timeout=args.timeout,
                max_workers=args.max_workers,
            )
            return
        parser.print_help()
        return

    if getattr(args, "subcommand", None) == "batch":
        if getattr(args, "batch_cmd", None) == "poll":
            from truthbot.config import settings
            from truthbot.verify.batch import BatchDispatcher

            st = BatchDispatcher(settings.metrics_dir).poll(args.run_id)
            print(st)
            return
        if getattr(args, "batch_cmd", None) == "reconcile":
            _run_publish_batch_reconcile(
                args.run_id,
                site_root=getattr(args, "site_root", None),
                validate_urls=bool(getattr(args, "validate_urls", False)),
            )
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
