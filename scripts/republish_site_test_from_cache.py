"""
Republish the local ``site-test/`` artifacts from the on-disk bundle cache,
populating the new 5-bucket coarse-axis projection fields without making any
LLM calls.

Why this script exists
----------------------
The 5-bucket coarse-axis projection (``coarse_lenient_*`` / ``coarse_strict_*``
on ``ConsensusVerdict``) was added *after* the existing ``site-test/`` artifacts
and the bundle cache were generated. Cached bundles therefore still hold a
``ConsensusVerdict`` with empty coarse fields, and a vanilla ``truthbot publish``
rerun would cache-hit those legacy bundles and render via the fine-axis
fallback. That hides the new headline pill / lens chip behaviour from the
GitHub Pages preview.

A full re-fire would be expensive (frontier LLM calls × 84 claims × 4 models).
This script is the cheap path: every bundle in the cache already contains the
full ``model_verdicts`` list, which is the only input ``_build_consensus``
needs to compute fresh consensus + projections. So we walk the cache, rebuild
each bundle's consensus in place, then sequentially re-render every report in
``site-test/data/reports.json`` from the refreshed cache.

What it does
------------
1. **Cache rebuild step** (``rebuild_consensus_in_cache``):
   Iterates every entry in the diskcache at ``$TRUTHBOT_CACHE_DIR/bundles``
   (default: ``truthbot_cache/bundles``). For each entry:
     - parse JSON → ``VerdictBundle``
     - ``new_consensus = _build_consensus(bundle.claim.id, bundle.model_verdicts)``
     - replace ``bundle.consensus`` with ``new_consensus``
     - write ``bundle.model_dump_json()`` back under the same key.
   Idempotent: re-running produces byte-identical output.

2. **Republish step** (``republish_site_test``):
   Loads ``site-test/data/reports.json`` and ``site-test/data/claims.json`` to
   recover the published report list and the (claim_id → report_id) mapping.
   Builds an ``id → VerdictBundle`` map by walking the (now-refreshed) cache.
   For each report it gathers its bundles by claim_id, constructs a
   ``SiteReport`` (preserving the original ``report_id`` so URLs stay stable),
   and calls ``SitePublisher.publish(...)`` against the ``site-test/`` root.

Usage
-----
    .venv/bin/python scripts/republish_site_test_from_cache.py
    .venv/bin/python scripts/republish_site_test_from_cache.py --rebuild-only
    .venv/bin/python scripts/republish_site_test_from_cache.py --skip-rebuild

Default site root is ``site-test/`` per project convention. Provide
``--site-root site`` (or any path) to retarget.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make src/ importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from truthbot.models import VerdictBundle  # noqa: E402
from truthbot.publish.site import SitePublisher, SiteReport  # noqa: E402
from truthbot.verify.engine import _build_consensus  # noqa: E402


# ── Cache rebuild ──────────────────────────────────────────────────────────────


def rebuild_consensus_in_cache(cache_dir: Path) -> dict[str, int]:
    """Walk the bundle cache and recompute consensus for every entry.

    Returns a small stats dict: ``{"total", "rewrote", "skipped", "errors"}``.

    Idempotency: ``_build_consensus`` is a pure function of ``model_verdicts``,
    so a second pass produces the same JSON and writes are no-ops in practice
    (diskcache will still touch the file but content is byte-identical).
    """
    import diskcache

    if not cache_dir.exists():
        raise SystemExit(f"Cache dir not found: {cache_dir}")

    stats = {"total": 0, "rewrote": 0, "skipped": 0, "errors": 0}
    cache = diskcache.Cache(str(cache_dir))
    try:
        keys = list(cache.iterkeys())
        stats["total"] = len(keys)
        print(f"[rebuild] {len(keys)} entries in {cache_dir}")
        for i, key in enumerate(keys, 1):
            raw = cache.get(key)
            if not raw:
                stats["skipped"] += 1
                continue
            try:
                bundle = VerdictBundle.model_validate_json(raw)
            except Exception as exc:
                print(f"[rebuild] {i:>3}/{len(keys)} key={key} parse-error: {exc}")
                stats["errors"] += 1
                continue

            if not bundle.model_verdicts:
                # Defensive: a bundle without per-model verdicts can't be
                # re-projected. Leave it alone.
                stats["skipped"] += 1
                continue

            new_consensus = _build_consensus(
                bundle.claim.id, bundle.model_verdicts
            )
            old = bundle.consensus
            bundle.consensus = new_consensus
            try:
                cache.set(key, bundle.model_dump_json())
                stats["rewrote"] += 1
            except Exception as exc:
                print(f"[rebuild] {i:>3}/{len(keys)} key={key} write-error: {exc}")
                stats["errors"] += 1
                continue

            if i % 25 == 0 or i == len(keys):
                print(
                    f"[rebuild] {i:>3}/{len(keys)} "
                    f"claim={bundle.claim.id[:8]} "
                    f"fine={old.consensus_label.value} -> {new_consensus.consensus_label.value} "
                    f"lenient={new_consensus.coarse_lenient_label!r}/"
                    f"{new_consensus.coarse_lenient_strength} "
                    f"strict={new_consensus.coarse_strict_label!r}/"
                    f"{new_consensus.coarse_strict_strength}"
                )
    finally:
        cache.close()

    print(
        f"[rebuild] done: total={stats['total']} rewrote={stats['rewrote']} "
        f"skipped={stats['skipped']} errors={stats['errors']}"
    )
    return stats


# ── Republish ──────────────────────────────────────────────────────────────────


def _load_bundle_index(
    cache_dir: Path,
) -> tuple[dict[str, VerdictBundle], dict[tuple[str, str, str], VerdictBundle]]:
    """Build two parallel lookup maps over the cache.

    Returns ``(by_claim_id, by_content_key)`` where:
      * ``by_claim_id`` maps each cached bundle's ``claim.id`` to the bundle.
      * ``by_content_key`` maps ``(claim_text_norm, speaker_norm, date_str)``
        to the bundle. This mirrors the engine's ``_cache_key`` triple and
        lets us recover bundles whose ``claim.id`` was rotated by a later
        run (fresh UUID per ``Claim``, but cache key is content-stable).

    The republisher tries ``by_claim_id`` first to preserve the existing
    claim URLs in ``site-test/``; on a miss it falls back to
    ``by_content_key`` so older reports whose claim ids were overwritten
    can still be re-rendered. Falling back to content rotates the claim
    id (and therefore the per-claim HTML filename) — that's an accepted
    trade-off: re-running the LLMs would do the same.
    """
    import diskcache

    cache = diskcache.Cache(str(cache_dir))
    by_id: dict[str, VerdictBundle] = {}
    by_content: dict[tuple[str, str, str], VerdictBundle] = {}
    try:
        for key in cache.iterkeys():
            raw = cache.get(key)
            if not raw:
                continue
            try:
                bundle = VerdictBundle.model_validate_json(raw)
            except Exception:
                continue
            by_id[bundle.claim.id] = bundle
            content_key = (
                (bundle.claim.text or "").strip().lower(),
                (bundle.speaker or "").strip().lower(),
                (bundle.date_str or "").strip(),
            )
            by_content[content_key] = bundle
    finally:
        cache.close()
    return by_id, by_content


def _parse_date(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None


def republish_site_test(
    site_root: Path,
    cache_dir: Path,
    reports_json: Path,
    claims_json: Path,
) -> dict[str, int]:
    """Re-render every report listed in ``reports_json`` from the cache."""
    if not reports_json.exists():
        raise SystemExit(f"reports.json not found: {reports_json}")
    if not claims_json.exists():
        raise SystemExit(f"claims.json not found: {claims_json}")

    reports_meta = json.loads(reports_json.read_text())
    claims_meta = json.loads(claims_json.read_text())

    print(
        f"[republish] {len(reports_meta)} reports / {len(claims_meta)} claims "
        f"from {reports_json}"
    )
    by_id, by_content = _load_bundle_index(cache_dir)
    print(
        f"[republish] indexed from {cache_dir}: "
        f"{len(by_id)} by claim_id, {len(by_content)} by content-key"
    )

    # claim_id -> claim record (we need the claim_text for content-key lookup)
    claim_record_by_id: dict[str, dict] = {}
    # report_id -> [claim_id, ...] in the order they appear in claims.json
    report_to_claims: dict[str, list[str]] = {}
    for c in claims_meta:
        cid = c.get("id")
        rid = c.get("report_id")
        if not cid or not rid:
            continue
        claim_record_by_id[cid] = c
        report_to_claims.setdefault(rid, []).append(cid)

    publisher = SitePublisher(site_root=str(site_root))

    stats = {"reports": 0, "bundles": 0, "missing": 0, "by_content": 0}
    for r in reports_meta:
        report_id = r["id"]
        speaker_norm = (r.get("speaker") or "").strip().lower()
        date_norm = (r.get("date") or "").strip()
        bundles: list[VerdictBundle] = []
        missing: list[str] = []
        by_content_hits = 0
        for cid in report_to_claims.get(report_id, []):
            b = by_id.get(cid)
            if b is None:
                # Fallback: look up by content key. The original bundle for
                # this (claim_text, speaker, date) was overwritten in the
                # cache by a newer run with a fresh UUID. Use the newer
                # bundle — re-running the LLMs would do the same thing.
                rec = claim_record_by_id.get(cid)
                if rec is not None:
                    content_key = (
                        (rec.get("claim_text") or "").strip().lower(),
                        speaker_norm,
                        date_norm,
                    )
                    b = by_content.get(content_key)
                    if b is not None:
                        by_content_hits += 1
            if b is None:
                missing.append(cid)
                continue
            bundles.append(b)

        if not bundles:
            print(
                f"[republish] WARN report={report_id[:8]} "
                f"no bundles found in cache (missing={len(missing)}); skipping"
            )
            continue

        site_report = SiteReport(
            report_id=report_id,
            speaker=r.get("speaker", ""),
            role=r.get("role", ""),
            date=_parse_date(r.get("date", "")),
            venue=r.get("venue", ""),
            transcript_source_url="",   # not persisted in reports.json; cosmetic
            bundles=bundles,
            video_source_url="",
            source_of_claims=r.get("source_of_claims", ""),
            source_of_claims_professional_public_title=r.get(
                "source_of_claims_professional_public_title", ""
            ),
            event=r.get("event", ""),
            channel=r.get("channel", ""),
        )

        path = publisher.publish(site_report)
        stats["reports"] += 1
        stats["bundles"] += len(bundles)
        stats["missing"] += len(missing)
        stats["by_content"] += by_content_hits
        print(
            f"[republish] wrote {path.relative_to(site_root.resolve())} "
            f"({len(bundles)} bundles, {by_content_hits} via content-key fallback, "
            f"{len(missing)} missing)"
        )

    summary = publisher.summary()
    print(
        f"[republish] done: reports={stats['reports']} "
        f"bundles={stats['bundles']} (content-key={stats['by_content']}) "
        f"missing={stats['missing']} site_root={summary['root']}"
    )
    return stats


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-root",
        default="site-test",
        help="Site output root (default: site-test, project convention).",
    )
    parser.add_argument(
        "--cache-dir",
        default="truthbot_cache/bundles",
        help="Bundle cache directory (default: truthbot_cache/bundles).",
    )
    parser.add_argument(
        "--reports-json",
        default=None,
        help="reports.json path (default: <site-root>/data/reports.json).",
    )
    parser.add_argument(
        "--claims-json",
        default=None,
        help="claims.json path (default: <site-root>/data/claims.json).",
    )
    parser.add_argument(
        "--rebuild-only",
        action="store_true",
        help="Refresh consensus in the bundle cache and stop (no republish).",
    )
    parser.add_argument(
        "--skip-rebuild",
        action="store_true",
        help="Skip the cache rebuild step; republish from the cache as-is.",
    )
    args = parser.parse_args()

    if args.rebuild_only and args.skip_rebuild:
        parser.error("--rebuild-only and --skip-rebuild are mutually exclusive")

    site_root = Path(args.site_root).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    reports_json = Path(args.reports_json) if args.reports_json else site_root / "data" / "reports.json"
    claims_json = Path(args.claims_json) if args.claims_json else site_root / "data" / "claims.json"

    if not args.skip_rebuild:
        rebuild_consensus_in_cache(cache_dir)

    if args.rebuild_only:
        return 0

    republish_site_test(
        site_root=site_root,
        cache_dir=cache_dir,
        reports_json=reports_json,
        claims_json=claims_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
