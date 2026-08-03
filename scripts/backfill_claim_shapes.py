#!/usr/bin/env python3
"""Claim-shape backfill for pre-role-axis pca_runs artifacts (remediation v2).

The published trump_2026 / biden_2022 / obama_2014 artifacts predate the
claim-shape axis (PR-A2.3): their claims[] carry no ``layer_a.claim_shape``,
so a Phase-3 rebuild would run them under the LEGACY evidential-role quota
while clinton_1998 / gwbush_2006 run role-aware — two methodologies in one
corpus. This script classifies each shapeless claim with the SAME Layer A
machinery that produced clinton/gwbush's shapes — ``classifier.classify``
(prompt ``A2_SYSTEM``, HydraMind ``single`` strategy, tier=cheap →
claude-haiku on the L-P proxy lane, identity response parser, tolerant
parse mode) — and records the shapes in a SIDECAR file. The source
artifacts are NEVER mutated (archive-never-delete); phase3_rebuild.py
merges the sidecar at run time via ``--shapes-sidecar``.

These claims are ALREADY published check-worthy claims. If the classifier
returns a non-check-worthy label (or no shape) for one, the claim is KEPT
and recorded with shape "" (legacy quota for that claim) plus a warning —
claims are never dropped.

Sidecar: metrics/remediation_v2/shapes_backfill_<speech>.json, schema
``truthbot-shape-backfill v1``. Written incrementally (atomic rewrite per
classified claim), so a crash resumes without re-spending: sids already in
the sidecar are never re-classified.

Usage (repo root):
  set -a; . ./.env; set +a          # only needed with --go
  PYTHONPATH=.:src .venv/bin/python scripts/backfill_claim_shapes.py \\
      --speech trump_2026 [--go] [--limit N]

Default (no --go): print the plan + cost estimate ($0, no key needed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import phase3_rebuild as p3  # noqa: E402  (speech registry / artifact map SSOT)

SCHEMA = "truthbot-shape-backfill v1"
SIDECAR_DIR = REPO / "metrics" / "remediation_v2"

# The exact production Layer A lane (mirrors publish_pipeline.build_pca_lane_fns):
# classifier.classify on the L-P proxy, HydraMind "single" strategy, tier=cheap
# → proxy alias "claude-haiku", identity response parser, tolerant parse mode.
A2_TIER = "cheap"
CLASSIFIER_ID = "claude-haiku (L-P proxy, hydramind single, tier=cheap)"
BATCH = 25              # same paced batching as build_pca_lane_fns
PAUSE_S = 1.0

# $0 estimator: claude-haiku fallback rates from hydramind.models
# RATE_TABLE_USD_PER_MTOK (proxy-priced in practice; this is the plan-mode
# projection only). ~4 chars/token; output is a one-line JSON verdict.
_HAIKU_RATES = (0.80, 4.00)      # USD per Mtok (in, out)
_OUT_TOKENS_EST = 80

# classify_fn signature: (sentences[{"sid","text","context"}]) -> A2 rows
ClassifyFn = Callable[[list[dict]], list[dict]]

_VOCAB = {"c-exist", "c-count", "c-eval", "c-third"}


# ── $0 helpers ───────────────────────────────────────────────────────────────

def sidecar_path(speech: str) -> Path:
    return SIDECAR_DIR / f"shapes_backfill_{speech}.json"


def shapeless_claims(claims: list[dict]) -> list[dict]:
    """Claims lacking a ``layer_a.claim_shape`` (the backfill's scope)."""
    return [c for c in claims
            if not (c.get("layer_a") or {}).get("claim_shape")]


def new_sidecar(speech: str, source_run: str) -> dict:
    return {"schema": SCHEMA, "speech_id": speech, "source_run": source_run,
            "classifier": CLASSIFIER_ID, "shapes": {}, "warnings": []}


def load_sidecar(path: Path, speech: str, source_run: str) -> dict:
    """Load an existing sidecar for resume; fail loudly on any mismatch
    (never silently mix shapes across speeches or artifact revisions)."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r} != {SCHEMA!r}")
    if doc.get("speech_id") != speech:
        raise ValueError(f"{path}: speech_id {doc.get('speech_id')!r} != {speech!r}")
    if doc.get("source_run") != source_run:
        raise ValueError(f"{path}: source_run {doc.get('source_run')!r} != "
                         f"{source_run!r} (artifact changed?)")
    doc.setdefault("shapes", {})
    doc.setdefault("warnings", [])
    return doc


def write_sidecar(path: Path, doc: dict) -> None:
    """Atomic rewrite (tmp + rename) — a crash never leaves a torn sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def row_to_shape(row: dict) -> tuple[str, Optional[str]]:
    """(shape, warning) for one classifier row. Published claims are never
    dropped: a non-check-worthy label, or a check-worthy row with no shape,
    keeps the claim at shape "" (legacy quota for that claim) with a warning."""
    sid = row.get("sid", "?")
    label = row.get("label")
    if label != "check-worthy":
        return "", (f"{sid}: classifier returned label={label!r} for a "
                    f"published claim — kept, shape=\"\" (legacy quota)")
    shape = row.get("claim_shape") or ""
    if not shape:
        return "", (f"{sid}: check-worthy but no claim_shape emitted — "
                    f"kept, shape=\"\" (legacy quota)")
    return shape, None


def lint_sidecar_shapes(doc: dict, text_by_sid: dict[str, str]) -> int:
    """Validate the recorded shapes with the deterministic shape lint
    (shape_lint.enforce_shape) + the vocabulary. parse_a2 already lints
    fresh rows; this pass also covers resumed/hand-edited entries. Any
    correction is applied in place and warned. Returns # corrections."""
    from truthbot.checkworthy.shape_lint import enforce_shape

    fixed = 0
    for sid, shape in list(doc["shapes"].items()):
        if not shape:
            continue
        if shape not in _VOCAB:
            doc["warnings"].append(
                f"{sid}: out-of-vocabulary shape {shape!r} — cleared to \"\"")
            doc["shapes"][sid] = ""
            fixed += 1
            continue
        linted = enforce_shape(text_by_sid.get(sid, ""), shape)
        if linted != shape:
            doc["warnings"].append(
                f"{sid}: shape lint forced {shape!r} -> {linted!r} "
                f"(ministerial shape with superlative/comparative/causal tokens)")
            doc["shapes"][sid] = linted
            fixed += 1
    return fixed


def shape_tally(doc: dict) -> dict[str, int]:
    tally: dict[str, int] = {}
    for shape in doc["shapes"].values():
        key = shape or '(none — legacy)'
        tally[key] = tally.get(key, 0) + 1
    return tally


def estimate_cost(pending: list[dict]) -> float:
    """$0 projection: A2_SYSTEM + sentence + context at ~4 chars/token,
    claude-haiku fallback rates."""
    from truthbot.checkworthy.classifier import A2_SYSTEM
    rin, rout = _HAIKU_RATES
    total = 0.0
    for c in pending:
        tin = (len(A2_SYSTEM) + len(c.get("text", ""))
               + len(c.get("context", ""))) / 4.0
        total += (tin * rin + _OUT_TOKENS_EST * rout) / 1e6
    return total


# ── the backfill core (classify_fn injected — offline-testable) ──────────────

def run_backfill(speech: str, art: dict, path: Path, classify_fn: ClassifyFn,
                 *, batch: int = BATCH, pause_s: float = PAUSE_S,
                 sleep_fn=None, limit: int = 0) -> dict:
    """Classify every shapeless claim not already in the sidecar; persist per
    claim (atomic sidecar rewrite) so a crash resumes without re-spending."""
    import time

    source_run = art.get("run_id", "")
    doc = (load_sidecar(path, speech, source_run) if path.exists()
           else new_sidecar(speech, source_run))

    lacking = shapeless_claims(art["claims"])
    pending = [c for c in lacking if c["sid"] not in doc["shapes"]]
    if len(pending) < len(lacking):
        print(f"resume: {len(lacking) - len(pending)} sid(s) already in "
              f"sidecar, {len(pending)} to classify")
    if limit:
        pending = pending[:limit]

    _sleep = sleep_fn or time.sleep
    sentences = [{"sid": c["sid"], "text": c["text"],
                  "context": c.get("context", "")} for c in pending]
    n = len(sentences)
    for i in range(0, n, max(1, batch)):
        chunk = sentences[i:i + batch]
        rows = classify_fn(chunk)
        got = {r.get("sid") for r in rows}
        for s in chunk:                      # a lane MUST answer every sid
            if s["sid"] not in got:
                raise RuntimeError(f"classifier returned no row for {s['sid']}")
        for row in rows:
            shape, warning = row_to_shape(row)
            doc["shapes"][row["sid"]] = shape
            if warning:
                doc["warnings"].append(warning)
                print(f"WARNING: {warning}")
            write_sidecar(path, doc)         # per-claim persistence
        print(f"  classified {min(i + batch, n)}/{n}")
        if pause_s and i + batch < n:
            _sleep(pause_s)

    text_by_sid = {c["sid"]: c["text"] for c in art["claims"]}
    if lint_sidecar_shapes(doc, text_by_sid):
        write_sidecar(path, doc)
    elif not path.exists():                  # nothing pending, still materialize
        write_sidecar(path, doc)
    return doc


def build_live_classify_fn() -> tuple[ClassifyFn, dict]:
    """The production Layer A lane, exactly as build_pca_lane_fns binds it:
    identity-parser HydraMind on the L-P proxy, classifier.classify at
    tier=cheap, tolerant parse mode. Returns (fn, cost holder)."""
    from truthbot.checkworthy import classifier
    from truthbot.verdict import proxy_lane

    hm = proxy_lane.build_hydramind(response_parser=None)  # Layer A lane
    cost = {"usd": 0.0, "calls": 0}

    def fn(sentences: list[dict]) -> list[dict]:
        rows, manifest = classifier.classify(
            hm, sentences, tier=A2_TIER, on_parse_error="default")
        cost["calls"] += len(sentences)
        try:                                 # best-effort usage note
            cost["usd"] += float(getattr(manifest, "total_cost_usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
        return rows

    return fn, cost


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--speech", required=True, choices=sorted(p3.SPEECHES))
    ap.add_argument("--go", action="store_true",
                    help="actually classify (else plan only, $0)")
    ap.add_argument("--limit", type=int, default=0,
                    help="classify only the first N pending claims (smoke)")
    args = ap.parse_args()

    art = p3.load_artifact(args.speech)
    path = sidecar_path(args.speech)
    lacking = shapeless_claims(art["claims"])
    done: dict = {}
    if path.exists():
        done = load_sidecar(path, args.speech, art.get("run_id", ""))["shapes"]
    pending = [c for c in lacking if c["sid"] not in done]

    from truthbot.verdict import proxy_lane
    print(f"Claim-shape backfill plan — {args.speech}")
    print(f"  source artifact: {p3.artifact_path(args.speech)} "
          f"(run {art.get('run_id', '?')[:8]})")
    print(f"  claims: {len(art['claims'])} total, {len(lacking)} lacking "
          f"layer_a.claim_shape, {len(done)} already in sidecar, "
          f"{len(pending)} to classify")
    print(f"  classifier: {CLASSIFIER_ID} — same A2_SYSTEM prompt + "
          f"shape vocabulary as clinton/gwbush; key env: "
          f"{proxy_lane.resolve_key_env()}")
    print(f"  sidecar: {path} (source artifact is never mutated)")
    print(f"  est cost: ~${estimate_cost(pending):.2f} "
          f"(claude-haiku fallback rates, ~4 chars/token)")

    if not args.go:
        print("\n($0 plan only — add --go to classify)")
        return
    if not pending:
        doc = run_backfill(args.speech, art, path, lambda s: [])  # lint + write
        print(f"nothing to classify; sidecar up to date: {path}")
        print(f"  shape tally: {shape_tally(doc)}")
        return
    if not proxy_lane.key_present():
        sys.exit(proxy_lane.BLOCKED_MSG)

    classify_fn, cost = build_live_classify_fn()
    doc = run_backfill(args.speech, art, path, classify_fn, limit=args.limit)
    print(f"\nsidecar written: {path}")
    print(f"  shape tally: {shape_tally(doc)}")
    print(f"  warnings: {len(doc['warnings'])}")
    print(f"  cost note (best-effort, manifest-reported): "
          f"${cost['usd']:.4f} over {cost['calls']} sentence(s) — the proxy "
          f"DB is the authoritative ledger")
    print(f"\nnext: PYTHONPATH=.:src .venv/bin/python scripts/phase3_rebuild.py "
          f"--speech {args.speech} --shapes-sidecar {path}")


if __name__ == "__main__":
    main()
