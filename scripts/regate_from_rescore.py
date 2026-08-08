#!/usr/bin/env python3
"""Re-run the DETERMINISTIC quality gate over the B1a re-scored evidence — $0.

This script makes NO model or API calls. It is pure arithmetic over data that
is already on disk: the five rebuilt run artifacts (metrics/pca_runs) and the
B1a re-score SIDECARS (metrics/remediation_v2/rescored_<speech>.json). Nothing
here needs a key, and nothing here should ever be given one.

WHY IT EXISTS
-------------
The v2 pack path never scored relevance or stance, so ``consolidator._bearing``
— which needs ``supports_claim in (True, False)`` — saw roughly a quarter of
every pack as unscored. Those items could not credit ``MIN_BEARING_T13=2``, so
the T2.4 gate forced Unverifiable for lack of SIGNAL rather than lack of
EVIDENCE (trump_2026:0469 is the worked example: NPR was the only credit while
AP, NBC and two govinfo records sat stanceless). B1a bought the missing stance
for the 4,344 stored items without re-retrieving anything. This script answers
the only question that matters next: with real scores, which claims does the
repair RELEASE, and which does it newly WITHHOLD?

The released set is what has to be re-adjudicated (B1b spend). A newly-withheld
claim costs nothing — withholding needs no panel call. So this file also sizes
the B1b bill.

HOW THE GATE IS RE-RUN (fidelity)
---------------------------------
The gate is not re-derived here. ``consolidator.consolidate()`` IS the gate, so
each sid's stored pack is fed back through it as a single shortlist. The stored
packs are already filtered, quota-trimmed and capped, so re-running the same
predicates over them is idempotent — and that is checked rather than assumed:
every speech's BEFORE recomputation is compared against the gate code the
artifact actually recorded, and the per-speech match count is reported as
``gate_reproduction``. On the five rebuilt runs this reproduces 529/529 rows
exactly, which is what makes the AFTER delta attributable to the re-score
alone rather than to drift in the surrounding code.

Two subtleties the reproduction depends on, both learned from the artifacts:

  * ROLE AWARENESS. ``consolidate`` takes the role-aware D11.2 quota branch
    only when ``claim_shape`` AND ``relation_of`` are both supplied. The
    rebuilt artifacts carry their claims VERBATIM from the ORIGINAL published
    run, so obama/biden/trump claims show no ``layer_a.claim_shape`` — but the
    rebuild merged ``shapes_backfill_<speech>.json`` in memory before running,
    so those legs WERE role-aware. Mirroring the original run therefore means
    merging the same shape sidecar (via phase3_rebuild's own loader/merger).
    Without it, 7 of 529 rows mis-reproduce. A speech with no shape sidecar and
    no artifact shapes falls back to the legacy branch, exactly as its run did.
  * SPEECH DATE. ``window_for`` and the era mode both read the registered
    utterance date, so each speech is registered from its artifact meta before
    any pack is gated.

Usage (repo root, always $0):
  PYTHONPATH=.:src .venv/bin/python scripts/regate_from_rescore.py
  PYTHONPATH=.:src .venv/bin/python scripts/regate_from_rescore.py --speech trump_2026

Speeches whose sidecar has not been written yet are SKIPPED and named in the
output — a partial B1a is a normal state to report from, never a silent gap.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# scripts/ is not a package, so these are path-based imports — the same way the
# phase-3 and B1a tests load their subjects. Everything reused here is $0.
from backfill_claim_shapes import sidecar_path as shapes_sidecar_path  # noqa: E402
from phase3_rebuild import (SPEECHES, load_sidecar_shapes,  # noqa: E402
                            merge_sidecar_shapes)
from rescore_stored_packs import (REBUILT_RUNS, SIDECAR_SCHEMA,  # noqa: E402
                                  artifact_path, load_artifact, sidecar_path)

OUT_DIR = REPO / "metrics" / "remediation_v2"
OUT_STEM = "regate_flipset"

#: The six claims jackie named for re-adjudication regardless of what the gate
#: says — they ride along with the released set in the B1b costing.
NAMED_EXTRAS: tuple[str, ...] = (
    "trump_2026:0030", "trump_2026:0031", "trump_2026:0023",
    "trump_2026:0024", "trump_2026:0343", "clinton_1998:0313",
)

#: Per-claim PCA re-adjudication cost band, USD (the DC-B1 packet's figure).
PER_CLAIM_USD = (0.0642, 0.0748)
#: Authorized ceiling for the whole remediation-v2 B leg, and the slice of it
#: PLANNED for B1a's re-scoring. The planned figure is only a fallback: when the
#: sidecars are on hand their own ``spend_usd`` is ledger truth and headroom is
#: computed against whichever is LARGER, so an overrun can never be reported as
#: though the money were still available.
BUDGET_CEILING_USD = 10.00
B1A_PLANNED_USD = 0.44

CLAIM_TEXT_TRUNC = 120

CLASSES = ("released", "still_gated", "newly_gated", "unchanged_decided")
#: A sid the sidecar has no entry for yet (B1a mid-flight). Its AFTER state is
#: UNKNOWN, so it is counted here and deliberately kept OUT of the four flip
#: classes — overlaying nothing would silently report it as "unchanged", which
#: is the difference between "the repair did not move this claim" and "we have
#: not asked yet".
NOT_RESCORED = "not_rescored"


# ── join ─────────────────────────────────────────────────────────────────────

def join_key(url: str) -> str:
    """Normalized source_url join key.

    Deliberately the SAME normalization ``consolidate`` dedups on
    (``url.rstrip("/").lower()``), so a pack can never hold two items that
    collide here — the join is one-to-one by construction."""
    return (url or "").strip().rstrip("/").lower()


def overlay_rescores(evidence: list, rescored: list[dict]) -> dict:
    """Overlay a sidecar sid's ``relevance_score`` / ``supports_claim`` onto the
    reconstructed ``Evidence`` objects, matching on ``source_url``.

    Mutates ``evidence`` in place (the caller owns freshly-built objects; the
    stored artifact is never touched) and returns join telemetry. NOTHING is
    dropped silently: an entry the artifact has no home for, and an artifact
    item the sidecar never scored, are both reported by URL."""
    by_key: dict[str, object] = {}
    for ev in evidence:
        by_key.setdefault(join_key(ev.source_url), ev)

    matched_keys: set[str] = set()
    unmatched_sidecar: list[str] = []
    for row in rescored or []:
        url = row.get("source_url") or ""
        ev = by_key.get(join_key(url))
        if ev is None:
            unmatched_sidecar.append(url)
            continue
        matched_keys.add(join_key(url))
        ev.relevance_score = row.get("relevance_score")
        ev.supports_claim = row.get("supports_claim")

    unmatched_artifact = [ev.source_url for ev in evidence
                          if join_key(ev.source_url) not in matched_keys]
    return {"items": len(evidence), "matched": len(matched_keys),
            "sidecar_unmatched": unmatched_sidecar,
            "artifact_unscored": unmatched_artifact}


# ── gate ─────────────────────────────────────────────────────────────────────

def claim_shape_map(artifact: dict, speech: str,
                    shapes_path: Optional[Path] = None) -> tuple[dict, int]:
    """``sid -> claim_shape`` exactly as the rebuild's shape registry saw it.

    Artifact shapes win; the ``shapes_backfill_<speech>.json`` sidecar fills the
    rest, through phase3_rebuild's OWN loader and merger (so the schema /
    speech / source-run guards that protect the rebuild protect this too). The
    shape sidecar is keyed to the ORIGINAL published run, not the rebuild, so
    it is validated against ``SPEECHES[speech]["run_id"]``. Returns
    (shapes, n_filled_from_sidecar)."""
    claims = [{"sid": c.get("sid"), "layer_a": dict(c.get("layer_a") or {})}
              for c in (artifact.get("claims") or [])]
    filled = 0
    path = shapes_path if shapes_path is not None else shapes_sidecar_path(speech)
    if Path(path).exists():
        source_run = (SPEECHES.get(speech) or {}).get("run_id", "")
        filled = merge_sidecar_shapes(
            claims, load_sidecar_shapes(Path(path), speech, source_run))
    return ({c["sid"]: (c.get("layer_a") or {}).get("claim_shape") or ""
             for c in claims}, filled)


def _contemporaneous(d: Optional[date], utterance: Optional[date],
                     window) -> Optional[bool]:
    """True/False for dated items, None for undated.

    NOTE: this mirrors the identically-named closure inside
    ``consolidator.consolidate`` — it is nested there, so it cannot be
    imported. It exists here only to reconstruct the era class the LENIENT
    ``_quota_credit`` branch needs for the printed arithmetic; the AUTHORITATIVE
    gate answer always comes from ``consolidate`` itself, and ``breakdown_for``
    cross-checks the two and reports any divergence rather than trusting this
    copy."""
    from truthbot.verdict import era_lint

    if d is None:
        return None
    if window is not None and not (window[0] <= d <= window[1]):
        return False
    if utterance is not None and d > era_lint.fair_game_end(utterance):
        return False
    return True


def _quota_credit(item, *, era_mode: str, utterance: Optional[date],
                  window) -> bool:
    """Mirror of ``consolidate``'s nested ``_quota_credit`` closure (private and
    un-importable for the same reason as ``_contemporaneous``). The predicates
    it is BUILT from — ``_bearing``, ``_T13``, ``SourceTier`` — are imported
    from the consolidator rather than restated."""
    from truthbot.models import SourceTier
    from truthbot.verdict import era_lint
    from truthbot.verdict.consolidator import _bearing, _T13

    if item.post_speech:
        return False
    if item.evidence.source_tier in _T13 and _bearing(item.evidence):
        return True
    if era_mode != "lenient" or item.evidence.source_tier != SourceTier.GOVERNMENT:
        return False
    d = era_lint.item_date(item.evidence.published_at, item.evidence.snippet or "")
    return _contemporaneous(d, utterance, window) is True


def breakdown_for(result, *, role_aware: bool, era_mode: str,
                  utterance: Optional[date], window) -> dict:
    """The credit arithmetic behind ``result.quota_met``, so a reviewer can see
    WHY a claim sits where it does: independent / corroborant / primary counts
    and the credit total, against ``MIN_BEARING_T13``.

    ``agrees`` records whether this reconstruction reaches the same verdict as
    ``consolidate`` did. It must always be True; it is surfaced rather than
    asserted so a divergence shows up as telemetry instead of a traceback."""
    from truthbot.verdict.consolidator import MIN_BEARING_T13, _bearing

    def credit(it):
        return _quota_credit(it, era_mode=era_mode, utterance=utterance,
                             window=window)

    items = result.items
    if not role_aware:
        independent = sum(1 for it in items if credit(it))
        corroborant = primary = 0
        credits = independent
        quota = credits >= MIN_BEARING_T13
    else:
        independent = sum(1 for it in items if it.role == "normal" and credit(it))
        corroborant = sum(1 for it in items if it.role == "corroborant"
                          and _bearing(it.evidence) and not it.post_speech)
        primary = sum(1 for it in items if it.role == "primary-record"
                      and _bearing(it.evidence) and not it.post_speech)
        credits = independent + corroborant + min(1, primary)
        quota = credits >= MIN_BEARING_T13 and (independent >= 1 or corroborant >= 1)
    return {"pack_items": len(items), "independent": independent,
            "corroborant": corroborant, "primary": primary, "credits": credits,
            "min_required": MIN_BEARING_T13, "role_aware": role_aware,
            "era_mode": era_mode, "quota_met": bool(result.quota_met),
            "agrees": bool(quota) == bool(result.quota_met)}


def gate_once(sid: str, evidence: list, *, utterance: Optional[date],
              claim_shape: str, relation_of, claim_text: str,
              utterance_record: Optional[bool] = None) -> tuple:
    """Run the real gate over one stored pack. Returns (result, breakdown).

    ``utterance_record`` is the D15 switch, handed straight to ``consolidate``.
    ``None`` — the default, and what THIS script always passes — means "obey the
    environment flag", which in production is OFF. The $0 D15 blast-radius
    measurement (``scripts/measure_d15.py``) drives this same entry point with
    an explicit True/False, so it can price the proposal without anything
    depending on ambient environment."""
    from truthbot.verdict import consolidator, era_lint
    from truthbot.verdict.consolidator import consolidate
    from truthbot.verdict.evidence_pack import window_for

    window = window_for(sid)
    era_mode = era_lint.era_mode_for(utterance, claim_text)
    role_aware = bool(claim_shape) and relation_of is not None
    result = consolidate(sid, [("stored", evidence)], utterance=utterance,
                         window=window, max_items=consolidator.PACK_CAP_V2,
                         era_mode=era_mode, claim_shape=claim_shape,
                         relation_of=relation_of,
                         utterance_record=utterance_record)
    return result, breakdown_for(result, role_aware=role_aware,
                                 era_mode=era_mode, utterance=utterance,
                                 window=window)


def classify(was_gated: bool, now_gated: bool) -> str:
    """The four-way flip classification."""
    if was_gated:
        return "still_gated" if now_gated else "released"
    return "newly_gated" if now_gated else "unchanged_decided"


def row_gate_code(row: dict) -> str:
    """The T2.4 gate marker on a stored verdict row. ``provenance_code`` is what
    the five rebuilt artifacts carry; ``evidence_gate`` is the newer field name
    the bridge also writes, accepted here for forward compatibility."""
    return str(row.get("evidence_gate") or row.get("provenance_code") or "")


# ── per-speech re-gate ───────────────────────────────────────────────────────

def load_rescore_sidecar(path: Path, speech: str, source_run: str) -> dict:
    """Load a B1a sidecar, refusing anything that was scored against a
    different speech or a different artifact revision."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SIDECAR_SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r} != "
                         f"{SIDECAR_SCHEMA!r}")
    if doc.get("speech_id") != speech:
        raise ValueError(f"{path}: speech_id {doc.get('speech_id')!r} != {speech!r}")
    if source_run and doc.get("source_run") != source_run:
        raise ValueError(f"{path}: source_run {doc.get('source_run')!r} != "
                         f"{source_run!r} — these scores belong to a different "
                         "artifact revision; joining them would attach stance "
                         "to the wrong evidence.")
    return doc


def regate_speech(speech: str, artifact: dict, sidecar: dict) -> dict:
    """Re-gate one speech. Pure: no I/O, no spend, no mutation of ``artifact``."""
    from truthbot.verdict import speech_context
    from truthbot.verdict.consolidator import GATE_INSUFFICIENT
    from truthbot.verdict.publish_pipeline import evidence_from_artifact_dict
    from truthbot.verify.principals import principal_relation

    meta = artifact.get("meta") or {}
    speaker = meta.get("speaker") or ""
    utterance = date.fromisoformat(meta["date"]) if meta.get("date") else None
    if utterance is not None:
        speech_context.register_speech_date(speech, utterance)

    relation_of = None
    if speaker and utterance is not None:
        def relation_of(ev):                      # noqa: F811 — mirrors pipeline
            return principal_relation(ev.source_url, speaker, utterance)

    claims = {c.get("sid"): c for c in (artifact.get("claims") or [])}
    shapes, n_shapes_filled = claim_shape_map(artifact, speech)
    rows = {r.get("sid"): r for r in (artifact.get("rows") or [])}
    scored = sidecar.get("sids") or {}

    counts = {k: 0 for k in CLASSES}
    counts[NOT_RESCORED] = 0
    flips: list[dict] = []
    join = {"items": 0, "matched": 0, "sidecar_unmatched": [],
            "artifact_unscored": [], "sids_without_scores": [],
            "sidecar_sids_not_in_artifact": sorted(
                set(scored) - set(artifact.get("evidence") or {}))}
    reproduced = mis = 0
    reproduction_mismatch: list[dict] = []
    breakdown_divergence: list[str] = []

    for sid, dumps in (artifact.get("evidence") or {}).items():
        claim = claims.get(sid) or {}
        text = (claim.get("text") or "").strip()
        shape = shapes.get(sid, "")
        was_gated = row_gate_code(rows.get(sid, {})) == GATE_INSUFFICIENT

        # BEFORE: the stored stances, untouched.
        before_ev = evidence_from_artifact_dict({sid: dumps})[sid]
        before, before_bd = gate_once(sid, before_ev, utterance=utterance,
                                      claim_shape=shape, relation_of=relation_of,
                                      claim_text=text)
        if (not before.quota_met) == was_gated:
            reproduced += 1
        else:
            mis += 1
            reproduction_mismatch.append(
                {"sid": sid, "artifact_gated": was_gated,
                 "recomputed_gated": not before.quota_met})

        # AFTER: a SECOND reconstruction, so the overlay can never leak back
        # into the BEFORE arithmetic.
        after_ev = evidence_from_artifact_dict({sid: dumps})[sid]
        rescored = scored.get(sid)
        if rescored is None:
            # B1a has not scored this sid yet. Nothing to overlay, so nothing
            # can be said about its AFTER state — do not manufacture one.
            join["sids_without_scores"].append(sid)
            counts[NOT_RESCORED] += 1
            continue
        j = overlay_rescores(after_ev, rescored)
        join["items"] += j["items"]
        join["matched"] += j["matched"]
        join["sidecar_unmatched"].extend(
            {"sid": sid, "source_url": u} for u in j["sidecar_unmatched"])
        join["artifact_unscored"].extend(
            {"sid": sid, "source_url": u} for u in j["artifact_unscored"])
        after, after_bd = gate_once(sid, after_ev, utterance=utterance,
                                    claim_shape=shape, relation_of=relation_of,
                                    claim_text=text)
        if not (before_bd["agrees"] and after_bd["agrees"]):
            breakdown_divergence.append(sid)

        cls = classify(was_gated, not after.quota_met)
        counts[cls] += 1
        if cls in ("released", "newly_gated"):
            flips.append({
                "sid": sid, "speech": speech, "class": cls,
                "claim": text[:CLAIM_TEXT_TRUNC],
                "claim_truncated": len(text) > CLAIM_TEXT_TRUNC,
                "old_verdict": (rows.get(sid) or {}).get("verdict"),
                "old_gate_code": row_gate_code(rows.get(sid, {})),
                "claim_shape": shape,
                "before": before_bd, "after": after_bd,
                # A flip is only attributable to the re-score when the BEFORE
                # recomputation matched what the artifact recorded.
                "baseline_reproduced": (not before.quota_met) == was_gated,
            })

    return {
        "speech": speech,
        "source_run": artifact.get("run_id"),
        "claims": len(artifact.get("evidence") or {}),
        # False = B1a is still writing this sidecar; the four flip classes cover
        # only the sids it has already scored.
        "sidecar_complete": counts[NOT_RESCORED] == 0,
        "counts": counts,
        "flips": flips,
        "join": join,
        "shapes_filled_from_sidecar": n_shapes_filled,
        "gate_reproduction": {"matched": reproduced, "mismatched": mis,
                              "mismatches": reproduction_mismatch},
        "breakdown_divergence": breakdown_divergence,
        "rescore_spend_usd": float(sidecar.get("spend_usd") or 0.0),
        "rescore_soft_failures": list(sidecar.get("soft_failures") or []),
    }


# ── costing ──────────────────────────────────────────────────────────────────

def costed_summary(released: list[str],
                   extras: tuple[str, ...] = NAMED_EXTRAS,
                   *, b1a_observed_usd: Optional[float] = None) -> dict:
    """Size the B1b re-adjudication bill.

    Only RELEASED claims (plus the named extras) need a panel call; a
    ``newly_gated`` claim costs $0, because withholding a verdict needs no
    adjudication. Extras already in the released set are counted ONCE.

    ``b1a_observed_usd`` is what the sidecars say B1a actually cost. Headroom is
    charged against ``max(planned, observed)`` so an overrun shows up as less
    money available, never as unspent budget."""
    rel = sorted(set(released))
    extra_new = sorted(set(extras) - set(rel))
    total = len(rel) + len(extra_new)
    lo, hi = PER_CLAIM_USD
    committed = B1A_PLANNED_USD
    if b1a_observed_usd is not None:
        committed = max(committed, float(b1a_observed_usd))
    remaining = BUDGET_CEILING_USD - committed
    return {
        "released": len(rel),
        "extras_named": len(extras),
        "extras_not_already_released": len(extra_new),
        "extras_overlapping_released": sorted(set(extras) & set(rel)),
        "claims_to_adjudicate": total,
        "per_claim_usd": [lo, hi],
        "cost_low_usd": round(total * lo, 2),
        "cost_high_usd": round(total * hi, 2),
        "ceiling_usd": BUDGET_CEILING_USD,
        "b1a_planned_usd": B1A_PLANNED_USD,
        "b1a_observed_usd": (None if b1a_observed_usd is None
                             else round(float(b1a_observed_usd), 4)),
        "b1a_overran_plan": (b1a_observed_usd is not None
                             and float(b1a_observed_usd) > B1A_PLANNED_USD),
        "committed_b1a_usd": round(committed, 2),
        "remaining_usd": round(remaining, 2),
        "fits_ceiling": round(total * hi, 2) <= round(remaining, 2),
    }


# ── report ───────────────────────────────────────────────────────────────────

def build_report(per_speech: list[dict], missing: list[str]) -> dict:
    corpus = {k: 0 for k in (*CLASSES, NOT_RESCORED)}
    for s in per_speech:
        for k in corpus:
            corpus[k] += s["counts"].get(k, 0)
    partial = [s["speech"] for s in per_speech if not s["sidecar_complete"]]
    released = [f["sid"] for s in per_speech for f in s["flips"]
                if f["class"] == "released"]
    newly = [f for s in per_speech for f in s["flips"]
             if f["class"] == "newly_gated"]
    return {
        "schema": "truthbot-regate-flipset v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "speeches": [s["speech"] for s in per_speech],
        "speeches_missing_sidecar": missing,
        "speeches_partial_sidecar": partial,
        "complete": not missing and not partial,
        "corpus_counts": corpus,
        "corpus_claims": sum(s["claims"] for s in per_speech),
        "per_speech": per_speech,
        "released_sids": sorted(released),
        "newly_gated_sids": sorted(f["sid"] for f in newly),
        "costed_b1b": costed_summary(
            released,
            b1a_observed_usd=sum(s["rescore_spend_usd"] for s in per_speech)),
    }


def _bd(b: dict) -> str:
    return (f"ind {b['independent']} / corr {b['corroborant']} / "
            f"prim {b['primary']} → {b['credits']} of {b['min_required']}")


def render_markdown(report: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# Re-gate flip set — remediation v2, B1a → B1b")
    A("")
    A(f"Generated {report['generated']} · $0 (no model calls; deterministic "
      "arithmetic over stored artifacts + B1a sidecars).")
    A("")
    if report["speeches_missing_sidecar"]:
        A("> **Partial run.** No B1a sidecar yet for: "
          + ", ".join(f"`{s}`" for s in report["speeches_missing_sidecar"])
          + ". Those speeches are excluded from every count below.")
        A("")
    if report["speeches_partial_sidecar"]:
        A("> **Partial sidecar.** B1a is still writing: "
          + ", ".join(f"`{s}`" for s in report["speeches_partial_sidecar"])
          + ". Their unscored claims are counted as `not re-scored`, NOT as "
            "`unchanged` — their AFTER state is genuinely unknown, and the "
            "released count for those speeches is a floor, not a total.")
        A("")

    A("## Flip set by speech")
    A("")
    A("| speech | claims | released | still gated | newly gated | "
      "unchanged decided | not re-scored |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for s in report["per_speech"]:
        c = s["counts"]
        A(f"| {s['speech']} | {s['claims']} | {c['released']} | "
          f"{c['still_gated']} | {c['newly_gated']} | {c['unchanged_decided']} | "
          f"{c[NOT_RESCORED]} |")
    c = report["corpus_counts"]
    A(f"| **corpus** | **{report['corpus_claims']}** | **{c['released']}** | "
      f"**{c['still_gated']}** | **{c['newly_gated']}** | "
      f"**{c['unchanged_decided']}** | **{c[NOT_RESCORED]}** |")
    A("")

    for label, cls in (("Released", "released"), ("Newly gated", "newly_gated")):
        rows = [f for s in report["per_speech"] for f in s["flips"]
                if f["class"] == cls]
        A(f"## {label} — {len(rows)} claim(s)")
        A("")
        if not rows:
            A("_none_")
            A("")
            continue
        if cls == "released":
            A("These were gate-forced Unverifiable for lack of stance signal. "
              "With B1a's scores their packs meet quota, so they need PCA "
              "re-adjudication (this is the B1b bill).")
        else:
            A("The repair WITHHOLDS these: they were decided, and with real "
              "stance their packs no longer meet quota. Re-adjudication cost "
              "$0 — a withheld claim needs no panel call.")
        A("")
        A("| sid | old verdict | shape | credit BEFORE | credit AFTER | claim |")
        A("|---|---|---|---|---|---|")
        for f in sorted(rows, key=lambda r: r["sid"]):
            txt = f["claim"].replace("|", "\\|")
            if f["claim_truncated"]:
                txt += "…"
            flag = "" if f["baseline_reproduced"] else " ⚠︎baseline-diverged"
            A(f"| `{f['sid']}`{flag} | {f['old_verdict']} | "
              f"{f['claim_shape'] or 'legacy'} | {_bd(f['before'])} | "
              f"{_bd(f['after'])} | {txt} |")
        A("")

    A("## Sidecar-join telemetry")
    A("")
    A("| speech | pack items | scores joined | sidecar rows unmatched | "
      "pack items unscored | sids w/o sidecar entry | gate reproduction |")
    A("|---|---:|---:|---:|---:|---:|---|")
    for s in report["per_speech"]:
        j, g = s["join"], s["gate_reproduction"]
        A(f"| {s['speech']} | {j['items']} | {j['matched']} | "
          f"{len(j['sidecar_unmatched'])} | {len(j['artifact_unscored'])} | "
          f"{len(j['sids_without_scores'])} | {g['matched']}/"
          f"{g['matched'] + g['mismatched']} rows |")
    A("")
    A("`gate reproduction` re-runs the gate over the STORED stances and compares "
      "it with the gate code the artifact recorded. A perfect score means the "
      "AFTER delta is attributable to the re-score alone.")
    A("")

    k = report["costed_b1b"]
    A("## Costed B1b summary")
    A("")
    if not report["complete"]:
        A("**Provisional** — B1a has not finished; these are lower bounds.")
        A("")
    A(f"- released (need re-adjudication): **{k['released']}**")
    A(f"- named extras: **{k['extras_named']}** "
      f"({k['extras_not_already_released']} not already released)")
    A(f"- **total claims to adjudicate: {k['claims_to_adjudicate']}**")
    A(f"- newly gated: {report['corpus_counts']['newly_gated']} — **$0** "
      "(withholding needs no panel call)")
    A(f"- implied cost at ${k['per_claim_usd'][0]:.4f}–"
      f"${k['per_claim_usd'][1]:.4f}/claim: "
      f"**${k['cost_low_usd']:.2f}–${k['cost_high_usd']:.2f}**")
    A(f"- ceiling ${k['ceiling_usd']:.2f}, of which ${k['committed_b1a_usd']:.2f} "
      f"is committed to B1a → ${k['remaining_usd']:.2f} remaining; "
      f"{'FITS' if k['fits_ceiling'] else 'DOES NOT FIT'}")
    if k["b1a_overran_plan"]:
        A(f"- ⚠︎ B1a was planned at ${k['b1a_planned_usd']:.2f} but the sidecars "
          f"record **${k['b1a_observed_usd']:.4f}** actually spent; headroom "
          "above is charged against the observed figure, not the plan.")
    A("")
    return "\n".join(L) + "\n"


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--speech", action="append", choices=sorted(REBUILT_RUNS),
                    help="limit to this speech (repeatable); default all five")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    speeches = args.speech or list(REBUILT_RUNS)
    per_speech, missing = [], []
    for speech in speeches:
        side = sidecar_path(speech)
        if not side.exists():
            missing.append(speech)
            print(f"SKIP {speech}: no B1a sidecar at {side} (still being written?)")
            continue
        artifact = load_artifact(artifact_path(speech))
        sidecar = load_rescore_sidecar(side, speech, REBUILT_RUNS[speech])
        res = regate_speech(speech, artifact, sidecar)
        per_speech.append(res)
        c = res["counts"]
        note = ("" if res["sidecar_complete"]
                else f"  [PARTIAL: {c[NOT_RESCORED]} sids not re-scored yet]")
        print(f"{speech}: released={c['released']} still_gated={c['still_gated']} "
              f"newly_gated={c['newly_gated']} unchanged={c['unchanged_decided']} "
              f"(gate reproduction {res['gate_reproduction']['matched']}/"
              f"{res['claims']}){note}")

    report = build_report(per_speech, missing)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{OUT_STEM}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / f"{OUT_STEM}.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"\nwrote {out / (OUT_STEM + '.json')}")
    print(f"wrote {out / (OUT_STEM + '.md')}")

    k = report["costed_b1b"]
    cc = report["corpus_counts"]
    print("\n── B1b costed summary ──────────────────────────────────────────")
    if missing:
        print(f"PARTIAL — no sidecar yet for: {', '.join(missing)}")
    if report["speeches_partial_sidecar"]:
        print("PARTIAL — B1a still writing: "
              f"{', '.join(report['speeches_partial_sidecar'])} "
              f"({cc[NOT_RESCORED]} sids not re-scored; counts are lower bounds)")
    print(f"released .......................... {k['released']}")
    print(f"named extras ...................... {k['extras_named']} "
          f"({k['extras_not_already_released']} not already released)")
    print(f"TOTAL claims to adjudicate ........ {k['claims_to_adjudicate']}")
    print(f"newly gated (cost $0) ............. {cc['newly_gated']}")
    print(f"implied cost @ ${k['per_claim_usd'][0]:.4f}-"
          f"${k['per_claim_usd'][1]:.4f}/claim  ${k['cost_low_usd']:.2f}-"
          f"${k['cost_high_usd']:.2f}")
    print(f"ceiling ${k['ceiling_usd']:.2f} - ${k['committed_b1a_usd']:.2f} "
          f"committed (B1a) = ${k['remaining_usd']:.2f} remaining  → "
          f"{'FITS' if k['fits_ceiling'] else 'DOES NOT FIT'}")
    if k["b1a_overran_plan"]:
        print(f"NOTE: B1a planned ${k['b1a_planned_usd']:.2f}, sidecars record "
              f"${k['b1a_observed_usd']:.4f} spent — headroom is charged "
              "against the observed figure.")
    print("no model calls were made; $0 spent.")
    return 0


if __name__ == "__main__":            # pragma: no cover
    raise SystemExit(main())
