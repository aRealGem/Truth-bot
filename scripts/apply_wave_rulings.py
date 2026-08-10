#!/usr/bin/env python3
"""Apply the 2026-08-10 wave rulings that cost nothing — $0, no model calls.

The adjudication wave (2026-08-09) left three things RECORDED but not APPLIED,
each for a stated reason. All three were then ruled on, and all three are
deterministic: they are arithmetic and text-copying over data already on disk.
This script does them in one pass so each speech gains ONE new artifact
generation rather than three.

WHAT IT APPLIES
---------------
1. **The deferred newly-gated set (R-2b).** The ratified D15/D16(alpha) rules,
   run over the B1a+B2 re-scored stance, GATE 65 claims that the wave did not
   touch. Withholding needs no panel call, so the wave recorded the sids in
   ``meta.wave.deferred_newly_gated`` and left them alone rather than applying
   them quietly. Applying them replaces each row with the adjudicator's own
   gate-forced Unverifiable row — the same row the gate would have produced had
   the rules been on when the pack was built.

   Each sid also gets a MECHANISM: re-score / D15 / D16(alpha) / both. That is
   not asserted, it is measured, by re-running the gate under four rule
   configurations (see :func:`mechanism_attribution`) and asking which
   configuration is the first to gate the claim. The result is written to
   ``deferred_gated_mechanism.json`` and feeds the corrections ledger.

2. **Blank rationales (R-3).** A row the stage-2 discriminator resolved out of
   a tie ships with a verdict and no reason. The ruled fix — the discriminator
   adopts the chosen seat's stored rationale verbatim — is in the pipeline as
   of ``discriminator.adopt_seat_rationale``, but it cannot repair rows that
   are ALREADY on disk: the wave discarded its seats' text at reduce time, so
   the wave artifact records the seats' LABELS (``by_role``) and none of their
   words.

   So the adoption walks the LINEAGE instead. For a blank row it looks back
   through ``rebuild_of`` for a stored panel output on the same sid that
   reached the same verdict, and adopts that run's rationale VERBATIM, with a
   provenance record naming the run and seat it came from. Nothing is
   synthesized, and an adoption that cannot be sourced does not happen.

3. **The coherence annotation (D14 disposition: ANNOTATE).** With rationales
   restored, ``verdict_audit.adjacent_coherence_conflicts`` can see the
   trump_2026:0023/:0024 contradiction again. The ruling for this publish is to
   ANNOTATE it, never to force the labels to agree: both rows get a
   ``coherence_note`` naming the other claim and the statistic they disagree
   about. The note is assembled from the checker's own output — sids, verdicts
   and shared tokens — so it states the conflict without characterising it.

WHAT IT DOES NOT DO
-------------------
No re-adjudication, no retrieval, no rendering. It never deletes an artifact:
every speech gets a NEW run id whose ``rebuild_of`` points at the wave artifact,
which stays on disk exactly as the wave wrote it.

Usage (repo root, always $0)::

    PYTHONPATH=.:src .venv/bin/python scripts/apply_wave_rulings.py --dry-run
    PYTHONPATH=.:src .venv/bin/python scripts/apply_wave_rulings.py --apply
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# scripts/ is not a package; path-based imports, as everywhere else in this
# directory. Everything imported here is $0.
from phase3_rebuild import build_verdict_diff, update_manifest  # noqa: E402
from regate_from_rescore import (PRE_RATIFICATION_RULES,  # noqa: E402
                                 merge_sidecars, regate_speech)
from rescore_stored_packs import (artifact_path, b2_sidecar_path,  # noqa: E402
                                  load_artifact, sidecar_path)

OUT_DIR = REPO / "metrics" / "remediation_v2"
RUNS_DIR = REPO / "metrics" / "pca_runs"

RULING_DATE = "2026-08-10"
RULING_TAG = "wave rulings R-2b/R-3/D14"
PIPELINE_GENERATION = "v2.3-role-axis-s5cap"

#: The four gate configurations the mechanism attribution runs, in the order a
#: cause is assigned. The FIRST configuration to gate a claim owns it: a claim
#: the re-score alone withholds is a re-score claim even if D15 would also have
#: caught it, because the re-score is what actually moved first.
LEGS: tuple[tuple[str, dict], ...] = (
    ("re-score", {"utterance_record": False, "statistical_release": False}),
    ("D15", {"utterance_record": True, "statistical_release": False}),
    ("D16alpha", {"utterance_record": False, "statistical_release": True}),
    ("both", {"utterance_record": True, "statistical_release": True}),
)

#: Mechanism label for a claim only the FULL configuration gates — neither rule
#: alone does it, so the two compose.
MECHANISM_INTERACTION = "D15+D16alpha (interaction)"


# ── mechanism attribution ($0) ───────────────────────────────────────────────

def mechanism_attribution(speeches: list[str]) -> dict:
    """sid → which mechanism first withholds it, measured not asserted.

    The reviewer's arithmetic for the deferred set was ``27 + 50 - overlap``,
    where 50 is D15's blast-radius ``gate_changed``. Those two numbers are not
    summable: the blast radius measures D15 against the ARTIFACT's recorded
    gate in isolation, while 27 comes from the composed gate on the re-scored
    stance. This runs all four configurations through the SAME gate the flip
    set uses, so the parts add up to the whole by construction."""
    legs: dict[str, dict[str, set[str]]] = {}
    per_speech: list[dict] = []
    for speech in speeches:
        art = load_artifact(artifact_path(speech))
        sidecar = _merged_sidecar(speech)
        row = {"speech": speech, "source_run": art.get("run_id"), "legs": {}}
        for name, rules in LEGS:
            res = regate_speech(speech, art, sidecar, **rules)
            gated = {f["sid"] for f in res["flips"] if f["class"] == "newly_gated"}
            legs.setdefault(name, {})[speech] = gated
            row["legs"][name] = sorted(gated)
        per_speech.append(row)

    union = {name: set().union(*by_speech.values()) if by_speech else set()
             for name, by_speech in legs.items()}
    shipped = union["both"]
    mechanism: dict[str, str] = {}
    for sid in sorted(shipped):
        for name, _rules in LEGS[:-1]:
            if sid in union[name]:
                mechanism[sid] = name
                break
        else:
            mechanism[sid] = MECHANISM_INTERACTION

    # D16(alpha) RELEASES; it is not a gating mechanism. Say so with the
    # measurement rather than by assertion: the claims the re-score would have
    # gated that the full configuration does not.
    rescued = sorted(union["re-score"] - shipped)
    by_mech: dict[str, int] = {}
    for m in mechanism.values():
        by_mech[m] = by_mech.get(m, 0) + 1
    return {
        "schema": "truthbot-deferred-gated-mechanism v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "ruling_date": RULING_DATE,
        "method": ("four gate configurations over the same consolidator the "
                   "flip set uses; the first configuration to withhold a claim "
                   "owns it. $0 — no model calls, no retrieval."),
        "leg_totals": {name: len(union[name]) for name, _ in LEGS},
        "shipped_total": len(shipped),
        "by_mechanism": by_mech,
        "rescued_by_d16alpha": rescued,
        "arithmetic": (
            f"{len(union['re-score'])} gated by the re-score alone, minus "
            f"{len(rescued)} the D16(alpha) statistical-release rule releases, "
            f"plus {by_mech.get('D15', 0)} D15 adds on top of the re-score, "
            f"= {len(shipped)}"),
        "mechanism": mechanism,
        "per_speech": per_speech,
    }


def _merged_sidecar(speech: str) -> dict:
    """B1a + B2 stance sidecars, merged the way the flip set merges them."""
    b1a = _read_sidecar(sidecar_path(speech), speech)
    b2 = _read_sidecar(b2_sidecar_path(speech), speech)
    return merge_sidecars(b1a, b2)


def _read_sidecar(path: Path, speech: str) -> Optional[dict]:
    from regate_from_rescore import load_rescore_sidecar
    art = load_artifact(artifact_path(speech))
    try:
        return load_rescore_sidecar(Path(path), speech, art.get("run_id") or "")
    except (FileNotFoundError, ValueError):
        return None


# ── 1. apply the deferred newly-gated set ────────────────────────────────────

def gate_row(sid: str, old_row: dict) -> dict:
    """The gate-forced Unverifiable row for a newly-gated claim.

    Built by ``adjudicator._forced_uv_row`` rather than hand-assembled, so an
    applied withholding is byte-identical to one the pipeline produces itself —
    including the gate's own rationale sentence, which is what keeps the
    no-blank-rationale lint satisfied on this path.

    The superseded verdict is kept on the row (``superseded``) because the
    corrections ledger has to state what the claim used to say, and an artifact
    that dropped it would make its own entry unverifiable."""
    from truthbot.verdict.adjudicator import _forced_uv_row

    row = _forced_uv_row(sid)
    row["superseded"] = {
        "verdict": old_row.get("verdict"),
        "reasoning": old_row.get("reasoning") or "",
        "votes": dict(old_row.get("votes") or {}),
        "by_role": dict(old_row.get("by_role") or {}),
        "at": RULING_DATE,
    }
    return row


def apply_deferred_gated(rows: list[dict], sids: set[str]) -> list[dict]:
    """Replace each deferred sid's row with its gate-forced row, in place of
    the original and in the original order. Non-deferred rows are untouched."""
    out = []
    for row in rows:
        sid = row.get("sid")
        out.append(gate_row(sid, row) if sid in sids else dict(row))
    return out


# ── 2. re-emit blank rationales from stored panel output ─────────────────────

def lineage(run: dict, runs_dir: Path = RUNS_DIR) -> list[dict]:
    """The artifact's ancestors, nearest first, following ``rebuild_of``."""
    out: list[dict] = []
    seen: set[str] = set()
    parent_id = str((run.get("meta") or {}).get("rebuild_of") or "")
    while parent_id and parent_id not in seen:
        seen.add(parent_id)
        path = runs_dir / f"{parent_id}.json"
        if not path.exists():
            break
        parent = json.loads(path.read_text(encoding="utf-8"))
        out.append(parent)
        parent_id = str((parent.get("meta") or {}).get("rebuild_of") or "")
    return out


def adopt_from_lineage(row: dict, ancestors: list[dict]) -> Optional[dict]:
    """Give a blank-rationaled row an ancestor run's rationale, VERBATIM.

    Only from a stored row for the SAME sid that reached the SAME verdict — a
    rationale written for a different verdict is not this verdict's reason, and
    adopting it would be fabrication with extra steps. Nearest ancestor wins.

    Returns the provenance record written, or None when no ancestor qualifies
    (in which case the row stays blank and the publish-blocking lint keeps it
    from shipping — which is the correct outcome, not a failure of this pass)."""
    sid = row.get("sid")
    verdict = str(row.get("verdict") or "").strip().upper()
    if not verdict or str(row.get("reasoning") or "").strip():
        return None
    for parent in ancestors:
        prior = next((r for r in (parent.get("rows") or [])
                      if r.get("sid") == sid), None)
        if prior is None:
            continue
        if str(prior.get("verdict") or "").strip().upper() != verdict:
            continue
        text = str(prior.get("reasoning") or "").strip()
        if not text:
            continue
        # Which seat wrote it: the arbiter if it voted the winning label,
        # else the first seat that did. Same precedence pca.reduce uses when
        # it picks the winning call's text.
        by_role = prior.get("by_role") or {}
        seat = next((role for role in ("arbiter", "proposer", "critic")
                     if verdict in [str(v).strip().upper()
                                    for v in (by_role.get(role) or [])]), "")
        prov = {
            "mode": "adopted-verbatim",
            "adopted_from": seat or "panel",
            "adopted_verdict": verdict,
            "adopted_from_run": str(parent.get("run_id") or ""),
            "resolver": "crm114-discriminator",
            "attribution": (
                f"adopted from the {seat or 'panel'} seat of run "
                f"{str(parent.get('run_id') or '')[:8]}"),
            "synthesized": False,
            "note": ("the wave run discarded its seats' rationale text at "
                     "reduce time, so the nearest stored panel output for this "
                     "sid and verdict is the source"),
        }
        row["reasoning"] = text
        row["rationale_provenance"] = prov
        return prov
    return None


def reemit_blank_rationales(rows: list[dict], ancestors: list[dict]) -> list[dict]:
    """Every blank-rationaled published row that an ancestor can source."""
    from truthbot.verdict import verdict_audit as va

    adopted = []
    blanks = {v["sid"] for v in va.blank_rationale_violations(rows)}
    for row in rows:
        if row.get("sid") not in blanks:
            continue
        prov = adopt_from_lineage(row, ancestors)
        if prov:
            adopted.append({"sid": row["sid"], "verdict": row.get("verdict"),
                            **prov})
    return adopted


# ── 3. annotate adjacent coherence conflicts (D14: ANNOTATE) ─────────────────

def coherence_note(conflict: dict, sid: str) -> str:
    """The annotation one row carries about its neighbour.

    Assembled from the checker's own record — the two sids, their verdicts and
    the tokens they share. It states that the pair rates the same statistic and
    that the published verdicts differ. It does not say which is right: the
    D14 disposition for this publish is ANNOTATE, never force the labels to
    agree, so a note that adjudicated the pair would be exceeding the ruling."""
    a, b = conflict["sids"]
    va_, vb = conflict["verdicts"]
    other, other_verdict = (b, vb) if sid == a else (a, va_)
    this_verdict = va_ if sid == a else vb
    shared = ", ".join(conflict.get("shared_tokens") or [])
    return (f"Adjacent-claim coherence: this claim ({this_verdict}) and "
            f"{other} ({other_verdict}) rate the same statistic"
            + (f" (shared terms: {shared})" if shared else "")
            + " and carry different verdicts. Both are published as adjudicated; "
              "the disagreement is disclosed here rather than resolved by "
              "forcing the labels to agree.")


def annotate_coherence(claims: list[dict], rows: list[dict]) -> list[dict]:
    """Stamp ``coherence_note`` on both sides of every unannotated conflict."""
    from truthbot.verdict import verdict_audit as va

    by_sid = {r.get("sid"): r for r in rows}
    conflicts = va.adjacent_coherence_conflicts(claims, rows)
    for conflict in conflicts:
        for sid in conflict["sids"]:
            row = by_sid.get(sid)
            if row is not None and not str(row.get("coherence_note") or "").strip():
                row["coherence_note"] = coherence_note(conflict, sid)
    return conflicts


# ── artifact writing ─────────────────────────────────────────────────────────

def write_artifact(source: dict, rows: list[dict], *, speech_id: str,
                   applied: dict, run_id: Optional[str] = None,
                   out_dir: Path = RUNS_DIR) -> tuple[Path, dict]:
    """A new artifact carrying the applied rows. The source is never touched."""
    run_id = run_id or str(uuid.uuid4())
    old_meta = source.get("meta") or {}
    meta = dict(old_meta)
    meta.update({
        "speech_id": speech_id,
        "cost_usd": 0.0,
        "rebuild_of": source.get("run_id", ""),
        "pipeline_generation": PIPELINE_GENERATION,
        "remediation": RULING_TAG,
        "rulings": applied,
    })
    payload = {
        "run_id": run_id,
        "meta": meta,
        "claims": list(source["claims"]),
        "rows": rows,
        "characterization": list(source.get("characterization") or []),
        "roster": source.get("roster") or {},
        "evidence": {sid: list(evs)
                     for sid, evs in (source.get("evidence") or {}).items()},
    }
    try:
        from truthbot.verdict.composition_telemetry import composition_report
        payload["composition"] = composition_report(payload["rows"],
                                                    payload["evidence"])
    except Exception:
        pass
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload, default=str, ensure_ascii=False),
                    encoding="utf-8")
    return path, payload


# ── driver ───────────────────────────────────────────────────────────────────

def wave_artifacts(runs_dir: Path = RUNS_DIR) -> dict[str, dict]:
    """speech_id → the newest artifact carrying a ``meta.wave`` block."""
    out: dict[str, dict] = {}
    for path in sorted(runs_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        meta = doc.get("meta") or {}
        if not meta.get("wave"):
            continue
        speech = meta.get("speech_id") or ""
        if speech:
            out[speech] = doc
    return out


def run(apply: bool, out_dir: Path = OUT_DIR,
        runs_dir: Path = RUNS_DIR) -> dict:
    arts = wave_artifacts(runs_dir)
    speeches = sorted(arts)
    if not speeches:
        raise SystemExit("no wave artifacts found — nothing to apply")

    mech = mechanism_attribution(speeches)
    report = {"schema": "truthbot-wave-rulings v1",
              "generated": datetime.now(timezone.utc).isoformat(),
              "ruling_date": RULING_DATE, "applied": bool(apply),
              "mechanism_summary": {"by_mechanism": mech["by_mechanism"],
                                    "arithmetic": mech["arithmetic"]},
              "per_speech": [], "spend_usd": 0.0}

    for speech in speeches:
        art = arts[speech]
        deferred = set((art["meta"]["wave"] or {}).get("deferred_newly_gated") or [])
        # Only ever gate sids the measured attribution also gates: the flip set
        # and the artifact were written at different times, and a sid in one and
        # not the other is a discrepancy to surface, not to act on.
        measured = {s for s in mech["mechanism"] if s.startswith(speech + ":")}
        unmeasured = sorted(deferred - measured)
        gating = deferred & measured

        rows = apply_deferred_gated(art["rows"], gating)
        ancestors = lineage(art, runs_dir)
        adopted = reemit_blank_rationales(rows, ancestors)
        conflicts = annotate_coherence(art["claims"], rows)

        from truthbot.verdict import verdict_audit as va
        blanks = va.blank_rationale_violations(rows)

        rec = {
            "speech": speech,
            "source_run": art.get("run_id"),
            "deferred_recorded": len(deferred),
            "deferred_applied": len(gating),
            "deferred_unmeasured": unmeasured,
            "rationales_adopted": adopted,
            "coherence_annotated": [c["sids"] for c in conflicts],
            "blank_rationales_remaining": blanks,
            "by_mechanism": {},
        }
        for sid in sorted(gating):
            m = mech["mechanism"][sid]
            rec["by_mechanism"][m] = rec["by_mechanism"].get(m, 0) + 1

        if apply:
            applied_meta = {
                "date": RULING_DATE,
                "deferred_newly_gated_applied": sorted(gating),
                "mechanism": {s: mech["mechanism"][s] for s in sorted(gating)},
                "rationales_adopted": [a["sid"] for a in adopted],
                "coherence_annotated": [c["sids"] for c in conflicts],
                "source": "scripts/apply_wave_rulings.py ($0, no model calls)",
            }
            path, payload = write_artifact(art, rows, speech_id=speech,
                                           applied=applied_meta,
                                           out_dir=runs_dir)
            update_manifest(payload["run_id"], speech)
            old_rows = {r.get("sid"): r for r in art["rows"]}
            changed = sorted(gating | {a["sid"] for a in adopted}
                             | {s for c in conflicts for s in c["sids"]})
            diff = build_verdict_diff([old_rows[s] for s in changed
                                       if s in old_rows],
                                      [r for r in rows
                                       if r.get("sid") in set(changed)],
                                      art["claims"])
            diff_out = {"speech_id": speech, "rebuild_of": art.get("run_id", ""),
                        "new_run_id": payload["run_id"], "rulings": True,
                        "ruling_sids": changed,
                        "mechanism": {s: mech["mechanism"].get(s, "")
                                      for s in changed},
                        **diff}
            diff_path = out_dir / f"rulings_{speech}_verdict_diff.json"
            diff_path.write_text(
                json.dumps(diff_out, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
            rec.update(new_run_id=payload["run_id"], artifact=str(path),
                       diff=str(diff_path), counts=diff["counts"])
        report["per_speech"].append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "deferred_gated_mechanism.json").write_text(
        json.dumps(mech, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if apply:
        (out_dir / "wave_rulings_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true",
                       help="measure and report; write no artifact")
    group.add_argument("--apply", action="store_true",
                       help="write the new artifacts, diffs and report")
    args = ap.parse_args(argv)

    report = run(apply=bool(args.apply))
    print(f"wave rulings {RULING_DATE} — "
          f"{'APPLIED' if args.apply else 'DRY RUN'} ($0, no model calls)\n")
    print("mechanism: " + report["mechanism_summary"]["arithmetic"])
    print("           " + json.dumps(report["mechanism_summary"]["by_mechanism"]))
    print()
    for rec in report["per_speech"]:
        print(f"{rec['speech']}: gated {rec['deferred_applied']}"
              f"/{rec['deferred_recorded']} deferred "
              f"{rec['by_mechanism']}, "
              f"{len(rec['rationales_adopted'])} rationale(s) adopted, "
              f"{len(rec['coherence_annotated'])} coherence annotation(s)")
        for adopted in rec["rationales_adopted"]:
            print(f"    rationale {adopted['sid']} ← {adopted['attribution']}")
        for sids in rec["coherence_annotated"]:
            print(f"    coherence {sids[0]} ↔ {sids[1]}")
        if rec["deferred_unmeasured"]:
            print(f"    DISCREPANCY — recorded but not measured: "
                  f"{rec['deferred_unmeasured']}")
        if rec["blank_rationales_remaining"]:
            print(f"    STILL BLANK: "
                  f"{[v['sid'] for v in rec['blank_rationales_remaining']]}")
        if rec.get("new_run_id"):
            print(f"    artifact {rec['new_run_id'][:8]} {rec['counts']}")
    print("\n$0 — no model calls were made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
