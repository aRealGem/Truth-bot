#!/usr/bin/env python3
"""The adjudication wave (remediation v2, B1b) — guardrailed, resumable.

ONE wave, one bill. Every claim the B1a+B2 stance re-score plus the 2026-08-09
ratification RELEASED from the T2.4 evidence gate, plus the handful the owner
named by hand, gets a production PCA panel call — and nothing else does.

WHAT MAKES THIS CHEAP: THERE IS NO RETRIEVAL
--------------------------------------------
The five rebuilt runs already hold every pack on disk. This script re-gates
those STORED packs under the ratified rules and hands the result straight to
``adjudicator.adjudicate`` through the ``pack_builder`` hook — a builder that
RETURNS a pack instead of building one. So:

  * no R1/R2/R3 calls, therefore no off-proxy spend and no list-price estimate
    to reconcile — every dollar this script spends is a LiteLLM proxy call the
    ledger already knows about, and ``proxy_key_spend()`` is the whole truth;
  * the evidence a claim is judged on is byte-identical to the evidence the
    re-gate reasoned about, so the flip set and the verdict diff describe the
    same packs.

THE STANCE VINTAGE
------------------
Stored packs carry the stance they were BUILT with, which for most items is
nothing at all — that is the defect B1a existed to repair. The repaired scores
live in sidecars, and the merge order matters (B1a first, B2 on top, per SID).
Rather than restate that, this script imports the selection and merge the final
re-gate used (``regate_from_rescore.merge_sidecars`` /
``load_rescore_sidecar`` / ``overlay_rescores`` / ``gate_once``). If the two
ever disagreed, a claim could be adjudicated on evidence the flip set never saw.

BOTH RATIFIED RULES ARE ON
--------------------------
D15 (utterance-record exclusion) and D16α (statistical release) were ratified
2026-08-09 and are default-on. This script passes them EXPLICITLY anyway and
prints the ambient default beside them, because "the default is on" is a claim
about the environment and the environment is not part of the artifact.

GUARDRAILS (mirroring scripts/phase3_rebuild.py)
------------------------------------------------
  * ``--budget USD`` is REQUIRED with ``--go``; it is a HARD cap, not a target;
  * the per-CLAIM circuit breaker fires inside the pack builder, i.e. BEFORE
    the chunk that claim belongs to is sent to the panel;
  * chunked (CHUNK_SIZE=5) with a chunk journal, so a halt loses nothing and a
    re-run re-spends only on unbanked sids;
  * a halt prints resume instructions and exits CLEAN — no traceback.

Usage (repo root):
  PYTHONPATH=.:src .venv/bin/python scripts/wave_adjudicate.py            # $0 plan
  set -a; . ~/.env; . ./.env; set +a
  PYTHONPATH=.:src .venv/bin/python scripts/wave_adjudicate.py --go --budget 3.28
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# scripts/ is not a package; these are path-based imports, the same way the
# phase-3 and re-gate tests load their subjects. Everything imported here is $0.
from phase3_rebuild import (BudgetHalt, ChunkFailed, CHUNK_SIZE,  # noqa: E402
                            _adjudicate_chunk, build_verdict_diff,
                            outcome_label, pending_claims, print_diff,
                            update_manifest)
from regate_from_rescore import (claim_shape_map, gate_once,  # noqa: E402
                                 load_rescore_sidecar, merge_sidecars,
                                 overlay_rescores)
from rescore_stored_packs import (REBUILT_RUNS, artifact_path,  # noqa: E402
                                  b2_sidecar_path, load_artifact, sidecar_path)

OUT_DIR = REPO / "metrics" / "remediation_v2"
JOURNAL_DIR = REPO / "metrics" / "journals"
FLIPSET_PATH = OUT_DIR / "regate_flipset.json"
EXHIBIT_PATH = (REPO / "metrics" / "computed_exhibits"
                / "cpilfesl_q4_2025_annualized.json")

#: Pipeline generation these artifacts are produced at. Same generation as the
#: rebuilds: the wave changes VERDICTS, not the method that produced the packs.
PIPELINE_GENERATION = "v2.3-role-axis-s5cap"
WAVE_TAG = "adjudication wave B1b"
WAVE_DATE = "2026-08-09"

#: Claims the owner named for re-adjudication regardless of the gate.
#: ``regate_from_rescore.NAMED_EXTRAS`` is the costing list and still contains
#: trump_2026:0343, which the ratified rules now GATE — a gated claim is
#: answered deterministically and for free, so paying a panel for it buys
#: nothing. The drop is derived, not hardcoded: any named extra in the flip
#: set's ``newly_gated_sids`` falls out (see :func:`wave_set`).
NAMED_EXTRAS: tuple[str, ...] = (
    "trump_2026:0030", "trump_2026:0031", "trump_2026:0023",
    "trump_2026:0024", "trump_2026:0343", "clinton_1998:0313",
)

#: trump_2026:0462 ships as a models-split with NO verdict, which no
#: deterministic re-gate can settle — and the acceptance suite carries a
#: strict xfail tied explicitly to "the adjudication wave". Adding it is what
#: lets that marker resolve one way or the other instead of outliving the wave
#: it names.
SPLIT_EXTRAS: tuple[str, ...] = ("trump_2026:0462",)

#: Claims the ratified computed exhibit is offered to. Whether it is actually
#: ATTACHED is decided per claim by ``computed_exhibit.is_admissible`` against
#: that claim's shape — see :func:`exhibit_for`. Offering is not attaching.
EXHIBIT_SIDS: tuple[str, ...] = ("trump_2026:0030", "trump_2026:0031")

#: The ratified rationale for trump_2026:0469, carried here so that if the
#: claim ever reaches a corrections ledger the reason travels with it. 0469 is
#: NOT in this wave (it stays Unverifiable by ratification, not by defect).
BECKSTROM_0469_RATIONALE = (
    "purposive clause uncheckable; factual core confirmed; sole purposive "
    "support is Political-tier"
)


# ── the claim set ($0) ───────────────────────────────────────────────────────

def wave_set(flipset: dict,
             named_extras: tuple[str, ...] = NAMED_EXTRAS,
             split_extras: tuple[str, ...] = SPLIT_EXTRAS) -> dict:
    """The wave's claim set, with every sid's reason for being in it.

    Three sources, de-duplicated in this precedence order (a sid released by
    the gate is recorded as released even if it is also a named extra):

      1. ``released_sids`` — the gate now admits them, so they are eligible for
         a substantive verdict and only a panel can give them one;
      2. named extras — owner-designated, minus any the ratified rules now
         GATE (answered for free; paying for them buys nothing);
      3. split extras — models-splits with no verdict, which nothing
         deterministic can settle.

    Returns ``{"sids": [...], "reason": {sid: why}, "dropped": {sid: why},
    "by_speech": {speech: [sids]}}``."""
    released = list(flipset.get("released_sids") or [])
    newly_gated = set(flipset.get("newly_gated_sids") or ())

    reason: dict[str, str] = {}
    dropped: dict[str, str] = {}
    for sid in sorted(released):
        reason[sid] = "released"
    for sid in named_extras:
        if sid in newly_gated:
            dropped[sid] = ("newly gated by the ratified rules — answered "
                            "deterministically, no panel call needed")
            continue
        reason.setdefault(sid, "named-extra")
    for sid in split_extras:
        if sid in newly_gated:
            dropped[sid] = "newly gated by the ratified rules"
            continue
        reason.setdefault(sid, "models-split extra")

    sids = sorted(reason)
    by_speech: dict[str, list[str]] = {}
    for sid in sids:
        by_speech.setdefault(sid.split(":", 1)[0], []).append(sid)
    return {"sids": sids, "reason": reason, "dropped": dropped,
            "by_speech": by_speech}


def print_wave_set(wave: dict) -> None:
    print(f"\nWAVE CLAIM SET — {len(wave['sids'])} claim(s)")
    for speech in sorted(wave["by_speech"]):
        sids = wave["by_speech"][speech]
        print(f"  {speech} ({len(sids)}):")
        for sid in sids:
            print(f"    {sid}  [{wave['reason'][sid]}]")
    for sid, why in sorted(wave["dropped"].items()):
        print(f"  DROPPED {sid}: {why}")


# ── the stored packs ($0) ────────────────────────────────────────────────────

def merged_sidecar(speech: str, *, use_b2: bool = True) -> dict:
    """The B1a+B2 merged stance sidecar for a speech — the SAME selection and
    merge the final re-gate ran (imported, not restated)."""
    b1a = load_rescore_sidecar(sidecar_path(speech), speech, REBUILT_RUNS[speech])
    b1a["pass_label"] = "b1a"
    b2 = None
    b2_path = b2_sidecar_path(speech)
    if use_b2 and b2_path.exists():
        b2 = load_rescore_sidecar(b2_path, speech, REBUILT_RUNS[speech])
        b2["pass_label"] = "b2"
    return merge_sidecars(b1a, b2)


def rules_default_state() -> dict:
    """What the AMBIENT flags say right now, so "both rules are on" is
    reported as an observation instead of an assumption."""
    from truthbot.verdict import statistical_release, utterance_record
    return {"utterance_record": bool(utterance_record.flag_enabled()),
            "statistical_release": bool(statistical_release.flag_enabled())}


def build_wave_packs(speech: str, artifact: dict, sidecar: dict,
                     sids: list[str], *,
                     utterance_record: bool = True,
                     statistical_release: bool = True) -> tuple[dict, list[dict]]:
    """Re-gate the STORED packs for ``sids`` and return (packs, telemetry).

    Pure and free: no retrieval, no model call, no mutation of ``artifact``.
    Each pack is rebuilt from the artifact's own Evidence dumps, overlaid with
    the merged stance vintage, run through the REAL gate
    (``regate_from_rescore.gate_once`` → ``consolidator.consolidate``) under
    the ratified rules, and assembled with
    ``evidence_pack_v2.pack_item_from_citation`` — the same function a live
    build uses, so a stored-pack item and a freshly-built one cannot drift.

    The gate is RE-RUN rather than assumed: a pack that fails it comes back
    carrying ``gate_code``, and ``adjudicate`` will force Unverifiable without
    spending a panel call. That is the correct outcome, and it must not be
    bypassed just because the flip set expected a release."""
    from truthbot.verdict import speech_context
    from truthbot.verdict.consolidator import scoring_telemetry
    from truthbot.verdict.evidence_pack import EvidencePack, window_for
    from truthbot.verdict.evidence_pack_v2 import pack_item_from_citation
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
    shapes, _ = claim_shape_map(artifact, speech)
    scored = sidecar.get("sids") or {}
    stored = artifact.get("evidence") or {}

    packs: dict = {}
    telemetry: list[dict] = []
    for sid in sids:
        text = ((claims.get(sid) or {}).get("text") or "").strip()
        evidence = evidence_from_artifact_dict({sid: stored.get(sid) or []})[sid]
        join = overlay_rescores(evidence, scored.get(sid) or [])
        result, breakdown = gate_once(
            sid, evidence, utterance=utterance, claim_shape=shapes.get(sid, ""),
            relation_of=relation_of, claim_text=text,
            utterance_record=utterance_record,
            statistical_release=statistical_release)
        items = [pack_item_from_citation(i, cit)
                 for i, cit in enumerate(result.items, start=1)]
        packs[sid] = EvidencePack(
            sid=sid, window=window_for(sid), items=items,
            gate_code=result.gate_code,
            excluded_fc=list(getattr(result, "excluded_fc", []) or []),
            quarantined=list(getattr(result, "quarantined", []) or []),
            scoring=scoring_telemetry(items))
        telemetry.append({
            "sid": sid, "speech": speech, "claim_shape": shapes.get(sid, ""),
            "stored_items": len(stored.get(sid) or []),
            "pack_items": len(items), "gate_code": result.gate_code,
            "quota_met": bool(result.quota_met),
            "scores_joined": join["matched"],
            "items_unscored": len(join["artifact_unscored"]),
            "credit": breakdown,
        })
    return packs, telemetry


# ── the computed exhibit ($0) ────────────────────────────────────────────────

def load_exhibit(path: Path = EXHIBIT_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exhibit_for(sid: str, exhibit: dict, claim_shape: str) -> tuple[Optional[dict], str]:
    """(exhibit-or-None, why-not) for one claim.

    The admissibility rule is the load-bearing one and it is NOT re-derived
    here: ``publish.computed_exhibit.is_admissible`` decides, against the SAME
    claim shape the gate used. A refusal is returned as a reason string rather
    than raised, because a wave that halted on it would strand 28 other claims
    over a display decision — but it is never silent: the reason is printed,
    journaled in the run report, and reported to the owner."""
    from truthbot.publish import computed_exhibit as ce

    if sid not in EXHIBIT_SIDS:
        return None, ""
    if ce.is_admissible(exhibit, claim_shape=claim_shape):
        return dict(exhibit), ""
    if not ce.is_well_formed(exhibit):
        return None, "exhibit is malformed (missing required fields)"
    return None, (
        f"INADMISSIBLE on claim_shape={claim_shape!r}: a computed exhibit is "
        "admissible only for a numeric claim-vs-series comparison, never on a "
        "C-EVAL judgment — arithmetic cannot settle an evaluative claim")


def exhibit_context(exhibit: dict) -> str:
    """The exhibit as the PANEL sees it: formula, both input levels, the
    vintage, and an explicit instruction that it is arithmetic on a published
    series and not a verdict. Appended to the claim context, which
    ``adjudicator.build_items`` puts into the payload after the temporal
    preamble."""
    inputs = "; ".join(f"{day} = {exhibit['inputs'][day]}"
                       for day in sorted(exhibit["inputs"]))
    return (
        "\n\nCOMPUTED EXHIBIT (arithmetic on a published data series, pinned "
        "to a data vintage — it is evidence about the NUMBER, not a verdict "
        "on the claim):\n"
        f"  series: {exhibit['source']} {exhibit['series']}\n"
        f"  data vintage: {exhibit['vintage_date']}\n"
        f"  inputs: {inputs}\n"
        f"  formula: {exhibit['formula']} = {float(exhibit['result']) * 100:.3f}%\n"
        + (f"  note: {exhibit['note']}\n" if exhibit.get("note") else "")
        + "  Use it to identify WHICH measure the claim is stating. It "
          "settles arithmetic only; it does not settle whether the claim's "
          "characterisation is fair.\n"
    )


# ── artifact writing ($0) ────────────────────────────────────────────────────

def merge_wave_rows(source_art: dict, wave_rows: list[dict]) -> list[dict]:
    """The new artifact's rows: the source artifact's rows with the wave's sids
    REPLACED in place, everything else verbatim and in the original order.

    Deliberately a replace, not a rebuild. This wave re-adjudicated 29 claims;
    it did not re-adjudicate the other 500, and an artifact that quietly
    restated them would be claiming work that was never done."""
    new_by_sid = {r.get("sid"): r for r in wave_rows}
    return [new_by_sid.get(r.get("sid"), r) for r in (source_art.get("rows") or [])]


def merge_wave_evidence(source_art: dict, packs: dict) -> dict:
    """The new artifact's evidence: the source artifact's, with the wave sids'
    packs replaced by the ones the panel actually saw (re-gated, stance
    overlaid). Non-wave sids keep their stored vintage — see
    :func:`merge_wave_rows` for why."""
    from truthbot.verdict import bridge as bridge_mod

    out = {sid: list(evs) for sid, evs in (source_art.get("evidence") or {}).items()}
    for sid, pack in (packs or {}).items():
        out[sid] = [ev.model_dump(mode="json")
                    for ev in bridge_mod._pack_to_evidence(sid, pack)]
    return out


def write_wave_artifact(source_art: dict, wave_rows: list[dict], packs: dict,
                        roster_note: dict, *, speech_id: str,
                        wave_sids: list[str], reasons: dict,
                        deferred_gated: list[str],
                        rules: dict, exhibits: dict,
                        run_id: Optional[str] = None,
                        out_dir: Optional[Path] = None,
                        cost_usd: float = 0.0) -> tuple[Path, dict]:
    """Write the wave's pca_runs artifact.

    Same shape ``rerender_pca_site.py`` consumes ({run_id, meta, claims, rows,
    characterization, roster, evidence}) — a sidecar would have needed a new
    consumer, and the renderer already reads artifacts. The SOURCE artifact is
    never touched: archive-never-delete means the rebuilt run stays exactly as
    it was and this is a new file with a new id and ``rebuild_of`` lineage."""
    import uuid

    run_id = run_id or str(uuid.uuid4())
    out_dir = Path(out_dir) if out_dir is not None else REPO / "metrics" / "pca_runs"
    old_meta = source_art.get("meta") or {}
    meta = {
        "speaker": old_meta.get("speaker", ""),
        "date": old_meta.get("date", ""),
        "speech_id": speech_id,
        "venue": old_meta.get("venue", ""),
        "roster": roster_note.get("name", "prod"),
        "n_sentences": old_meta.get("n_sentences"),
        "n_check_worthy": old_meta.get("n_check_worthy"),
        "cost_usd": round(cost_usd, 4),
        "rebuild_of": source_art.get("run_id", ""),
        "pipeline_generation": PIPELINE_GENERATION,
        "remediation": WAVE_TAG,
        "wave": {
            "date": WAVE_DATE,
            "rules": dict(rules),
            "stance_vintage": "b1a+b2 merged re-score sidecars",
            "retrieval": "none — stored packs re-gated, never re-retrieved",
            "sids_adjudicated": list(wave_sids),
            "reasons": {sid: reasons.get(sid, "") for sid in wave_sids},
            "computed_exhibits": dict(exhibits),
            # Honesty about what this artifact does NOT do: the ratified rules
            # also newly GATE claims outside the wave. Applying that is a
            # separate decision (it collides with a passing acceptance case),
            # so the sids are RECORDED here and left un-applied rather than
            # applied quietly or forgotten.
            "deferred_newly_gated": sorted(deferred_gated),
        },
    }
    payload = {
        "run_id": run_id,
        "meta": meta,
        "claims": list(source_art["claims"]),
        "rows": wave_rows,
        "characterization": list(source_art.get("characterization") or []),
        "roster": roster_note,
        "evidence": merge_wave_evidence(source_art, packs),
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


# ── the funded path ──────────────────────────────────────────────────────────

def go_refusal(budget: Optional[float]) -> Optional[str]:
    """The one --go refusal. There is no retrieval in this wave, so the
    TRUTHBOT_R2_MODEL economy guard phase-3 needs does not apply — no R2 call
    is possible. The budget cap still is, and it is required."""
    if budget is None or budget <= 0:
        return ("REFUSING to spend: --budget USD is REQUIRED with --go (it is "
                "the halt cap for the per-claim breaker and the between-chunk "
                "checks). No spend attempted.")
    return None


def run_wave(args) -> int:
    from hydramind.rosters import get_roster
    from truthbot.verdict import adjudicator, proxy_lane, publish_pipeline

    if not proxy_lane.key_present():
        sys.exit(proxy_lane.BLOCKED_MSG)
    refusal = go_refusal(args.budget)
    if refusal:
        sys.exit(refusal)

    flipset = json.loads(FLIPSET_PATH.read_text(encoding="utf-8"))
    wave = wave_set(flipset)
    print_wave_set(wave)
    if args.sids:
        keep = set(args.sids)
        wave["sids"] = [s for s in wave["sids"] if s in keep]
        wave["by_speech"] = {}
        for sid in wave["sids"]:
            wave["by_speech"].setdefault(sid.split(":", 1)[0], []).append(sid)
        print(f"\n--sids slice: {len(wave['sids'])} claim(s)")

    rules = {"utterance_record": True, "statistical_release": True}
    print(f"\nrules: D15/D16(alpha) passed explicitly as {rules}; "
          f"ambient default reads {rules_default_state()}")

    exhibit = load_exhibit()
    hm = proxy_lane.build_hydramind(response_parser=adjudicator.parse_verdict)
    roster_note = {"name": "prod", "seats": dict(get_roster("prod").seats)}
    start_spend = proxy_lane.proxy_key_spend()
    print(f"proxy key spend at start: ${start_spend:.4f} "
          f"(HARD cap ${args.budget:.2f})")

    newly_gated = sorted(flipset.get("newly_gated_sids") or ())
    report = {"schema": "truthbot-wave-report v1",
              "generated": datetime.now(timezone.utc).isoformat(),
              "wave_date": WAVE_DATE, "rules": rules,
              "claim_set": wave["sids"], "reasons": wave["reason"],
              "dropped": wave["dropped"], "per_speech": [],
              "exhibit_decisions": {}, "halted": ""}
    halted = ""
    banked_total = 0.0

    for speech in sorted(wave["by_speech"]):
        sids = wave["by_speech"][speech]
        art = load_artifact(artifact_path(speech))
        sidecar = merged_sidecar(speech)
        packs, pack_tel = build_wave_packs(speech, art, sidecar, sids, **rules)
        claims_by_sid = {c.get("sid"): c for c in art["claims"]}
        shapes = {t["sid"]: t["claim_shape"] for t in pack_tel}

        journal = JOURNAL_DIR / f"{speech}_wave.jsonl"
        done_rows, _, banked_cost, _ = publish_pipeline.load_chunk_journal(journal)
        banked_total += banked_cost

        claims = []
        for sid in sids:
            src = claims_by_sid[sid]
            context = src.get("context", "") or ""
            ex, why_not = exhibit_for(sid, exhibit, shapes.get(sid, ""))
            if sid in EXHIBIT_SIDS:
                report["exhibit_decisions"][sid] = {
                    "claim_shape": shapes.get(sid, ""),
                    "attached": bool(ex), "reason": why_not}
                print(f"  computed exhibit {sid}: "
                      + ("ATTACHED" if ex else f"NOT attached — {why_not}"))
            if ex:
                context = context + exhibit_context(ex)
            claims.append({"sid": sid, "text": src["text"], "context": context})

        todo = pending_claims(claims, done_rows)
        if done_rows:
            print(f"{speech} resume: {len(done_rows)} banked "
                  f"(${banked_cost:.4f}), {len(todo)} to run")

        def pack_builder(sid: str, text: str, context: str):
            spent = (proxy_lane.proxy_key_spend() - start_spend) + banked_total
            if spent >= args.budget:
                raise BudgetHalt(f"${spent:.4f} >= cap ${args.budget:.2f} "
                                 f"(before the panel call for {sid})")
            return packs[sid]

        chunks = [todo[i:i + CHUNK_SIZE] for i in range(0, len(todo), CHUNK_SIZE)]
        all_rows = list(done_rows)
        for idx, chunk in enumerate(chunks, 1):
            running = (proxy_lane.proxy_key_spend() - start_spend) + banked_total
            if running >= args.budget:
                halted = (f"BUDGET HALT before {speech} chunk {idx}: "
                          f"${running:.4f} >= cap ${args.budget:.2f}")
                print(halted)
                break
            t0, s0 = time.time(), proxy_lane.proxy_key_spend()
            try:
                rows, _manifest, notes = _adjudicate_chunk(
                    adjudicator, hm, chunk, pack_builder, idx)
            except BudgetHalt as exc:
                halted = f"BUDGET HALT mid-chunk {idx} ({speech}): {exc}"
                print(halted)
                break
            except ChunkFailed as exc:
                halted = f"TRANSIENT HALT at {speech} chunk {idx}: {exc}"
                print(halted)
                break
            s1, t1 = proxy_lane.proxy_key_spend(), time.time()
            publish_pipeline.append_chunk_journal(
                journal, idx, rows, notes.get("packs") or {}, s1 - s0,
                roster=roster_note if not done_rows and idx == 1 else None)
            all_rows.extend(rows)
            running = (proxy_lane.proxy_key_spend() - start_spend) + banked_total
            print(f"{speech} chunk {idx}/{len(chunks)} ({len(chunk)} claims): "
                  f"${s1 - s0:.4f} · wave running ${running:.4f} / "
                  f"${args.budget:.2f} · {t1 - t0:.0f}s")

        complete = {c["sid"] for c in claims} <= {r.get("sid") for r in all_rows}
        wave_rows = [r for r in all_rows if r.get("sid") in set(sids)]
        speech_rec = {"speech": speech, "sids": sids, "complete": complete,
                      "rows_banked": len(wave_rows), "packs": pack_tel,
                      "journal": str(journal)}
        if not complete:
            print(f"{speech}: INCOMPLETE — {len(sids) - len(wave_rows)} claim(s) "
                  f"unbanked; no artifact written, journal keeps the rest")
            report["per_speech"].append(speech_rec)
            break

        merged_rows = merge_wave_rows(art, wave_rows)
        exhibits = {sid: d for sid, d in report["exhibit_decisions"].items()
                    if d["attached"] and sid in set(sids)}
        for row in merged_rows:
            if row.get("sid") in exhibits:
                row["computed_exhibit"] = dict(exhibit)
        out_path, payload = write_wave_artifact(
            art, merged_rows, packs, roster_note, speech_id=speech,
            wave_sids=sids, reasons=wave["reason"],
            deferred_gated=[s for s in newly_gated if s.startswith(speech + ":")],
            rules=rules, exhibits=exhibits,
            cost_usd=(proxy_lane.proxy_key_spend() - start_spend))
        update_manifest(payload["run_id"], speech)

        old_rows = {r.get("sid"): r for r in art["rows"]}
        diff = build_verdict_diff([old_rows[s] for s in sids if s in old_rows],
                                  wave_rows, art["claims"])
        print(f"\n{speech}: artifact {out_path.name} (rebuild_of "
              f"{art.get('run_id', '')[:8]})")
        print_diff(diff)
        diff_out = {"speech_id": speech, "rebuild_of": art.get("run_id", ""),
                    "new_run_id": payload["run_id"], "wave": True,
                    "wave_sids": sids, "reasons": wave["reason"], **diff}
        diff_path = OUT_DIR / f"wave_{speech}_verdict_diff.json"
        diff_path.write_text(json.dumps(diff_out, indent=2, ensure_ascii=False)
                             + "\n", encoding="utf-8")
        speech_rec.update(new_run_id=payload["run_id"],
                          artifact=str(out_path), diff=str(diff_path),
                          counts=diff["counts"])
        report["per_speech"].append(speech_rec)
        if halted:
            break

    total = (proxy_lane.proxy_key_spend() - start_spend) + banked_total
    report["halted"] = halted
    report["spend_usd"] = round(total, 4)
    report["cap_usd"] = args.budget
    (OUT_DIR / "wave_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSPEND: ${total:.4f} of cap ${args.budget:.2f} "
          "(all on-proxy — no retrieval, so the ledger is the whole bill)")
    print(f"wave report → {OUT_DIR / 'wave_report.json'}")
    if halted:
        print("\nRESUME (re-spends only on unbanked sids):")
        print("  PYTHONPATH=.:src .venv/bin/python scripts/wave_adjudicate.py "
              "--go --budget <USD>")
        return 1
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--go", action="store_true",
                    help="actually spend (else print the plan, $0)")
    ap.add_argument("--budget", type=float, default=None,
                    help="HARD halt cap in USD — REQUIRED with --go")
    ap.add_argument("--sids", nargs="*", default=None,
                    help="restrict to these sids (must already be in the set)")
    args = ap.parse_args(argv)

    flipset = json.loads(FLIPSET_PATH.read_text(encoding="utf-8"))
    wave = wave_set(flipset)
    print(f"Adjudication wave plan — flip set generated "
          f"{flipset.get('generated', '?')[:19]}, rules "
          f"{flipset.get('rules', {}).get('after')}")
    print_wave_set(wave)
    if not args.go:
        print("\n($0 plan only — add --go --budget USD to spend)")
        return 0
    return run_wave(args)


if __name__ == "__main__":            # pragma: no cover
    raise SystemExit(main())
