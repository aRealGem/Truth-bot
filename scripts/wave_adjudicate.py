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

THE AUDITED ESCAPE (``--extra-sids``)
-------------------------------------
``--sids`` slices the wave set and REFUSES anything outside it, which is the
right default: the wave set is derived from the flip set, and a claim that
wanders in by hand is a claim nobody costed. But a publish-blocking defect can
sit OUTSIDE the flip set — a verdict that ships with no rationale, which the R-3
ruling makes unpublishable and which no deterministic re-gate can repair.

``--extra-sids`` is the one way in, and it is audited rather than silent:

  * it REQUIRES ``--reason``, and the reason is written into the run report's
    provenance and printed as a banner, so the escape can never be reconstructed
    only from a shell history;
  * it REQUIRES its own ``--tag``, because an escape run writes a report, a
    verdict diff and a chunk journal, and under the default tag those would
    overwrite the wave's. Prior artifacts are never mutated;
  * it does NOT widen the wave set. ``wave_set`` returns exactly what it
    returned before; the escape run's claim set is the named sids and nothing
    else, each recorded with the escape reason;
  * it sources each speech's artifact from the current PUBLISHING HEAD rather
    than the phase-3 rebuild, because an escape happens after the wave and must
    not silently discard the rulings that landed in between.

Usage (repo root):
  PYTHONPATH=.:src .venv/bin/python scripts/wave_adjudicate.py            # $0 plan
  set -a; . ~/.env; . ./.env; set +a
  PYTHONPATH=.:src .venv/bin/python scripts/wave_adjudicate.py --go --budget 3.28
  PYTHONPATH=.:src .venv/bin/python scripts/wave_adjudicate.py --go --budget 0.25 \
      --extra-sids biden_2022:0432 --tag r3 --reason "publish-blocking blank rationale"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
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


#: The default output tag — the wave's own. An escape run may not use it (see
#: :func:`escape_refusal`): the wave's report, diffs and journals are prior
#: artifacts and this script never overwrites them.
DEFAULT_TAG = "wave"

#: How an escaped sid's membership is recorded in the report and the artifact.
ESCAPE_REASON = "extra-sid escape"


def report_path(tag: str = DEFAULT_TAG) -> Path:
    return OUT_DIR / f"{tag}_report.json"


def diff_path(speech: str, tag: str = DEFAULT_TAG) -> Path:
    return OUT_DIR / f"{tag}_{speech}_verdict_diff.json"


def journal_path(speech: str, tag: str = DEFAULT_TAG) -> Path:
    return JOURNAL_DIR / f"{speech}_{tag}.jsonl"


def sids_refusal(available: list[str], requested: list[str]) -> Optional[str]:
    """The ``--sids`` contract, ENFORCED: every requested sid must already be in
    the run's claim set.

    It used to be documented ("must already be in the set") and then silently
    intersected, so a typo — or a sid the flip set never released — produced a
    smaller run and no complaint. A slice that quietly drops what it was asked
    for is the same failure mode as a widened wave, just in the other
    direction. To adjudicate something outside the set, name it with
    ``--extra-sids`` and say why."""
    unknown = [s for s in requested if s not in set(available)]
    if not unknown:
        return None
    return ("REFUSING --sids: not in this run's claim set: "
            + ", ".join(sorted(unknown))
            + ". --sids SLICES the set, it does not extend it. A claim outside "
              "the set (e.g. a publish-blocking defect the flip set never "
              "released) needs --extra-sids with a --reason, which is recorded "
              "in the run report. No spend attempted.")


def escape_refusal(extra_sids: Optional[list[str]], reason: str,
                   tag: str) -> Optional[str]:
    """Everything the audited escape demands before it may cost a cent.

    Three conditions, each of them the fix for a way this could go wrong
    quietly: a reason (or the escape is unreconstructible after the fact), a
    tag of its own (or it overwrites the wave's report/diffs/journals), and
    well-formed sids (or a typo becomes a KeyError halfway through a metered
    run)."""
    if not extra_sids:
        if (reason or "").strip():
            return ("--reason is only meaningful with --extra-sids: there is "
                    "nothing to justify. No spend attempted.")
        return None
    if not (reason or "").strip():
        return ("REFUSING --extra-sids: --reason is REQUIRED. The escape "
                "admits a claim the flip set never released, and the reason is "
                "written into the run report's provenance — an unexplained "
                "escape is indistinguishable from a widened wave. No spend "
                "attempted.")
    if tag == DEFAULT_TAG:
        return (f"REFUSING --extra-sids under --tag {DEFAULT_TAG!r}: an escape "
                f"run writes a report, a verdict diff and a chunk journal, and "
                f"under the wave's own tag those would OVERWRITE the wave's. "
                f"Give it its own --tag. No spend attempted.")
    if not tag or not tag.replace("_", "").replace("-", "").isalnum():
        return (f"REFUSING --tag {tag!r}: it names output files, so it must be "
                "alphanumeric (dashes and underscores allowed). No spend "
                "attempted.")
    malformed = [s for s in extra_sids if len(s.split(":")) != 2 or not all(s.split(":"))]
    if malformed:
        return ("REFUSING --extra-sids: malformed sid(s) "
                + ", ".join(sorted(malformed))
                + " — expected speech_id:claim_id. No spend attempted.")
    return None


def escape_set(extra_sids: list[str], reason: str) -> dict:
    """The claim set of an AUDITED ESCAPE run: EXACTLY the named sids.

    Deliberately not ``wave_set(...) + extras``. The wave set is derived from
    the flip set and was costed as a unit; bolting a hand-named claim onto it
    would re-run 29 claims nobody asked for and would make "the wave" mean two
    different things in two reports. So the escape is its own run over its own
    claims, and ``wave_set`` keeps returning exactly what it returned before —
    which is what "must not silently widen the wave set" means in code."""
    sids = sorted(dict.fromkeys(extra_sids))
    by_speech: dict[str, list[str]] = {}
    for sid in sids:
        by_speech.setdefault(sid.split(":", 1)[0], []).append(sid)
    return {"sids": sids,
            "reason": {sid: f"{ESCAPE_REASON}: {reason.strip()}" for sid in sids},
            "dropped": {}, "by_speech": by_speech}


def escape_provenance(extra_sids: list[str], reason: str, tag: str,
                      wave_sids: list[str]) -> dict:
    """The escape's record, carried in the run report AND in the artifact meta.

    It states what was admitted, why, under what tag — and, explicitly, that
    the wave set is unchanged, with its size, so a later reader can check that
    claim instead of taking it on trust."""
    return {
        "kind": ESCAPE_REASON,
        "reason": reason.strip(),
        "sids": sorted(dict.fromkeys(extra_sids)),
        "tag": tag,
        "wave_set_widened": False,
        "wave_set_size": len(wave_sids),
        "source_artifact": "current publishing head (post-wave, post-rulings)",
        "note": ("--extra-sids admits claims the flip set never released. It "
                 "is for a publish-blocking defect outside the flip set; the "
                 "wave's own claim set, report and diffs are untouched."),
    }


def print_escape(prov: dict) -> None:
    """The banner. Loud on purpose: a metered run that admits hand-named claims
    should be impossible to miss in a log."""
    bar = "=" * 72
    print(f"\n{bar}\nEXTRA-SIDS ESCAPE — {len(prov['sids'])} claim(s) admitted "
          f"OUTSIDE the wave set\n{bar}")
    for sid in prov["sids"]:
        print(f"  {sid}")
    print(f"  reason : {prov['reason']}")
    print(f"  tag    : {prov['tag']} (own report/diffs/journals — the wave's "
          "are not touched)")
    print(f"  source : {prov['source_artifact']}")
    print(f"  wave set UNCHANGED at {prov['wave_set_size']} claim(s)\n{bar}")


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

def d17c_sidecar_path(speech: str) -> Path:
    return REPO / "metrics" / "remediation_v2" / f"rescored_d17c_{speech}.json"


def merged_sidecar(speech: str, *, use_b2: bool = True,
                   use_d17c: bool = True) -> dict:
    """The B1a+B2(+D17-c) merged stance sidecar — the SAME selection and merge
    the final re-gate ran (imported, not restated).

    D17-c is a THIRD pass and merges last, so where it disagrees it wins. That
    precedence rests on the same argument as B1a→B2: the later pass saw
    strictly more evidence — the series rows themselves — so it is better
    informed, not merely newer. Its control arm produced zero flips, so the
    stances it moved are attributable to the rows rather than to re-scoring.

    Without it the gate never sees the flips Stage A paid to measure:
    ``trump_2026:0054`` fails T2.4 with ``insufficient-qualifying-evidence``
    and is forced Unverifiable before any panel call.
    """
    b1a = load_rescore_sidecar(sidecar_path(speech), speech, REBUILT_RUNS[speech])
    b1a["pass_label"] = "b1a"
    b2 = None
    b2_path = b2_sidecar_path(speech)
    if use_b2 and b2_path.exists():
        b2 = load_rescore_sidecar(b2_path, speech, REBUILT_RUNS[speech])
        b2["pass_label"] = "b2"
    d17c = None
    d17c_path = d17c_sidecar_path(speech)
    if use_d17c and d17c_path.exists():
        d17c = load_rescore_sidecar(d17c_path, speech, REBUILT_RUNS[speech])
        d17c["pass_label"] = "d17c"
    return merge_sidecars(b1a, b2, d17c)


def source_artifact(speech: str, *, head: bool = False) -> tuple[Path, dict]:
    """The artifact a run re-adjudicates on top of.

    The wave's source is the PHASE-3 REBUILD (``artifact_path``), pinned, so the
    wave is reproducible against the vintage it was costed on.

    An escape run's source is the current PUBLISHING HEAD, selected by lineage
    (``reshape_rerun_0031.shipping_artifact``). An escape happens after the wave
    and after the rulings pass; re-adjudicating on the rebuild would produce an
    artifact that silently discarded the 65 applied withholdings and the R-1
    shape correction — a repair that undoes two prior repairs is not a repair.

    The import is deferred because ``reshape_rerun_0031`` imports this module;
    by the time this runs, both are loaded."""
    if not head:
        path = artifact_path(speech)
        return path, load_artifact(path)
    from reshape_rerun_0031 import shipping_artifact
    return shipping_artifact(speech)


def rules_default_state() -> dict:
    """What the AMBIENT flags say right now, so "both rules are on" is
    reported as an observation instead of an assumption."""
    from truthbot.verdict import statistical_release, utterance_record
    return {"utterance_record": bool(utterance_record.flag_enabled()),
            "statistical_release": bool(statistical_release.flag_enabled())}


#: D17-c goldens: the committed series excerpts, keyed (claim_sid, evidence_id).
GOLDENS_PATH = (REPO / "metrics" / "remediation_v2" / "d17c_stage0"
                / "goldens.json")

#: R2: this window does not reach the period its claim compares against, so the
#: rows are shown with a warning and cannot settle the claim.
SERIES_PERIOD_MISMATCH = {("obama_2014:0189", "E4")}


def _series_rows_index() -> dict:
    """``(sid, source_url) -> series_rows`` for every committed wave-1 excerpt.

    Keyed on the URL, NOT on the E-number. The golden's ``evidence_id`` is the
    item's position in the SHIPPED pack; the gate re-orders and can drop items,
    so ``E7`` after re-gating is not necessarily ``E7`` before it. Attaching a
    series to the wrong item would be a silent, load-bearing error — the panel
    would be handed a table that does not belong to the source beside it.
    """
    if not GOLDENS_PATH.exists():
        return {}
    from mint_d17c_successors import series_index
    # Resolved against the shipped head via the golden's E-number, NOT via the
    # golden's ``full_table``: those differ (obama_2014:0189 is stored at
    # ``fred.stlouisfed.org/data/cpiaucsl``, while full_table is
    # ``/series/CPIAUCSL``), and keying on full_table silently dropped that
    # excerpt — the one carrying the period-mismatch warning.
    return series_index()


def attach_series_rows(sid: str, items: list) -> list:
    """Attach D17-c series excerpts to the pack items they belong to.

    A no-op for every claim without a committed excerpt, which is all but seven
    of them. ``PackItem`` is frozen, so this replaces rather than mutates.

    Raises on an ambiguous match: if a pack ever carries two items at the same
    source URL, the right rows cannot be chosen and guessing would put a table
    under the wrong citation.
    """
    from dataclasses import replace

    index = _series_rows_index()
    if not index:
        return items
    out = []
    for it in items:
        rows = index.get((sid, it.source_url))
        if rows is None:
            out.append(it)
            continue
        dupes = [x for x in items if x.source_url == it.source_url]
        if len(dupes) > 1:
            raise ValueError(
                f"{sid}: {len(dupes)} pack items share {it.source_url} — cannot "
                "decide which carries the series excerpt")
        out.append(replace(it, series_rows=rows))
    return out


def build_wave_packs(speech: str, artifact: dict, sidecar: dict,
                     sids: list[str], *,
                     utterance_record: bool = True,
                     statistical_release: bool = True,
                     shapes_override: Optional[dict] = None
                     ) -> tuple[dict, list[dict]]:
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
    bypassed just because the flip set expected a release.

    ``shapes_override`` (sid → claim_shape) replaces the shape the registry
    derived, for the one case where a shape is CORRECTED rather than read: the
    Layer-A backfill assigned a shape, a ruling found it wrong, and the gate
    has to run on the corrected one. It is a parameter and not an edit to the
    shape sidecar on purpose — the sidecar is the record of what the classifier
    produced, and overwriting it would erase the evidence that a correction
    happened. Whatever it changes shows up in the telemetry's ``claim_shape``,
    so a re-shaped claim can never be gated on a shape nobody can see."""
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
    shapes.update({k: v for k, v in (shapes_override or {}).items() if v})
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
        items = attach_series_rows(sid, items)
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
    def _rows(block: dict) -> str:
        return "; ".join(f"{day} = {block['inputs'][day]}"
                         for day in sorted(block["inputs"]))

    comp = exhibit.get("comparison") or {}
    comp_text = ""
    if comp.get("formula") and comp.get("result") is not None:
        # The directional row (R-1). Same series, same vintage, same formula,
        # adjacent window — so "down to" is checked against arithmetic on a
        # published series instead of riding on the panel's own recall.
        comp_text = (
            f"  ALSO, {comp.get('label') or 'the preceding window'}:\n"
            f"    inputs: {_rows(comp)}\n"
            f"    formula: {comp['formula']} = "
            f"{float(comp['result']) * 100:.3f}%\n"
            + (f"    change: {float(comp['delta_pp']):+.2f} percentage points\n"
               if comp.get("delta_pp") is not None else ""))
    return (
        "\n\nCOMPUTED EXHIBIT (arithmetic on a published data series, pinned "
        "to a data vintage — it is evidence about the NUMBER, not a verdict "
        "on the claim):\n"
        f"  series: {exhibit['source']} {exhibit['series']}\n"
        f"  data vintage: {exhibit['vintage_date']}\n"
        f"  inputs: {_rows(exhibit)}\n"
        f"  formula: {exhibit['formula']} = {float(exhibit['result']) * 100:.3f}%\n"
        + comp_text
        + (f"  note: {exhibit['note']}\n" if exhibit.get("note") else "")
        + "  Use it to identify WHICH measure the claim is stating, and — "
          "where a second row is given — the DIRECTION of the change between "
          "the two windows. It settles arithmetic only; it does not settle "
          "whether the claim's characterisation is fair.\n"
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


#: Namespace for derived successor ids. Fixed forever: changing it would make
#: every previously derived id irreproducible, which is the whole property.
SUCCESSOR_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def successor_run_id(rebuild_of: str, tag: str, sids) -> str:
    """A run id DERIVED from (parent, tag, claim set) rather than minted fresh.

    A ``uuid4`` here means an identical re-mint produces a different id, so any
    record that already cites the old one — a committed verdict diff, a report,
    a manifest row — ends up pointing at an artifact that never existed. That
    is not hypothetical: it is exactly what the first D17-c wave-2 mint did.

    Same argument as stable claim ids: an identifier for a thing that is a pure
    function of its inputs should be a pure function of those inputs. ``uuid5``
    keeps the UUID shape, so nothing downstream needs to know it is derived.
    """
    basis = "\n".join([rebuild_of or "", tag or "", *sorted(sids or [])])
    return str(uuid.uuid5(SUCCESSOR_NAMESPACE, basis))


def write_wave_artifact(source_art: dict, wave_rows: list[dict], packs: dict,
                        roster_note: dict, *, speech_id: str,
                        wave_sids: list[str], reasons: dict,
                        deferred_gated: list[str],
                        rules: dict, exhibits: dict,
                        run_id: Optional[str] = None,
                        out_dir: Optional[Path] = None,
                        cost_usd: float = 0.0,
                        escape: Optional[dict] = None,
                        remediation: str = WAVE_TAG,
                        inherit_meta: bool = False) -> tuple[Path, dict]:
    """Write the wave's pca_runs artifact.

    Same shape ``rerender_pca_site.py`` consumes ({run_id, meta, claims, rows,
    characterization, roster, evidence}) — a sidecar would have needed a new
    consumer, and the renderer already reads artifacts. The SOURCE artifact is
    never touched: archive-never-delete means the rebuilt run stays exactly as
    it was and this is a new file with a new id and ``rebuild_of`` lineage.

    ``inherit_meta`` keeps the SOURCE artifact's other meta keys (a rulings
    block, a cost note, an earlier wave block) rather than writing a fresh meta
    from a fixed field list. The wave built on the phase-3 rebuilds, whose meta
    held nothing worth carrying; an escape run builds on the publishing head,
    whose meta records the rulings that landed after the wave — dropping that
    would leave the newest artifact the least documented one."""
    run_id = run_id or successor_run_id(
        source_art.get("run_id", ""), remediation, wave_sids)
    out_dir = Path(out_dir) if out_dir is not None else REPO / "metrics" / "pca_runs"
    old_meta = source_art.get("meta") or {}
    meta = dict(old_meta) if inherit_meta else {}
    meta.update({
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
        "remediation": remediation,
    })
    run_block = {
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
    }
    if escape:
        # An escape run gets its OWN meta key. Writing it into ``wave`` would
        # overwrite the inherited block and leave the artifact claiming that
        # the 2026-08-09 wave adjudicated two claims.
        run_block["escape"] = dict(escape)
        run_block["date"] = date.today().isoformat()
        meta["escape_run"] = run_block
    else:
        meta["wave"] = run_block
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

#: Seconds to wait before re-reading the proxy key's spend counter. It is
#: written ASYNCHRONOUSLY, so an immediate read can report $0 for a call that
#: cost money — which is exactly what happened on the 2026-08-10 R-1 run
#: (banked $0.0000, true cost $0.0036, patched afterwards by hand).
SPEND_SETTLE_S = 20


def settled_delta(proxy_lane, start_spend: float, *,
                  settle_s: float = SPEND_SETTLE_S) -> float:
    """Spend since ``start_spend``, read twice and rounded UP to the larger.

    A cost report may never round DOWN: under-reporting a bill is the one
    direction that looks like good news."""
    first = proxy_lane.proxy_key_spend() - start_spend
    time.sleep(settle_s)
    return max(first, proxy_lane.proxy_key_spend() - start_spend)


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

    extra_sids = list(getattr(args, "extra_sids", None) or [])
    tag = getattr(args, "tag", DEFAULT_TAG) or DEFAULT_TAG
    reason = getattr(args, "reason", "") or ""
    refusal = escape_refusal(extra_sids, reason, tag)
    if refusal:
        sys.exit(refusal)

    flipset = json.loads(FLIPSET_PATH.read_text(encoding="utf-8"))
    full_wave = wave_set(flipset)
    print_wave_set(full_wave)
    escape = None
    if extra_sids:
        escape = escape_provenance(extra_sids, reason, tag, full_wave["sids"])
        wave = escape_set(extra_sids, reason)
        print_escape(escape)
    else:
        wave = full_wave
    if args.sids:
        refusal = sids_refusal(wave["sids"], args.sids)
        if refusal:
            sys.exit(refusal)
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
          f"(HARD cap ${args.budget:.2f})", flush=True)

    newly_gated = sorted(flipset.get("newly_gated_sids") or ())
    report = {"schema": "truthbot-wave-report v1",
              "generated": datetime.now(timezone.utc).isoformat(),
              "wave_date": WAVE_DATE, "rules": rules, "tag": tag,
              "escape": escape,
              "claim_set": wave["sids"], "reasons": wave["reason"],
              "dropped": wave["dropped"], "per_speech": [],
              "exhibit_decisions": {}, "halted": ""}
    halted = ""
    banked_total = 0.0

    for speech in sorted(wave["by_speech"]):
        sids = wave["by_speech"][speech]
        src_path, art = source_artifact(speech, head=bool(escape))
        print(f"\n{speech}: source artifact {str(art.get('run_id'))[:8]} "
              f"({src_path.name})")
        sidecar = merged_sidecar(speech)
        packs, pack_tel = build_wave_packs(speech, art, sidecar, sids, **rules)
        claims_by_sid = {c.get("sid"): c for c in art["claims"]}
        shapes = {t["sid"]: t["claim_shape"] for t in pack_tel}

        journal = journal_path(speech, tag)
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
                  f"${args.budget:.2f} · {t1 - t0:.0f}s", flush=True)

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
            # An escape run builds on the publishing head, where the deferred
            # newly-gated sids have ALREADY been applied by the rulings pass.
            # Re-listing them as deferred would re-open a closed item.
            deferred_gated=([] if escape else
                            [s for s in newly_gated if s.startswith(speech + ":")]),
            rules=rules, exhibits=exhibits, escape=escape,
            inherit_meta=bool(escape),
            remediation=(f"{ESCAPE_REASON} ({tag})" if escape else WAVE_TAG),
            cost_usd=settled_delta(proxy_lane, start_spend) + banked_total)
        update_manifest(payload["run_id"], speech)

        old_rows = {r.get("sid"): r for r in art["rows"]}
        diff = build_verdict_diff([old_rows[s] for s in sids if s in old_rows],
                                  wave_rows, art["claims"])
        print(f"\n{speech}: artifact {out_path.name} (rebuild_of "
              f"{art.get('run_id', '')[:8]})")
        print_diff(diff)
        diff_out = {"speech_id": speech, "rebuild_of": art.get("run_id", ""),
                    "new_run_id": payload["run_id"], "wave": not escape,
                    "escape": escape, "tag": tag,
                    "wave_sids": sids, "reasons": wave["reason"], **diff}
        diff_file = diff_path(speech, tag)
        diff_file.write_text(json.dumps(diff_out, indent=2, ensure_ascii=False)
                             + "\n", encoding="utf-8")
        speech_rec.update(new_run_id=payload["run_id"],
                          artifact=str(out_path), diff=str(diff_file),
                          counts=diff["counts"])
        report["per_speech"].append(speech_rec)
        if halted:
            break

    total = settled_delta(proxy_lane, start_spend) + banked_total
    report["halted"] = halted
    report["spend_usd"] = round(total, 4)
    report["cap_usd"] = args.budget
    out_report = report_path(tag)
    out_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSPEND: ${total:.4f} of cap ${args.budget:.2f} "
          "(all on-proxy — no retrieval, so the ledger is the whole bill)")
    print(f"run report → {out_report}")
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
                    help="restrict to these sids — SLICES the claim set and "
                         "REFUSES any sid outside it")
    ap.add_argument("--extra-sids", nargs="+", default=None,
                    help="AUDITED ESCAPE: adjudicate these sids INSTEAD of the "
                         "wave set (for a publish-blocking defect the flip set "
                         "never released). Requires --reason and its own --tag; "
                         "the wave set is not widened")
    ap.add_argument("--reason", default="",
                    help="why the --extra-sids are admitted — REQUIRED with "
                         "--extra-sids, recorded in the run report and the "
                         "artifact's provenance")
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help=f"names this run's report, diffs and journals "
                         f"(default {DEFAULT_TAG!r}; an escape run must use "
                         f"its own so it cannot overwrite the wave's)")
    args = ap.parse_args(argv)

    refusal = escape_refusal(args.extra_sids, args.reason, args.tag)
    if refusal:
        print(refusal)
        return 2

    flipset = json.loads(FLIPSET_PATH.read_text(encoding="utf-8"))
    wave = wave_set(flipset)
    print(f"Adjudication wave plan — flip set generated "
          f"{flipset.get('generated', '?')[:19]}, rules "
          f"{flipset.get('rules', {}).get('after')}")
    print_wave_set(wave)
    if args.extra_sids:
        print_escape(escape_provenance(args.extra_sids, args.reason,
                                       args.tag, wave["sids"]))
    if not args.go:
        print("\n($0 plan only — add --go --budget USD to spend)")
        return 0
    return run_wave(args)


if __name__ == "__main__":            # pragma: no cover
    raise SystemExit(main())
