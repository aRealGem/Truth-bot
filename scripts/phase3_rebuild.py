#!/usr/bin/env python3
"""Phase-3 rebuild runner (remediation v2, DC-5(b)) — guardrailed, resumable.

Per-speech rebuild under the current pipeline generation: take the EXISTING
claims from a published pca_runs artifact (claim identity preserved, so the
verdict diff and the corrections ledger stay meaningful), rebuild every
evidence pack FRESH under the v2.3 rules (S5 political tier + <=3 cap, role
axis, era fail-closed, fact-check exclusion), adjudicate with the production
PCA roster, and write a NEW metrics/pca_runs artifact compatible with
scripts/rerender_pca_site.py. Run five times; gwbush_2006 is the calibration
speech (smallest claim set).

Guardrail machinery is copied from scripts/rescue_gated_s5_p131.py:
  * proxy key gate + refuse --go without TRUTHBOT_R2_MODEL=gpt-5-mini
    (the economy config; the 2026-08-01 leg ran R2 on default gpt-5.5 and
    overspent ~2.5x — this guard makes that unrepeatable by accident);
  * --budget REQUIRED with --go; per-claim BudgetHalt raised inside the
    pack builder BEFORE retrieval (proxy_key_spend delta + off-proxy
    MODEL_RATES estimate vs the cap), plus between-chunk cap checks;
  * chunked adjudication (CHUNK_SIZE=5) with chunk-journal resume — a halt
    loses nothing, re-running the same command re-spends only on unbanked
    sids.

Usage (repo root):
  set -a; . ./.env; . ~/.env; set +a
  PYTHONPATH=.:src TRUTHBOT_R2_MODEL=gpt-5-mini \\
      .venv/bin/python scripts/phase3_rebuild.py --speech gwbush_2006 \\
      [--estimate] [--go --budget USD] [--limit N]

Default (no --go/--estimate): print the plan ($0) and exit.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Callable, Mapping, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

# ── speech registry ──────────────────────────────────────────────────────────
# Speaker strings + dates verified against site-pca/data/reports.json; run ids
# are the published artifact per speech (methodology_manifest.json).
SPEECHES: dict[str, dict] = {
    "trump_2026": {"speaker": "Donald Trump", "date": date(2026, 2, 24),
                   "run_id": "23939712-59ea-449d-93f7-a0a0b449efd8"},
    "biden_2022": {"speaker": "Joe Biden", "date": date(2022, 3, 1),
                   "run_id": "7208bbbb-c802-4155-932f-d0cc66803b24"},
    "obama_2014": {"speaker": "Barack Obama", "date": date(2014, 1, 28),
                   "run_id": "28965cdf-046e-4c87-a5d1-d21b6529c625"},
    "clinton_1998": {"speaker": "Bill Clinton", "date": date(1998, 1, 27),
                     "run_id": "7c59e9e0-0062-487d-84e3-4af15ab94aab"},
    "gwbush_2006": {"speaker": "George W. Bush", "date": date(2006, 1, 31),
                    "run_id": "92f39851-8870-4609-97f6-458798d5dbb8"},
}

# List prices, USD per Mtok, PER MODEL (litellm price map 2026-07-23) — the
# off-proxy estimator for R2 (OpenAI browsing) / R3 (grok) usage, identical to
# the rescue script. Unknown model → priced pessimistically.
MODEL_RATES = {
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5.5": (5.00, 30.00),
    "grok-4.3": (1.25, 2.50),
}
_DEFAULT_RATE = (5.00, 30.00)
CHUNK_SIZE = 5

# Per-claim cost projection for --estimate, derived from the 2026-08-01
# clinton_1998/gwbush_2006 full-stack actuals at the gpt-5-mini economy
# config: proxy chunk-journal actuals (R1 retrieval + panel) ran
# $0.366/48 claims (gwbush) and $0.936/92 (clinton) ≈ $0.008-0.010/claim,
# with the R2 off-proxy browsing estimate + grok rescue rounds carrying the
# rest → retrieval R1 proxy + R2 off-proxy + panel ≈ $0.065-0.08/claim.
PER_CLAIM_EST = (0.065, 0.08)

# Mirrors truthbot.verdict.consolidator.GATE_INSUFFICIENT (asserted equal in
# tests); kept literal here so the diff helpers stay import-light.
GATE_INSUFFICIENT = "insufficient-qualifying-evidence"

PIPELINE_GENERATION = "v2.3-role-axis-s5cap"
REMEDIATION_TAG = "phase-3 DC-5(b)"

PCA_RUNS_DIR = REPO / "metrics" / "pca_runs"
MANIFEST_PATH = PCA_RUNS_DIR / "methodology_manifest.json"
DIFF_DIR = REPO / "metrics" / "remediation_v2"


class BudgetHalt(RuntimeError):
    """Raised from inside the pack builder when the running estimate crosses
    the cap — the per-CLAIM circuit breaker (retrieval is where the money
    goes, so the cap is enforced BEFORE each claim's retrieval, not just per
    chunk)."""


# ── $0 helpers (import-safe: no proxy/key imports at module level) ───────────

def artifact_path(speech: str) -> Path:
    return PCA_RUNS_DIR / f"{SPEECHES[speech]['run_id']}.json"


def journal_paths(speech: str) -> tuple[Path, Path]:
    base = REPO / "metrics" / "journals"
    return base / f"{speech}_p3rebuild.jsonl", base / f"{speech}_p3rebuild_packs.jsonl"


def load_artifact(speech: str) -> dict:
    return json.loads(artifact_path(speech).read_text(encoding="utf-8"))


SHAPE_SIDECAR_SCHEMA = "truthbot-shape-backfill v1"


def load_sidecar_shapes(path: Path, speech: str, source_run: str) -> dict[str, str]:
    """Load a scripts/backfill_claim_shapes.py sidecar and return its non-empty
    ``{sid: shape}`` map. Fails loudly on a schema/speech/source-run mismatch —
    never silently apply another speech's (or another artifact revision's)
    shapes."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SHAPE_SIDECAR_SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r} != "
                         f"{SHAPE_SIDECAR_SCHEMA!r}")
    if doc.get("speech_id") != speech:
        raise ValueError(f"{path}: speech_id {doc.get('speech_id')!r} != "
                         f"{speech!r}")
    if source_run and doc.get("source_run") != source_run:
        raise ValueError(f"{path}: source_run {doc.get('source_run')!r} != "
                         f"{source_run!r} (sidecar built from a different "
                         "artifact)")
    return {sid: shape for sid, shape in (doc.get("shapes") or {}).items()
            if shape}


def merge_sidecar_shapes(claims: list[dict], shapes: Mapping[str, str]) -> int:
    """Fill ``layer_a.claim_shape`` from the sidecar for claims LACKING one —
    in memory only (the on-disk source artifact is never touched, and
    ``write_new_artifact`` re-reads the artifact, so the new artifact's
    claims stay verbatim). A shape already present in the artifact is NEVER
    overridden. Returns how many claims were filled."""
    n = 0
    for c in claims:
        la = c.get("layer_a") or {}
        if la.get("claim_shape"):
            continue                       # artifact shape wins, always
        shape = shapes.get(c.get("sid", ""))
        if shape:
            la["claim_shape"] = shape
            c["layer_a"] = la
            n += 1
    return n


def shape_refusal(n_shaped: int, n_claims: int,
                  legacy_ok: bool) -> Optional[str]:
    """--go guard for the one-methodology goal: a speech with shapeless
    claims (pre-role-axis artifact, or a partial sidecar) must not be
    rebuilt under the legacy quota by ACCIDENT. None = clear to run."""
    if legacy_ok or n_shaped >= n_claims:
        return None
    return (f"REFUSING --go: {n_claims - n_shaped}/{n_claims} claims have no "
            "claim shape, so this rebuild would run the LEGACY evidential-"
            "role quota — breaking one-methodology-corpus-wide. Backfill "
            "first (scripts/backfill_claim_shapes.py --speech <id> --go) and "
            "pass --shapes-sidecar PATH, or pass --legacy-quota-ok to run "
            "legacy DELIBERATELY. No spend attempted.")


def go_refusal(environ: Mapping[str, str], budget: Optional[float]) -> Optional[str]:
    """The two --go refusals, testable without argparse. None = clear to run."""
    if budget is None or budget <= 0:
        return ("REFUSING to spend: --budget USD is REQUIRED with --go (it is "
                "the halt cap for the per-claim breaker and the between-chunk "
                "checks). No spend attempted.")
    if environ.get("TRUTHBOT_R2_MODEL") != "gpt-5-mini":
        return ("REFUSING to spend: TRUTHBOT_R2_MODEL=gpt-5-mini is not set "
                "(the economy config). The 2026-08-01 leg ran R2 on default "
                "gpt-5.5 and overspent ~2.5x; this guard makes that "
                "impossible to repeat by accident.")
    return None


def outcome_label(row: dict) -> str:
    """Canonical bucket for a verdict-contract row. Gate-forced UV rows are
    EXPECTED outcomes (evidence_gate/provenance_code carries the T2.4 code),
    labeled apart from panel UNVERIFIABLE so the diff can count them."""
    gate = row.get("evidence_gate") or row.get("provenance_code") or ""
    if gate == GATE_INSUFFICIENT:
        return "gated-UNVERIFIABLE"
    verdict = row.get("verdict")
    if verdict is not None:
        return str(verdict)
    return "Models split" if row.get("split") else "No verdict"


def is_decided(label: str) -> bool:
    return label not in ("gated-UNVERIFIABLE", "Models split", "No verdict")


def classify_change(old_label: str, new_label: str) -> str:
    """Diff category for one sid (counts keys of the verdict diff)."""
    if old_label == new_label:
        return "unchanged"
    if new_label == "gated-UNVERIFIABLE":
        return "newly_gated"
    if is_decided(new_label) and not is_decided(old_label):
        return "newly_decided"
    if is_decided(new_label) and is_decided(old_label):
        return "decided_to_decided_changed"
    if new_label == "Models split" or old_label == "Models split":
        return "split_changes"
    return "other"


def build_verdict_diff(old_rows: list[dict], new_rows: list[dict],
                       claims: Optional[list[dict]] = None) -> dict:
    """Per-sid old→new labels + category counts, over the sids present in
    ``new_rows`` (partial-run safe). Gate-forced UV rows are counted, not
    treated as errors (failure honesty)."""
    old_by_sid = {r.get("sid"): r for r in old_rows}
    text_by_sid = {c.get("sid"): (c.get("text") or "") for c in (claims or [])}
    per_sid: list[dict] = []
    counts = {"unchanged": 0, "decided_to_decided_changed": 0,
              "newly_gated": 0, "newly_decided": 0, "split_changes": 0,
              "other": 0}
    n_gated_new = 0
    for row in new_rows:
        sid = row.get("sid")
        old = old_by_sid.get(sid)
        old_label = outcome_label(old) if old else "(no old row)"
        new_label = outcome_label(row)
        n_gated_new += new_label == "gated-UNVERIFIABLE"
        cat = (classify_change(old_label, new_label) if old else "other")
        counts[cat] += 1
        per_sid.append({"sid": sid, "old": old_label, "new": new_label,
                        "category": cat,
                        "text": text_by_sid.get(sid, "")[:120]})
    return {"n_compared": len(per_sid), "counts": counts,
            "gate_forced_new": n_gated_new, "per_sid": per_sid}


def pending_claims(claims: list[dict], done_rows: list[dict]) -> list[dict]:
    """Resume filter: claims whose sid is already banked in the chunk journal
    are never re-run (never re-spent on)."""
    done_sids = {r.get("sid") for r in done_rows}
    return [c for c in claims if c["sid"] not in done_sids]


def make_pack_builder(*, build_pack: Callable[[str, str, str], object],
                      cap: float, start_spend: float,
                      offproxy_est: Callable[[], float] = lambda: 0.0,
                      banked_cost: float = 0.0,
                      packs_journal: Optional[Path] = None):
    """Wrap a raw pack builder with the per-claim budget breaker + Phase-R
    packs journaling. The breaker fires BEFORE retrieval: on-proxy delta
    (proxy_key_spend ledger) + off-proxy estimate + banked prior-session cost
    >= cap → BudgetHalt, and no retriever runs for that sid."""
    def pack_builder(sid: str, text: str, context: str):
        from truthbot.verdict import proxy_lane, publish_pipeline
        spent = ((proxy_lane.proxy_key_spend() - start_spend)
                 + offproxy_est() + banked_cost)
        if spent >= cap:
            raise BudgetHalt(f"${spent:.2f} >= cap ${cap:.2f} "
                             f"(before retrieving {sid})")
        pack = build_pack(sid, text, context)
        if packs_journal is not None:
            publish_pipeline.append_packs_journal(packs_journal, sid, pack)
        return pack
    return pack_builder


def write_new_artifact(source_art: dict, new_rows: list[dict], packs: dict,
                       roster_note: dict, *, speech_id: str,
                       run_id: Optional[str] = None,
                       out_dir: Optional[Path] = None,
                       cost_usd: float = 0.0) -> tuple[Path, dict]:
    """Write the rebuild's pca_runs artifact — same shape `_persist_pca_run`
    emits and `rerender_pca_site.py` consumes: {run_id, meta, claims, rows,
    characterization, roster, evidence, composition}. claims (and the Layer A
    characterization, which does not depend on verdicts) carry over VERBATIM
    from the source artifact; rows/evidence/roster are the rebuild's; meta
    records the lineage (rebuild_of + pipeline generation)."""
    import uuid

    from truthbot.verdict import bridge as bridge_mod

    run_id = run_id or str(uuid.uuid4())
    out_dir = Path(out_dir) if out_dir is not None else PCA_RUNS_DIR
    old_meta = source_art.get("meta") or {}
    # Stable claim order in rows (matches the source claims[] order).
    rows_by_sid = {r.get("sid"): r for r in new_rows}
    rows = [rows_by_sid[c["sid"]] for c in source_art["claims"]
            if c["sid"] in rows_by_sid]
    evidence = {sid: [ev.model_dump(mode="json")
                      for ev in bridge_mod._pack_to_evidence(sid, pack)]
                for sid, pack in (packs or {}).items()}
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
        "remediation": REMEDIATION_TAG,
    }
    payload = {
        "run_id": run_id,
        "meta": meta,
        "claims": list(source_art["claims"]),          # verbatim — identity
        "rows": rows,
        "characterization": list(source_art.get("characterization") or []),
        "roster": roster_note,
        "evidence": evidence,
    }
    try:  # composition-bias telemetry, same best-effort as _persist_pca_run
        from truthbot.verdict.composition_telemetry import composition_report
        payload["composition"] = composition_report(rows, evidence)
    except Exception:
        pass
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload, default=str, ensure_ascii=False),
                    encoding="utf-8")
    return path, payload


def update_manifest(run_id: str, speech_id: str,
                    manifest_path: Optional[Path] = None) -> None:
    """Add the rebuild to the methodology manifest at the current generation,
    published=false. Never touches the old run's row — un-publishing (site
    cutover) is a separate, human-gated step."""
    p = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH
    manifest = json.loads(p.read_text(encoding="utf-8"))
    manifest["runs"][run_id] = {"speech_id": speech_id,
                                "generation": PIPELINE_GENERATION,
                                "published": False}
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                 encoding="utf-8")


def estimate_report(speech_ids: list[str]) -> str:
    """$0 cost projection: claims × the 2026-08-01 per-claim actuals."""
    lo_rate, hi_rate = PER_CLAIM_EST
    lines = [
        "Phase-3 rebuild cost projection ($0 — constants only, no calls):",
        f"  per-claim ${lo_rate:.3f}-{hi_rate:.3f} (2026-08-01 clinton/gwbush "
        "actuals @ gpt-5-mini economy config: retrieval R1 proxy + R2 "
        "off-proxy + panel)",
    ]
    total_n = 0
    for sp in speech_ids:
        n = len(load_artifact(sp)["claims"])
        total_n += n
        lines.append(f"  {sp:<13} {n:>3} claims  ->  "
                     f"${n * lo_rate:.2f} - ${n * hi_rate:.2f}")
    if len(speech_ids) > 1:
        lines.append(f"  {'TOTAL':<13} {total_n:>3} claims  ->  "
                     f"${total_n * lo_rate:.2f} - ${total_n * hi_rate:.2f}")
    return "\n".join(lines)


def print_diff(diff: dict, *, partial: bool = False) -> None:
    tag = "PARTIAL VERDICT DIFF" if partial else "VERDICT DIFF"
    c = diff["counts"]
    print(f"\n{tag} (old -> new, {diff['n_compared']} sid(s)):")
    for e in diff["per_sid"]:
        if e["category"] != "unchanged":
            print(f"  [{e['sid']}] {e['old']} -> {e['new']}  ({e['category']})")
    print(f"  unchanged {c['unchanged']} · decided->decided changed "
          f"{c['decided_to_decided_changed']} · newly-gated {c['newly_gated']}"
          f" · newly-decided {c['newly_decided']} · split changes "
          f"{c['split_changes']} · other {c['other']}")
    print(f"  gate-forced UV in new rows: {diff['gate_forced_new']} "
          "(expected outcomes of the T2.4 gate, not errors)")


# ── the funded path ──────────────────────────────────────────────────────────

def run_rebuild(args) -> None:
    import os

    from hydramind.rosters import get_roster
    from truthbot.verdict import (adjudicator, proxy_lane, publish_pipeline,
                                  shape_registry)
    from truthbot.verdict.evidence_pack_v2 import build_evidence_pack_v2
    from truthbot.verify import retrievers as R
    from truthbot.verify.principals import principal_relation

    if not proxy_lane.key_present():
        sys.exit(proxy_lane.BLOCKED_MSG)
    refusal = go_refusal(os.environ, args.budget)
    if refusal:
        sys.exit(refusal)

    speech = args.speech
    spec = SPEECHES[speech]
    art = load_artifact(speech)
    claims = [{"sid": c["sid"], "text": c["text"],
               "context": c.get("context", "")} for c in art["claims"]]
    if args.limit:
        claims = claims[:args.limit]
    chunk_journal, packs_journal = journal_paths(speech)

    # Resume: sids already banked in the chunk journal are never re-run.
    done_rows, done_packs, banked_cost, _ = \
        publish_pipeline.load_chunk_journal(chunk_journal)
    todo = pending_claims(claims, done_rows)
    if done_rows:
        print(f"resume: {len(done_rows)} sids banked "
              f"(${banked_cost:.4f} prior proxy spend), {len(todo)} to run")

    # Off-proxy estimation (R2/R3 usage at list price) — rescue-script pattern.
    usage: dict[str, list] = {"R2": [], "R3": []}

    class MeteredR2(R.OpenAIBrowsingRetriever):
        def _post(self, model, prompt):
            doc = super()._post(model, prompt)
            usage["R2"].append({"model": model, "usage": doc.get("usage") or {}})
            return doc

    class MeteredR3(R.GrokSearchRetriever):
        def _post(self, model, prompt, tool):
            doc = super()._post(model, prompt, tool)
            usage["R3"].append({"model": model, "usage": doc.get("usage") or {}})
            return doc

    def _offproxy_est() -> float:
        total = 0.0
        for entries in usage.values():
            for e in entries:
                rates = MODEL_RATES.get(str(e.get("model") or ""), _DEFAULT_RATE)
                u = e["usage"]
                tin = int(u.get("input_tokens") or u.get("prompt_tokens") or 0)
                tout = int(u.get("output_tokens") or u.get("completion_tokens") or 0)
                total += (tin * rates[0] + tout * rates[1]) / 1e6
        return total

    # Economy config (same as the rescue script / the 2026-08-01 full-stack
    # runs): R1+R2 primary, grok joins only the T2.4 rescue round.
    primary = (R.ClaudeWorkerRetriever(), MeteredR2())
    retry = primary + (MeteredR3(model="grok-4.3"),)

    # Role-axis wiring, exactly like the CLI's _build_v2_pack_builder
    # (PR-A2.3/A2.5): the principal relation closes over (speaker, utterance)
    # and each claim's Layer A shape comes from the shape registry (registered
    # from the artifact claims in main()).
    speaker, utterance = spec["speaker"], spec["date"]

    def relation_of(ev):
        return principal_relation(ev.source_url, speaker, utterance)

    def build_pack(sid: str, text: str, context: str):
        return build_evidence_pack_v2(
            sid, text, primary, retry_retrievers=retry, context=context,
            claim_shape=shape_registry.shape_for(sid), relation_of=relation_of)

    start_spend = proxy_lane.proxy_key_spend()
    pack_builder = make_pack_builder(
        build_pack=build_pack, cap=args.budget, start_spend=start_spend,
        offproxy_est=_offproxy_est, banked_cost=banked_cost,
        packs_journal=packs_journal)

    hm = proxy_lane.build_hydramind(response_parser=adjudicator.parse_verdict)
    roster_note = {"name": "prod", "seats": dict(get_roster("prod").seats)}
    print(f"proxy key spend at start: ${start_spend:.4f} "
          f"(budget ${args.budget:.2f} incl. off-proxy est + banked)")

    chunks = [todo[i:i + CHUNK_SIZE] for i in range(0, len(todo), CHUNK_SIZE)]
    all_rows = list(done_rows)
    all_packs = dict(done_packs)
    halted = ""
    for idx, chunk in enumerate(chunks, 1):
        total_so_far = ((proxy_lane.proxy_key_spend() - start_spend)
                        + _offproxy_est() + banked_cost)
        if total_so_far >= args.budget:
            halted = (f"BUDGET HALT before chunk {idx}: "
                      f"${total_so_far:.2f} >= cap ${args.budget:.2f}")
            print(halted)
            break
        t0, s0 = time.time(), proxy_lane.proxy_key_spend()
        try:
            rows, manifest, notes = adjudicator.adjudicate(
                hm, chunk, roster="prod", pack_builder=pack_builder,
                two_stage=True)
        except BudgetHalt as exc:
            halted = f"BUDGET HALT mid-chunk {idx}: {exc}"
            print(halted)
            break
        s1, t1 = proxy_lane.proxy_key_spend(), time.time()
        chunk_packs = notes.get("packs") or {}
        publish_pipeline.append_chunk_journal(
            chunk_journal, idx, rows, chunk_packs, s1 - s0,
            roster=roster_note if not done_rows and idx == 1 else None)
        all_rows.extend(rows)
        all_packs.update(chunk_packs)
        print(f"chunk {idx}/{len(chunks)} ({len(chunk)} claims): "
              f"proxy ${s1 - s0:.4f}, off-proxy est ${_offproxy_est():.4f}, "
              f"{t1 - t0:.0f}s")

    proxy_total = proxy_lane.proxy_key_spend() - start_spend
    off_total = _offproxy_est()
    run_cost = proxy_total + off_total + banked_cost
    print(f"\nSPEND: proxy ${proxy_total:.4f} + off-proxy est ${off_total:.4f}"
          f" + banked ${banked_cost:.4f} = ${run_cost:.4f} "
          f"(cap ${args.budget:.2f})")

    # The artifact is written ONLY when every claim of the speech completed —
    # partial runs (halt, --limit slice) stay in the journals.
    full_sids = {c["sid"] for c in art["claims"]}
    have_sids = {r.get("sid") for r in all_rows}
    complete = full_sids <= have_sids and not halted
    diff = build_verdict_diff(art["rows"], all_rows, art["claims"])
    if not complete:
        print_diff(diff, partial=True)
        missing = len(full_sids - have_sids)
        print(f"\nINCOMPLETE ({missing} of {len(full_sids)} claims not yet "
              "banked) — no artifact written. Everything completed is "
              "journaled:")
        print(f"  chunk journal: {chunk_journal}")
        print(f"  packs journal: {packs_journal}")
        print("Resume (re-spends only on unbanked sids):")
        print(f"  TRUTHBOT_R2_MODEL=gpt-5-mini PYTHONPATH=.:src "
              f".venv/bin/python scripts/phase3_rebuild.py --speech {speech} "
              f"--go --budget <USD>")
        return

    out_path, payload = write_new_artifact(
        art, all_rows, all_packs, roster_note, speech_id=speech,
        cost_usd=run_cost)
    update_manifest(payload["run_id"], speech)
    print(f"\nartifact written: {out_path}")
    print(f"manifest updated: {MANIFEST_PATH} (+{payload['run_id']} @ "
          f"{PIPELINE_GENERATION}, published=false — publishing is a "
          "separate human-gated step)")

    print_diff(diff)
    DIFF_DIR.mkdir(parents=True, exist_ok=True)
    diff_path = DIFF_DIR / f"phase3_{speech}_verdict_diff.json"
    diff_out = {"speech_id": speech, "rebuild_of": art.get("run_id", ""),
                "new_run_id": payload["run_id"],
                "old_tally": _tally(art["rows"]), "new_tally": _tally(all_rows),
                **diff}
    diff_path.write_text(json.dumps(diff_out, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"diff written: {diff_path}")


def _tally(rows: list[dict]) -> dict:
    from truthbot.verdict.publish_pipeline import verdict_bucket_tally
    return verdict_bucket_tally(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--speech", choices=sorted(SPEECHES),
                    help="speech to rebuild (required except for a "
                         "five-speech --estimate)")
    ap.add_argument("--go", action="store_true",
                    help="actually spend (else plan/estimate only, $0)")
    ap.add_argument("--budget", type=float, default=None,
                    help="halt cap in USD — REQUIRED with --go "
                         "(on-proxy delta + off-proxy est + banked)")
    ap.add_argument("--estimate", action="store_true",
                    help="$0 cost projection (all five speeches when "
                         "--speech is omitted) and exit")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N claims only (smoke slice; artifact is NOT "
                         "written until the full speech is banked)")
    ap.add_argument("--shapes-sidecar", default=None, metavar="PATH",
                    help="shapes_backfill_<speech>.json from "
                         "scripts/backfill_claim_shapes.py — fills claim "
                         "shapes for claims LACKING one (never overrides an "
                         "artifact shape)")
    ap.add_argument("--legacy-quota-ok", action="store_true",
                    help="allow --go on a speech with shapeless claims "
                         "(legacy evidential-role quota) — deliberate only")
    args = ap.parse_args()

    if args.estimate:
        speeches = [args.speech] if args.speech else list(SPEECHES)
        print(estimate_report(speeches))
        return
    if not args.speech:
        ap.error("--speech is required (or use --estimate for the projection)")

    spec = SPEECHES[args.speech]
    # Era gate fails closed now — register the utterance date at startup so
    # temporal grounding resolves for every rebuilt pack.
    from truthbot.verdict import shape_registry, speech_context
    speech_context.register_speech_date(args.speech, spec["date"])

    art = load_artifact(args.speech)
    n_from_sidecar = 0
    if args.shapes_sidecar:
        sidecar_shapes = load_sidecar_shapes(
            Path(args.shapes_sidecar), args.speech, art.get("run_id", ""))
        n_from_sidecar = merge_sidecar_shapes(art["claims"], sidecar_shapes)
    n_shapes = shape_registry.register_claim_shapes(art["claims"])
    chunk_journal, packs_journal = journal_paths(args.speech)
    from truthbot.verdict import publish_pipeline
    done_rows, _, banked_cost, _ = \
        publish_pipeline.load_chunk_journal(chunk_journal)

    n_claims = len(art["claims"])
    print(f"Phase-3 rebuild plan — {args.speech} "
          f"({spec['speaker']}, {spec['date'].isoformat()})")
    print(f"  source artifact: {artifact_path(args.speech)} "
          f"(run {art.get('run_id', '?')[:8]})")
    print(f"  claims: {n_claims} (identity preserved verbatim)"
          + (f"; --limit slice: first {args.limit}" if args.limit else ""))
    print(f"  claim shapes registered: {n_shapes}/{n_claims} "
          f"({n_from_sidecar} from sidecar)"
          + ("" if n_shapes else " (pre-role-axis artifact — legacy quota; "
             "relation_of still applies)"))
    print(f"  chunk journal: {chunk_journal}")
    print(f"  packs journal: {packs_journal}")
    print(f"  resume state: {len(done_rows)}/{n_claims} rows banked"
          + (f" (${banked_cost:.4f} prior proxy spend)" if done_rows else ""))
    print(f"  old verdict tally: {_tally(art['rows'])}")
    print(f"  old gate-forced UV: "
          f"{sum(1 for r in art['rows'] if outcome_label(r) == 'gated-UNVERIFIABLE')}")

    if not args.go:
        print("\n($0 plan only — add --estimate for the cost projection, or "
              "--go --budget USD to spend)")
        return
    refusal = shape_refusal(n_shapes, n_claims, args.legacy_quota_ok)
    if refusal:
        sys.exit(refusal)
    run_rebuild(args)


if __name__ == "__main__":
    main()
