#!/usr/bin/env python3
"""Re-score STORED pack items (remediation v2, B1a) — resumable, budget-capped.

The v2 evidence path never called ``verify.relevance.score_evidence``, so all
4,344 items in the five rebuilt runs carry the 0.5 relevance default and 20.5-
30.2%% carry a null stance. ``consolidator._bearing()`` needs True/False, so
those nulls cannot credit ``MIN_BEARING_T13=2`` and the T2.4 gate forces
Unverifiable on packs that hold good evidence. B1b fixes that going FORWARD (the
``scorer`` hook in ``build_evidence_pack_v2``); this script is the one-off that
fixes what is ALREADY on disk — without re-retrieving anything, so it pays only
for the cheap scoring call and never again for search.

Never mutates the stored artifact. Results land in a SIDECAR
(metrics/remediation_v2/rescored_<speech>.json) because artifacts are the
record: archive-never-delete. A later step joins the sidecar to decide which
claims actually flip the gate and therefore deserve re-adjudication.

Guardrails are modeled on scripts/phase3_rebuild.py:
  * refuses to spend without an explicit --go, and --budget USD is REQUIRED;
  * per-CLAIM budget breaker checked BEFORE each scoring call, plus a
    between-batch check;
  * the sidecar is rewritten after EVERY sid, so a crash or halt resumes
    without re-spending on anything already scored;
  * a sid that comes back unscored is retried with phase-3's backoff, then left
    UNBANKED (so a resume retries it rather than recording coverage that was
    never obtained), and a run of them halts CLEANLY with resume instructions —
    no traceback. Note ``score_evidence`` fails soft and never raises, so the
    failure signal is "nothing changed", not an exception; see ``_score_one``.

The lane is the cheap one on purpose: ``score_evidence`` runs through
``relevance.build_proxy_llm()`` = Haiku over the LiteLLM proxy. That is
ON-PROXY, so ``proxy_lane.proxy_key_spend()`` is ledger truth and the breaker
needs no off-proxy estimate (unlike phase3_rebuild, which had to estimate R2/R3).

Usage (repo root):
  # $0 — price the job from the actual stored payloads:
  PYTHONPATH=.:src .venv/bin/python scripts/rescore_stored_packs.py --estimate
  # $0 — plan one speech:
  PYTHONPATH=.:src .venv/bin/python scripts/rescore_stored_packs.py --speech trump_2026
  # SPENDS MONEY (DC-B1-gated):
  set -a; . ./.env; . ~/.env; set +a
  PYTHONPATH=.:src .venv/bin/python scripts/rescore_stored_packs.py \\
      --speech trump_2026 --go --budget 2.00

Default (no --go/--estimate): print the plan ($0) and exit.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Retry count + backoff are phase3_rebuild's, reused rather than re-derived —
# same lane, same blips, same operational contract. (scripts/ is not a package,
# so this is a path-based import, exactly how the phase-3 tests load it.)
# Its ``_is_transient`` classifier is deliberately NOT used: see ``_score_one``.
from phase3_rebuild import CHUNK_BACKOFF_S, CHUNK_RETRIES  # noqa: E402

# ── the rebuilt runs (remediation v2 phase 3) ────────────────────────────────
# These are the REBUILT artifacts, not the originally-published ones that
# phase3_rebuild.SPEECHES points at: the rebuilds are what the 4,344-item
# census and the DC-B1 estimate are computed over.
REBUILT_RUNS: dict[str, str] = {
    "gwbush_2006":  "74a89c5f-54c4-47dd-a6eb-624c69fcdd4b",
    "clinton_1998": "d0010426-2e8f-4449-b839-60f85d923d56",
    "obama_2014":   "4de8a551-ea99-440d-aca0-12a133e620e3",
    "biden_2022":   "37744fc8-41f1-4375-9f8a-01ca17d0327f",
    "trump_2026":   "4ee5a251-9b3c-49be-9283-0ac062ac2c10",
}

PCA_RUNS_DIR = REPO / "metrics" / "pca_runs"
SIDECAR_DIR = REPO / "metrics" / "remediation_v2"
SIDECAR_SCHEMA = "truthbot-rescore-sidecar v1"

DEFAULT_MODEL = "claude-haiku"
#: sids per progress batch — the unit for the between-batch cap check and the
#: per-batch spend print. Each sid is its own call; batching is reporting only.
BATCH_SIZE = 10

#: Consecutive soft failures (a sid coming back unscored after its retries)
#: that mean the LANE is down rather than one claim being odd. Halting there
#: stops a dead proxy from grinding through every remaining sid.
SOFT_FAIL_HALT = 3

#: Cost constants are NOT defined here any more. This script's own $0 estimate
#: came in 2.42x low on B1a (est $0.4391, actual $1.0632) because it carried a
#: private chars/4 constant and a private guess at the reply shape;
#: b2_primary_series imported those guesses and shipped the same miss again at
#: 2.35x. They now live ONCE, in ``truthbot.costs``, fitted to both runs'
#: ledger actuals — so a recalibration moves every estimator at the same time.
#: (Imported lazily inside estimate_speech, like the rest of the $0 path, so
#: this module still imports clean with nothing configured.)


# ── $0 helpers (no proxy import at module level — this file must import clean
#    with no key present, so the estimator can never touch a lane) ────────────

def artifact_path(speech: str) -> Path:
    return PCA_RUNS_DIR / f"{REBUILT_RUNS[speech]}.json"


def sidecar_path(speech: str) -> Path:
    return SIDECAR_DIR / f"rescored_{speech}.json"


def b2_sidecar_path(speech: str) -> Path:
    """The B2 re-score lands in its OWN sidecar, beside B1a's.

    Deliberately not a merge-in-place. ``score_evidence`` rewrites every item in
    a pack, so writing B2 into the B1a file would silently replace B1a's scores
    for items B2 was not targeting — and there would be no way afterwards to
    tell which vintage any given row came from. Two files, merged downstream in
    a defined order (B1a first, B2 on top), keeps both records intact and keeps
    the two spends separately attributable."""
    return SIDECAR_DIR / f"rescored_b2_{speech}.json"


def load_artifact(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def claim_texts(artifact: dict) -> dict[str, str]:
    """sid → claim text, from the artifact's claims[]. Scoring is only as good
    as the claim it scores against, so a sid with no claim text is SKIPPED
    rather than scored against an empty string."""
    return {c.get("sid", ""): (c.get("text") or "").strip()
            for c in (artifact.get("claims") or [])}


def load_sidecar(path: Path, speech: str, source_run: str) -> dict:
    """Load an existing sidecar for resume, or start a fresh one. Fails LOUD on
    a schema/speech/source-run mismatch — never merge another speech's scores,
    or scores taken against a different artifact revision, into this one."""
    p = Path(path)
    if not p.exists():
        return {"schema": SIDECAR_SCHEMA, "speech_id": speech,
                "source_run": source_run, "model": "", "generated": "",
                "spend_usd": 0.0, "sids": {}, "soft_failures": []}
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("schema") != SIDECAR_SCHEMA:
        raise ValueError(f"{p}: schema {doc.get('schema')!r} != {SIDECAR_SCHEMA!r}")
    if doc.get("speech_id") != speech:
        raise ValueError(f"{p}: speech_id {doc.get('speech_id')!r} != {speech!r}")
    if source_run and doc.get("source_run") != source_run:
        raise ValueError(
            f"{p}: source_run {doc.get('source_run')!r} != {source_run!r} — this "
            "sidecar was scored against a different artifact revision. Move it "
            "aside; merging it would attach scores to the wrong evidence.")
    doc.setdefault("sids", {})
    doc.setdefault("soft_failures", [])
    doc.setdefault("spend_usd", 0.0)
    return doc


def write_sidecar(path: Path, doc: dict) -> None:
    """Atomic rewrite (tmp + replace) after every sid — a crash mid-write must
    not leave a truncated sidecar that loses everything already paid for."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(p)


def scored_rows(evidence: list) -> list[dict]:
    """The persisted shape: identity (source_url) + what scoring produced.
    source_url is the join key back onto the artifact — stable, and it is what
    the pack dedup already keys on.

    ``one_line_why`` and ``arithmetic_hinge`` (the B2 contract) are written
    ONLY when the scorer actually produced them, so a B1a-vintage sidecar row
    and a B2-vintage row stay visibly different on disk. Readers ignore keys
    they do not know, which is what lets both vintages share one schema."""
    rows = []
    for ev in evidence:
        row = {"source_url": ev.source_url,
               "relevance_score": ev.relevance_score,
               "supports_claim": ev.supports_claim}
        if getattr(ev, "one_line_why", None):
            row["one_line_why"] = ev.one_line_why
        if getattr(ev, "arithmetic_hinge", False):
            row["arithmetic_hinge"] = True
        rows.append(row)
    return rows


def load_only_sids(path: Optional[str]) -> Optional[set]:
    """Read a targeting list (a JSON array of sids) written by
    ``scripts/b2_primary_series.py --write-sids``. None = no restriction."""
    if not path:
        return None
    sids = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(sids, list) or not all(isinstance(s, str) for s in sids):
        raise ValueError(f"{path}: expected a JSON array of sid strings")
    return set(sids)


def pending_sids(artifact: dict, sidecar: dict, texts: dict[str, str],
                 only: Optional[set] = None) -> list[str]:
    """Resume filter: a sid already in the sidecar is never re-scored (never
    re-spent on). Sids with no evidence or no claim text are skipped outright.

    ``only`` narrows to a derived subset (B2). It is applied ON TOP of the
    resume filter, never instead of it, so a targeted re-run still cannot pay
    twice for the same sid in the same sidecar."""
    done = set(sidecar.get("sids") or {})
    return [sid for sid, evs in (artifact.get("evidence") or {}).items()
            if evs and sid not in done and texts.get(sid)
            and (only is None or sid in only)]


def go_refusal(budget: Optional[float]) -> Optional[str]:
    """The --go refusal, testable without argparse. None = clear to run."""
    if budget is None or budget <= 0:
        return ("REFUSING to spend: --budget USD is REQUIRED with --go (it is "
                "the halt cap for the per-claim breaker and the between-batch "
                "checks). No spend attempted.")
    return None


# ── $0 estimator ─────────────────────────────────────────────────────────────

def estimate_speech(artifact: dict, *, model: str = DEFAULT_MODEL,
                    freetext: bool = True) -> dict:
    """Price one speech's re-score from its ACTUAL stored payloads.

    Builds the exact prompt ``score_evidence`` would send for every sid
    (``relevance.score_payload`` — the same function the funded path calls), so
    the INPUT volume is measured, not guessed. Sends nothing.

    The OUTPUT volume is where this estimator was wrong twice, and it cannot be
    measured the same way: the reply does not exist yet, and the model's own
    formatting (which is billed) is not ours to predict. It is therefore priced
    from ``truthbot.costs``, whose per-item and per-free-text-character loads
    are back-solved from what B1a and B2 actually cost.

    ``freetext=True`` (the default) prices the CURRENT scorer contract, which
    asks for ``one_line_why`` and so pays for prose on every item. Pass False
    only to price the pre-B2 three-key reply."""
    from truthbot import costs
    from truthbot.verdict.publish_pipeline import evidence_from_artifact_dict
    from truthbot.verify.relevance import _SCORE_SYSTEM, score_payload

    texts = claim_texts(artifact)
    by_sid = evidence_from_artifact_dict(artifact.get("evidence") or {})
    in_chars = 0
    n_calls = n_items = 0
    skipped: list[str] = []
    for sid, evs in by_sid.items():
        if not evs:
            continue
        text = texts.get(sid, "")
        if not text:
            skipped.append(sid)
            continue
        in_chars += len(_SCORE_SYSTEM) + len(score_payload(text, evs))
        n_calls += 1
        n_items += len(evs)

    est = costs.estimate_scoring_cost(
        prompt_chars=in_chars, items=n_items, model=model,
        freetext_chars=None if freetext else 0)
    est.update(calls=n_calls, skipped_no_claim_text=skipped)
    return est


def estimate_report(speeches: list[str], *, model: str = DEFAULT_MODEL,
                    freetext: bool = True) -> tuple[str, dict]:
    """Per-speech + total projection. Returns (printable, machine-readable)."""
    from truthbot import costs

    rows = {sp: estimate_speech(load_artifact(artifact_path(sp)), model=model,
                                freetext=freetext)
            for sp in speeches}
    tot_calls = sum(r["calls"] for r in rows.values())
    tot_items = sum(r["items"] for r in rows.values())
    tot_cost = sum(r["cost_usd_est"] for r in rows.values())
    tok_in = sum(r["tokens_in_est"] for r in rows.values())
    tok_out = sum(r["tokens_out_est"] for r in rows.values())

    any_row = next(iter(rows.values()))
    lines = [
        f"B1a re-score cost estimate — model {model}, LiteLLM proxy (Haiku lane)",
        "$0: measured from the STORED payloads, nothing sent.",
        "  method: the exact score_evidence prompt per sid "
        f"(relevance.score_payload), converted at {costs.CHARS_PER_TOKEN} "
        f"chars/token; the reply at {costs.REPLY_TOKENS_PER_ITEM} output "
        f"tokens/item" + (f" plus {costs.FREETEXT_CHARS_PER_ITEM} chars/item of "
                          "one_line_why free text" if freetext else
                          " (pre-B2 reply, no free text)") + "; priced at "
        f"{any_row['rate_in_usd_per_mtok']}/{any_row['rate_out_usd_per_mtok']} "
        f"USD per Mtok in/out. Calibration {costs.CALIBRATION_ID} "
        "(truthbot.costs)",
        "",
        f"  {'speech':<14}{'calls':>6}{'items':>7}{'tok_in':>10}{'tok_out':>9}{'est USD':>10}",
    ]
    for sp in speeches:
        r = rows[sp]
        lines.append(f"  {sp:<14}{r['calls']:>6}{r['items']:>7}"
                     f"{r['tokens_in_est']:>10}{r['tokens_out_est']:>9}"
                     f"{r['cost_usd_est']:>10.4f}")
    lines.append(f"  {'TOTAL':<14}{tot_calls:>6}{tot_items:>7}"
                 f"{tok_in:>10}{tok_out:>9}{tot_cost:>10.4f}")
    skipped = {sp: r["skipped_no_claim_text"] for sp, r in rows.items()
               if r["skipped_no_claim_text"]}
    if skipped:
        lines.append(f"  (skipped, no claim text: {skipped})")
    lines += ["", "  " + costs.uncertainty_note(model=model)]
    summary = {"model": model,
               "calibration_id": costs.CALIBRATION_ID,
               "chars_per_token": costs.CHARS_PER_TOKEN,
               "reply_tokens_per_item": costs.REPLY_TOKENS_PER_ITEM,
               "freetext_priced": bool(freetext),
               "per_speech": rows, "total_calls": tot_calls,
               "total_items": tot_items, "total_cost_usd_est": round(tot_cost, 4)}
    return "\n".join(lines), summary


# ── the funded path ──────────────────────────────────────────────────────────

def _score_one(llm, text: str, evs: list) -> bool:
    """Score one sid's items in place, with bounded retries. True iff it worked.

    The retry is driven by the RESULT, not by an exception, and that is not a
    stylistic choice: ``score_evidence`` catches every exception internally and
    returns, leaving the neutral defaults ("fails SOFT ... so a scoring hiccup
    degrades to the old tier-only ranking"). Nothing ever propagates out of it,
    so ``phase3_rebuild._is_transient`` has nothing to classify here — an
    exception-based retry around it would be dead code. What a failure actually
    looks like from outside is "the items came back unchanged", so that is what
    is retried, with phase-3's retry count and backoff.

    A pack that never changes despite room to change is reported as a soft
    failure by the caller and deliberately NOT banked, so a resume retries it
    instead of recording coverage that was never obtained."""
    from truthbot.verdict.consolidator import DEFAULT_RELEVANCE_SCORE
    from truthbot.verify.relevance import score_evidence

    before = [(ev.relevance_score, ev.supports_claim) for ev in evs]
    # Nothing to detect if every item is already scored — accept and move on.
    if not any(r is None or r == DEFAULT_RELEVANCE_SCORE for r, _ in before):
        score_evidence(llm, text, evs)
        return True

    for attempt in range(1, CHUNK_RETRIES + 1):
        score_evidence(llm, text, evs)
        if [(ev.relevance_score, ev.supports_claim) for ev in evs] != before:
            return True
        if attempt == CHUNK_RETRIES:
            break
        wait = CHUNK_BACKOFF_S[min(attempt - 1, len(CHUNK_BACKOFF_S) - 1)]
        print(f"  {text[:40]!r}: scoring returned nothing usable — attempt "
              f"{attempt}/{CHUNK_RETRIES}, retrying in {wait}s", flush=True)
        time.sleep(wait)
    return False


def run_rescore(args) -> int:
    from truthbot.verdict import proxy_lane
    from truthbot.verdict.publish_pipeline import evidence_from_artifact_dict
    from truthbot.verify.relevance import build_proxy_llm

    if not proxy_lane.key_present():
        print(proxy_lane.BLOCKED_MSG)
        return 1
    refusal = go_refusal(args.budget)
    if refusal:
        print(refusal)
        return 1

    speech = args.speech
    art_path = Path(args.artifact) if args.artifact else artifact_path(speech)
    art = load_artifact(art_path)
    source_run = art.get("run_id", "")
    out_path = Path(args.out) if args.out else sidecar_path(speech)
    sidecar = load_sidecar(out_path, speech, source_run)

    texts = claim_texts(art)
    by_sid = evidence_from_artifact_dict(art.get("evidence") or {})
    only = load_only_sids(getattr(args, "only_sids", None))
    todo = pending_sids(art, sidecar, texts, only)
    if args.limit:
        todo = todo[:args.limit]
    banked = float(sidecar.get("spend_usd") or 0.0)
    if sidecar["sids"]:
        print(f"resume: {len(sidecar['sids'])} sids already scored "
              f"(${banked:.4f} prior proxy spend), {len(todo)} to go")
    if not todo:
        print("nothing to do — every sid is already scored. No spend attempted.")
        return 0

    llm = build_proxy_llm(args.model)
    if llm is None:
        print("REFUSING: no LiteLLM proxy key — the scoring lane is "
              "unavailable. No spend attempted.")
        return 1

    start_spend = proxy_lane.proxy_key_spend()
    sidecar["model"] = args.model
    print(f"proxy key spend at start: ${start_spend:.4f} "
          f"(cap ${args.budget:.2f}, incl. ${banked:.4f} banked)")

    def spent() -> float:
        return (proxy_lane.proxy_key_spend() - start_spend) + banked

    halted = ""
    n_done = 0
    consecutive_failures = 0
    batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    for bi, batch in enumerate(batches, 1):
        b0 = spent()
        if b0 >= args.budget:
            halted = f"BUDGET HALT before batch {bi}: ${b0:.2f} >= cap ${args.budget:.2f}"
            break
        t0 = time.time()
        for sid in batch:
            # Per-claim breaker, BEFORE the call — no sid is ever scored past
            # the cap.
            now = spent()
            if now >= args.budget:
                halted = (f"BUDGET HALT: ${now:.2f} >= cap ${args.budget:.2f} "
                          f"(before scoring {sid})")
                break
            evs = by_sid[sid]
            if not _score_one(llm, texts[sid], evs):
                # NOT banked: score_evidence fails soft, so banking here would
                # record coverage that was never obtained and a resume would
                # skip it forever. Leaving it unbanked means the resume retries.
                if sid not in sidecar["soft_failures"]:
                    sidecar["soft_failures"].append(sid)
                write_sidecar(out_path, sidecar)
                consecutive_failures += 1
                if consecutive_failures >= SOFT_FAIL_HALT:
                    halted = (f"SOFT-FAILURE HALT at {sid}: "
                              f"{consecutive_failures} sids in a row came back "
                              "unscored — that is a lane outage, not bad luck")
                    break
                continue
            consecutive_failures = 0
            if sid in sidecar["soft_failures"]:
                sidecar["soft_failures"].remove(sid)     # a retry rescued it
            sidecar["sids"][sid] = scored_rows(evs)
            n_done += 1
            # Bank after EVERY sid: a crash here must not re-spend on it.
            sidecar["spend_usd"] = round(spent(), 6)
            sidecar["generated"] = datetime.now(timezone.utc).isoformat()
            write_sidecar(out_path, sidecar)
        print(f"batch {bi}/{len(batches)}: {len(batch)} sids, "
              f"proxy ${spent() - b0:.4f}, {time.time() - t0:.0f}s "
              f"(total ${spent():.4f} / cap ${args.budget:.2f})")
        if halted:
            break

    total = spent()
    print(f"\nSPEND: ${total - banked:.4f} this session + ${banked:.4f} banked "
          f"= ${total:.4f} (cap ${args.budget:.2f})")
    print(f"scored {n_done} sids this session; sidecar: {out_path}")
    if sidecar["soft_failures"]:
        print(f"SOFT FAILURES ({len(sidecar['soft_failures'])} sids came back "
              "unchanged — score_evidence fails soft and keeps defaults): "
              f"{sidecar['soft_failures'][:10]}"
              + (" …" if len(sidecar['soft_failures']) > 10 else ""))
    if halted:
        print(f"\n{halted}")
        remaining = len(pending_sids(art, sidecar, texts, only))
        print(f"HALTED CLEANLY — {remaining} sids still unscored. Everything "
              "paid for is banked in the sidecar. Resume (re-spends only on "
              "unscored sids):")
        print(f"  PYTHONPATH=.:src .venv/bin/python scripts/rescore_stored_packs.py "
              f"--speech {speech} --go --budget <USD>")
        return 2
    print("\nComplete. The artifact was NOT modified — join the sidecar to it "
          "downstream to find the gate flips.")
    return 0


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--speech", choices=sorted(REBUILT_RUNS),
                    help="speech to re-score (required except for a "
                         "five-speech --estimate)")
    ap.add_argument("--artifact", default=None, metavar="PATH",
                    help="explicit metrics/pca_runs/<uuid>.json (overrides the "
                         "--speech lookup; --speech still names the sidecar)")
    ap.add_argument("--go", action="store_true",
                    help="actually spend (else plan/estimate only, $0)")
    ap.add_argument("--budget", type=float, default=None,
                    help="halt cap in USD — REQUIRED with --go. Haiku is "
                         "on-proxy, so this is checked against the ledger")
    ap.add_argument("--estimate", action="store_true",
                    help="$0 cost estimate from the stored payloads (all five "
                         "speeches when --speech is omitted) and exit")
    ap.add_argument("--estimate-json", default=None, metavar="PATH",
                    help="also write the --estimate numbers as JSON")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"scoring model on the proxy (default {DEFAULT_MODEL} "
                         "— the cheap lane; changing it changes the bill)")
    ap.add_argument("--out", default=None, metavar="PATH",
                    help="sidecar path (default "
                         "metrics/remediation_v2/rescored_<speech>.json)")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N unscored sids only (smoke slice)")
    ap.add_argument("--only-sids", default=None, metavar="PATH",
                    help="JSON array of sids to restrict to — the B2 targeting "
                         "list from scripts/b2_primary_series.py --write-sids. "
                         "Applied ON TOP of the resume filter, so a targeted "
                         "run still never pays twice for the same sid")
    args = ap.parse_args(argv)

    if args.estimate:
        speeches = [args.speech] if args.speech else list(REBUILT_RUNS)
        text, summary = estimate_report(speeches, model=args.model)
        print(text)
        if args.estimate_json:
            Path(args.estimate_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.estimate_json).write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(f"\nwrote {args.estimate_json}")
        return 0

    if not args.speech:
        ap.error("--speech is required (or use --estimate for the projection)")

    art_path = Path(args.artifact) if args.artifact else artifact_path(args.speech)
    art = load_artifact(art_path)
    out_path = Path(args.out) if args.out else sidecar_path(args.speech)
    sidecar = load_sidecar(out_path, args.speech, art.get("run_id", ""))
    texts = claim_texts(art)
    only = load_only_sids(getattr(args, "only_sids", None))
    todo = pending_sids(art, sidecar, texts, only)
    n_items = sum(len(art["evidence"].get(sid) or []) for sid in todo)

    print(f"B1a re-score plan — {args.speech}")
    print(f"  artifact: {art_path} (run {art.get('run_id', '?')[:8]})")
    print(f"  sidecar:  {out_path}"
          + (" (exists — resume)" if out_path.exists() else " (new)"))
    print(f"  packs with evidence: {len(art.get('evidence') or {})}")
    print(f"  already scored: {len(sidecar['sids'])}")
    print(f"  to score: {len(todo)} sids / {n_items} items"
          + (f"; --limit slice: first {args.limit}" if args.limit else ""))
    print(f"  model: {args.model} (LiteLLM proxy, on-proxy → ledger-true cost)")
    print("  the stored artifact is NEVER modified; results go to the sidecar")

    if not args.go:
        print("\n($0 plan only — add --estimate for the cost estimate, or "
              "--go --budget USD to spend)")
        return 0
    return run_rescore(args)


if __name__ == "__main__":
    sys.exit(main())
