#!/usr/bin/env python3
"""Agreed-verdict audit (P67.6 / remediation T1.2) — Severity-tier model pass
over every non-escalated decided claim.

The audit's F8 class: when proposer and critic agree, the claim is never
escalated and never reviewed — so a SHARED misread (tense, units, baseline)
ships as a confident verdict. This harness re-reads each such claim with the
Severity-tier model (claude-sonnet-4-6 via the LiteLLM proxy) against an
explicit error-class checklist:

  (a) tense/temporal misreads (a posthumous tribute is not false)
  (b) quarterly-vs-annual and unit confusion
  (c) baseline selection ("before I took office" = the fiscal year ending
      as the term begins, not an earlier one)
  (d) colloquial-recency falsification ("recently" spanning a few months
      is not a falsehood)
  (e) invented referents (reading an entity into the claim that is not there)

For referent grounding ONLY, the prompt includes the two preceding transcript
sentences (from the artifact's own segmented transcript — claims +
characterization cover every sid), marked NON-EVIDENTIARY.

Crash-safe by construction (P67.3 lesson): results append to a JSONL file
after every claim; re-running with the same --out skips already-audited sids,
so proxy phantom-budget legs resume cleanly.

Usage (repo root):
  PYTHONPATH=. .venv/bin/python scripts/audit_agreed_verdicts.py \
      --out metrics/audits/agreed_verdicts_2026-07-21.jsonl [--limit 10] \
      [--speech trump_2026] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import requests

DECIDED = {"TRUE", "FALSE", "MISLEADING"}

SYSTEM_PROMPT = """\
You are a senior fact-check reviewer auditing verdicts produced by a panel of
smaller models. The panel judged each claim AS OF THE UTTERANCE DATE from a
fixed evidence pack. Your job is NOT to re-litigate the evidence from scratch:
it is to catch a specific family of reasoning errors that survive when two
models agree:

(a) tense/temporal misreads — e.g. treating a posthumous tribute as a false
    present-tense claim, or misreading narrative sequence;
(b) quarterly-vs-annual and unit confusion — e.g. rejecting a correct annual
    figure because quarterly annualized rates differ;
(c) baseline selection — e.g. "before I took office" compared against the
    wrong prior year;
(d) colloquial-recency falsification — "recently"/"just" spanning weeks or a
    few months is not thereby false;
(e) invented referents — the panel's rationale relies on an entity or event
    the claim does not actually reference.

Also note (f) any OTHER clear reasoning error, but do not stretch: if the
verdict is defensible on the evidence, mark it sound. Judge as of the
utterance date. The two preceding transcript sentences are provided ONLY to
ground referents (who "she"/"that" is); they are NON-EVIDENTIARY and must not
be treated as evidence for or against the claim.

Respond with STRICT JSON, no prose outside it:
{"verdict_sound": true|false,
 "error_classes": ["a".."f"],            // empty when sound
 "suggested_verdict": "TRUE"|"FALSE"|"MISLEADING"|"UNVERIFIABLE"|"UNCHANGED",
 "confidence": 0.0-1.0,
 "explanation": "<= 3 sentences"}"""


def latest_artifacts() -> list[Path]:
    candidates = sorted((REPO / "metrics" / "pca_runs").glob("*.json"),
                        key=lambda p: p.stat().st_mtime)
    latest: dict[str, Path] = {}
    for p in candidates:
        d = json.loads(p.read_text(encoding="utf-8"))
        if "evidence" not in d:
            continue
        latest[(d.get("meta") or {}).get("speech_id") or p.stem] = p
    return list(latest.values())


def sid_index(sid: str) -> int:
    return int(sid.rsplit(":", 1)[1])


def build_sentence_map(artifact: dict) -> dict[str, str]:
    m: dict[str, str] = {}
    for c in artifact.get("claims") or []:
        m[c["sid"]] = c.get("text", "")
    for ch in artifact.get("characterization") or []:
        if ch.get("sid"):
            m[ch["sid"]] = ch.get("text", "")
    return m


def preceding_sentences(sid: str, sent_map: dict[str, str], n: int = 2) -> list[str]:
    speech, idx = sid.rsplit(":", 1)[0], sid_index(sid)
    out = []
    for i in range(max(0, idx - n), idx):
        text = sent_map.get(f"{speech}:{i:04d}")
        if text:
            out.append(text)
    return out


def claim_prompt(row: dict, claim: dict, evidence: list[dict],
                 utterance: str, preceding: list[str]) -> str:
    pack_lines = []
    for i, ev in enumerate(evidence, start=1):
        date_note = ev.get("published_at") or ""
        stance = ev.get("supports_claim")
        stance_s = {True: "supports", False: "refutes"}.get(stance, "")
        pack_lines.append(
            f"[E{i}] ({ev.get('source_tier', '?')}{' · ' + stance_s if stance_s else ''}"
            f"{' · ' + str(date_note)[:10] if date_note else ''}) "
            f"{ev.get('source_url', '')}\n     {ev.get('snippet', '')}")
    preceding_block = ""
    if preceding:
        preceding_block = (
            "PRECEDING TRANSCRIPT SENTENCES (NON-EVIDENTIARY — referent "
            "grounding only):\n" + "\n".join(f"  · {s}" for s in preceding) + "\n\n")
    crm = row.get("crm114") or {}
    crm_note = (f"\nSeverity-stage override already applied: "
                f"{crm.get('stage1')} → {crm.get('final')}" if crm else "")
    return (
        f"UTTERANCE DATE: {utterance}\n\n"
        f"{preceding_block}"
        f"CLAIM ({row['sid']}): {claim.get('text', '')}\n\n"
        f"EVIDENCE PACK the panel judged from:\n" + "\n".join(pack_lines) +
        f"\n\nPANEL VERDICT: {row.get('verdict')} "
        f"(votes {row.get('votes')}, non-escalated){crm_note}\n"
        f"PANEL REASONING: {row.get('reasoning', '')}\n\n"
        "Audit this verdict against the error-class checklist."
    )


def load_key() -> str:
    for var in ("LITELLM_TRUTHBOT_KEY", "LITELLM_PCA_KEY", "LITELLM_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    envf = REPO / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            if line.startswith("LITELLM_TRUTHBOT_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    sys.exit("no LiteLLM key found (env or repo .env)")


def call_model(base: str, key: str, prompt: str, model: str) -> dict:
    r = requests.post(
        f"{base}/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "temperature": 0,
              "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "content": prompt}]},
        timeout=180)
    r.raise_for_status()
    body = r.json()
    text = body["choices"][0]["message"]["content"].strip()
    usage = body.get("usage") or {}
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):text.rfind("}") + 1]
    start, end = text.find("{"), text.rfind("}")
    parsed = json.loads(text[start:end + 1])
    parsed["_usage"] = {k: usage.get(k) for k in
                        ("prompt_tokens", "completion_tokens")}
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="JSONL results path (appended; resume-safe)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N new audits (metering)")
    ap.add_argument("--speech", default="", help="restrict to one speech_id")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--dry-run", action="store_true",
                    help="print selection counts + the first prompt; no calls")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                done.add(json.loads(line)["sid"])
            except Exception:
                pass

    base = os.environ.get("LITELLM_BASE_URL", "http://127.0.0.1:4141")
    key = None if args.dry_run else load_key()

    total_new = 0
    for path in latest_artifacts():
        artifact = json.loads(path.read_text(encoding="utf-8"))
        meta = artifact.get("meta") or {}
        speech = meta.get("speech_id", "")
        if args.speech and speech != args.speech:
            continue
        utterance = meta.get("date", "")
        sent_map = build_sentence_map(artifact)
        claims_by_sid = {c["sid"]: c for c in artifact.get("claims") or []}
        evidence = artifact.get("evidence") or {}

        rows = [r for r in artifact.get("rows") or []
                if not r.get("escalated") and (r.get("verdict") or "") in DECIDED]
        print(f"{speech}: {len(rows)} non-escalated decided claims "
              f"({len(done & {r['sid'] for r in rows})} already audited)")

        for row in rows:
            sid = row["sid"]
            if sid in done:
                continue
            if args.limit and total_new >= args.limit:
                print(f"limit {args.limit} reached — stopping (resume with same --out)")
                return
            prompt = claim_prompt(
                row, claims_by_sid.get(sid, {}), evidence.get(sid) or [],
                utterance, preceding_sentences(sid, sent_map))
            if args.dry_run:
                print("\n----- SAMPLE PROMPT -----\n" + prompt[:3000])
                return
            try:
                result = call_model(base, key, prompt, args.model)
            except Exception as e:  # noqa: BLE001 — log & continue; resume re-tries
                print(f"  ! {sid}: {type(e).__name__}: {e}")
                time.sleep(2)
                continue
            record = {
                "sid": sid, "speech": speech,
                "shipped_verdict": row.get("verdict"),
                "escalated": False,
                **{k: result.get(k) for k in
                   ("verdict_sound", "error_classes", "suggested_verdict",
                    "confidence", "explanation", "_usage")},
                "model": args.model, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_new += 1
            flag = "" if result.get("verdict_sound") else \
                f"  ⚑ {result.get('error_classes')} → {result.get('suggested_verdict')}"
            print(f"  {sid}: {row.get('verdict')} sound={result.get('verdict_sound')}{flag}")

    print(f"\ndone — {total_new} new audits appended to {out}")


if __name__ == "__main__":
    main()
