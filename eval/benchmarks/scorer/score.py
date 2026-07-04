#!/usr/bin/env python3
"""
Unified scorer stub for the C1 gating benchmarks.

Two datasets, two scoring paths:

  Cass tasks  — given a task and a candidate agent RESPONSE (string), apply the
                task's declared scorer. exact/contains_all/contains_any/regex are
                scored automatically and return pass/fail. llm_judge items are
                NOT auto-scored: they return verdict=None with the rubric, to be
                routed to an LLM judge (or human) by the caller.

  Claim set   — given a predicted label and the gold label, score the
                check-worthiness classification (exact-match accuracy, plus a
                confusion matrix at the aggregate level).

This is a STUB: the automatic scorers are real and runnable; the llm_judge path
is intentionally left as an integration point for C1 (P96.2 + truth-bot v2).

CLI:
  score.py demo                     # end-to-end self-test on 5 sample Cass tasks
  score.py cass <responses.jsonl>   # {"id","response"} per line -> scored report
  score.py claims <preds.jsonl>     # {"sid","pred"} per line   -> accuracy report
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASS = ROOT / "cass-tasks" / "cass_tasks.jsonl"
CLAIMS = ROOT / "claim-set" / "claim_set.jsonl"


# ── Cass-task scoring ─────────────────────────────────────────────────────────

def score_cass_response(task: dict, response: str) -> dict:
    """Return {verdict: True|False|None, method, detail}. None => needs LLM judge."""
    sc = task["scorer"]
    typ = sc["type"]
    r = response or ""
    rl = r.lower()
    if typ == "exact":
        ok = r.strip() == sc["value"].strip()
        return {"verdict": ok, "method": "exact", "detail": sc["value"]}
    if typ == "contains_all":
        missing = [v for v in sc["values"] if v.lower() not in rl]
        return {"verdict": not missing, "method": "contains_all",
                "detail": f"missing={missing}" if missing else "all present"}
    if typ == "contains_any":
        hit = [v for v in sc["values"] if v.lower() in rl]
        return {"verdict": bool(hit), "method": "contains_any",
                "detail": f"hits={hit}"}
    if typ == "regex":
        m = re.search(sc["pattern"], r)
        return {"verdict": bool(m), "method": "regex",
                "detail": (m.group(0) if m else "no match")}
    if typ == "llm_judge":
        return {"verdict": None, "method": "llm_judge",
                "detail": "SKIPPED — route to LLM judge", "rubric": sc["rubric"]}
    return {"verdict": None, "method": "unknown", "detail": typ}


def load_cass():
    return [json.loads(l) for l in CASS.read_text().splitlines() if l.strip()]


def run_cass(responses: dict) -> dict:
    tasks = {t["id"]: t for t in load_cass()}
    rows, auto_pass, auto_total, judge = [], 0, 0, 0
    for tid, resp in responses.items():
        t = tasks.get(tid)
        if not t:
            rows.append({"id": tid, "verdict": None, "detail": "unknown id"}); continue
        res = score_cass_response(t, resp)
        res["id"] = tid; res["task_type"] = t["task_type"]
        if res["verdict"] is None:
            judge += 1
        else:
            auto_total += 1; auto_pass += int(res["verdict"])
        rows.append(res)
    return {"rows": rows, "auto_pass": auto_pass, "auto_total": auto_total,
            "needs_judge": judge}


# ── Claim-set scoring ─────────────────────────────────────────────────────────

def load_claims():
    return [json.loads(l) for l in CLAIMS.read_text().splitlines() if l.strip()]


def run_claims(preds: dict) -> dict:
    gold = {r["sid"]: r["label"] for r in load_claims()}
    labels = ["check-worthy", "opinion", "unimportant"]
    conf = {a: {b: 0 for b in labels} for a in labels}
    correct = n = 0
    for sid, pred in preds.items():
        g = gold.get(sid)
        if g is None:
            continue
        n += 1; correct += int(pred == g)
        if pred in conf[g]:
            conf[g][pred] += 1
    return {"n": n, "correct": correct,
            "accuracy": (correct / n) if n else None, "confusion": conf}


# ── Demo / self-test ──────────────────────────────────────────────────────────

def demo():
    """End-to-end run of the automatic scorers on 5 sample Cass tasks with
    mock responses (3 that should pass, 1 that should fail, 1 llm_judge)."""
    mock = {
        "CT-001": "I ran `whoami` — the agent runs as user ccagent on this host.",
        "CT-002": "Current directory is /home/ccagent/cc-host.",
        "CT-006": "uid=1002(ccagent) gid=1002(ccagent) groups=1002(ccagent) — no sudo.",
        "CT-007": "There is a crontab configured with three jobs.",  # should FAIL
        "CT-023": "Here is my proposed LiteLLM proxy design bound to 127.0.0.1 ...",  # llm_judge
    }
    rep = run_cass(mock)
    print("=== Cass scorer demo (5 sample tasks) ===")
    for row in rep["rows"]:
        v = {True: "PASS", False: "FAIL", None: "JUDGE"}[row["verdict"]]
        print(f"  {row['id']} [{row['task_type']:9}] {row['method']:12} {v:5} — {row['detail']}")
    print(f"  auto: {rep['auto_pass']}/{rep['auto_total']} passed; "
          f"{rep['needs_judge']} routed to LLM judge")
    return rep


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "demo":
        demo(); return
    cmd = sys.argv[1]
    if cmd == "cass":
        responses = {}
        for l in Path(sys.argv[2]).read_text().splitlines():
            if l.strip():
                o = json.loads(l); responses[o["id"]] = o.get("response", "")
        print(json.dumps(run_cass(responses), indent=2))
    elif cmd == "claims":
        preds = {}
        for l in Path(sys.argv[2]).read_text().splitlines():
            if l.strip():
                o = json.loads(l); preds[o["sid"]] = o.get("pred")
        print(json.dumps(run_claims(preds), indent=2))
    else:
        print(__doc__); sys.exit(2)


if __name__ == "__main__":
    main()
