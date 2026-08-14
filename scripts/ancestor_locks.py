#!/usr/bin/env python3
"""Ancestor immutability — the direct check, replacing a generational proxy.

"Prior artifacts are untouched" used to be asserted through a PROXY: the
publishing head's parent should look unscored (``scored_rate < 0.1``), because
if a merge had edited in place the parent's rate would have moved with the
child's. That worked while every head sat exactly one rebuild above the unscored
phase-3 artifact, and it is wrong the moment a chain is deeper — a legitimate
successor to a scored head has a SCORED parent, and the proxy calls that
mutation. It also only ever looked one link up.

This is the property the proxy was standing in for, checked directly and at any
depth: **every artifact in a ``rebuild_of`` chain still has the bytes it had
when it was locked.** A merge that edits a prior artifact fails this no matter
how it disguises itself, and a merge that writes a new child passes trivially.

The lock file is the record. Locking is deliberate and append-only: an artifact
is locked once, and a lock is never rewritten to match changed bytes — that
would be the failure this exists to catch, performed by the tool meant to
detect it.

Usage (repo root):
  PYTHONPATH=src .venv/bin/python scripts/ancestor_locks.py --verify
  PYTHONPATH=src .venv/bin/python scripts/ancestor_locks.py --lock-new
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "metrics" / "pca_runs"
LOCKS = REPO / "metrics" / "ancestor_locks.json"
SCHEMA = "truthbot-ancestor-locks v1"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_locks() -> dict:
    if not LOCKS.exists():
        return {"schema": SCHEMA, "locks": {}}
    doc = json.loads(LOCKS.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{LOCKS}: unknown schema {doc.get('schema')!r}")
    return doc


def artifact_path(run_id: str) -> Path:
    return RUNS / f"{run_id}.json"


def chain_of(run_id: str) -> list[str]:
    """``run_id`` and every ancestor, oldest last. Stops at a missing parent."""
    out, seen = [], set()
    cur = run_id
    while cur and cur not in seen:
        seen.add(cur)
        out.append(cur)
        p = artifact_path(cur)
        if not p.exists():
            break
        cur = (json.loads(p.read_text(encoding="utf-8")).get("meta")
               or {}).get("rebuild_of")
    return out


def verify(run_ids: "list[str] | None" = None) -> list[str]:
    """Violations, as human-readable strings. Empty list = every lock holds."""
    doc = load_locks()
    locks = doc["locks"]
    targets = run_ids if run_ids is not None else list(locks)
    problems = []
    for rid in targets:
        recorded = locks.get(rid)
        if recorded is None:
            continue
        path = artifact_path(rid)
        if not path.exists():
            problems.append(f"{rid}: LOCKED artifact is missing from disk")
            continue
        actual = sha256_of(path)
        if actual != recorded["sha256"]:
            problems.append(
                f"{rid}: bytes changed since it was locked "
                f"({recorded['sha256'][:12]} -> {actual[:12]}) — a prior "
                "artifact was mutated instead of a new child being written")
    return problems


def lock_new() -> dict:
    """Lock any artifact not yet locked. Never rewrites an existing lock."""
    doc = load_locks()
    locks = doc["locks"]
    added = {}
    for path in sorted(RUNS.glob("*.json")):
        rid = path.stem
        if rid in locks or rid == "methodology_manifest":
            continue
        locks[rid] = {"sha256": sha256_of(path)}
        added[rid] = locks[rid]["sha256"]
    doc["schema"] = SCHEMA
    LOCKS.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return added


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--lock-new", action="store_true")
    args = ap.parse_args()
    if args.lock_new:
        added = lock_new()
        print(f"locked {len(added)} new artifact(s)")
        for rid, sha in sorted(added.items()):
            print(f"  {rid}  {sha[:16]}")
    problems = verify()
    if problems:
        print("\nANCESTOR LOCK VIOLATIONS:")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"\nall {len(load_locks()['locks'])} locks hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
