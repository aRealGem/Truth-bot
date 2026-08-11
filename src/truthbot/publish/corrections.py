"""Public corrections — the fact-checking-norm changelog (P67.6 / T1.5).

``data/corrections.json`` at the repo root is the system of record:

    {"schema": "truthbot-corrections v1",
     "entries": [
        {"sid": "biden_2022:0115",
         "speech_id": "biden_2022",
         "old_verdict": "FALSE",
         "new_verdict": "TRUE",
         "reason": "Panel confused quarterly annualized GDP rates with the correct 5.7% annual figure.",
         "date": "2026-07-21",
         "source": "agreed-verdict-audit-2026-07-21"}]}

Entries are ONLY added with jackie's explicit approval (remediation T1.4
halt). Applying them:

* ``apply_to_artifact`` rewrites the matching adjudication rows' verdicts
  in-memory before the bridge runs, stamping ``row['corrected']`` so the
  bundle's provenance carries a correction note (rendered on the claim's
  provenance strip).
* ``_render_corrections`` (site.py) publishes the changelog page; every
  page footer links it, and the feed template carries a related link.

A correction never silently overwrites history: the page lists claim id,
old → new verdict, reason, and date, per fact-checking norms.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA = "truthbot-corrections v1"
_REQUIRED = ("sid", "speech_id", "old_verdict", "new_verdict", "reason", "date")
_VALID_VERDICTS = {"TRUE", "FALSE", "MISLEADING", "UNVERIFIABLE"}


class CorrectionsError(ValueError):
    """corrections.json is malformed — fail the build, don't guess."""


def load_notes(path: Path) -> list[dict]:
    """Editorial notes for the Corrections page ({date, text} dicts) — the
    D2 band-change explanation lives here as DATA, not hand-typed HTML."""
    path = Path(path)
    if not path.exists():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    return [n for n in (doc.get("notes") or []) if n.get("date") and n.get("text")]


def load_corrections(path: Path) -> list[dict]:
    """Load + validate corrections. Missing file → no corrections (empty)."""
    path = Path(path)
    if not path.exists():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise CorrectionsError(f"{path}: unknown schema {doc.get('schema')!r}")
    entries = doc.get("entries") or []
    seen: set[str] = set()
    for e in entries:
        missing = [k for k in _REQUIRED if not e.get(k)]
        if missing:
            raise CorrectionsError(f"{path}: entry {e.get('sid', '?')} missing {missing}")
        for k in ("old_verdict", "new_verdict"):
            if e[k].upper() not in _VALID_VERDICTS:
                raise CorrectionsError(f"{path}: {e['sid']} bad {k}={e[k]!r}")
        if e["old_verdict"].upper() == e["new_verdict"].upper():
            raise CorrectionsError(f"{path}: {e['sid']} corrects to the same verdict")
        if e["sid"] in seen:
            raise CorrectionsError(f"{path}: duplicate correction for {e['sid']}")
        seen.add(e["sid"])
    return entries


def note_for(entry: dict) -> str:
    return (f"Corrected {entry['old_verdict'].upper()} → "
            f"{entry['new_verdict'].upper()} ({entry['date']}): {entry['reason']}")


def apply_to_artifact(artifact: dict, entries: list[dict]) -> int:
    """Apply corrections to a pca_runs artifact IN MEMORY (the persisted
    artifact on disk is never rewritten — it remains the record of what the
    panel actually produced). Returns the number of rows corrected.

    A correction whose ``old_verdict`` does not match the row's current
    verdict is an error: it means the entry was written against a different
    run and must be re-approved, not silently re-targeted."""
    speech_id = ((artifact.get("meta") or {}).get("speech_id")) or ""
    by_sid = {e["sid"]: e for e in entries if e["speech_id"] == speech_id}
    if not by_sid:
        return 0
    applied = 0
    for row in artifact.get("rows") or []:
        entry = by_sid.get(row.get("sid"))
        if entry is None:
            continue
        current = (row.get("verdict") or "").strip().upper()
        if current != entry["old_verdict"].upper():
            raise CorrectionsError(
                f"correction for {entry['sid']} expects old verdict "
                f"{entry['old_verdict']!r} but the artifact row says {current!r}")
        row["verdict"] = entry["new_verdict"].upper()
        row["corrected"] = {
            "old": entry["old_verdict"].upper(),
            "new": entry["new_verdict"].upper(),
            "date": entry["date"],
            "reason": entry["reason"],
            "source": entry.get("source", ""),
            "note": note_for(entry),
        }
        applied += 1
    missing = set(by_sid) - {r.get("sid") for r in artifact.get("rows") or []}
    if missing:
        raise CorrectionsError(
            f"corrections reference sids absent from the {speech_id} artifact: "
            f"{sorted(missing)}")
    return applied


def load_resolution_changes(path: Path) -> list[dict]:
    """F9: the ``resolution_state_changes`` block of data/corrections.json — the
    net-visible non-ledger moves (verdict crossed a models-split boundary).
    Missing file or key → empty."""
    path = Path(path)
    if not path.exists():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    return [e for e in (doc.get("resolution_state_changes") or []) if e.get("sid")]


def annotate_to_artifact(artifact: dict, entries: list[dict],
                         resolution: list[dict] | None = None) -> int:
    """F12: SKIP-mode annotation. The staged head already carries the corrected
    verdict, so the verdict is NOT rewritten — but the corrections page promises
    every corrected claim keeps a visible note on its provenance strip, so this
    stamps ``row['corrected']`` (which the bridge surfaces as the strip's
    correction note) by joining the ledger to the rows by sid.

    Guard is the MIRROR of :func:`apply_to_artifact`: a ledger entry must match
    the row's CURRENT (baked) verdict, so a stale ledger describing a run that is
    not being shipped is still caught. Resolution-state changes (models-split
    boundary) carry no vocabulary verdict, so they are annotated by sid without a
    verdict assertion. Returns the number of rows annotated."""
    speech_id = ((artifact.get("meta") or {}).get("speech_id")) or ""
    by_sid = {e["sid"]: e for e in entries if e.get("speech_id") == speech_id}
    res_by = {e["sid"]: e for e in (resolution or [])
              if e.get("speech_id") == speech_id}
    rows = {r.get("sid"): r for r in artifact.get("rows") or []}
    annotated = 0
    for sid, e in by_sid.items():
        row = rows.get(sid)
        if row is None:
            raise CorrectionsError(
                f"correction for {sid} absent from the {speech_id} artifact")
        current = (row.get("verdict") or "").strip().upper()
        if current != e["new_verdict"].upper():
            raise CorrectionsError(
                f"skip-mode annotation for {sid} expects the baked verdict "
                f"{e['new_verdict']!r} but the artifact row says {current!r}")
        row["corrected"] = {
            "old": e["old_verdict"].upper(), "new": e["new_verdict"].upper(),
            "date": e["date"], "reason": e["reason"],
            "source": e.get("source", ""), "note": note_for(e),
        }
        annotated += 1
    for sid, e in res_by.items():
        row = rows.get(sid)
        if row is None:
            continue
        row["corrected"] = {
            "old": e["old_verdict"], "new": e["new_verdict"],
            "date": e.get("date", ""), "reason": e.get("reason", ""),
            "source": e.get("source", ""),
            "note": (f"Resolution-state change {e['old_verdict']} → "
                     f"{e['new_verdict']} ({e.get('date', '')})"),
        }
        annotated += 1
    return annotated
