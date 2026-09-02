"""Canonical claim count — one number, with the exclusions named (A9).

The record disagreed with itself: 529 in the handoff, 530 in commit e268dec's
DC-4' tally, 183 vs 182 Trump rows. Every one of those was true of *something*,
which is why the argument never closed. These tests pin the reconciliation so
it cannot drift back open:

* the rebuilt artifacts have ZERO rows without a matching claim record;
* the published 530 is exactly 529 + one named orphan row;
* the fold rules and the "decided" definition documented in
  ``docs/run-schema.md`` reproduce the published figures from the artifacts
  alone — which is the actual claim the note makes to an external reviewer.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO / "metrics" / "pca_runs"
SITE = REPO / "site-pca"
DOC = REPO / "docs" / "run-schema.md"

#: The five STAGED rebuilds, by run-id prefix.
RUNS = {
    "gwbush_2006": "74a89c5f",
    "clinton_1998": "d0010426",
    "obama_2014": "4de8a551",
    "biden_2022": "37744fc8",
    "trump_2026": "4ee5a251",
}

#: The pre-remediation runs the rebuild replaced.
OLD_RUNS = {
    "gwbush_2006": "92f39851",
    "clinton_1998": "7c59e9e0",
    "obama_2014": "28965cdf",
    "biden_2022": "7208bbbb",
    "trump_2026": "23939712",
}

CANONICAL_CLAIMS = 529

#: What the PUBLISHED site carries, which is no longer the same number.
#: 529 (the DC-6' presidential record) + 103 (warren_2025-04-29) + 6
#: (cruz_2026-06-24) = 638. FR-0901-02 D3, clerical: the 529 canonical
#: reconciliation and its named orphan are untouched -- that record is about
#: the presidential remediation and does not move when the corpus grows.
#: budd and tillis are registered but published:false, so they add nothing.
PUBLISHED_SITE_CLAIMS = 638
PUBLISHED_RECORDS = 530
ORPHAN_SID = "trump_2026:0311"
PLACEHOLDER_TEXT = "(claim text unavailable)"

ABSTAIN = {"UNVERIFIABLE", "gated-UNVERIFIABLE", "Models split", "No verdict"}

_SPEC = importlib.util.spec_from_file_location(
    "dc6_package_a9", REPO / "scripts" / "dc6_package.py")
dc6 = importlib.util.module_from_spec(_SPEC)
sys.modules["dc6_package_a9"] = dc6
_SPEC.loader.exec_module(dc6)


def _path(prefix: str) -> Path | None:
    hits = sorted(RUNS_DIR.glob(f"{prefix}*.json"))
    return hits[0] if hits else None


_MISSING = [s for s, p in {**RUNS, **OLD_RUNS}.items() if _path(p) is None]
pytestmark = pytest.mark.skipif(
    bool(_MISSING), reason=f"run artifacts absent: {sorted(set(_MISSING))}")


def _load(prefix: str) -> dict:
    return json.loads(_path(prefix).read_text("utf-8"))


def _orphans(run: dict) -> list[str]:
    """Rows with NO matching claim record — the whole 530-vs-529 story."""
    sids = {c.get("sid") for c in run.get("claims") or []}
    return [r.get("sid") for r in run.get("rows") or []
            if r.get("sid") not in sids]


def _label(row: dict) -> str:
    """The vocabulary documented in docs/run-schema.md §2, in rule order."""
    if (row.get("evidence_gate") or row.get("provenance_code") or "") == \
            "insufficient-qualifying-evidence":
        return "gated-UNVERIFIABLE"
    if row.get("verdict") is not None:
        return str(row["verdict"])
    return "Models split" if row.get("split") else "No verdict"


# ── the invariant ────────────────────────────────────────────────────────────

def test_rebuilt_artifacts_contain_zero_orphan_rows():
    """THE invariant. A row with no claim record publishes a card with nothing
    on it and inflates every count derived from rows."""
    for speech_id, prefix in RUNS.items():
        assert _orphans(_load(prefix)) == [], speech_id


def test_the_canonical_count_is_529_claims_and_529_rows():
    claims = rows = 0
    for prefix in RUNS.values():
        run = _load(prefix)
        claims += len(run["claims"])
        rows += len(run["rows"])
    assert claims == CANONICAL_CLAIMS
    assert rows == CANONICAL_CLAIMS      # rows == claims once orphans are gone


def test_the_old_artifacts_carry_exactly_one_named_orphan():
    """529 claims / 530 rows — and the extra row has a name."""
    claims = rows = 0
    orphans: list[str] = []
    for prefix in OLD_RUNS.values():
        run = _load(prefix)
        claims += len(run["claims"])
        rows += len(run["rows"])
        orphans += _orphans(run)
    assert claims == CANONICAL_CLAIMS
    assert rows == CANONICAL_CLAIMS + 1
    assert orphans == [ORPHAN_SID]


@pytest.mark.skipif(not (SITE / "data" / "claims.json").exists(),
                    reason="site-pca tree not present")
def test_the_published_site_is_638_and_the_orphan_is_ledgered_not_placeheld():
    """After the DC-6' publish (rev 5) the committed site carries EXACTLY the 529
    canonical claims. The pre-remediation orphan row (trump_2026:0311, which the
    old run published as a "(claim text unavailable)" placeholder to reach 530) is
    gone from the page — the rebuild emits rows only for real claims — and is
    disclosed instead in the net ledger's dropped_rows. A count correction, not a
    silent drop: no reader ever saw a claim there."""
    published = json.loads((SITE / "data" / "claims.json").read_text("utf-8"))
    placeholders = [c for c in published
                    if PLACEHOLDER_TEXT in (c.get("claim_text") or "")]
    assert len(published) == PUBLISHED_SITE_CLAIMS
    assert placeholders == []
    net = json.loads((REPO / "metrics" / "remediation_v2"
                      / "dc6_net_ledger.json").read_text("utf-8"))
    assert [d["sid"] for d in net["dropped_rows"]] == [ORPHAN_SID]


# ── the packager agrees, and says the same thing in words ────────────────────

def test_canonical_counts_reports_the_same_reconciliation():
    counts = dc6.canonical_counts(dc6.load_diffs(), RUNS_DIR, SITE)
    assert counts["canonical_claims"] == CANONICAL_CLAIMS
    assert counts["new"] == {"claims": CANONICAL_CLAIMS,
                             "rows": CANONICAL_CLAIMS, "orphan_rows": []}
    assert counts["old"]["rows"] == CANONICAL_CLAIMS + 1
    assert counts["old"]["orphan_rows"] == [ORPHAN_SID]
    # Rev 5: the published site is now the DC-6' render — 529 records, no
    # placeholder; the orphan is still NAMED as an excluded row in the
    # reconciliation, just no longer rendered as a page.
    assert counts["published"]["records"] == PUBLISHED_SITE_CLAIMS
    assert counts["published"]["placeholder_records"] == 0
    assert [e["sid"] for e in counts["named_exclusions"]] == [ORPHAN_SID]
    assert str(CANONICAL_CLAIMS) in counts["statement"]
    assert ORPHAN_SID in counts["statement"]


def test_the_phantom_drop_is_ledgered_not_silent():
    """A dropped row has no old→new verdict pair, so it can never be a public
    correction entry — but it moves a PUBLISHED COUNT, and the count is what
    the record disagreed about. It gets its own section."""
    dropped = dc6.dropped_rows(dc6.load_diffs(), RUNS_DIR)
    assert [d["sid"] for d in dropped] == [ORPHAN_SID]
    entry = dropped[0]
    assert entry["kind"] == "orphan_row"
    assert entry["speech_id"] == "trump_2026"
    assert PLACEHOLDER_TEXT in entry["reason"]

    ledgered = json.loads(
        (REPO / "metrics" / "remediation_v2"
         / "dc6_corrections_entries.json").read_text("utf-8"))
    assert ledgered["dropped_total"] == 1
    assert [d["sid"] for d in ledgered["dropped_rows"]] == [ORPHAN_SID]
    # and it is NOT smuggled into the verdict corrections
    assert ORPHAN_SID not in {e["sid"] for e in ledgered["entries"]}
    assert ORPHAN_SID not in {e["sid"] for e in ledgered["non_ledger_changes"]}


# ── the note is reproducible, which is the point of writing it ──────────────

def test_run_schema_note_exists_and_states_the_canonical_count():
    text = DOC.read_text("utf-8")
    for needed in ("gated-UNVERIFIABLE", "Models split", "No verdict",
                   "anecdote-adjusted", ORPHAN_SID, "529",
                   "(claim text unavailable)"):
        assert needed in text, needed


def test_the_documented_fold_reproduces_the_published_decided_rate():
    """§4 of the note: decided = not in {Unverifiable, Models split} after the
    §3 fold. Recomputed straight from the artifacts — if this drifts, the note
    is telling an external reviewer something false."""
    decided = total = 0
    for prefix in RUNS.values():
        for row in _load(prefix)["rows"]:
            total += 1
            decided += _label(row) not in ABSTAIN
    assert total == CANONICAL_CLAIMS
    assert decided == 420
    assert round(decided / total, 4) == 0.794

    review = json.loads(
        (REPO / "metrics" / "remediation_v2" / "dc6_review.json")
        .read_text("utf-8"))
    corpus = review["distributions"]["corpus"]["new_decided"]
    assert (corpus["decided"], corpus["total"]) == (decided, total)


def test_the_two_unverifiables_fold_to_one_published_bucket():
    """The fold §3 documents, asserted against the packager's own table."""
    assert dc6.display("UNVERIFIABLE") == "Unverifiable"
    assert dc6.display("gated-UNVERIFIABLE") == "Unverifiable"
    assert dc6.display("Models split") == "Models split"
    assert dc6.display("No verdict") == "Models split"
    assert "Mostly True" in dc6.DISPLAY_ORDER      # carried even at zero
    assert dc6.ABSTAIN == {"Unverifiable", "Models split"}
