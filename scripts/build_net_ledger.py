#!/usr/bin/env python3
"""DC-6' net corrections ledger — LIVE-vs-STAGED, per sid, mechanism per hop (F6).

$0, no model calls. Reads the per-hop correction records the DC-6 packager
already produced — Phase-3 rebuild, the 2026-08-09 wave, the 2026-08-10 rulings,
the R-1 shape correction, the R-3 escape run — and FOLDS them per sid into the
NET change the reader sees: the verdict a publish used to show (the first hop's
``old_verdict``) versus the verdict the staged head shows now (the last hop's
``new_verdict``), with every intermediate hop attributed as its own mechanism.

Why compose rather than diff two rendered sites: composition is what carries the
mechanism per hop (re-score / D15 / D16a / wave / discriminator / ruling /
panel). The intermediate split states collapse — e.g. trump_2026:0462 went
TRUE → models-split → UNVERIFIABLE across two hops and folds to the SINGLE
resolved-UNVERIFIABLE entry D-A ratified, with no special-casing.

Writes:
  data/corrections.json                      superseded publication record
  data/corrections-archive-<date>.json       the ledger this replaces
  metrics/remediation_v2/dc6_net_ledger.json full net record (all sections)

The staged heads already carry these verdicts, so the final publish renders with
``--corrections skip``: data/corrections.json is the changelog the corrections
page reads, NOT an input to apply_to_artifact (applying it would fail the
old_verdict check — the "0026 crash class"). Every ledger-eligible net verdict is
cross-checked against the head the resolver actually publishes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from truthbot.publish.heads import publishing_heads  # noqa: E402

DIFF_DIR = REPO / "metrics" / "remediation_v2"
DATA = REPO / "data" / "corrections.json"
GENERATION = "v2.3-role-axis-s5cap"
PUBLISH_DATE = "2026-08-10"
VALID = {"TRUE", "FALSE", "MISLEADING", "UNVERIFIABLE"}

#: Hops in the order they were adjudicated. The mechanism family is the hop's
#: own; ``rulings`` refines per-sid from the measured attribution below.
HOPS = [
    ("phase3", "dc6_corrections_entries.json", "re-score/rebuild"),
    ("wave", "wave_corrections_entries.json", "wave (D16a release)"),
    ("rulings", "rulings_corrections_entries.json", "ruling"),
    ("r1", "r1_corrections_entries.json", "shape correction"),
    ("r3", "r3_corrections_entries.json", "panel ruling"),
]


def _norm(v: str | None) -> str:
    return (v or "").strip().upper()


def _valid(v: str | None) -> bool:
    return _norm(v) in VALID


def _load(name: str) -> dict:
    p = DIFF_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _rulings_mechanism() -> dict[str, str]:
    d = _load("deferred_gated_mechanism.json")
    return d.get("mechanism") or {}


def collect_appearances() -> tuple[dict[str, list[dict]], set[str], list[dict], list[dict]]:
    """``sid -> [appearance,…]`` in hop order, plus dropped sids and the
    verdict-unchanged provenance/shape changes (documented, not ledgered)."""
    rul_mech = _rulings_mechanism()
    by_sid: dict[str, list[dict]] = {}
    dropped: dict[str, dict] = {}
    prov_shape: list[dict] = []
    for idx, (hop, fname, family) in enumerate(HOPS):
        doc = _load(fname)
        if not doc:
            continue
        for section in ("entries", "non_ledger_changes"):
            for e in doc.get(section) or []:
                sid = e.get("sid")
                if not sid:
                    continue
                mech = family
                if hop == "rulings":
                    mech = rul_mech.get(sid, family)
                if hop == "r3" and sid == "trump_2026:0462":
                    # F10: 0462 was resolved by a FRESH R-3 escape panel call
                    # (2026-08-10) and the outcome owner-ratified (D-A, 2-1). It
                    # is NOT adopted from another run — run 4ee5a251 is
                    # trump_2026:0023's adopted-rationale source, not this one.
                    mech = ("fresh R-3 escape panel (2026-08-10); "
                            "owner-ratified adoption (D-A, 2-1)")
                by_sid.setdefault(sid, []).append({
                    "hop": hop, "order": idx, "section": section,
                    "old": e.get("old_verdict"), "new": e.get("new_verdict"),
                    "old_label": e.get("old_label"), "new_label": e.get("new_label"),
                    "mechanism": mech, "reason": e.get("reason", ""),
                    "speech_id": e.get("speech_id", ""),
                    "claim_text": e.get("claim_text", ""),
                })
        for e in doc.get("dropped_rows") or []:
            if e.get("sid"):
                dropped[e["sid"]] = {**e, "hop": hop}
        # verdict-unchanged records: provenance (rationale/coherence) and shape.
        for e in doc.get("provenance_changes") or []:
            prov_shape.append({**e, "hop": hop, "kind": "provenance"})
        for e in doc.get("shape_changes") or []:
            prov_shape.append({**e, "hop": hop, "kind": "shape"})
    return by_sid, set(dropped), list(dropped.values()), prov_shape


def fold(by_sid: dict[str, list[dict]]) -> tuple[list[dict], list[dict], list[str]]:
    """Fold each sid's hop appearances into one net record. Returns
    (ledger_eligible, non_ledger, composability_warnings)."""
    ledger, non_ledger, warnings = [], [], []
    for sid, apps in by_sid.items():
        apps = sorted(apps, key=lambda a: a["order"])
        # Composability: each hop's old should equal the prior hop's new.
        for a, b in zip(apps, apps[1:]):
            if _norm(a["new"]) != _norm(b["old"]):
                warnings.append(
                    f"{sid}: {a['hop']}.new={a['new']!r} != {b['hop']}.old={b['old']!r}")
        first, last = apps[0], apps[-1]
        net = {
            "sid": sid, "speech_id": last["speech_id"] or first["speech_id"],
            "old_verdict": first["old"], "new_verdict": last["new"],
            "claim_text": next((a["claim_text"] for a in reversed(apps)
                                if a["claim_text"]), ""),
            "reason": last["reason"],
            "date": PUBLISH_DATE,
            "source": f"dc6-net-ledger-{PUBLISH_DATE} ({GENERATION})",
            "mechanism_trail": [
                {"hop": a["hop"], "from": a["old"], "to": a["new"],
                 "mechanism": a["mechanism"]} for a in apps],
        }
        if _valid(first["old"]) and _valid(last["new"]) \
                and _norm(first["old"]) != _norm(last["new"]):
            net["old_verdict"] = _norm(first["old"])
            net["new_verdict"] = _norm(last["new"])
            ledger.append(net)
        else:
            # split-involved, or churned-but-net-equal (e.g. biden_2022:0432).
            net["net_unchanged"] = _norm(first["old"]) == _norm(last["new"])
            non_ledger.append(net)
    ledger.sort(key=lambda e: e["sid"])
    non_ledger.sort(key=lambda e: e["sid"])
    return ledger, non_ledger, warnings


def head_verdicts() -> dict[str, str]:
    out: dict[str, str] = {}
    for sid, path in publishing_heads().items():
        doc = json.loads(path.read_text(encoding="utf-8"))
        for r in doc.get("rows") or []:
            out[r.get("sid")] = _norm(r.get("verdict"))
    return out


def build() -> dict:
    by_sid, dropped_sids, dropped_rows, prov_shape = collect_appearances()
    ledger, non_ledger, warnings = fold(by_sid)
    # Ground-truth: every ledger-eligible net verdict must match the head a
    # publish actually renders. A mismatch means the ledger describes a run that
    # is not being shipped — fail loud rather than mislead the changelog.
    heads = head_verdicts()
    mismatches = [f"{e['sid']}: ledger says {e['new_verdict']} but head says "
                  f"{heads.get(e['sid'])!r}"
                  for e in ledger if heads.get(e["sid"]) != e["new_verdict"]]
    changed = set(by_sid) | dropped_sids
    ledgered = ({e["sid"] for e in ledger} | {e["sid"] for e in non_ledger}
                | dropped_sids)
    # F9: the set that must appear on corrections.html — ledger-eligible entries
    # plus the net-VISIBLE non-ledger moves (verdict crossed a split boundary).
    net_visible = [e["sid"] for e in non_ledger if not e.get("net_unchanged")]
    published_expected = sorted({e["sid"] for e in ledger} | set(net_visible))
    return {
        "schema": "truthbot-dc6-net-ledger v1",
        "generated": PUBLISH_DATE,
        "generation": GENERATION,
        "publish_date": PUBLISH_DATE,
        "basis": "LIVE (first-hop old verdict) vs STAGED (publishing head), "
                 "net per sid, mechanism attributed per hop.",
        "corrections_mode": "skip — the staged heads already carry these "
                            "verdicts; data/corrections.json is the publication "
                            "record, never an input to apply_to_artifact.",
        "hops": [h[0] for h in HOPS],
        "changed_total": len(changed),
        "ledger_eligible": len(ledger),
        "non_ledger_total": len(non_ledger),
        "net_visible_total": len(net_visible),
        "published_expected": published_expected,
        "dropped_total": len(dropped_rows),
        "provenance_shape_total": len(prov_shape),
        "completeness_ok": changed == ledgered,
        "completeness_missing": sorted(changed - ledgered),
        "completeness_phantom": sorted(ledgered - changed),
        "head_mismatches": mismatches,
        "composability_warnings": warnings,
        "entries": ledger,
        "non_ledger_changes": non_ledger,
        "dropped_rows": dropped_rows,
        "provenance_shape_changes": prov_shape,
    }


def net_visible_changes(net: dict) -> list[dict]:
    """F9: the non-ledger changes whose net verdict actually moved (old != new) —
    the Models-split boundary crossings the reader would see change on the page.
    The net-UNCHANGED non-ledger churn (30) stays prose-only, off the table."""
    return [{"sid": e["sid"], "speech_id": e["speech_id"],
             "old_verdict": e["old_verdict"], "new_verdict": e["new_verdict"],
             "reason": e["reason"], "date": e.get("date", PUBLISH_DATE)}
            for e in net["non_ledger_changes"] if not e.get("net_unchanged")]


def public_ledger(net: dict, framing_draft: str | None = None) -> dict:
    """The superseded data/corrections.json.

    ``entries``: the ledger-eligible net corrections (valid old != new verdict),
    strict truthbot-corrections v1 schema. ``resolution_state_changes`` (F9): the
    net-visible non-ledger moves whose verdict crossed into or out of a
    Models-split state — rendered on corrections.html as their own section.
    Both editorial notes ship draft=true (F11): nothing ccagent-authored renders
    as final framing prose; the owner's approved wording replaces them, flagged
    final."""
    entries = [{"sid": e["sid"], "speech_id": e["speech_id"],
                "old_verdict": e["old_verdict"], "new_verdict": e["new_verdict"],
                "reason": e["reason"], "date": e["date"], "source": e["source"]}
               for e in net["entries"]]
    resolution = net_visible_changes(net)
    note = (
        f"On {PUBLISH_DATE} the five-speech corpus finished re-adjudication on "
        f"the unified {GENERATION} pipeline across five recorded hops "
        f"(re-score/rebuild, the D16(alpha) release wave, the D15/D16 rulings, "
        f"the R-1 shape correction and the R-3 escape run). {len(entries)} "
        f"claims now publish a verdict that differs from the previously "
        f"published run; a further {len(resolution)} crossed into or out of a "
        f"models-split state and are listed separately. Prior entries described "
        f"the superseded runs and are archived verbatim.")
    # F11: the factual note is ccagent-authored, so it too ships as a draft
    # (HTML comment) until the owner supplies approved wording.
    notes = [{"date": PUBLISH_DATE, "draft": True,
              "text": "DRAFT - OWNER RED-PEN REQUIRED: " + note}]
    if framing_draft:
        notes.append({"date": PUBLISH_DATE, "draft": True,
                      "text": "DRAFT - OWNER RED-PEN REQUIRED: " + framing_draft})
    return {"schema": "truthbot-corrections v1", "notes": notes,
            "entries": entries, "resolution_state_changes": resolution}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help="write data/corrections.json (+archive) and the net ledger")
    ap.add_argument("--framing-draft", default=None,
                    help="framing prose to add behind a draft flag (S-8)")
    args = ap.parse_args()

    net = build()
    print(f"changed={net['changed_total']} ledger={net['ledger_eligible']} "
          f"non_ledger={net['non_ledger_total']} dropped={net['dropped_total']} "
          f"prov/shape={net['provenance_shape_total']}")
    print(f"completeness set(changed)==set(ledgered): {net['completeness_ok']}")
    if net["completeness_missing"]:
        print("  MISSING:", net["completeness_missing"])
    if net["head_mismatches"]:
        print("  HEAD MISMATCHES:", net["head_mismatches"])
    if net["composability_warnings"]:
        print("  composability warnings:", net["composability_warnings"])
    for sid in ("trump_2026:0554", "biden_2022:0432", "trump_2026:0462"):
        where = ("entry" if any(e["sid"] == sid for e in net["entries"])
                 else "non_ledger" if any(e["sid"] == sid
                                          for e in net["non_ledger_changes"])
                 else "ABSENT")
        row = next((e for sec in ("entries", "non_ledger_changes")
                    for e in net[sec] if e["sid"] == sid), None)
        vv = f"{row['old_verdict']} -> {row['new_verdict']}" if row else "?"
        print(f"  {sid}: {where} ({vv})")

    if not (net["completeness_ok"] and not net["head_mismatches"]):
        raise SystemExit("net ledger failed its own gate — refusing to write")

    if args.write:
        (DIFF_DIR / "dc6_net_ledger.json").write_text(
            json.dumps(net, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        current = json.loads(DATA.read_text(encoding="utf-8")) if DATA.exists() else {}
        archive = REPO / "data" / f"corrections-archive-{PUBLISH_DATE}.json"
        if current and not archive.exists():
            archive.write_text(json.dumps(current, indent=2, ensure_ascii=False)
                               + "\n", encoding="utf-8")
        DATA.write_text(json.dumps(public_ledger(net, args.framing_draft),
                                   indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        print(f"wrote {DATA} ({net['ledger_eligible']} entries) and dc6_net_ledger.json")


if __name__ == "__main__":
    main()
