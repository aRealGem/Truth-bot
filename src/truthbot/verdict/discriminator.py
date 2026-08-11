"""CRM-114 — the stage-2 FALSE-vs-MISLEADING discriminator (Phase 3, P67.2).

Gate-2 diagnosis: given open-book evidence the PCA panel *unanimously* softens
severity, and the Phase-3 A/B showed a single 4-way prompt can't hold both the FALSE
and MISLEADING boundaries at once (fix one, break the other — a see-saw). CRM-114
isolates the hard boundary into its own binary call: stage 1 is the normal open-book
panel; stage 2 re-decides ONLY the claims the panel put in the FALSE-or-MISLEADING
bucket, asking a single focused question — is the core assertion CONTRADICTED (FALSE)
or merely OVERSTATED (MISLEADING)?

Rides the ``single`` strategy with a STRONGER seat (default sonnet, tier=standard) on
the BINARY question — the economical shape: stage 1 (the cheap 3-seat panel) runs over
every claim, but stage 2 fires only on the small FALSE-or-MISLEADING bucket, so paying
for one stronger judge there costs almost nothing in aggregate. Prior tries with cheap
seats failed: a single cheap seat under-flipped (net-zero), and a cheap 3-seat vote was
worse still (voting amplifies the seats' shared MISLEADING bias). The fix is a better
judge, not more cheap votes. Because the choice is binary (no TRUE/UNVERIFIABLE escape),
the seat cannot hedge to a milder non-adverse label. Speaker-blind (I3): linted at
import. The discriminator only re-labels within {FALSE, MISLEADING}; it never invents
TRUE/UNVERIFIABLE and never touches a claim the panel resolved as TRUE or abstained on —
stage-1 citations are preserved.
"""
from __future__ import annotations

from typing import Optional

from hydramind import HydraMind
from hydramind.invariants import lint_template_for_speaker_conditionals

_ADVERSE = {"FALSE", "MISLEADING"}

# Binary contract. The payload (claim + evidence pack) is the SAME one stage 1 saw, so
# the discriminator judges on identical evidence — only the question is narrowed.
CRM114_SYSTEM = (
    "You are the CRM-114 DISCRIMINATOR. A prior panel already judged this factual claim, "
    "on the evidence, to be NOT fully true — it is either FALSE or MISLEADING. Your ONLY "
    "job is to decide which, by the claim's CORE assertion:\n"
    "- FALSE: the evidence CONTRADICTS the core assertion — the stated fact did not "
    "happen, or the reverse is true. A contradicted core is FALSE even if some peripheral "
    "detail is accurate. ABSOLUTE-CLAIM RULE: when the core assertion is an absolute or "
    "universal (zero, none, only, all, every, ended, eliminated, completely stopped or "
    "destroyed, biggest/lowest in history), evidence of material counterexamples "
    "contradicts that core and the verdict is FALSE — a real underlying trend does not "
    "soften an absolute.\n"
    "- MISLEADING: the core assertion is REAL but exaggerated, cherry-picked, stripped of "
    "context, or spun to create a false impression. Overstating a true underlying fact is "
    "MISLEADING, not FALSE — unless the claim states an absolute (rule above).\n"
    'Evidence items are provided under "evidence" (each with an id, source, trust tier, '
    "and a dated snippet); weigh higher-trust tiers above lower ones. Distinguish "
    "contradiction (FALSE) from overstatement of a real fact (MISLEADING); do not default "
    "to the milder label. Return JSON only: "
    '{"verdict": "FALSE" | "MISLEADING", "confidence": 0.0-1.0, "reasoning": "one clause"}.'
)

lint_template_for_speaker_conditionals("CRM114", CRM114_SYSTEM)


def discriminate(hm: HydraMind, items: list[dict], *, tier: str = "standard",
                 tune: Optional[dict] = None) -> dict[str, str]:
    """Run the single-seat binary discriminator over ``items`` ([{item_id, payload}],
    the same evidence-pack payloads stage 1 used). ``tier`` picks the anthropic seat
    strength — "standard" = sonnet (default), "cheap" = haiku (the failed v1). Returns
    {sid: "FALSE"|"MISLEADING"} for items that came back a valid binary label; an item
    the seat labeled TRUE/UNVERIFIABLE is omitted (caller keeps the stage-1 label).
    Requires a live proxy lane."""
    if not items:
        return {}
    run_tune = {"prompt": CRM114_SYSTEM, "roles.solo.tier": tier}
    run_tune.update(tune or {})
    result, _manifest = hm.run("discriminate", items, "single", tune=run_tune)
    out: dict[str, str] = {}
    for r in result.items:
        v = ((r.value or {}).get("verdict") or "").strip().upper()
        if v in _ADVERSE:
            out[r.item_id] = v
    return out


#: Provenance marker written onto a row whose rationale was ADOPTED rather than
#: authored by the resolver that set the verdict. Read by the publish bridge so
#: the strip can attribute the text instead of implying the resolver wrote it.
ADOPTED_PREFIX = "adopted from"


def adopt_seat_rationale(row: dict, final: str) -> Optional[dict]:
    """Give ``row`` the CHOSEN SEAT's stored rationale, VERBATIM (R-3 ruling,
    2026-08-10). Returns the provenance record it wrote, or None.

    The stage-2 discriminator resolves a tie by NAMING a label; it does not
    write prose, and until this existed the resolved row carried an empty
    ``reasoning``. That published a fact-check that could not say why, and it
    blinded ``verdict_audit.adjacent_coherence_conflicts``, which links claims
    partly through rationale text.

    The fix is structural and adds NO new text to the corpus. Among the seats
    that voted ``final``, the one with a non-empty rationale is adopted (arbiter
    first — it is the seat that saw the split — then the panel order). The text
    is copied UNCHANGED; the attribution lives in ``rationale_provenance``, not
    in the prose, so the sentence a reader sees is exactly the sentence a model
    wrote. If no seat voted ``final`` with text to give, nothing is invented and
    the row is left for the no-blank-rationale lint to catch.

    A row that already carries a rationale is never overwritten."""
    if str(row.get("reasoning") or "").strip():
        return None
    seats = [s for s in (row.get("seat_rationales") or [])
             if str(s.get("verdict") or "").strip().upper() == str(final).strip().upper()
             and str(s.get("reasoning") or "").strip()]
    if not seats:
        return None
    seats.sort(key=lambda s: 0 if str(s.get("role")) == "arbiter" else 1)
    chosen = seats[0]
    prov = {
        "mode": "adopted-verbatim",
        "adopted_from": str(chosen.get("role") or "seat"),
        "adopted_verdict": str(chosen.get("verdict") or ""),
        "resolver": "crm114-discriminator",
        "attribution": f"{ADOPTED_PREFIX} {chosen.get('role') or 'seat'} seat",
        "synthesized": False,
    }
    row["reasoning"] = str(chosen.get("reasoning"))
    row["rationale_provenance"] = prov
    return prov


def apply_tie_routing(rows: list[dict], disc: dict[str, str]) -> list[dict]:
    """Resolve DISAGREEMENT-flagged adverse-severity ties with the discriminator's
    binary call (in place). Only fires for rows the caller routed (i.e. present in
    ``disc``); the row becomes resolved with ``crm114 = {"stage1": "DISAGREEMENT",
    "final": ...}`` and its vote tally intact, so the tie and its adjudication are
    both readable from the artifact — an explicit stage-2 decision, never a silent
    tie-break (I2). Confidence stays None (no seat consensus to average) and
    citations stay [] (the tie had no winning seat to take citations from).

    R-3 (2026-08-10): the resolved row also ADOPTS the chosen seat's stored
    rationale verbatim (:func:`adopt_seat_rationale`). Before this, every
    tie-routed row published with a blank rationale."""
    for row in rows:
        if row.get("status") != "disagreement":
            continue
        final = disc.get(row["sid"])
        if final in _ADVERSE:
            row["crm114"] = {"stage1": "DISAGREEMENT", "final": final}
            row["status"] = "resolved"
            row["verdict"] = final
            adopt_seat_rationale(row, final)
    return rows


def apply_discrimination(rows: list[dict], disc: dict[str, str]) -> list[dict]:
    """Override stage-1 FALSE/MISLEADING labels with the discriminator's call (in place).

    Only a RESOLVED row whose verdict is FALSE or MISLEADING is eligible; a changed row
    records ``crm114`` = {"stage1", "final"} for telemetry. Rows the panel resolved as
    TRUE, or abstained on, are never touched (they never entered the adverse bucket).

    R-3 (2026-08-10): a flipped row whose rationale is BLANK adopts the chosen
    seat's stored text (:func:`adopt_seat_rationale`). A row that already carries
    the stage-1 winner's rationale keeps it — re-attributing an existing rationale
    after a severity flip is a separate question, logged as a D17 candidate."""
    for row in rows:
        if row.get("status") != "resolved" or row.get("verdict") not in _ADVERSE:
            continue
        final = disc.get(row["sid"])
        if final in _ADVERSE and final != row["verdict"]:
            row["crm114"] = {"stage1": row["verdict"], "final": final}
            row["verdict"] = final
            adopt_seat_rationale(row, final)
    return rows
