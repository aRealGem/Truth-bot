"""CRM-114 — the stage-2 FALSE-vs-MISLEADING discriminator (Phase 3, P67.2).

Gate-2 diagnosis: given open-book evidence the PCA panel *unanimously* softens
severity, and the Phase-3 A/B showed a single 4-way prompt can't hold both the FALSE
and MISLEADING boundaries at once (fix one, break the other — a see-saw). CRM-114
isolates the hard boundary into its own binary call: stage 1 is the normal open-book
panel; stage 2 re-decides ONLY the claims the panel put in the FALSE-or-MISLEADING
bucket, asking a single focused question — is the core assertion CONTRADICTED (FALSE)
or merely OVERSTATED (MISLEADING)?

Rides a mini ``pca`` panel (roster.dev) on the BINARY question — a 3-seat vote, not a
single cheap seat: the v1 single-haiku discriminator under-flipped (it carries the same
MISLEADING bias the full panel showed), so the boundary gets a vote. Because the choice
is binary (no TRUE/UNVERIFIABLE escape), the panel cannot hedge to a milder non-adverse
label. Speaker-blind (I3): linted at import. The discriminator only re-labels within
{FALSE, MISLEADING}; it never invents TRUE/UNVERIFIABLE and never touches a claim the
panel resolved as TRUE or abstained on — stage-1 citations are preserved. A binary
disagreement with no plurality leaves the stage-1 label untouched.
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
    "detail is accurate.\n"
    "- MISLEADING: the core assertion is REAL but exaggerated, cherry-picked, stripped of "
    "context, or spun to create a false impression. Overstating a true underlying fact is "
    "MISLEADING, not FALSE.\n"
    'Evidence items are provided under "evidence" (each with an id, source, trust tier, '
    "and a dated snippet); weigh higher-trust tiers above lower ones. Distinguish "
    "contradiction (FALSE) from overstatement of a real fact (MISLEADING); do not default "
    "to the milder label. Return JSON only: "
    '{"verdict": "FALSE" | "MISLEADING", "confidence": 0.0-1.0, "reasoning": "one clause"}.'
)

lint_template_for_speaker_conditionals("CRM114", CRM114_SYSTEM)


def discriminate(hm: HydraMind, items: list[dict], *, roster: str = "dev",
                 tune: Optional[dict] = None) -> dict[str, str]:
    """Run the binary discriminator over ``items`` ([{item_id, payload}], the same
    evidence-pack payloads stage 1 used) as a mini pca panel. Returns
    {sid: "FALSE"|"MISLEADING"} for items the panel resolved to a valid binary label;
    an item that came back TRUE/UNVERIFIABLE or disagreement-flagged is omitted (caller
    keeps the stage-1 label). Requires a live proxy lane."""
    if not items:
        return {}
    run_tune = {"prompts": {r: CRM114_SYSTEM for r in ("proposer", "critic", "arbiter")}}
    run_tune.update(tune or {})
    result, _manifest = hm.run("discriminate", items, "pca", roster=roster, tune=run_tune)
    out: dict[str, str] = {}
    for r in result.items:
        v = ((r.value or {}).get("verdict") or "").strip().upper()   # {} for disagreement
        if v in _ADVERSE:
            out[r.item_id] = v
    return out


def apply_discrimination(rows: list[dict], disc: dict[str, str]) -> list[dict]:
    """Override stage-1 FALSE/MISLEADING labels with the discriminator's call (in place).

    Only a RESOLVED row whose verdict is FALSE or MISLEADING is eligible; a changed row
    records ``crm114`` = {"stage1", "final"} for telemetry. Rows the panel resolved as
    TRUE, or abstained on, are never touched (they never entered the adverse bucket)."""
    for row in rows:
        if row.get("status") != "resolved" or row.get("verdict") not in _ADVERSE:
            continue
        final = disc.get(row["sid"])
        if final in _ADVERSE and final != row["verdict"]:
            row["crm114"] = {"stage1": row["verdict"], "final": final}
            row["verdict"] = final
    return rows
