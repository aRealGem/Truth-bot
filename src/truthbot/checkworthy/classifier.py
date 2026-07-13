"""
Layer A / A2 — LLM check-worthiness + speech-act classifier (spec §4).

Rides the HydraMind `single` strategy (one cheap completion per sentence).
Emits EXACTLY the claim-set label contract:
  {check-worthy(+claim_type), opinion, unimportant}
matching `src/truthbot/extract/claims.py` and the 277-row eval set.

The prompt is speaker-blind and is linted for I3 (no speaker/source conditionals)
at import — a violation fails the module load, not a run.
"""
from __future__ import annotations

import json
from typing import Optional

from hydramind import HydraMind, TaskItem
from hydramind.invariants import lint_template_for_speaker_conditionals

_VALID_LABELS = {"check-worthy", "opinion", "unimportant"}
_VALID_CLAIM_TYPES = {"statistical", "historical", "attribution", "comparison", "other", None}

A2_SYSTEM = """You classify a single sentence from a political transcript for a fact-checking \
pipeline. Decide whether it should be verified. Output EXACTLY one label:

- "check-worthy": a SPECIFIC, verifiable factual assertion of public consequence — a statistic, \
a historical or current event, a quantitative comparison or superlative, a causal claim, or a \
claim about what a person/entity did, said, or funded. Keep it check-worthy even if the fact is \
well known or dramatically phrased, as long as the assertion is specific and consequential. Also \
return claim_type in {statistical, historical, attribution, comparison, other}.
- "opinion": label opinion when the sentence's MAIN speech-act is a value judgment, rhetoric, \
aspiration, promise, prediction, or a proposal/recommendation ("we should...", "let's...", \
"let X do Y", "I think..."). It stays opinion even if it embeds a factual premise — the premise \
is context, not the claim being asserted. claim_type=null.
- "unimportant": factual but with no public stakes — a greeting, thanks, ceremony, procedure, \
personal aside, sentence fragment, sports score, or a trivial undisputed truism (e.g. a famous \
death date). If the content is specific and consequential, it is check-worthy, not unimportant. \
claim_type=null.

Judge the DOMINANT speech-act, and whether the checkable content is specific and consequential. \
Do NOT consider who the speaker is.

Examples:
- "Core inflation fell to 1.7 percent in the last quarter." -> check-worthy (specific statistic)
- "Our forces overwhelmed all defenses and defeated the enemy to end the dictator's reign." -> \
check-worthy (specific consequential event, even if dramatically phrased)
- "There has been a law on the books for almost a century supporting American jobs." -> \
check-worthy (specific historical/legal claim, even though well known)
- "Let Medicare negotiate lower drug prices, like the VA already does." -> opinion (main act is a \
policy proposal; the "VA already does" premise is incidental, not the assertion)
- "I think he's a nice guy." -> opinion (value judgment)
- "Thomas Jefferson drew his last breath." -> unimportant (undisputed truism, no public stakes)
- "Thank you all for being here tonight." -> unimportant (greeting)

Return JSON only: {"label": "...", "claim_type": "... or null", "confidence": 0.0-1.0, \
"rationale": "one clause"}"""

# I3 guard at load — a speaker conditional in this template must fail the import.
lint_template_for_speaker_conditionals("A2_SYSTEM", A2_SYSTEM)


def parse_a2(raw: dict) -> dict:
    """Normalize a model output dict to the label contract; fail closed on
    an invalid label (better to surface than to silently mislabel)."""
    label = (raw.get("label") or "").strip().lower()
    if label not in _VALID_LABELS:
        # tolerate minor variants
        label = {"checkworthy": "check-worthy", "check worthy": "check-worthy"}.get(label, label)
    if label not in _VALID_LABELS:
        raise ValueError(f"A2 emitted invalid label {raw.get('label')!r}")
    ct = raw.get("claim_type")
    ct = ct.strip().lower() if isinstance(ct, str) else None
    if ct not in _VALID_CLAIM_TYPES:
        ct = "other"
    if label != "check-worthy":
        ct = None
    conf = raw.get("confidence")
    return {"label": label, "claim_type": ct,
            "confidence": float(conf) if conf is not None else None,
            "rationale": raw.get("rationale", "")}


def classify(hm: HydraMind, sentences: list[dict], tune: Optional[dict] = None):
    """sentences: [{"sid","text","context"}]. Returns (list[normalized], manifest).
    Requires a live L-P/L-B lane (proxy virtual key from repo .env)."""
    items = [TaskItem(item_id=s["sid"],
                      payload={"sentence": s["text"], "context": s.get("context", "")})
             for s in sentences]
    run_tune = {"prompt": A2_SYSTEM, "roles.solo.tier": "cheap"}
    run_tune.update(tune or {})
    result, manifest = hm.run("classify", items, "single", tune=run_tune)
    by_sid = {s["sid"]: s for s in sentences}
    out = []
    for r in result.items:
        norm = parse_a2(r.value)
        norm["sid"] = r.item_id
        # Carry the claim text (and context) through so Layer B can consume the
        # check-worthy queue directly — run_layer_b needs {"sid","text"} per row.
        src = by_sid.get(r.item_id, {})
        norm["text"] = src.get("text", "")
        norm["context"] = src.get("context", "")
        out.append(norm)
    return out, manifest
