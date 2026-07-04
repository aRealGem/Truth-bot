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

- "check-worthy": a factual, verifiable assertion of public importance (statistic, historical \
event, quantitative comparison, causal attribution, or a claim about what someone/some entity \
did or said). Also return claim_type in \
{statistical, historical, attribution, comparison, other}.
- "opinion": opinion, value judgment, rhetoric, aspiration, promise, or a prediction about the \
future. claim_type=null.
- "unimportant": literally factual but trivial (greeting, ceremony, procedure, personal aside, \
truism), not worth a fact-check budget. claim_type=null.

Judge only the proposition and its speech-act form. Do NOT consider who the speaker is.

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
    out = []
    for r in result.items:
        norm = parse_a2(r.value)
        norm["sid"] = r.item_id
        out.append(norm)
    return out, manifest
