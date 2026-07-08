"""
HydraMind engine-enforced invariants (design §3.3).

These are HARD GUARDS, not config. A YAML typo cannot defeat them and load-time
guards FAIL (raise), never warn. Split by enforcement point:

  load-time (registry.load):   I1 (grok pool), I3 (no speaker conditionals)
  runtime   (strategy.reduce): I2 (material tie ⇒ disagreement_flagged)
  runtime   (verdict emit):    I4 (citations ⊆ evidence pack)
  structural (evidence enter): I5 (provenance required, via Layer C only)
  release    (heldout access): I6 (read once per RC)

I2 lives in the strategies' reduce(); the helpers here are the single source of
truth those call sites use, so the rule can't drift between strategies.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

# ── exceptions ────────────────────────────────────────────────────────────────


class InvariantError(Exception):
    """Base for all hard-guard failures. Fail closed."""
    code = "I0"


class I1GrokPoolError(InvariantError):
    code = "I1"


class I3SpeakerConditionalError(InvariantError):
    code = "I3"


class I4CitationError(InvariantError):
    code = "I4"


class I5ProvenanceError(InvariantError):
    code = "I5"


class I6HeldoutReuseError(InvariantError):
    code = "I6"


# ── I1: grok never proposes or arbitrates (critic-only if present) ────────────

_PROPOSE_ARBITRATE_ROLES = ("proposer", "arbiter")


def check_i1_grok_pool(roles: dict) -> None:
    """`grok ∉ providers(proposer|arbiter)`. Registry load fails otherwise."""
    for role in _PROPOSE_ARBITRATE_ROLES:
        rs = roles.get(role)
        if rs is None:
            continue
        providers = getattr(rs, "providers", None)
        if providers is None and isinstance(rs, dict):
            providers = rs.get("providers", [])
        if any(p.lower() == "grok" for p in (providers or ())):
            raise I1GrokPoolError(
                f"I1 violation: 'grok' present in {role} provider pool "
                f"({list(providers)}); grok may be critic-only."
            )


# ── I3: no source/speaker conditioning anywhere ───────────────────────────────

# Keys that would let a spec branch on who is being analyzed. The schema forbids
# per-source keys; this catches them defensively at load.
_FORBIDDEN_SPEC_KEY_RX = re.compile(
    r"(?i)\b(speaker|per[_-]?speaker|per[_-]?source|source_id|by_speaker|"
    r"speaker_conditional|persona_of_subject)\b"
)

# Template constructs that condition on the speaker/source identity of the
# *subject under analysis*. Model-provenance conditionals (which MODEL produced
# an output) are allowed (Principle 2) — hence we key on speaker/source words.
_TEMPLATE_SPEAKER_COND_RX = re.compile(
    r"\{%\s*if\s+[^%]*\b(speaker|source|who_said|is_trump|is_biden|party|"
    r"politician|subject_name)\b[^%]*%\}"
    r"|\bif\s+speaker\s*(==|is|in)\b"
    r"|\bwhen\s+(the\s+)?(speaker|source)\s+is\b",
    re.IGNORECASE | re.DOTALL,
)


def _walk_keys(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}.{k}", k
            yield from _walk_keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_keys(v, f"{path}[{i}]")


def check_i3_no_speaker_conditionals(spec_raw: dict) -> None:
    """Schema-level guard: reject any spec key that branches on subject identity."""
    for path, key in _walk_keys(spec_raw):
        if _FORBIDDEN_SPEC_KEY_RX.search(str(key)):
            raise I3SpeakerConditionalError(
                f"I3 violation: spec key '{path}' conditions on source/speaker identity."
            )


def lint_template_for_speaker_conditionals(name: str, template: str) -> None:
    """Template linter (design §3.3 I3): reject speaker/source conditionals in a
    prompt template. Run at PromptRef registration / classifier load."""
    m = _TEMPLATE_SPEAKER_COND_RX.search(template)
    if m:
        raise I3SpeakerConditionalError(
            f"I3 violation: template '{name}' contains a speaker/source "
            f"conditional near: {m.group(0)!r}"
        )


# ── I2: material tie ⇒ disagreement_flagged ───────────────────────────────────

def is_material_disagreement(
    label_a: Optional[str],
    label_b: Optional[str],
    conf_a: Optional[float],
    conf_b: Optional[float],
    threshold: float,
) -> bool:
    """Material disagreement = label mismatch OR |Δconfidence| ≥ threshold
    (spec §3.3 flow.gate). This is the single definition both the pca gate and
    the I2 tie-check use, so they cannot drift apart."""
    if label_a is not None and label_b is not None and label_a != label_b:
        return True
    if conf_a is not None and conf_b is not None:
        if abs(conf_a - conf_b) >= threshold:
            return True
    return False


# ── I4: verdict citations ⊆ evidence pack ─────────────────────────────────────

def check_i4_citations(citations: Iterable[str], pack_item_ids: Iterable[str]) -> None:
    pack = set(pack_item_ids)
    unknown = [c for c in citations if c not in pack]
    if unknown:
        raise I4CitationError(
            f"I4 violation: verdict cites evidence not in pack: {unknown}"
        )


# ── I5: evidence must carry provenance and enter via Layer C ──────────────────

_REQUIRED_PROVENANCE = ("url", "retrieved_at", "sha256", "tier")


def check_i5_provenance(evidence_item: dict) -> None:
    missing = [k for k in _REQUIRED_PROVENANCE if not evidence_item.get(k)]
    if missing:
        raise I5ProvenanceError(
            f"I5 violation: evidence item missing provenance {missing}; "
            f"evidence enters only via Layer C with {list(_REQUIRED_PROVENANCE)}."
        )


# ── I6: heldout read once per release candidate ───────────────────────────────

class HeldoutGuard:
    """Tracks heldout dataset reads per release-candidate id. A second read of
    the same heldout split under the same RC id fails closed."""

    def __init__(self) -> None:
        self._seen: set[tuple[str, str]] = set()

    def read(self, dataset_id: str, rc_id: str) -> None:
        key = (dataset_id, rc_id)
        if key in self._seen:
            raise I6HeldoutReuseError(
                f"I6 violation: heldout '{dataset_id}' already read for RC '{rc_id}'. "
                f"Heldout is read once per release candidate."
            )
        self._seen.add(key)
