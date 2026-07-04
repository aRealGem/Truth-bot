"""
Strategy registry — loads YAML specs into immutable `Spec` objects and enforces
the load-time invariants (I1 grok pool, I3 no speaker conditionals). Load FAILS
(raises) on any violation; it never warns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from .types import Cap, RoleSpec, Spec
from . import invariants as inv

SPECS_DIR = Path(__file__).parent / "specs"

_VALID_TIERS = {"cheap", "standard", "frontier"}


def _parse_roles(raw_roles: dict) -> dict[str, RoleSpec]:
    roles: dict[str, RoleSpec] = {}
    for name, rd in (raw_roles or {}).items():
        if "tier" not in rd or "providers" not in rd:
            raise ValueError(f"role '{name}' must define tier and providers")
        if rd["tier"] not in _VALID_TIERS:
            raise ValueError(f"role '{name}': invalid tier {rd['tier']!r}")
        roles[name] = RoleSpec(
            tier=rd["tier"],
            providers=tuple(rd["providers"]),
            rotation=rd.get("rotation"),
        )
    return roles


def build_spec(raw: dict) -> Spec:
    """Validate a raw spec dict and return an immutable Spec. Enforces I1 + I3."""
    if "name" not in raw:
        raise ValueError("spec missing 'name'")

    # I3 (schema guard) runs FIRST on the untouched raw dict — before we trust
    # any key — so a speaker-conditional key can't sneak through parsing.
    inv.check_i3_no_speaker_conditionals(raw)

    roles = _parse_roles(raw.get("roles", {}))

    # I1 hard guard at load.
    inv.check_i1_grok_pool(roles)

    caps = frozenset(Cap(c) for c in raw.get("caps", []))

    flow = raw.get("flow", {})
    # A multi_round spec must declare a gate; a batch-only single need not.
    if Cap.MULTI_ROUND in caps and "gate" not in flow:
        raise ValueError(f"spec '{raw['name']}': multi_round requires flow.gate")

    return Spec(
        name=raw["name"],
        caps=caps,
        roles=roles,
        flow=flow,
        batch=raw.get("batch", {}),
        cost=raw.get("cost", {}),
        gate_threshold=float(raw.get("gate_threshold", 0.25)),
        tie_policy=raw.get("tie_policy", "flag_disagreement"),
        evidence=raw.get("evidence", {}),
        raw=raw,
    )


def load_spec_file(path: str | Path) -> Spec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return build_spec(raw)


def load_registry(specs_dir: str | Path = SPECS_DIR) -> dict[str, Spec]:
    """Load every *.yaml in specs_dir. Any single invalid spec fails the whole
    load (fail closed — a bad spec must not be silently skipped)."""
    specs: dict[str, Spec] = {}
    for p in sorted(Path(specs_dir).glob("*.yaml")):
        spec = load_spec_file(p)
        if spec.name in specs:
            raise ValueError(f"duplicate spec name '{spec.name}' ({p})")
        specs[spec.name] = spec
    return specs


def get_spec(name: str, specs_dir: str | Path = SPECS_DIR) -> Spec:
    reg = load_registry(specs_dir)
    if name not in reg:
        raise KeyError(f"unknown strategy '{name}'; known: {sorted(reg)}")
    return reg[name]
