"""
Strategy base — shared helpers. The `Strategy` protocol itself lives in
hydramind.types; concrete strategies (single, pca) implement first/next/reduce.
"""
from __future__ import annotations

from ..types import ModelBinding, RoleSpec
from ..models import binding_for


def resolve_binding(role_spec: RoleSpec, rotation_index: int = 0) -> ModelBinding:
    """Pick a concrete provider+model for a role.

    - rotation == "round_robin": rotate across the pool by rotation_index
      (used by the arbiter so gated items spread across frontier providers).
    - otherwise: the first provider in the pool is the decided default
      (e.g. critic → mistral, the decided cross-vendor Western critic).
    """
    providers = role_spec.providers
    if not providers:
        raise ValueError("role has empty provider pool")
    if role_spec.rotation == "round_robin":
        provider = providers[rotation_index % len(providers)]
    else:
        provider = providers[0]
    return binding_for(provider, role_spec.tier)
