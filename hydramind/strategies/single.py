"""
`single` strategy — one cheap solo completion per item, one wave. This is the
strategy truth-bot v2 Layer A (A2 classifier) rides.
"""
from __future__ import annotations

from typing import Optional

from ..types import (
    Call, Wave, TaskBundle, RunState, Spec, PromptRef, Cap,
    StrategyResult, ItemResult, StrategyResultKind,
)
from .base import resolve_binding

# The prompt template is injected by the caller (e.g. the A2 classifier) via
# spec.raw["prompt"] or task item payloads; the strategy stays task-agnostic.
_DEFAULT_TEMPLATE = "{input}"


class SingleStrategy:
    name = "single"
    caps = frozenset({Cap.BATCH})

    def first(self, task: TaskBundle, spec: Spec) -> Wave:
        role_spec = spec.roles["solo"]
        template = spec.raw.get("prompt", _DEFAULT_TEMPLATE)
        prompt = PromptRef.of(f"{spec.name}:solo", template)
        calls = [
            Call(
                role="solo",
                item_id=item.item_id,
                prompt=prompt,
                binding=resolve_binding(role_spec),
                inputs=item.payload,
            )
            for item in task.items
        ]
        return Wave(calls=calls, batchable=True, tag="wave1")

    def next(self, st: RunState) -> Optional[Wave]:
        return None  # single wave, done after first

    def reduce(self, st: RunState) -> StrategyResult:
        items = []
        for item_id, results in st.by_item(role="solo").items():
            out = results[0].output
            items.append(ItemResult(
                item_id=item_id,
                kind=StrategyResultKind.RESOLVED,
                value=out,
            ))
        return StrategyResult(items=items)
