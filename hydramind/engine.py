"""
HydraMind engine (design §3.1–§3.2).

    hm = HydraMind.load("hydramind.yaml")          # or .from_specs_dir()
    result, manifest = hm.run(task="classify", items=sents, strategy="single",
                              tune={"roles.solo.tier": "cheap", "batch.min_lot": 100})

The engine owns the wave loop and transport; strategies stay pure. Tuning is
dotted-path overrides into the spec's raw YAML, after which the spec is REBUILT —
so every invariant (I1/I3) is re-checked and a tune can never defeat a guard.
"""
from __future__ import annotations

import copy
from typing import Any, Optional

from .types import TaskBundle, TaskItem, RunState, Spec, StrategyResult
from .registry import load_registry, build_spec
from .transport import Transport, ProxyCompletion
from .manifest import RunManifest, SpendSink, NullSpendSink
from .strategies.single import SingleStrategy
from .strategies.pca import PcaStrategy

STRATEGY_CLASSES = {
    SingleStrategy.name: SingleStrategy,
    PcaStrategy.name: PcaStrategy,
}


def _set_by_path(d: dict, dotted: str, value: Any) -> None:
    cur = d
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


class HydraMind:
    def __init__(self, registry: dict[str, Spec], transport: Transport,
                 spend_sink: Optional[SpendSink] = None, project: str = "hydramind"):
        self.registry = registry
        self.transport = transport
        self.spend_sink = spend_sink or NullSpendSink()
        self.project = project

    @classmethod
    def from_specs_dir(cls, specs_dir=None, transport: Optional[Transport] = None,
                       **kw) -> "HydraMind":
        reg = load_registry(specs_dir) if specs_dir else load_registry()
        transport = transport or Transport(completion_fn=ProxyCompletion())
        return cls(reg, transport, **kw)

    # convenience alias matching the design snippet
    load = from_specs_dir

    def resolve_spec(self, strategy: str, tune: Optional[dict],
                     roster: Optional[str] = None) -> Spec:
        if strategy not in self.registry:
            raise KeyError(f"unknown strategy '{strategy}'")
        raw = copy.deepcopy(self.registry[strategy].raw)
        for path, val in (tune or {}).items():
            _set_by_path(raw, path, val)
        if roster:
            from .rosters import get_roster       # validates roles_allowed + completeness
            raw["roster_resolved"] = get_roster(roster).seats
            raw["roster_name"] = roster
        return build_spec(raw)   # re-validates I1/I3 on the tuned spec

    def explain(self, strategy: str, tune: Optional[dict] = None) -> dict:
        """`hydra explain pca` — resolved spec + knob inventory."""
        spec = self.resolve_spec(strategy, tune)
        return {"resolved_spec": spec.snapshot(),
                "knobs": sorted(_knobs(spec.raw))}

    def run(self, task: str, items: list, strategy: str,
            tune: Optional[dict] = None, rc_id: Optional[str] = None,
            roster: Optional[str] = None
            ) -> tuple[StrategyResult, RunManifest]:
        spec = self.resolve_spec(strategy, tune, roster=roster)
        strat = STRATEGY_CLASSES[strategy]()

        bundle = TaskBundle(task=task, items=[_coerce_item(i) for i in items])
        st = RunState(spec=spec, task=bundle)
        manifest = RunManifest.start(strategy, spec, task, bundle.hash(),
                                     n_items=len(bundle.items), project=self.project)

        ceiling = spec.cost.get("ceiling_usd")
        on_breach = spec.cost.get("on_breach", "halt_and_flag")

        wave = strat.first(bundle, spec)
        while wave:
            results = self.transport.dispatch(wave, spec)
            manifest.record(results)
            st.absorb(results)
            if ceiling is not None and manifest.total_cost_usd > float(ceiling):
                if on_breach == "halt_and_flag":
                    manifest.halted = True
                    manifest.halt_reason = (
                        f"cost ceiling ${ceiling} exceeded "
                        f"(${manifest.total_cost_usd:.4f})")
                    break
            wave = strat.next(st)

        result = strat.reduce(st)
        self.spend_sink.push(manifest.to_spend_records())
        return result, manifest


def _coerce_item(i) -> TaskItem:
    if isinstance(i, TaskItem):
        return i
    if isinstance(i, dict):
        return TaskItem(item_id=str(i["item_id"]), payload=i.get("payload", i),
                        meta=i.get("meta", {}))
    raise TypeError(f"cannot coerce item of type {type(i)}")


def _knobs(raw: dict, prefix="") -> list[str]:
    out = []
    for k, v in raw.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            out += _knobs(v, path + ".")
        else:
            out.append(path)
    return out
