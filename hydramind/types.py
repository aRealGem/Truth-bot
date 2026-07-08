"""
HydraMind L2 — core types (design §3.1).

A *wave* is a set of independent single completions. Strategies define topology
(which roles, in what waves, gated how) and reduction; the engine executes waves
and owns transport-lane choice. Because every wave element is an independent
completion, batch eligibility is a property of the wave, not the call.
"""
from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, NamedTuple, Optional, Protocol, runtime_checkable


class Kind(enum.Enum):
    COMPLETION = "completion"   # routine completion  -> L-P / L-B
    TOOL_TASK = "tool_task"     # agentic executor    -> L-W (on hold in C1)


class Cap(enum.Enum):
    BATCH = "batch"
    MULTI_ROUND = "multi_round"
    PERSONA = "persona"


class Lane(enum.Enum):
    L_P = "L-P"   # proxy single completions (LiteLLM, routine only)
    L_B = "L-B"   # native provider batch
    L_T = "L-T"   # native provider tools (web_search) — stub in C1
    L_W = "L-W"   # Claude Code worker (Max sub) — on hold in C1


class ModelBinding(NamedTuple):
    """A resolved provider+model after pool/rotation rules are applied."""
    provider: str
    model: str
    tier: str                      # "cheap" | "standard" | "frontier"


class PromptRef(NamedTuple):
    """A versioned template reference. `template` is the raw text; `version` a
    content hash so the run manifest can pin exactly what was sent."""
    name: str
    version: str
    template: str

    @staticmethod
    def of(name: str, template: str) -> "PromptRef":
        v = hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]
        return PromptRef(name=name, version=v, template=template)


class Call(NamedTuple):
    role: str                       # "proposer" | "critic" | "arbiter" | "solo" | ...
    item_id: str                    # element of the task bundle
    prompt: PromptRef               # versioned template
    binding: ModelBinding           # resolved provider+model (post pool rules)
    kind: Kind = Kind.COMPLETION
    # rendered prompt vars / payload (never speaker-conditional). Defaults to
    # None (not {}) to avoid a shared-mutable NamedTuple default; read via
    # `call.inputs or {}`.
    inputs: Optional[dict] = None


class Wave(NamedTuple):
    calls: list[Call]
    batchable: bool                 # strategy declares; engine still checks min_lot
    tag: str = ""                   # e.g. "wave1", "wave2" — for lane eligibility


# ── task / result payloads ────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskItem:
    """One unit under analysis. `payload` carries the content; `meta` carries
    non-conditioning metadata (speaker, source) that machinery MUST NOT branch on."""
    item_id: str
    payload: dict
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TaskBundle:
    task: str                       # "classify" | "verdict" | ...
    items: list[TaskItem]

    def hash(self) -> str:
        blob = json.dumps(
            [(i.item_id, i.payload) for i in self.items],
            sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


@dataclass
class CallResult:
    call: Call
    output: dict                    # parsed model output
    lane: Lane
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    returned_model: str = ""     # model id reported by the provider/proxy (fallback detection)
    raw: Any = None
    cost_source: str = "none"    # provenance of cost_usd: "proxy" | "table" | "none"


class StrategyResultKind(enum.Enum):
    RESOLVED = "resolved"
    DISAGREEMENT_FLAGGED = "disagreement_flagged"   # I2 — material tie


@dataclass
class ItemResult:
    item_id: str
    kind: StrategyResultKind
    value: dict                     # e.g. {"label": "...", ...} or {"verdict": "...", "citations": [...]}
    agreement: dict = field(default_factory=dict)   # agreement stats (Principle 5)


@dataclass
class StrategyResult:
    items: list[ItemResult]
    notes: dict = field(default_factory=dict)


# ── Spec (resolved YAML) ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class RoleSpec:
    tier: str
    providers: tuple[str, ...]
    rotation: Optional[str] = None       # e.g. "round_robin" for arbiter


@dataclass(frozen=True)
class Spec:
    """A resolved, immutable strategy spec (loaded + validated from YAML)."""
    name: str
    caps: frozenset[Cap]
    roles: dict[str, RoleSpec]
    flow: dict[str, Any]
    batch: dict[str, Any]
    cost: dict[str, Any] = field(default_factory=dict)
    gate_threshold: float = 0.25
    tie_policy: str = "flag_disagreement"
    evidence: dict[str, Any] = field(default_factory=dict)
    raw: dict = field(default_factory=dict)   # original dict, for manifest snapshot

    def snapshot(self) -> dict:
        """Deterministic dict for the run manifest."""
        return json.loads(json.dumps(self.raw, sort_keys=True))


# ── RunState ──────────────────────────────────────────────────────────────────

@dataclass
class RunState:
    spec: Spec
    task: TaskBundle
    results: list[CallResult] = field(default_factory=list)
    round: int = 0
    scratch: dict = field(default_factory=dict)     # strategy-private working memory

    def absorb(self, call_results: list[CallResult]) -> None:
        self.results.extend(call_results)
        self.round += 1

    def by_item(self, role: Optional[str] = None) -> dict[str, list[CallResult]]:
        out: dict[str, list[CallResult]] = {}
        for r in self.results:
            if role is not None and r.call.role != role:
                continue
            out.setdefault(r.call.item_id, []).append(r)
        return out


@runtime_checkable
class Strategy(Protocol):
    name: ClassVar[str]
    caps: ClassVar[frozenset[Cap]]

    def first(self, task: TaskBundle, spec: Spec) -> Wave: ...
    def next(self, st: RunState) -> Optional[Wave]: ...      # None ⇒ done
    def reduce(self, st: RunState) -> StrategyResult: ...
