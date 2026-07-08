"""
Layer A pipeline (spec §4): A1 lexical prefilter → (ambiguous band) → A2 classifier,
into two sinks. `check-worthy` → Layer B queue; everything else → characterization
stream WITH its speech-act label (Principle 4 — the characterization layer is the
product, not a discard pile).

Speaker-blind: neither A1 nor A2 reads speaker/source metadata (I3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from . import prefilter

# A2 classify fn signature: (list[{"sid","text","context"}]) -> list[{"sid","label",...}]
ClassifyFn = Callable[[list[dict]], list[dict]]


@dataclass
class LayerAResult:
    check_worthy_queue: list[dict] = field(default_factory=list)      # → Layer B
    characterization_stream: list[dict] = field(default_factory=list)  # → product
    a1_routes: dict = field(default_factory=dict)                     # sid -> bucket
    n_to_a2: int = 0


def run_layer_a(sentences: list[dict], classify_fn: Optional[ClassifyFn] = None,
                tau_low: float = 0.45, tau_high: float = 0.70,
                full_speech: bool = False) -> LayerAResult:
    """sentences: [{"sid","text","context"}].

    Steady state: only the A1 AMBIGUOUS band reaches A2. `full_speech=True` sends
    everything to A2 (calibration runs). A1 PASS routes straight to the
    check-worthy queue; A1 DROP routes to characterization. If classify_fn is
    None (no live lane), ambiguous items are parked as `needs_a2`."""
    res = LayerAResult()
    to_a2: list[dict] = []

    for s in sentences:
        r = prefilter.route(s["sid"], s["text"], tau_low, tau_high)
        res.a1_routes[s["sid"]] = r.bucket
        if full_speech:
            to_a2.append(s)
            continue
        if r.bucket == "pass":
            res.check_worthy_queue.append({**s, "label": "check-worthy",
                                           "source": "A1", "a1_score": r.score})
        elif r.bucket == "drop":
            res.characterization_stream.append({**s, "label": "non-check-worthy",
                                                "source": "A1", "a1_score": r.score})
        else:
            to_a2.append(s)

    res.n_to_a2 = len(to_a2)
    if not to_a2:
        return res

    if classify_fn is None:
        for s in to_a2:
            res.characterization_stream.append({**s, "label": "needs_a2", "source": "A1"})
        return res

    for verdict in classify_fn(to_a2):
        row = {**verdict, "source": "A2"}
        if verdict["label"] == "check-worthy":
            res.check_worthy_queue.append(row)
        else:
            res.characterization_stream.append(row)
    return res
