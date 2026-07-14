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
                full_speech: bool = False, confirm_pass: bool = True) -> LayerAResult:
    """sentences: [{"sid","text","context"}].

    A1 lexical prefilter routes each sentence:
      - DROP  → characterization (never reaches A2; cheap, obvious non-claims).
      - PASS/AMBIGUOUS → A2 (the LLM decides check-worthy/opinion/unimportant).

    `confirm_pass` (default True) is the fix for the 2026-07-13 finding that A1-PASS
    was going STRAIGHT to the check-worthy queue with no LLM review — so A1's lexical
    false positives (e.g. a "we should…" proposal, or a ceremonial truism) reached the
    expensive PCA panel unchecked. With confirm_pass, A2 confirms/vetoes the PASS band
    too; set it False to restore the old A1-PASS→queue shortcut. `full_speech=True`
    sends everything to A2 (calibration). No-lane (classify_fn=None): PASS falls back to
    the queue (A1's call), AMBIGUOUS parks as `needs_a2`."""
    res = LayerAResult()
    to_a2: list[dict] = []
    pass_sids: set[str] = set()

    for s in sentences:
        r = prefilter.route(s["sid"], s["text"], tau_low, tau_high)
        res.a1_routes[s["sid"]] = r.bucket
        if full_speech:
            to_a2.append(s)
            continue
        if r.bucket == "pass":
            if confirm_pass:
                pass_sids.add(s["sid"])
                to_a2.append({**s, "a1_score": r.score})
            else:
                res.check_worthy_queue.append({**s, "label": "check-worthy",
                                               "source": "A1", "a1_score": r.score})
        elif r.bucket == "drop":
            res.characterization_stream.append({**s, "label": "non-check-worthy",
                                                "source": "A1", "a1_score": r.score})
        else:
            to_a2.append({**s, "a1_score": r.score})

    res.n_to_a2 = len(to_a2)
    if not to_a2:
        return res

    if classify_fn is None:
        for s in to_a2:
            if s["sid"] in pass_sids:      # no lane → fall back to A1's PASS decision
                res.check_worthy_queue.append({**s, "label": "check-worthy", "source": "A1"})
            else:
                res.characterization_stream.append({**s, "label": "needs_a2", "source": "A1"})
        return res

    for verdict in classify_fn(to_a2):
        # a1_pass=True marks a row A1 would have auto-passed; when A2 sends it to
        # characterization, that is A2 vetoing an A1 lexical false positive.
        row = {**verdict, "source": "A2", "a1_pass": verdict["sid"] in pass_sids}
        if verdict["label"] == "check-worthy":
            res.check_worthy_queue.append(row)
        else:
            res.characterization_stream.append(row)
    return res
