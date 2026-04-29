"""Model panel insights — pure data, no rendering.

Generalizes [`eval/opus_vs_rest_scan.py`](../../../eval/opus_vs_rest_scan.py)
from "Opus vs the rest" into "every model vs the rest". Computes:

* Per-model **dissent rate** against the consensus verdict.
* Per-model **truthy bias** (avg signed distance from the panel mean
  on the Truthy axis; +ve = more lenient than the panel, -ve = more
  strict).
* Per-model **lone-optimist** / **lone-pessimist** counts on the
  same Truthy-axis extreme-split definition the Opus scan uses.
* **Pairwise agreement** matrix entries.
* **Top extreme splits** across all models, deduped by claim text.

The site renderer ([`src/truthbot/publish/site.py`](site.py)
``_render_model_insights``) consumes this output to populate the
landing-page Insights strip + the dedicated ``model-insights.html``
deep-dive page. The eval-side scan retains its own copy of the
threshold + label scoring so the analysis stays runnable without
the publish path; constants are kept in lockstep by
``tests/test_insights.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from statistics import median
from typing import Iterable


# Truthy-axis score per fine label. Higher == more truthy.
# Mirrors LABEL_SCORE in eval/opus_vs_rest_scan.py — kept in lockstep
# by test_insights.test_label_score_matches_opus_scan.
LABEL_SCORE: dict[str, int] = {
    "True":         +2,
    "Mostly True":  +1,
    "Exaggerated":  -1,
    "Misleading":   -1,
    "False":        -2,
    "Unverifiable":  0,
    # Defensive — model_verdicts_summary shouldn't carry the engine's
    # coarse "Models split" label, but tolerate it.
    "Models split":  0,
}

# Diff (in label-score points) at which we call a panel an "extreme split":
#   Opus = Mostly True (+1) vs rest median = False (-2) -> diff = 3.
EXTREME_DIFF_THRESHOLD = 3

# Brand display names. ``_ADAPTER_BRAND`` in site.py uses "grok" for
# xAI, but published claims.json uses "xai" as the adapter key, so
# we keep an insights-local map that handles both.
_ADAPTER_BRAND: dict[str, str] = {
    "anthropic": "Anthropic",
    "openai":    "OpenAI",
    "gemini":    "Google",
    "grok":      "xAI",
    "xai":       "xAI",
}


def _adapter_brand(adapter: str) -> str:
    a = (adapter or "").lower()
    return _ADAPTER_BRAND.get(a, a.title() if a else "Model")


# ── Output dataclasses ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelStat:
    """Per-model summary across the corpus."""
    adapter: str
    pretty_name: str
    claims_seen: int
    dissent_count: int
    dissent_rate: float
    truthy_bias: float
    extreme_lone_optimist: int
    extreme_lone_pessimist: int


@dataclass(frozen=True)
class PairAgreement:
    a: str
    b: str
    agreement_rate: float
    claims_both_present: int


@dataclass(frozen=True)
class ExtremeSplit:
    """One claim where exactly one model is the lone optimist/pessimist."""
    diff: int
    odd_one_out: str          # adapter id, lowercase
    odd_pretty: str
    direction: str            # "optimist" | "pessimist"
    odd_label: str
    other_labels: dict[str, str]
    claim_id: str
    claim_text: str
    speaker: str
    date: str
    claim_url: str
    occurrences: int          # how many duplicate (claim_text, panel) tuples we saw


@dataclass(frozen=True)
class ModelPanelInsights:
    per_model: list[ModelStat]
    pairwise: list[PairAgreement]    # sorted by agreement_rate desc
    top_extreme_splits: list[ExtremeSplit]
    total_claims: int                # distinct claims (deduped by text+date+speaker)

    # Convenience accessors used by the renderer to produce one-line
    # highlight cards on the landing-page Insights strip.

    @property
    def most_divergent(self) -> ModelStat | None:
        if not self.per_model:
            return None
        return max(self.per_model, key=lambda m: m.dissent_rate)

    @property
    def most_lenient(self) -> ModelStat | None:
        if not self.per_model:
            return None
        return max(self.per_model, key=lambda m: m.truthy_bias)

    @property
    def most_strict(self) -> ModelStat | None:
        if not self.per_model:
            return None
        return min(self.per_model, key=lambda m: m.truthy_bias)

    @property
    def top_pair(self) -> PairAgreement | None:
        return self.pairwise[0] if self.pairwise else None


# ── Core computation ─────────────────────────────────────────────────────────


def compute_model_panel_insights(
    claims: Iterable[dict],
    *,
    reports_by_id: dict[str, dict] | None = None,
    extreme_diff_threshold: int = EXTREME_DIFF_THRESHOLD,
    top_splits_limit: int = 10,
) -> ModelPanelInsights:
    """Build a ``ModelPanelInsights`` from a list of claim dicts.

    Each claim dict must follow the shape produced by
    ``SitePublisher._claim_meta`` and persisted to ``claims.json``:

        {
          "id": ..., "report_id": ..., "claim_text": ...,
          "consensus_verdict": "<fine label>",
          "model_verdicts_summary": [
              {"adapter": "anthropic", "label": "Mostly True", "confidence": "High"},
              ...
          ],
          "url": "claims/<uuid>.html",
        }

    ``reports_by_id`` is optional; when provided we backfill speaker/
    date metadata onto each ``ExtremeSplit`` so the renderer can show
    where the split happened without re-querying. Without it, those
    fields fall back to empty strings.
    """
    claims_list = [dict(c) for c in claims]
    reports_by_id = reports_by_id or {}

    # Dedupe by normalized claim text — claims.json carries the same
    # underlying claim text across multiple report runs.
    seen_texts: set[str] = set()
    distinct_claims: list[dict] = []
    for c in claims_list:
        key = (c.get("claim_text") or "").strip().lower()
        if not key or key in seen_texts:
            continue
        seen_texts.add(key)
        distinct_claims.append(c)

    # Per-claim score and label tables.
    per_claim_scores: list[dict[str, int]] = []
    per_claim_labels: list[dict[str, str]] = []
    consensus_labels: list[str] = []
    for c in distinct_claims:
        scores: dict[str, int] = {}
        labels: dict[str, str] = {}
        for mv in c.get("model_verdicts_summary", []):
            adapter = (mv.get("adapter") or "").lower()
            label = mv.get("label") or ""
            if adapter and label and label in LABEL_SCORE:
                scores[adapter] = LABEL_SCORE[label]
                labels[adapter] = label
        per_claim_scores.append(scores)
        per_claim_labels.append(labels)
        consensus_labels.append((c.get("consensus_verdict") or "").strip())

    # ── Per-model aggregates ─────────────────────────────────────────────
    adapter_universe: set[str] = set()
    for s in per_claim_scores:
        adapter_universe.update(s.keys())

    per_model: list[ModelStat] = []
    extreme_split_records: list[ExtremeSplit] = []

    for adapter in sorted(adapter_universe):
        seen = 0
        dissent_count = 0
        bias_accum: list[float] = []
        lone_optimist = 0
        lone_pessimist = 0
        for i, scores in enumerate(per_claim_scores):
            if adapter not in scores:
                continue
            seen += 1
            self_score = scores[adapter]
            self_label = per_claim_labels[i][adapter]
            consensus = consensus_labels[i]
            if consensus and self_label and self_label != consensus:
                dissent_count += 1
            others = [v for a, v in scores.items() if a != adapter]
            if others:
                others_mean = sum(others) / len(others)
                bias_accum.append(self_score - others_mean)
                if len(others) >= 2:
                    others_med = median(others)
                    diff = self_score - others_med
                    if abs(diff) >= extreme_diff_threshold:
                        direction = "optimist" if diff > 0 else "pessimist"
                        if direction == "optimist":
                            lone_optimist += 1
                        else:
                            lone_pessimist += 1
                        # Only record the split if exactly ONE model
                        # is an outlier on this claim — otherwise the
                        # narrative "lone X" is misleading.
                        outliers_this_claim = sum(
                            1
                            for a, sc in scores.items()
                            if a != adapter
                            and abs(sc - median([s for aa, s in scores.items() if aa != a])) >= extreme_diff_threshold
                        )
                        if outliers_this_claim == 0:
                            claim = distinct_claims[i]
                            report = reports_by_id.get(claim.get("report_id", ""), {})
                            extreme_split_records.append(
                                ExtremeSplit(
                                    diff=int(abs(diff)),
                                    odd_one_out=adapter,
                                    odd_pretty=_adapter_brand(adapter),
                                    direction=direction,
                                    odd_label=self_label,
                                    other_labels={
                                        a: per_claim_labels[i][a]
                                        for a in per_claim_labels[i]
                                        if a != adapter
                                    },
                                    claim_id=str(claim.get("id", "")),
                                    claim_text=(claim.get("claim_text") or "").strip(),
                                    speaker=str(
                                        report.get("speaker") or claim.get("speaker") or ""
                                    ),
                                    date=str(report.get("date") or ""),
                                    claim_url=str(claim.get("url") or ""),
                                    occurrences=1,
                                )
                            )
        truthy_bias = (sum(bias_accum) / len(bias_accum)) if bias_accum else 0.0
        per_model.append(
            ModelStat(
                adapter=adapter,
                pretty_name=_adapter_brand(adapter),
                claims_seen=seen,
                dissent_count=dissent_count,
                dissent_rate=(dissent_count / seen) if seen else 0.0,
                truthy_bias=truthy_bias,
                extreme_lone_optimist=lone_optimist,
                extreme_lone_pessimist=lone_pessimist,
            )
        )

    # Sort by sortable identity for stable rendering (vendor-then-model).
    per_model.sort(key=lambda m: (m.pretty_name.lower(), m.adapter))

    # ── Pairwise agreement ───────────────────────────────────────────────
    pairwise_records: list[PairAgreement] = []
    for a, b in combinations(sorted(adapter_universe), 2):
        agree = 0
        seen = 0
        for labels in per_claim_labels:
            if a in labels and b in labels:
                seen += 1
                if labels[a] == labels[b]:
                    agree += 1
        if seen:
            pairwise_records.append(
                PairAgreement(
                    a=a,
                    b=b,
                    agreement_rate=agree / seen,
                    claims_both_present=seen,
                )
            )
    pairwise_records.sort(key=lambda p: (-p.agreement_rate, p.a, p.b))

    # ── Top extreme splits (sorted by diff desc, then occurrences desc) ──
    extreme_split_records.sort(key=lambda e: (-e.diff, -e.occurrences))
    top = extreme_split_records[: top_splits_limit if top_splits_limit > 0 else None]

    return ModelPanelInsights(
        per_model=per_model,
        pairwise=pairwise_records,
        top_extreme_splits=top,
        total_claims=len(distinct_claims),
    )
