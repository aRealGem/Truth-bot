#!/usr/bin/env python3
"""One-off scan over ``site-test/data/claims.json`` for claims where the
Anthropic verdict (Claude Opus 4.7) diverges sharply from the rest of
the frontier panel.

Motivation: user noted on 2026-04-29 that Opus sometimes votes True (or
Mostly True) on claims where ≥2 other frontier models vote False — and
the inverse. That's an "extreme split" and worth a closer look because
it likely points at one of:

  * A real subtle factual issue Opus is catching that the others miss
    (or vice versa).
  * Prompt cache divergence on a particular claim shape.
  * A specific evidence-grounding pattern (e.g. Anthropic
    ``model_reported_sources`` empty + tool retrieval hit different
    domains than the rest).
  * The Strict-vs-Lenient mapping bouncing certain claims around the
    Truthy/Falsey divide unevenly across providers.

Output: ``eval/opus_vs_rest_extreme_splits.md``.

This is pure analysis (no LLM calls, no code path under test).
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_JSON = ROOT / "site-test" / "data" / "claims.json"
REPORTS_JSON = ROOT / "site-test" / "data" / "reports.json"
OUT_MD = Path(__file__).parent / "opus_vs_rest_extreme_splits.md"

# Truthy-axis score per fine label. Higher == more truthy.
LABEL_SCORE: dict[str, int] = {
    "True":         +2,
    "Mostly True":  +1,
    "Exaggerated":  -1,
    "Misleading":   -1,
    "False":        -2,
    "Unverifiable":  0,
    # Defensive: model_verdicts shouldn't carry the engine's coarse
    # "Models split" label, but tolerate it.
    "Models split":  0,
}

# Difference on the truthy axis that qualifies as "extreme".
# Opus = Mostly True (+1) vs rest median = False (-2) -> diff = 3.
# Opus = True (+2)        vs rest median = False (-2) -> diff = 4.
EXTREME_DIFF_THRESHOLD = 3


def _load_claims() -> list[dict]:
    raw = json.loads(CLAIMS_JSON.read_text())
    return raw if isinstance(raw, list) else raw.get("claims", [])


def _load_report_index() -> dict[str, dict]:
    raw = json.loads(REPORTS_JSON.read_text())
    reports = raw if isinstance(raw, list) else raw.get("reports", [])
    return {r["id"]: r for r in reports}


def _classify(claim: dict) -> tuple[int, dict[str, str], dict[str, int], int, int] | None:
    """Compute (diff, label_by_adapter, score_by_adapter, opus_score, rest_median)
    for a claim, or None if Anthropic isn't on the panel."""
    by_adapter_label: dict[str, str] = {}
    by_adapter_score: dict[str, int] = {}
    for mv in claim.get("model_verdicts_summary", []):
        adapter = (mv.get("adapter") or "").lower()
        label = mv.get("label") or ""
        if adapter and label and label in LABEL_SCORE:
            by_adapter_label[adapter] = label
            by_adapter_score[adapter] = LABEL_SCORE[label]
    if "anthropic" not in by_adapter_score:
        return None
    rest_scores = [s for a, s in by_adapter_score.items() if a != "anthropic"]
    if len(rest_scores) < 2:
        return None  # not enough peers to call it a split
    opus = by_adapter_score["anthropic"]
    rest = int(median(rest_scores))
    return abs(opus - rest), by_adapter_label, by_adapter_score, opus, rest


def main() -> None:
    claims = _load_claims()
    reports = _load_report_index()

    # Dedupe by normalized claim text — claims.json carries the same
    # underlying claim text across multiple SOTU 2026 report runs (each
    # extraction generates fresh claim IDs even when the wording is
    # identical), so without dedup the same Opus-vs-rest split shows up
    # 4-5× and crowds out genuinely distinct splits.
    extreme_by_text: dict[str, tuple[int, dict, dict[str, str], int, int, int]] = {}
    total_with_opus = 0
    seen_texts: set[str] = set()
    for c in claims:
        out = _classify(c)
        if out is None:
            continue
        text_norm = (c.get("claim_text") or "").strip().lower()
        if text_norm not in seen_texts:
            seen_texts.add(text_norm)
            total_with_opus += 1
        diff, labels, _scores, opus, rest = out
        if diff >= EXTREME_DIFF_THRESHOLD:
            prev = extreme_by_text.get(text_norm)
            occurrences = (prev[5] if prev else 0) + 1
            if prev is None or diff > prev[0]:
                extreme_by_text[text_norm] = (diff, c, labels, opus, rest, occurrences)
            else:
                extreme_by_text[text_norm] = (
                    prev[0], prev[1], prev[2], prev[3], prev[4], occurrences
                )

    extreme = sorted(
        extreme_by_text.values(),
        key=lambda x: (-x[0], -x[5]),
    )

    # Direction tally: Opus more truthy vs Opus more falsey.
    opus_more_truthy = sum(1 for d, _, _, op, rest, _occ in extreme if op > rest)
    opus_more_falsey = sum(1 for d, _, _, op, rest, _occ in extreme if op < rest)

    lines: list[str] = []
    lines.append("# Opus vs the rest — extreme-split scan")
    lines.append("")
    lines.append("**Source:** [`site-test/data/claims.json`](../site-test/data/claims.json) · "
                 f"**Generated:** {OUT_MD.parent.parent.name}/`{OUT_MD.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("Each fine-axis verdict label is mapped to a Truthy-axis score:")
    lines.append("")
    lines.append("| Label | Score |")
    lines.append("|-------|------:|")
    for label, score in sorted(LABEL_SCORE.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {label} | {score:+d} |")
    lines.append("")
    lines.append(
        f"For each claim with Anthropic on the panel and at least two other "
        f"adapters, we compute `|opus_score - median(rest_scores)|`. A claim "
        f"is **extreme** when that diff ≥ {EXTREME_DIFF_THRESHOLD} points "
        f"(roughly: Opus calls it Mostly True/True while ≥half of the rest "
        f"call it False/Misleading, or vice versa)."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    pct = (len(extreme) / total_with_opus * 100) if total_with_opus else 0.0
    lines.append(f"* **{len(extreme)} distinct extreme splits** out of "
                 f"**{total_with_opus} distinct claims** with Anthropic on "
                 f"the panel ({pct:.1f}%).")
    lines.append(f"* Opus is the **more-truthy** voice in **{opus_more_truthy}** of those splits.")
    lines.append(f"* Opus is the **more-falsey** voice in **{opus_more_falsey}** of those splits.")
    lines.append("")
    if opus_more_truthy >= 3 * max(opus_more_falsey, 1):
        lines.append(
            "> **Asymmetry note.** Opus is the lone optimist roughly "
            f"{opus_more_truthy}× more often than the lone pessimist. That "
            "matches the user's 2026-04-29 hunch and is consistent with "
            "Claude's tendency toward charitable interpretation of partisan "
            "claims; worth a closer prompt-engineering look if it persists "
            "across speakers/topics."
        )
        lines.append("")
    lines.append("")
    if not extreme:
        lines.append("_No claims hit the extreme-split threshold. Either Opus is well-calibrated "
                     "with the rest of the panel here, or the threshold needs tightening._")
    else:
        lines.append("## Top splits (deduped by claim text, sorted by magnitude then recurrence)")
        lines.append("")
        for i, (diff, claim, labels, opus, rest, occurrences) in enumerate(extreme, 1):
            report_id = claim.get("report_id", "")
            report = reports.get(report_id, {})
            speaker = report.get("speaker") or claim.get("speaker") or ""
            date    = report.get("date") or ""
            text    = (claim.get("claim_text") or "").replace("\n", " ").strip()
            url     = claim.get("url") or ""
            occurrences_note = (
                f" · seen in **{occurrences}** report runs of this speech"
                if occurrences > 1 else ""
            )
            lines.append(f"### {i}. diff = {diff} — Opus says **{labels.get('anthropic')}**{occurrences_note}")
            lines.append("")
            lines.append(f"_{speaker} · {date}_")
            lines.append("")
            lines.append(f"> {text}")
            lines.append("")
            label_table = " | ".join(
                f"**{a}**: {labels[a]}" for a in ["anthropic", "openai", "gemini", "xai"]
                if a in labels
            )
            lines.append(label_table)
            lines.append("")
            if url:
                lines.append(f"[Open claim page](../site-test/{url})")
                lines.append("")
            lines.append("---")
            lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "* This is **pure label scoring** — it doesn't read the per-model "
        "explanations. A diff-3 split where Opus has a citation the others "
        "missed is qualitatively different from a diff-3 split where Opus "
        "is over-charitable. Spot-check the worst offenders by hand."
    )
    lines.append(
        "* Median-of-rest is robust to a single outlier on the rest side, "
        "but with only 3 peers (OpenAI + Gemini + xAI) one outlier still "
        "shifts the median noticeably. Read the row of labels, not just the "
        "diff."
    )
    lines.append(
        "* The reference set in [`eval/sotu-2026/reference.json`](sotu-2026/"
        "reference.json) is the only ground truth we trust; if a flagged "
        "split lines up with a reference claim, cross-check there before "
        "concluding which side is wrong."
    )
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print(f"[opus-scan] wrote {OUT_MD} — {len(extreme)} extreme splits / "
          f"{total_with_opus} claims with Anthropic")


if __name__ == "__main__":
    main()
