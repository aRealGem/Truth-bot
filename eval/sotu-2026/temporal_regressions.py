"""Loader + schema validator for the temporal-regressions pin set.

The data lives in ``eval/sotu-2026/temporal-regressions.json``; this
module is the canonical entry point for tools and tests that need to
read it. Schema is intentionally narrow — we want a future contributor
who adds a new regression case to fail loudly if any required field is
missing or any label is outside the canonical Truthy-scale alphabet.

See :doc:`temporal-regressions-runbook.md` for the operator-only
live-run procedure (cached HTML cannot reproduce the OpenAI / Gemini
temporal-dismissal failure mode).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REGRESSIONS_PATH = Path(__file__).parent / "temporal-regressions.json"

# Canonical 5-bucket Strict / Lenient labels (Truthy scale). Mirror of
# COARSE_VERDICT_ORDER in src/truthbot/verify/engine.py.
_STRICT_LABELS = {"True", "Truthy", "Models split", "Falsey", "False"}
_LENIENT_LABELS = {"True", "Truthy", "Models split", "Falsey", "False"}

# Fine-axis labels we expect in test_acceptance.fine_label_in.
_FINE_LABELS = {
    "True", "Mostly True", "Exaggerated", "Misleading",
    "Unverifiable", "False", "Models split",
}

# Confidence buckets the engine emits.
_CONFIDENCE_BUCKETS = {"Low", "Medium", "High"}


@dataclass(frozen=True)
class RegressionCase:
    id: str
    source_run_claim_id: int
    topic: str
    claim: str
    published_label_2026_04: str
    verdict: str
    ground_truth_strict: str
    ground_truth_lenient: str
    rationale: str
    primary_source_pattern: str
    failure_mode: str
    test_acceptance: dict


def _validate_case(raw: dict) -> RegressionCase:
    required = {
        "id", "source_run_claim_id", "topic", "claim",
        "published_label_2026_04", "verdict",
        "ground_truth_strict", "ground_truth_lenient",
        "rationale", "primary_source_pattern", "failure_mode",
        "test_acceptance",
    }
    missing = required - raw.keys()
    if missing:
        raise ValueError(
            f"temporal-regressions case {raw.get('id', '<unknown>')} "
            f"missing required fields: {sorted(missing)}"
        )

    if raw["ground_truth_strict"] not in _STRICT_LABELS:
        raise ValueError(
            f"case {raw['id']}: ground_truth_strict "
            f"{raw['ground_truth_strict']!r} is not a canonical Strict "
            f"label (allowed: {sorted(_STRICT_LABELS)})"
        )
    if raw["ground_truth_lenient"] not in _LENIENT_LABELS:
        raise ValueError(
            f"case {raw['id']}: ground_truth_lenient "
            f"{raw['ground_truth_lenient']!r} is not a canonical Lenient "
            f"label (allowed: {sorted(_LENIENT_LABELS)})"
        )

    acceptance = raw["test_acceptance"]
    if not isinstance(acceptance, dict):
        raise ValueError(
            f"case {raw['id']}: test_acceptance must be a dict"
        )
    for required_acceptance in ("fine_label_in", "strict_label_in", "min_confidence"):
        if required_acceptance not in acceptance:
            raise ValueError(
                f"case {raw['id']}: test_acceptance missing "
                f"{required_acceptance!r}"
            )
    if acceptance["min_confidence"] not in _CONFIDENCE_BUCKETS:
        raise ValueError(
            f"case {raw['id']}: min_confidence "
            f"{acceptance['min_confidence']!r} is not in "
            f"{sorted(_CONFIDENCE_BUCKETS)}"
        )
    bad_fine = set(acceptance["fine_label_in"]) - _FINE_LABELS
    if bad_fine:
        raise ValueError(
            f"case {raw['id']}: fine_label_in has unknown labels "
            f"{sorted(bad_fine)}"
        )
    bad_strict = set(acceptance["strict_label_in"]) - _STRICT_LABELS
    if bad_strict:
        raise ValueError(
            f"case {raw['id']}: strict_label_in has unknown labels "
            f"{sorted(bad_strict)}"
        )

    return RegressionCase(
        id=raw["id"],
        source_run_claim_id=raw["source_run_claim_id"],
        topic=raw["topic"],
        claim=raw["claim"],
        published_label_2026_04=raw["published_label_2026_04"],
        verdict=raw["verdict"],
        ground_truth_strict=raw["ground_truth_strict"],
        ground_truth_lenient=raw["ground_truth_lenient"],
        rationale=raw["rationale"],
        primary_source_pattern=raw["primary_source_pattern"],
        failure_mode=raw["failure_mode"],
        test_acceptance=acceptance,
    )


def load_temporal_regressions(
    path: Path | None = None,
) -> tuple[dict[str, Any], list[RegressionCase]]:
    """Load and validate the regression pin set.

    Returns a ``(metadata, cases)`` tuple where ``metadata`` is the JSON
    object minus the ``regressions`` array (so consumers can read
    ``schema_version`` / ``source_run`` / ``_comment``) and ``cases`` is
    the validated list. Raises ``ValueError`` on schema violations so
    CI catches typos before they reach a live run.
    """
    p = path or REGRESSIONS_PATH
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"{p} top-level must be a JSON object, got {type(raw).__name__}"
        )
    if "regressions" not in raw or not isinstance(raw["regressions"], list):
        raise ValueError(f"{p} missing 'regressions' list")
    if raw.get("schema_version") != 1:
        raise ValueError(
            f"{p} schema_version must be 1 "
            f"(got {raw.get('schema_version')!r})"
        )
    cases = [_validate_case(c) for c in raw["regressions"]]
    metadata = {k: v for k, v in raw.items() if k != "regressions"}
    return metadata, cases


def case_by_id(case_id: str, *, path: Path | None = None) -> RegressionCase:
    _, cases = load_temporal_regressions(path)
    for c in cases:
        if c.id == case_id:
            return c
    raise KeyError(case_id)
