"""Schema + content pins for the temporal-regressions data set.

The regression set lives at ``eval/sotu-2026/temporal-regressions.json``
and is the canonical pin for Part A of ``findings-review.md`` (the four
materially-wrong cases the 2026-04 SOTU run published as 'False' when
ground truth lives in the [Truthy, True] band). These tests do NOT
exercise the live verification pipeline — they lock the data file's
schema and content so future contributors can't silently corrupt the
pin.

Operator-only live-run procedure for actually running the 4 cases
through the pipeline lives at
``eval/sotu-2026/temporal-regressions-runbook.md``.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "sotu-2026"))
from temporal_regressions import (  # type: ignore[import]
    REGRESSIONS_PATH,
    RegressionCase,
    case_by_id,
    find_matching_bundle,
    load_temporal_regressions,
)


# ── Schema pins ───────────────────────────────────────────────────────────────


def test_data_file_exists_and_loads() -> None:
    assert REGRESSIONS_PATH.exists()
    metadata, cases = load_temporal_regressions()
    assert metadata["schema_version"] == 2
    assert len(cases) == 4, (
        "Part A in findings-review.md is exactly 4 cases. If you added a "
        "fifth, update both this assertion and the docstring at the top "
        "of temporal-regressions.json."
    )


def test_metadata_carries_provenance_pointers() -> None:
    metadata, _ = load_temporal_regressions()
    assert "_comment" in metadata
    assert "findings-review.md" in metadata["_comment"]
    assert "findings-review.md" in metadata["source_run"]["findings_review"]


def test_each_case_is_a_validated_RegressionCase() -> None:
    _, cases = load_temporal_regressions()
    assert all(isinstance(c, RegressionCase) for c in cases)


def test_loader_raises_on_unknown_strict_label(tmp_path) -> None:
    """Schema guard: reject typos in the canonical Truthy-scale alphabet."""
    import json as _json
    metadata, cases = load_temporal_regressions()
    bad = {
        "schema_version": 2,
        "regressions": [
            {
                "id": "x", "source_run_claim_id": 1, "topic": "t",
                "claim": "c", "published_label_2026_04": "False",
                "verdict": "TRUE",
                "ground_truth_strict": "Mostly True",  # not a Strict label
                "ground_truth_lenient": "Truthy",
                "rationale": "r", "primary_source_pattern": "s",
                "failure_mode": "m",
                "match_keywords": ["x"],
                "test_acceptance": {
                    "fine_label_in": ["True"],
                    "strict_label_in": ["True"],
                    "min_confidence": "Medium",
                },
            }
        ],
    }
    p = tmp_path / "bad.json"
    p.write_text(_json.dumps(bad))
    with pytest.raises(ValueError, match="canonical Strict label"):
        load_temporal_regressions(p)


def test_loader_raises_on_missing_required_field(tmp_path) -> None:
    import json as _json
    bad = {
        "schema_version": 2,
        "regressions": [
            {"id": "x"}  # missing nearly everything
        ],
    }
    p = tmp_path / "bad.json"
    p.write_text(_json.dumps(bad))
    with pytest.raises(ValueError, match="missing required fields"):
        load_temporal_regressions(p)


def test_loader_raises_on_wrong_schema_version(tmp_path) -> None:
    import json as _json
    bad = {"schema_version": 99, "regressions": []}
    p = tmp_path / "bad.json"
    p.write_text(_json.dumps(bad))
    with pytest.raises(ValueError, match="schema_version must be 2"):
        load_temporal_regressions(p)


# ── Content pins (one assertion per Part-A row) ───────────────────────────────


def test_rubio_case_pinned_with_pre_cutoff_failure_mode() -> None:
    """#99 — Rubio 100% confirmation. Pre-cutoff, NOT a temporal-grounding
    case. Failure cause is consensus tie-break / caveat-vs-verdict gap."""
    c = case_by_id("rubio-100-percent-2026")
    assert c.source_run_claim_id == 99
    assert c.published_label_2026_04 == "False"
    assert c.ground_truth_strict == "Truthy"
    assert "NOT a temporal-grounding case" in c.failure_mode
    assert "True" in c.test_acceptance["fine_label_in"]
    assert "Mostly True" in c.test_acceptance["fine_label_in"]


def test_trumprx_case_pinned_with_post_cutoff_dependency() -> None:
    """#109 — TrumpRx.gov. Post-cutoff event; passing depends on the
    trust-when-fired fallback (commit ea10e34) plus working web search."""
    c = case_by_id("trumprx-mfn-2026-02")
    assert c.source_run_claim_id == 109
    assert c.published_label_2026_04 == "False"
    assert c.ground_truth_strict == "True"
    assert "post-cutoff" in c.failure_mode.lower()
    # Requires tools to actually fire on at least 3 of 4 adapters.
    assert c.test_acceptance.get("min_adapters_with_tool_calls") == 3


def test_venezuela_tech_case_allows_falsey_on_thousands_of_soldiers() -> None:
    """#107 — Venezuela raid. Permits Falsey on the 'thousands of soldiers'
    modifier; pinning is that the verdict isn't a flat 'False' that
    dismisses the operation as fictional."""
    c = case_by_id("venezuela-russian-chinese-tech-2026")
    assert c.source_run_claim_id == 107
    assert "Falsey" in c.test_acceptance["strict_label_in"]
    # 'False' must NOT be in the acceptable set.
    assert "False" not in c.test_acceptance["strict_label_in"]
    assert "speculative fiction" in c.failure_mode.lower() or "temporal-dismissal" in c.failure_mode.lower()


def test_helicoide_case_pinned_to_truthy() -> None:
    """#108 — Helicoide closure. Same root cause as #107 (temporal-dismissal);
    ground truth Truthy with no Falsey escape hatch."""
    c = case_by_id("helicoide-prisoner-release-2026")
    assert c.source_run_claim_id == 108
    assert c.ground_truth_strict == "Truthy"
    # No 'Falsey' or 'False' on this one — claim text is more clearly
    # supported than #107.
    assert set(c.test_acceptance["strict_label_in"]) == {"True", "Truthy"}


def test_all_four_cases_published_as_false() -> None:
    """The whole point of this regression set: every case was published
    as 'False' by the 2026-04 run. If a future PR changes any of these
    to a different baseline, the test forces the change to be deliberate."""
    _, cases = load_temporal_regressions()
    assert all(c.published_label_2026_04 == "False" for c in cases), (
        "All four Part A cases were published as 'False' in the "
        "~117-claim 2026-04 SOTU run. If you're changing one, you're "
        "either re-baselining the regression set (do that intentionally) "
        "or you've corrupted the pin."
    )


def test_all_four_cases_have_truthy_or_better_strict_ground_truth() -> None:
    """Every Part A case's ground truth is in the [Truthy, True] band. If
    any case ever needs Falsey/False as ground truth, that's not a
    materially-wrong-label regression any more — it should leave the set."""
    _, cases = load_temporal_regressions()
    for c in cases:
        assert c.ground_truth_strict in {"True", "Truthy"}, (
            f"case {c.id}: ground_truth_strict {c.ground_truth_strict!r} "
            f"is below Truthy — this case no longer fits the Part A "
            f"materially-wrong-label definition."
        )


def test_case_by_id_raises_key_error_on_unknown_id() -> None:
    with pytest.raises(KeyError):
        case_by_id("nope-not-a-case")


# ── match_keywords + find_matching_bundle (schema v2) ────────────────────────
#
# The 2026-05-01 live run of the regression set (run cbc335a1-…) showed the
# runbook's first-30-char substring matcher failed on all 4 cases because the
# extractor split compound sentences and normalized "100%" → "100 percent".
# match_keywords + find_matching_bundle replace the brittle anchor with an
# AND-match on case-insensitive substrings — robust to extractor splits and
# common normalization passes.


def test_each_case_has_non_empty_match_keywords() -> None:
    """Schema v2 requires every regression case to carry a non-empty list of
    match_keywords. Without these the runbook scorer can't reliably find the
    bundle for a case after the extractor splits / rephrases the prompt."""
    _, cases = load_temporal_regressions()
    for c in cases:
        assert c.match_keywords, f"case {c.id} has empty match_keywords"
        assert all(isinstance(k, str) and k for k in c.match_keywords)


def test_loader_raises_when_match_keywords_missing(tmp_path) -> None:
    import json as _json
    bad = {
        "schema_version": 2,
        "regressions": [
            {
                "id": "x", "source_run_claim_id": 1, "topic": "t",
                "claim": "c", "published_label_2026_04": "False",
                "verdict": "TRUE",
                "ground_truth_strict": "True",
                "ground_truth_lenient": "True",
                "rationale": "r", "primary_source_pattern": "s",
                "failure_mode": "m",
                # match_keywords intentionally omitted
                "test_acceptance": {
                    "fine_label_in": ["True"],
                    "strict_label_in": ["True"],
                    "min_confidence": "Medium",
                },
            }
        ],
    }
    p = tmp_path / "bad.json"
    p.write_text(_json.dumps(bad))
    with pytest.raises(ValueError, match="missing required fields"):
        load_temporal_regressions(p)


def test_loader_raises_when_match_keywords_empty(tmp_path) -> None:
    import json as _json
    bad = {
        "schema_version": 2,
        "regressions": [
            {
                "id": "x", "source_run_claim_id": 1, "topic": "t",
                "claim": "c", "published_label_2026_04": "False",
                "verdict": "TRUE",
                "ground_truth_strict": "True",
                "ground_truth_lenient": "True",
                "rationale": "r", "primary_source_pattern": "s",
                "failure_mode": "m",
                "match_keywords": [],
                "test_acceptance": {
                    "fine_label_in": ["True"],
                    "strict_label_in": ["True"],
                    "min_confidence": "Medium",
                },
            }
        ],
    }
    p = tmp_path / "bad.json"
    p.write_text(_json.dumps(bad))
    with pytest.raises(ValueError, match="match_keywords must be a non-empty list"):
        load_temporal_regressions(p)


def test_find_matching_bundle_handles_extractor_normalization() -> None:
    """The 2026-05-01 live run produced bundles with claim text like:
    'Marco Rubio received 100 percent of Senate confirmation votes...'
    even though the runbook's mini-transcript wrote '100%'. The Rubio case
    keywords ['rubio', '100', 'confirm'] should match this normalized text
    without any string-equality."""
    case = case_by_id("rubio-100-percent-2026")
    bundles = [
        {"claim": {"text": "Some unrelated claim about inflation."}},
        {"claim": {"text": "Marco Rubio received 100 percent of Senate confirmation votes."}},
        {"claim": {"text": "Trump announced TrumpRx.gov in February 2026."}},
    ]
    match = find_matching_bundle(case, bundles)
    assert match is not None
    assert "Rubio" in match["claim"]["text"]


def test_find_matching_bundle_handles_compound_claim_splits() -> None:
    """The TrumpRx pin's keywords ['trumprx', 'mfn'] should match the
    extractor's split-out 'White House announced TrumpRx.gov ... MFN ...'
    fragment even when the original mini-transcript bundled additional
    qualifiers that were extracted into separate claims."""
    case = case_by_id("trumprx-mfn-2026-02")
    bundles = [
        {"claim": {"text": "White House announced TrumpRx.gov in February 2026 with MFN drug pricing."}},
        {"claim": {"text": "Drug-pricing reform was discussed."}},  # no trumprx
    ]
    match = find_matching_bundle(case, bundles)
    assert match is not None
    assert "TrumpRx" in match["claim"]["text"]


def test_find_matching_bundle_returns_none_when_no_match() -> None:
    case = case_by_id("helicoide-prisoner-release-2026")
    bundles = [{"claim": {"text": "Something completely unrelated."}}]
    assert find_matching_bundle(case, bundles) is None


def test_find_matching_bundle_is_case_insensitive() -> None:
    case = case_by_id("venezuela-russian-chinese-tech-2026")
    bundles = [
        {"claim": {"text": "VENEZUELA OPERATION INVOLVED RUSSIAN MILITARY EQUIPMENT."}},
    ]
    match = find_matching_bundle(case, bundles)
    assert match is not None


def test_find_matching_bundle_requires_all_keywords() -> None:
    """AND-match: every keyword must appear. A bundle with only one keyword
    of the case must NOT be returned (prevents Helicoide from matching a
    Venezuela bundle that mentions 'prisoner' but not 'helicoide')."""
    case = case_by_id("helicoide-prisoner-release-2026")
    # Only the 'helicoide' keyword exists in the case (single-keyword case);
    # confirm narrow-keyword case works correctly.
    bundles = [
        {"claim": {"text": "Hundreds of political prisoners were released."}},  # missing helicoide
    ]
    assert find_matching_bundle(case, bundles) is None
