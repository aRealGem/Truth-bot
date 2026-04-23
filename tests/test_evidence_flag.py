"""--no-inject-evidence: user message omits evidence snippets."""

from __future__ import annotations

from truthbot.models import Claim, Evidence, SourceTier
from truthbot.verify.adapters.base import build_user_message


def test_build_user_message_no_inject_skips_snippets(sample_claim, sample_evidence):
    msg = build_user_message(sample_claim, [sample_evidence], inject_evidence=False)
    assert sample_evidence.snippet not in msg
    assert "No pre-gathered evidence was supplied" in msg


def test_build_user_message_inject_includes_snippet(sample_claim, sample_evidence):
    msg = build_user_message(sample_claim, [sample_evidence], inject_evidence=True)
    assert "BLS" in msg
    assert sample_evidence.snippet[:20] in msg


def test_build_user_message_inject_empty_evidence(sample_claim):
    msg = build_user_message(sample_claim, [], inject_evidence=True)
    assert "No pre-gathered evidence available" in msg


def test_evidence_tier_in_message(sample_claim):
    ev = Evidence(
        claim_id=sample_claim.id,
        source_name="Gov",
        source_url="https://gov/x",
        source_tier=SourceTier.GOVERNMENT,
        snippet="data",
    )
    msg = build_user_message(sample_claim, [ev], inject_evidence=True)
    assert "Government" in msg
