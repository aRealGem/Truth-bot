"""Run the build-time consistency checker over the committed ``site-pca/``
tree so hand-typed or drifted figures cannot merge (T0.8; this is the test the
``consistency.py`` docstring has promised since P67.4 — added in PR-A2.0
together with the distribution-sum invariants).

The remediation-v2 strict lints (index Sources-chip buckets incl. the
political tier, bucket-sum invariants) run against a FRESH render below —
the committed tree predates the remediation regeneration (its cards were
rendered without the political bucket), so it is linted with
``strict_buckets=False``; the Phase-2 regen flips it to True."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from truthbot.models import (Claim, Confidence, ConsensusVerdict,
                             ModelVerdict, VerdictBundle, VerdictLabel)
from truthbot.publish.consistency import check_site
from truthbot.publish.site import SitePublisher, SiteReport

_SITE = Path(__file__).resolve().parent.parent / "site-pca"


@pytest.mark.skipif(not (_SITE / "data" / "reports.json").exists(),
                    reason="site-pca tree not present")
def test_committed_site_has_no_consistency_violations() -> None:
    # committed tree predates remediation regen; Phase-2 regen flips this
    # to strict_buckets=True (and deletes the flag once the tree is fresh).
    violations = check_site(_SITE, strict_buckets=False)
    assert violations == [], "\n".join(violations)


# ── Fresh-render companion: the strict lints PASS on new output ──────────────


def _bundle(fine: VerdictLabel, coarse_lenient: str, coarse_strict: str,
            verdict: str | None = None,
            urls: list[str] | None = None) -> VerdictBundle:
    claim = Claim(
        transcript_id="t",
        text=f"Synthetic claim {uuid.uuid4().hex[:8]}.",
        speaker="Synthetic Speaker",
        context="",
        category="economy",
        is_checkable=True,
    )
    mvs = [
        ModelVerdict(
            adapter_name="pca",
            model_id="reconciled",
            claim_id=claim.id,
            label=fine,
            confidence=Confidence.HIGH,
            explanation="Synthetic reasoning.",
            web_sources=list(urls or []),
        )
    ]
    consensus = ConsensusVerdict(
        claim_id=claim.id,
        model_verdicts=mvs,
        consensus_label=fine,
        consensus_verdict=verdict or fine.value,
        confidence=Confidence.HIGH,
        agreement=True,
        consensus_strength="strong",
        explanation="Synthetic.",
        coarse_lenient_label=coarse_lenient,
        coarse_lenient_strength="strong",
        coarse_strict_label=coarse_strict,
        coarse_strict_strength="strong",
    )
    return VerdictBundle(
        claim=claim,
        speaker="Synthetic Speaker",
        date_str="2026-03-04",
        model_verdicts=mvs,
        consensus=consensus,
    )


def test_fresh_render_passes_strict_lints(tmp_path) -> None:
    """Render ONE report end-to-end into a temp site root and assert the
    FULL strict check_site pass (political Sources bucket included) is
    clean — this is the invariant the Phase-2 regeneration must meet before
    the committed-tree test above can flip to strict_buckets=True."""
    bundles = [
        _bundle(VerdictLabel.TRUE, "True", "True",
                urls=["https://www.bls.gov/cpi.htm",
                      "https://www.whitehouse.gov/briefing-room/x"]),
        _bundle(VerdictLabel.MOSTLY_TRUE, "Truthy", "Truthy",
                urls=["https://apnews.com/article/abc"]),
        _bundle(VerdictLabel.FALSE, "False", "False",
                urls=["https://example-blog.net/post"]),
        _bundle(VerdictLabel.UNVERIFIABLE, "Unverifiable", "Unverifiable"),
        _bundle(VerdictLabel.UNVERIFIABLE, "Models split", "Models split",
                verdict="Models split"),
    ]
    sr = SiteReport(
        report_id=str(uuid.uuid4()),
        speaker="Synthetic Speaker",
        role="President",
        date=datetime(2026, 3, 4),
        venue="Test Hall",
        transcript_source_url="https://example.org/transcript",
        bundles=bundles,
        generated_at=datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
        speech_id="synthetic_2026",
    )
    pub = SitePublisher(site_root=str(tmp_path))
    pub.publish(sr)
    violations = check_site(tmp_path, strict_buckets=True)
    assert violations == [], "\n".join(violations)

    # The political bucket actually rendered on the index card (the exact
    # regression the strict lint exists for).
    index_html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "press/political" in index_html
    assert 'data-tier-counts="' in index_html
