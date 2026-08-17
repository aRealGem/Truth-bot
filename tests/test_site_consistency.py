"""Run the build-time consistency checker over the committed ``site-pca/``
tree so hand-typed or drifted figures cannot merge (T0.8; this is the test the
``consistency.py`` docstring has promised since P67.4 — added in PR-A2.0
together with the distribution-sum invariants).

The remediation-v2 strict lints (index Sources-chip buckets incl. the
political tier, bucket-sum invariants, feed, no-lens-UI, rendered tiers) used to
be exempted here, because the committed tree predated the remediation
regeneration. The DC-6' publish (2026-08-11) replaced it with the
post-remediation render, so the committed tree is now held to the SAME strict
standard as a fresh render — no exemption, no flag."""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from truthbot.models import (Claim, Confidence, ConsensusVerdict,
                             ModelVerdict, VerdictBundle, VerdictLabel)
from truthbot.publish.consistency import _check_no_lens_ui, check_site
from truthbot.publish.site import SitePublisher, SiteReport

_SITE = Path(__file__).resolve().parent.parent / "site-pca"


@pytest.mark.skipif(not (_SITE / "data" / "reports.json").exists(),
                    reason="site-pca tree not present")
def test_committed_site_has_no_consistency_violations() -> None:
    # STRICT (the default) since the DC-6' publish: the committed tree is the
    # post-remediation render, so it gets no exemption a fresh render would not.
    violations = check_site(_SITE)
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


# ── R-1: no lens UI anywhere ─────────────────────────────────────────────────


def test_fresh_render_has_no_lens_ui_anywhere(tmp_path) -> None:
    """R-1: a fresh render carries NO lens UI on ANY page — the word, the
    chip class, the paired-axis attributes, or the toggle's JS constant.

    Swept over every HTML/CSS/JS file the publisher writes, not a hand-picked
    page list: the two previous removal passes each left a remnant on a
    surface nobody thought to re-check (the status-bar chip, then the
    ``[data-lens-axis][hidden]`` rule in the stylesheet)."""
    sr = SiteReport(
        report_id=str(uuid.uuid4()),
        speaker="Synthetic Speaker",
        role="President",
        date=datetime(2026, 3, 4),
        venue="Test Hall",
        transcript_source_url="https://example.org/transcript",
        bundles=[_bundle(VerdictLabel.TRUE, "True", "True",
                         urls=["https://www.bls.gov/cpi.htm"]),
                 _bundle(VerdictLabel.FALSE, "False", "False",
                         urls=["https://example-blog.net/post"])],
        generated_at=datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
        speech_id="synthetic_2026",
    )
    SitePublisher(site_root=str(tmp_path)).publish(sr)

    assert _check_no_lens_ui(tmp_path) == []
    # Belt and braces: the raw sweep, independent of the lint's pattern list.
    offenders = [p.relative_to(tmp_path).as_posix()
                 for p in sorted(tmp_path.rglob("*"))
                 if p.is_file() and p.suffix.lower() in {".html", ".css", ".js"}
                 and re.search(r"\bLens\b", p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_lens_lint_is_strict_gated_and_actually_fires(tmp_path) -> None:
    """The lint has teeth (it flags a reintroduced chip) AND is gated, so the
    committed pre-remediation ``site-pca/`` tree — which still renders the
    chip on every page — stays lintable at ``strict_buckets=False``."""
    page = tmp_path / "regression.html"
    page.write_text(
        '<button class="editorial-lens" data-lens="strict">'
        '<span class="lens-label">Lens:</span>'
        '<span class="lens-value">Strict</span></button>',
        encoding="utf-8")
    fired = _check_no_lens_ui(tmp_path)
    assert fired and all(v.startswith("regression.html:") for v in fired)

    # Word-boundary matching: a source URL or headline naming Zelenskyy must
    # not be mistaken for the chip (``Lens`` is a substring of ``Lenskyy``).
    page.write_text('<a href="https://x/zelenskyy-speech">Zelenskyy</a>',
                    encoding="utf-8")
    assert _check_no_lens_ui(tmp_path) == []


@pytest.mark.skipif(not (_SITE / "about.html").exists(),
                    reason="site-pca tree not present")
def test_committed_site_carries_no_lens_ui() -> None:
    """The Q-3 tripwire, inverted after the DC-6' publish.

    This test used to assert the OPPOSITE — that the committed tree still carried
    the lens chip — as the standing reason the lens lint had to be strict-gated.
    The publish replaced site-pca with the post-remediation render, so the premise
    is gone and the assertion is now simply: the committed tree carries no lens
    UI. The lint keeps its teeth via
    test_lens_lint_is_strict_gated_and_actually_fires."""
    assert _check_no_lens_ui(_SITE) == []
