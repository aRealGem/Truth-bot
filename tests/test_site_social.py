"""Tests for social sharing infrastructure in the site publisher.

Covers:
- Tier bucket classification (gov/wire/news/fc/other)
- Tier-count aggregation + URL deduping across bundles
- _social_head emits the correct favicon/OG/Twitter/feed blocks
- Per-shell HTML: favicon links, OG tags, feed link (index only), footer
  prompt-hash, and report-card src-tiers chip
- SitePublisher copies favicon.ico to site root and writes feed.xml with
  the [SITE_URL] placeholder preserved
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from truthbot.models import (
    Claim,
    ConsensusVerdict,
    Confidence,
    ModelVerdict,
    VerdictBundle,
    VerdictLabel,
)
from truthbot.publish.site import (
    FEED_XML_TEMPLATE,
    SiteReport,
    SitePublisher,
    _page_about,
    _page_index,
    _page_report,
    _page_truthy,
    _prompt_hash,
    _render_404,
    _render_about,
    _render_claim_page,
    _render_index,
    _render_report,
    _render_truthy,
    _report_card,
    _social_head,
    _tier_bucket,
    _tier_counts_for_report,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_bundle(
    claim_text: str = "Unemployment is at a 50-year low.",
    urls: list[str] | None = None,
    label: VerdictLabel = VerdictLabel.MOSTLY_TRUE,
    n_models: int = 2,
) -> VerdictBundle:
    """Build a minimal VerdictBundle suitable for site-renderer tests."""
    claim = Claim(
        transcript_id="test-transcript",
        text=claim_text,
        speaker="Test Politician",
        context=claim_text,
        category="economy",
        is_checkable=True,
    )
    urls = urls or []
    mvs = [
        ModelVerdict(
            adapter_name=f"adapter-{i}",
            model_id=f"model-{i}",
            claim_id=claim.id,
            label=label,
            confidence=Confidence.HIGH,
            explanation="Test explanation.",
            web_sources=list(urls),
        )
        for i in range(n_models)
    ]
    consensus = ConsensusVerdict(
        claim_id=claim.id,
        model_verdicts=mvs,
        consensus_label=label,
        consensus_verdict=label.value,
        confidence=Confidence.HIGH,
        agreement=True,
        consensus_strength="weak" if n_models == 2 else "strong",
        explanation="Consensus explanation.",
    )
    return VerdictBundle(
        claim=claim,
        speaker="Test Politician",
        date_str="2026-03-04",
        model_verdicts=mvs,
        consensus=consensus,
    )


@pytest.fixture
def site_report() -> SiteReport:
    gov_urls = [
        "https://bls.gov/release.htm",
        "https://www.bls.gov/different-release.htm",
        "https://whitehouse.gov/statement",
    ]
    wire_urls = ["https://apnews.com/article"]
    news_urls = ["https://nytimes.com/opinion/piece"]
    b1 = _make_bundle(
        claim_text="Unemployment claim.",
        urls=gov_urls + wire_urls,
        label=VerdictLabel.MOSTLY_TRUE,
    )
    # Second claim shares one gov URL with the first — dedup should collapse it.
    b2 = _make_bundle(
        claim_text="Deficit claim.",
        urls=[gov_urls[0], news_urls[0], "https://factcheck.org/review"],
        label=VerdictLabel.FALSE,
    )
    return SiteReport(
        report_id="11111111-aaaa-bbbb-cccc-222222222222",
        speaker="Test Politician",
        role="President",
        date=datetime(2026, 3, 4),
        venue="Capitol",
        transcript_source_url="https://example.gov/transcript",
        bundles=[b1, b2],
        generated_at=datetime(2026, 4, 22, tzinfo=timezone.utc),
        source_of_claims="Test Politician",
        source_of_claims_professional_public_title="President",
        event="Joint Session",
        channel="Broadcast",
    )


# ── _tier_bucket ────────────────────────────────────────────────────────────


class TestTierBucket:
    def test_gov_urls(self):
        assert _tier_bucket("https://bls.gov/news") == "gov"
        assert _tier_bucket("https://www.whitehouse.gov/a") == "gov"
        assert _tier_bucket("https://army.mil/unit") == "gov"

    def test_wire_urls(self):
        assert _tier_bucket("https://apnews.com/article") == "wire"
        assert _tier_bucket("https://reuters.com/world") == "wire"

    def test_news_urls(self):
        assert _tier_bucket("https://nytimes.com/2026/03/04") == "news"
        assert _tier_bucket("https://washingtonpost.com/politics") == "news"
        assert _tier_bucket("https://www.bbc.com/news") == "news"
        assert _tier_bucket("https://npr.org/story") == "news"

    def test_factcheck_urls(self):
        assert _tier_bucket("https://politifact.com/check") == "fc"
        assert _tier_bucket("https://www.factcheck.org/article") == "fc"
        assert _tier_bucket("https://snopes.com/fact-check") == "fc"

    def test_other_urls(self):
        assert _tier_bucket("https://random-blog.example/post") == "other"
        assert _tier_bucket("https://twitter.com/someone") == "other"


# ── _tier_counts_for_report ─────────────────────────────────────────────────


class TestTierCountsForReport:
    def test_returns_all_buckets(self, site_report):
        counts = _tier_counts_for_report(site_report)
        assert set(counts.keys()) == {"gov", "wire", "news", "fc", "other"}

    def test_dedupes_across_bundles_and_models(self, site_report):
        # Both bundles reference https://bls.gov/release.htm. Two models each.
        # Unique gov URLs: bls.gov/release.htm, bls.gov/different-release.htm,
        # whitehouse.gov/statement -> 3, not more.
        counts = _tier_counts_for_report(site_report)
        assert counts["gov"] == 3
        assert counts["wire"] == 1
        assert counts["news"] == 1
        assert counts["fc"] == 1
        assert counts["other"] == 0

    def test_empty_report(self):
        empty = SiteReport(
            report_id="empty",
            speaker="Nobody",
            role="",
            date=None,
            venue="",
            transcript_source_url="",
            bundles=[],
        )
        counts = _tier_counts_for_report(empty)
        assert counts == {"gov": 0, "wire": 0, "news": 0, "fc": 0, "other": 0}


# ── _social_head ────────────────────────────────────────────────────────────


class TestSocialHead:
    def test_emits_favicon_links(self):
        html = _social_head("./", "T", "D")
        assert '<link rel="icon" href="./favicon.ico" sizes="any">' in html
        assert '<link rel="icon" href="./assets/favicon-32.png"' in html
        assert '<link rel="apple-touch-icon" href="./assets/apple-touch-icon.png">' in html

    def test_emits_og_block(self):
        html = _social_head("../", "My Title", "My desc", og_type="article")
        assert '<meta property="og:type" content="article">' in html
        assert '<meta property="og:site_name" content="truth-bot">' in html
        assert '<meta property="og:title" content="My Title">' in html
        assert '<meta property="og:description" content="My desc">' in html
        assert '<meta property="og:image" content="../assets/social-card.png">' in html
        assert '<meta property="og:image:width" content="1200">' in html
        assert '<meta property="og:image:height" content="630">' in html
        assert 'property="og:image:alt"' in html

    def test_emits_twitter_block(self):
        html = _social_head("./", "T", "D")
        assert '<meta name="twitter:card" content="summary_large_image">' in html
        assert '<meta name="twitter:title" content="T">' in html
        assert '<meta name="twitter:description" content="D">' in html
        assert '<meta name="twitter:image" content="./assets/social-card.png">' in html
        assert 'name="twitter:image:alt"' in html

    def test_feed_link_opt_in_only(self):
        without = _social_head("./", "T", "D")
        assert "atom+xml" not in without
        with_feed = _social_head("./", "T", "D", include_feed_link=True)
        assert (
            '<link rel="alternate" type="application/atom+xml" '
            'title="truth-bot feed" href="./feed.xml">'
        ) in with_feed

    def test_relative_prefix_used_for_subdir_pages(self):
        html = _social_head("../", "T", "D")
        assert '"../favicon.ico"' in html
        assert '"../assets/favicon-32.png"' in html
        assert '"../assets/social-card.png"' in html
        assert '"./favicon.ico"' not in html

    def test_html_escapes_title_and_description(self):
        html = _social_head("./", "A & B", 'Line <br> with "quotes"')
        assert "A &amp; B" in html
        assert "&lt;br&gt;" in html
        # Raw double quotes inside the description must be escaped so they
        # don't close the surrounding content="…" attribute.
        assert '"quotes"' not in html
        assert "&quot;quotes&quot;" in html


# ── Page shells ─────────────────────────────────────────────────────────────


class TestPageShells:
    def test_index_includes_feed_link(self):
        html = _page_index("Latest", "<p>hi</p>", "footer")
        assert 'type="application/atom+xml"' in html
        assert 'href="./feed.xml"' in html
        assert 'property="og:type" content="website"' in html

    def test_report_shell_uses_parent_rel(self):
        html = _page_report(
            "Speaker — March 04, 2026",
            "<p>body</p>",
            "footer",
            og_description="desc",
        )
        assert '"../favicon.ico"' in html
        assert '"../assets/social-card.png"' in html
        assert 'property="og:type" content="article"' in html
        # No feed link on non-index pages
        assert "atom+xml" not in html

    def test_about_shell_uses_root_rel(self):
        html = _page_about("About", "<p>body</p>", "footer")
        assert '"./favicon.ico"' in html
        assert 'property="og:title" content="About — truth-bot"' in html

    def test_truthy_shell_has_social_head(self):
        html = _page_truthy("Meet Truthy", "<p>body</p>", "footer")
        assert 'property="og:type" content="website"' in html
        assert '"./favicon.ico"' in html


# ── _report_card src-tiers chip ─────────────────────────────────────────────


class TestReportCardSrcTiers:
    def test_renders_chip_when_counts_present(self):
        r = {
            "speaker": "Speaker",
            "url": "reports/x.html",
            "verdict_distribution": {"False": 3, "True": 1},
            "claim_count": 4,
            "tier_counts": {"gov": 3, "wire": 2, "news": 0, "fc": 1, "other": 0},
        }
        html = _report_card(r)
        assert '<span class="src-tiers">' in html
        assert "3 gov" in html
        assert "2 wire" in html
        assert "1 fc" in html

    def test_omits_zero_and_other_buckets(self):
        r = {
            "speaker": "S",
            "url": "reports/x.html",
            "verdict_distribution": {"True": 1},
            "claim_count": 1,
            "tier_counts": {"gov": 2, "wire": 0, "news": 0, "fc": 0, "other": 99},
        }
        html = _report_card(r)
        assert '<span class="src-tiers">' in html
        assert "2 gov" in html
        # Zero buckets are suppressed
        assert "0 wire" not in html
        assert "0 news" not in html
        assert "0 fc" not in html
        # "other" is never surfaced in the chip
        assert "99 other" not in html
        assert " other" not in html.split('class="src-tiers"')[1]

    def test_no_chip_when_all_counts_zero(self):
        r = {
            "speaker": "S",
            "url": "reports/x.html",
            "verdict_distribution": {"True": 1},
            "claim_count": 1,
            "tier_counts": {"gov": 0, "wire": 0, "news": 0, "fc": 0, "other": 0},
        }
        html = _report_card(r)
        assert 'class="src-tiers"' not in html

    def test_no_chip_when_tier_counts_missing(self):
        r = {
            "speaker": "S",
            "url": "reports/x.html",
            "verdict_distribution": {"True": 1},
            "claim_count": 1,
        }
        html = _report_card(r)
        assert 'class="src-tiers"' not in html


# ── End-to-end renderers ────────────────────────────────────────────────────


class TestRenderers:
    def test_render_index_has_social_and_feed(self):
        reports = [
            {
                "speaker": "X",
                "url": "reports/x.html",
                "verdict_distribution": {"True": 1},
                "claim_count": 1,
                "tier_counts": {"gov": 1, "wire": 0, "news": 0, "fc": 0, "other": 0},
            }
        ]
        html = _render_index(reports, {"total_claims": 1, "total_leaders": 1, "avg_consensus": 1.0})
        assert 'property="og:title"' in html
        assert "atom+xml" in html
        assert 'class="footer-hash"' in html
        assert _prompt_hash() in html
        assert 'class="src-tiers"' in html

    def test_render_report_og_and_footer(self, site_report):
        html = _render_report(site_report)
        assert 'property="og:type" content="article"' in html
        # Speaker + display date appear in the OG title
        assert "Test Politician" in html
        assert "March 04, 2026" in html
        # Description mentions claim count
        assert "2 claims checked" in html
        # Footer hash link points to about.html with ../
        assert 'class="footer-hash" href="../about.html#prompt"' in html

    def test_render_claim_page_uses_claim_text_and_model_agreement(self, site_report):
        bundle = site_report.bundles[0]
        html = _render_claim_page(bundle, site_report)
        assert 'property="og:type" content="article"' in html
        assert "Claim:" in html
        # Truncated claim text appears in the OG title (first 60 chars)
        assert bundle.claim.text[:30] in html
        # "2 of 2 models agree" since both adapters return the same label
        assert "2 of 2 model" in html
        assert 'class="footer-hash" href="../about.html#prompt"' in html

    def test_render_about_has_prompt_anchor(self):
        html = _render_about()
        assert 'id="prompt"' in html
        assert 'class="footer-hash" href="./about.html#prompt"' in html

    def test_render_404_has_social_head_no_footer(self):
        html = _render_404()
        assert 'property="og:title" content="404 Not Found — truth-bot"' in html
        # 404 intentionally has no footer by design
        assert 'class="footer-hash"' not in html

    def test_render_truthy_has_social_head_and_footer_hash(self):
        html = _render_truthy()
        assert 'property="og:title"' in html
        assert 'class="footer-hash" href="./about.html#prompt"' in html


# ── SitePublisher asset + feed writer ───────────────────────────────────────


class TestPublisherAssets:
    def test_feed_template_has_placeholder(self):
        assert "[SITE_URL]" in FEED_XML_TEMPLATE
        assert "<feed" in FEED_XML_TEMPLATE

    def test_copy_assets_places_social_files_and_feed(self, tmp_dir):
        pub = SitePublisher(site_root=str(tmp_dir))
        pub._ensure_structure()
        pub._copy_assets()

        assert (tmp_dir / "favicon.ico").exists()
        assert (tmp_dir / "feed.xml").exists()
        assert (tmp_dir / "assets" / "social-card.png").exists()
        assert (tmp_dir / "assets" / "favicon-32.png").exists()
        assert (tmp_dir / "assets" / "apple-touch-icon.png").exists()

        feed_text = (tmp_dir / "feed.xml").read_text(encoding="utf-8")
        assert "[SITE_URL]" in feed_text
        assert "truth-bot" in feed_text

    def test_report_meta_includes_tier_counts(self, site_report, tmp_dir):
        pub = SitePublisher(site_root=str(tmp_dir))
        meta = pub._report_meta(site_report)
        assert "tier_counts" in meta
        assert set(meta["tier_counts"].keys()) == {"gov", "wire", "news", "fc", "other"}
        assert meta["tier_counts"]["gov"] == 3
        assert meta["tier_counts"]["wire"] == 1
