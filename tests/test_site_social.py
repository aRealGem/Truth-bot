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


# Default public base URL (no TRUTHBOT_SITE_URL in the test env) — absolute
# canonical/og:url/og:image links resolve against this (1.10).
_BASE = "https://raw.githack.com/aRealGem/Truth-bot/main/site-pca"


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
        assert _tier_bucket("https://army.mil/unit") == "gov"

    def test_whitehouse_is_political_not_gov(self):
        """Claim Eval v3 / D7: the executive's communications shop is S5, not
        top-tier Government. This URL used to bucket as "gov"."""
        assert _tier_bucket("https://www.whitehouse.gov/a") == "political"

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
        assert set(counts.keys()) == {"gov", "wire", "news", "fc", "political", "other"}

    def test_dedupes_across_bundles_and_models(self, site_report):
        # Both bundles reference https://bls.gov/release.htm. Two models each.
        # Unique gov URLs: bls.gov/release.htm, bls.gov/different-release.htm
        # -> 2. whitehouse.gov/statement used to make that 3; under D7 it is
        # S5 political communications and tallies in its own bucket.
        counts = _tier_counts_for_report(site_report)
        assert counts["gov"] == 2
        assert counts["political"] == 1
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
        assert counts == {"gov": 0, "wire": 0, "news": 0, "fc": 0,
                          "political": 0, "other": 0}


# ── _social_head ────────────────────────────────────────────────────────────


class TestSocialHead:
    def test_emits_favicon_links(self):
        html = _social_head("./", "T", "D")
        assert '<link rel="icon" href="./favicon.svg" type="image/svg+xml">' in html
        assert '<link rel="icon" href="./favicon.ico" sizes="any">' in html
        assert '<link rel="icon" href="./assets/favicon-32.png"' in html
        assert '<link rel="apple-touch-icon" href="./assets/apple-touch-icon.png">' in html

    def test_emits_og_block(self):
        html = _social_head("../", "My Title", "My desc", og_type="article")
        assert '<meta property="og:type" content="article">' in html
        assert '<meta property="og:site_name" content="truth-bot">' in html
        assert '<meta property="og:title" content="My Title">' in html
        assert '<meta property="og:description" content="My desc">' in html
        # Images are ABSOLUTE (1.10): crawlers don't resolve relative og:image.
        assert (f'<meta property="og:image" content="{_BASE}/assets/social-card.png">'
                in html)
        assert '<meta property="og:image:width" content="1200">' in html
        assert '<meta property="og:image:height" content="630">' in html
        assert 'property="og:image:alt"' in html

    def test_emits_twitter_block(self):
        html = _social_head("./", "T", "D")
        assert '<meta name="twitter:card" content="summary_large_image">' in html
        assert '<meta name="twitter:title" content="T">' in html
        assert '<meta name="twitter:description" content="D">' in html
        assert (f'<meta name="twitter:image" content="{_BASE}/assets/social-card.png">'
                in html)
        assert 'name="twitter:image:alt"' in html

    def test_meta_description_defaults_to_og_description_and_escapes(self):
        html = _social_head("./", "T", 'Desc & "quoted" <text>')
        assert ('<meta name="description" content="Desc &amp; '
                '&quot;quoted&quot; &lt;text&gt;">') in html
        override = _social_head("./", "T", "OG desc",
                                meta_description="Meta only")
        assert '<meta name="description" content="Meta only">' in override
        assert '<meta property="og:description" content="OG desc">' in override

    def test_canonical_and_og_url_absolute_when_page_path_given(self):
        html = _social_head("../", "T", "D",
                            page_path="reports/2026-03-04-x-abc123.html")
        assert (f'<link rel="canonical" '
                f'href="{_BASE}/reports/2026-03-04-x-abc123.html">') in html
        assert (f'<meta property="og:url" '
                f'content="{_BASE}/reports/2026-03-04-x-abc123.html">') in html
        # Index (page_path="") canonicalizes to the site root with a slash.
        index = _social_head("./", "T", "D", page_path="")
        assert f'<link rel="canonical" href="{_BASE}/">' in index
        # No page_path → no canonical/og:url at all (e.g. the 404 page).
        bare = _social_head("./", "T", "D")
        assert "canonical" not in bare
        assert "og:url" not in bare

    def test_site_url_env_override_respected(self, monkeypatch):
        monkeypatch.setenv("TRUTHBOT_SITE_URL", "https://truthbot.example.org/")
        html = _social_head("./", "T", "D", page_path="about.html")
        assert ('<link rel="canonical" '
                'href="https://truthbot.example.org/about.html">') in html
        assert ('<meta property="og:image" content='
                '"https://truthbot.example.org/assets/social-card.png">') in html

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
        assert f'"{_BASE}/assets/social-card.png"' in html
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


# Readability pass (site.py Section 4): the homepage card's source-tier
# chip (.src-tiers) was removed from ``_report_card`` entirely — that detail
# now lives only on the report page itself, not the index card. The
# TestReportCardSrcTiers class that pinned the chip's rendering has been
# removed along with the feature it tested (not weakened — the behavior it
# asserted no longer exists by design).


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

    def test_render_404_has_no_canonical_or_og_url(self):
        """The 404 page is the one shell WITHOUT a canonical/og:url (1.10):
        it serves at arbitrary URLs, so any canonical would be a lie."""
        html = _render_404()
        assert 'rel="canonical"' not in html
        assert 'property="og:url"' not in html

    def test_render_report_canonical_matches_slug(self, site_report):
        html = _render_report(site_report)
        expected = f"{_BASE}/{site_report.report_url}"
        assert f'<link rel="canonical" href="{expected}">' in html
        assert f'<meta property="og:url" content="{expected}">' in html
        assert '<meta name="description" content=' in html

    def test_render_claim_page_canonical_matches_claim_id(self, site_report):
        bundle = site_report.bundles[0]
        html = _render_claim_page(bundle, site_report)
        expected = f"{_BASE}/claims/{bundle.claim.id}.html"
        assert f'<link rel="canonical" href="{expected}">' in html
        assert f'<meta property="og:url" content="{expected}">' in html

    def test_render_index_canonical_is_site_root(self):
        html = _render_index([], {"total_claims": 0, "total_leaders": 0,
                                  "avg_consensus": 0.0})
        assert f'<link rel="canonical" href="{_BASE}/">' in html
        assert f'<meta property="og:url" content="{_BASE}/">' in html
        # The banned index phrase must not ride in via the meta description
        # (consistency bans "primary sources" on index; the description
        # reuses the page's own og_description, never the module default).
        assert "primary sources" not in html

    def test_render_about_canonical(self):
        html = _render_about()
        assert f'<link rel="canonical" href="{_BASE}/about.html">' in html

    def test_render_truthy_has_social_head_and_footer_hash(self):
        html = _render_truthy()
        assert 'property="og:title"' in html
        assert 'class="footer-hash" href="./about.html#prompt"' in html


# ── SitePublisher asset + feed writer ───────────────────────────────────────


class TestPublisherAssets:
    def test_copy_assets_places_social_files(self, tmp_dir):
        pub = SitePublisher(site_root=str(tmp_dir))
        pub._ensure_structure()
        pub._copy_assets()

        assert (tmp_dir / "favicon.ico").exists()
        assert (tmp_dir / "favicon.svg").exists()
        assert (tmp_dir / "assets" / "social-card.png").exists()
        assert (tmp_dir / "assets" / "favicon-32.png").exists()
        assert (tmp_dir / "assets" / "apple-touch-icon.png").exists()
        # The feed is DATA, not a static asset (remediation v2, 1.5): it
        # renders from the reports index inside publish(), so the asset
        # copier alone writes none. Rendering covered in tests/publish/test_feed.py.
        assert not (tmp_dir / "feed.xml").exists()

    def test_publish_writes_feed_with_report_entry(self, site_report, tmp_dir):
        pub = SitePublisher(site_root=str(tmp_dir))
        pub.publish(site_report)
        feed_text = (tmp_dir / "feed.xml").read_text(encoding="utf-8")
        assert "[SITE_URL]" not in feed_text
        assert "<entry>" in feed_text
        assert "Test Politician" in feed_text

    def test_report_meta_includes_tier_counts(self, site_report, tmp_dir):
        pub = SitePublisher(site_root=str(tmp_dir))
        meta = pub._report_meta(site_report)
        assert "tier_counts" in meta
        assert set(meta["tier_counts"].keys()) == {"gov", "wire", "news", "fc",
                                                   "political", "other"}
        assert meta["tier_counts"]["gov"] == 2          # whitehouse.gov -> political (D7)
        assert meta["tier_counts"]["political"] == 1
        assert meta["tier_counts"]["wire"] == 1
