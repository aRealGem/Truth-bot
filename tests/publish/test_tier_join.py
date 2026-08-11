"""Render the tier the pipeline STORED, not one re-derived at render time
(remediation v2 Phase A, A5 / owner ruling C-3(a)).

``ModelVerdict.web_sources`` is a bare list of URL strings with no tier on it.
The renderer used to hand each one to ``classify_tier`` at publish time, so a
report displayed whatever the registry said the day it rendered — while the
artifact beside it stored the tiers the panel actually adjudicated on. On the
two published modern reports those disagreed on 414 of 1543 and 272 of 918
items: the page was showing a different evidence hierarchy than the one its
verdicts were written from.

C-3(a) is a join, not a rewrite: match web_sources to the claim's evidence-pack
items by NORMALIZED url, take the stored tier on a hit, fall back to the rules
on a miss, and COUNT the misses. The count is the point — a miss is legitimate
(a model can cite a URL that never entered the pack), so the only way to know
the join is healthy is to publish its rate.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone

from truthbot.models import (Claim, Confidence, ConsensusVerdict, ModelVerdict,
                             SourceTier, VerdictBundle, VerdictLabel)
from truthbot.publish.aggregation import normalize_url
from truthbot.publish.consistency import (_check_rendered_tiers,
                                          tier_render_telemetry)
from truthbot.publish.site import (SitePublisher, SiteReport, _claim_card,
                                   _new_tier_tally, _resolve_tier,
                                   _tier_counts_for_report, stored_tier_index,
                                   tier_counts_with_join, tier_join_rate)

# ── The normalizer ───────────────────────────────────────────────────────────
#
# Fable's note on C-3(a): "normalize before the join or the fallback metric is
# noise." Every case below is a difference that must NOT split one document
# into two join keys.


class TestNormalizeUrl:
    def test_scheme_is_dropped_so_http_and_https_key_alike(self) -> None:
        assert (normalize_url("http://bls.gov/cpi")
                == normalize_url("https://bls.gov/cpi") == "bls.gov/cpi")

    def test_leading_www_is_dropped(self) -> None:
        assert (normalize_url("https://www.bls.gov/cpi")
                == normalize_url("https://bls.gov/cpi"))

    def test_host_is_case_folded_but_path_is_not(self) -> None:
        """Hosts are case-insensitive; paths are not — folding both would join
        two genuinely different documents."""
        assert normalize_url("https://BLS.GOV/CPI") == "bls.gov/CPI"
        assert normalize_url("https://bls.gov/cpi") != normalize_url(
            "https://bls.gov/CPI")

    def test_trailing_slash_is_dropped(self) -> None:
        assert (normalize_url("https://bls.gov/news/")
                == normalize_url("https://bls.gov/news") == "bls.gov/news")
        # A bare host normalizes to just the host, not "host/".
        assert normalize_url("https://bls.gov/") == "bls.gov"

    def test_fragment_is_dropped(self) -> None:
        assert (normalize_url("https://bls.gov/cpi#table-3")
                == normalize_url("https://bls.gov/cpi"))

    def test_tracking_params_are_dropped(self) -> None:
        for junk in ("utm_source=twitter", "utm_medium=social", "utm_campaign=x",
                     "utm_content=a", "utm_term=b", "fbclid=IwAR123",
                     "gclid=abc", "ref=hp", "mc_cid=deadbeef"):
            assert normalize_url(f"https://bls.gov/cpi?{junk}") == "bls.gov/cpi", junk

    def test_meaningful_params_survive_and_order_does_not_matter(self) -> None:
        """A query that identifies the document is part of its identity; the
        ORDER it was written in is not."""
        assert (normalize_url("https://fred.stlouisfed.org/graph?id=CPI&start=2020")
                == normalize_url("https://fred.stlouisfed.org/graph?start=2020&id=CPI"))
        assert "id=CPI" in normalize_url(
            "https://fred.stlouisfed.org/graph?id=CPI")

    def test_default_ports_and_userinfo_are_dropped(self) -> None:
        assert normalize_url("http://bls.gov:80/cpi") == "bls.gov/cpi"
        assert normalize_url("https://bls.gov:443/cpi") == "bls.gov/cpi"
        assert normalize_url("https://user:pw@bls.gov/cpi") == "bls.gov/cpi"

    def test_nonstandard_port_survives(self) -> None:
        assert normalize_url("https://bls.gov:8443/cpi") == "bls.gov:8443/cpi"

    def test_scheme_less_input_still_keys(self) -> None:
        assert normalize_url("www.bls.gov/cpi") == "bls.gov/cpi"

    def test_empty_input_is_unjoinable_not_a_shared_key(self) -> None:
        """Empty must be falsy so callers skip it — if it returned a stable
        key, every URL-less item would join to every other one."""
        assert normalize_url("") == ""
        assert normalize_url("   ") == ""
        assert normalize_url(None) == ""  # type: ignore[arg-type]

    def test_the_full_cosmetic_stack_collapses_to_one_key(self) -> None:
        assert normalize_url(
            "HTTP://WWW.Example-Blog.net:80/post/?utm_source=x&fbclid=y#top"
        ) == normalize_url("https://example-blog.net/post")


# ── Fixtures ─────────────────────────────────────────────────────────────────
#
# The stored tier is deliberately set to something classify_tier would NEVER
# return for that URL, so "did the render use the stored value" has an
# unambiguous answer.

_PACK_URL = "https://example-blog.net/post"
_PACK_STORED_TIER = SourceTier.GOVERNMENT      # rules say OTHER (T6)
_CITED_VARIANT = "http://www.example-blog.net/post/?utm_source=newsletter#top"
_UNPACKED_URL = "https://apnews.com/article/xyz"   # never in the pack → T2·Wire


def _bundle(*, web_sources: list[str], pack: list[dict]) -> VerdictBundle:
    claim = Claim(
        transcript_id="t",
        text=f"Synthetic claim {uuid.uuid4().hex[:8]}.",
        speaker="Synthetic Speaker",
        context="",
        category="economy",
        is_checkable=True,
    )
    mvs = [ModelVerdict(
        adapter_name="pca", model_id="reconciled", claim_id=claim.id,
        label=VerdictLabel.TRUE, confidence=Confidence.HIGH,
        explanation="Synthetic reasoning.", web_sources=list(web_sources))]
    consensus = ConsensusVerdict(
        claim_id=claim.id, model_verdicts=mvs, consensus_label=VerdictLabel.TRUE,
        consensus_verdict="True", confidence=Confidence.HIGH, agreement=True,
        consensus_strength="strong", explanation="Synthetic.",
        coarse_lenient_label="True", coarse_lenient_strength="strong",
        coarse_strict_label="True", coarse_strict_strength="strong")
    return VerdictBundle(
        claim=claim, speaker="Synthetic Speaker", date_str="2026-03-04",
        model_verdicts=mvs, consensus=consensus, sources_consulted=list(pack))


def _pack_item(url: str, tier: SourceTier, pack_id: str = "E1") -> dict:
    return {"id": pack_id, "source": "Example Source", "url": url,
            "tier": tier.value, "snippet": "Synthetic snippet.",
            "supports_claim": True, "relevance_score": 0.9}


def _joined_bundle() -> VerdictBundle:
    return _bundle(web_sources=[_CITED_VARIANT, _UNPACKED_URL],
                   pack=[_pack_item(_PACK_URL, _PACK_STORED_TIER)])


def _site_report(bundles: list[VerdictBundle]) -> SiteReport:
    return SiteReport(
        report_id=str(uuid.uuid4()), speaker="Synthetic Speaker",
        role="President", date=datetime(2026, 3, 4), venue="Test Hall",
        transcript_source_url="https://example.org/transcript",
        bundles=bundles,
        generated_at=datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
        speech_id="synthetic_2026")


# ── The join itself ──────────────────────────────────────────────────────────


def test_stored_tier_index_keys_on_the_normalized_url() -> None:
    index = stored_tier_index(_joined_bundle())
    assert index == {normalize_url(_PACK_URL): _PACK_STORED_TIER}
    # The cited variant differs cosmetically in four ways and still hits.
    assert index[normalize_url(_CITED_VARIANT)] is _PACK_STORED_TIER


def test_stored_tier_index_skips_a_bogus_stored_value() -> None:
    """An unparseable tier must become a join MISS (visible in the fallback
    rate), never a silently invented tier."""
    b = _bundle(web_sources=[], pack=[{"id": "E1", "url": _PACK_URL,
                                       "tier": "NotATier", "source": "x",
                                       "snippet": ""}])
    assert stored_tier_index(b) == {}


def test_resolve_tier_prefers_stored_and_flags_the_fallback() -> None:
    index = stored_tier_index(_joined_bundle())
    tally = _new_tier_tally()

    tier, hit = _resolve_tier(_CITED_VARIANT, index, tally)
    assert (tier, hit) == (SourceTier.GOVERNMENT, True)

    tier, hit = _resolve_tier(_UNPACKED_URL, index, tally)
    assert (tier, hit) == (SourceTier.WIRE, False)

    assert tally == {"joined": 1, "fallback": 1}


def test_tier_counts_use_the_stored_tier_not_the_render_time_rules() -> None:
    """The regression in one assertion: without the join this URL counts as
    "other" (the rules read example-blog.net as untiered); with it, it counts
    where the pipeline actually filed it."""
    counts, tally = tier_counts_with_join(_site_report([_joined_bundle()]))
    assert counts["gov"] == 1        # stored Government, NOT rules-derived
    assert counts["other"] == 0
    assert counts["wire"] == 1       # the unpacked URL, legitimately re-derived
    assert tally == {"joined": 1, "fallback": 1}
    # The back-compat wrapper still returns just the counts.
    assert _tier_counts_for_report(_site_report([_joined_bundle()])) == counts


def test_dedup_is_on_the_normalized_url() -> None:
    """Two cosmetic spellings of one document counted twice before — dedup on
    the raw string could not see they were the same source."""
    b = _bundle(web_sources=["https://www.bls.gov/cpi/",
                             "http://bls.gov/cpi?utm_source=x"],
                pack=[])
    counts, tally = tier_counts_with_join(_site_report([b]))
    assert sum(counts.values()) == 1
    assert tally["fallback"] == 1


def test_tier_join_rate_shape() -> None:
    assert tier_join_rate({"joined": 3, "fallback": 1}) == {
        "joined": 3, "fallback": 1, "total": 4, "fallback_rate": 0.25}
    assert tier_join_rate(_new_tier_tally())["fallback_rate"] == 0.0


# ── What the reader actually sees ────────────────────────────────────────────


def test_rendered_badges_show_the_stored_tier() -> None:
    tally = _new_tier_tally()
    html = _claim_card(_joined_bundle(), 1, 1, tier_tally=tally)

    # Pack item: stored Government renders T1·Gov and advertises the source.
    assert 'data-stored-tier="Government"' in html
    assert re.search(r'data-stored-tier="Government"[^>]*>.*?'
                     r'data-tier-src="stored">T1·Gov<', html, re.S)
    # The rules would have said T6 for this domain — prove they did not win.
    assert "T6" not in html

    # Cited-URL list: the cosmetically different spelling joined to the pack,
    # the unpacked AP URL fell back to the rules and says so.
    assert 'data-tier-src="classified">T2·Wire<' in html
    assert tally["joined"] >= 2 and tally["fallback"] == 1


def test_unjoined_url_still_renders_a_tier() -> None:
    """Fallback is a fallback, not a hole: a cited URL that never entered the
    pack still gets a badge, just an honestly-labelled one."""
    html = _claim_card(_bundle(web_sources=[_UNPACKED_URL], pack=[]), 1, 1)
    assert 'data-tier-src="classified">T2·Wire<' in html


# ── Telemetry: the number has to be published, not just computed ─────────────


def test_reports_json_publishes_the_fallback_rate(tmp_path) -> None:
    SitePublisher(site_root=str(tmp_path)).publish(
        _site_report([_joined_bundle()]))
    rows = json.loads((tmp_path / "data" / "reports.json").read_text("utf-8"))
    tel = rows[0]["tier_fallback"]
    assert tel == {"joined": 1, "fallback": 1, "total": 2, "fallback_rate": 0.5}
    assert rows[0]["tier_counts"]["gov"] == 1


def test_render_prints_the_fallback_rate(tmp_path, capsys) -> None:
    """C-3(a) asks for the rate to be VISIBLE during the render — a broken
    join should be obvious in the log, not only on forensic inspection of
    reports.json afterwards."""
    SitePublisher(site_root=str(tmp_path)).publish(
        _site_report([_joined_bundle()]))
    out = capsys.readouterr().out
    assert "tier join · synthetic_2026" in out
    assert "fallback rate 50.0%" in out


def test_rendered_telemetry_is_recomputable_from_the_html(tmp_path) -> None:
    """The lint's figure comes off the shipped HTML, independently of whatever
    the publisher chose to write into reports.json."""
    SitePublisher(site_root=str(tmp_path)).publish(
        _site_report([_joined_bundle()]))
    rows = json.loads((tmp_path / "data" / "reports.json").read_text("utf-8"))
    tel = tier_render_telemetry(tmp_path, rows)
    assert tel["total"]["joined"] >= 2
    assert tel["total"]["fallback"] == 1
    assert 0.0 < tel["total"]["fallback_rate"] < 1.0


# ── The lint has teeth ───────────────────────────────────────────────────────


def test_lint_is_clean_on_a_fresh_render(tmp_path) -> None:
    SitePublisher(site_root=str(tmp_path)).publish(
        _site_report([_joined_bundle()]))
    rows = json.loads((tmp_path / "data" / "reports.json").read_text("utf-8"))
    assert _check_rendered_tiers(tmp_path, rows) == []


def test_lint_catches_a_badge_that_contradicts_the_stored_tier(tmp_path) -> None:
    """Simulates the exact defect A5 exists for: the page says one tier, the
    artifact stored another."""
    SitePublisher(site_root=str(tmp_path)).publish(
        _site_report([_joined_bundle()]))
    rows = json.loads((tmp_path / "data" / "reports.json").read_text("utf-8"))
    page_path = tmp_path / rows[0]["url"]
    page_path.write_text(
        page_path.read_text("utf-8").replace(
            'data-tier-src="stored">T1·Gov<',
            'data-tier-src="stored">T6<'),
        encoding="utf-8")

    violations = _check_rendered_tiers(tmp_path, rows)
    assert violations and "renders T6" in violations[0]
    assert "adjudicated on" in violations[0]


def test_lint_catches_a_stored_item_rendered_as_re_derived(tmp_path) -> None:
    """A stored tier that renders with ``data-tier-src="classified"`` means the
    join was skipped for that item — a silent regression to the old behavior
    even when the two happen to agree."""
    SitePublisher(site_root=str(tmp_path)).publish(
        _site_report([_joined_bundle()]))
    rows = json.loads((tmp_path / "data" / "reports.json").read_text("utf-8"))
    page_path = tmp_path / rows[0]["url"]
    page_path.write_text(
        page_path.read_text("utf-8").replace(
            'data-tier-src="stored">T1·Gov<',
            'data-tier-src="classified">T1·Gov<'),
        encoding="utf-8")

    violations = _check_rendered_tiers(tmp_path, rows)
    assert violations and "the join was skipped" in violations[0]
