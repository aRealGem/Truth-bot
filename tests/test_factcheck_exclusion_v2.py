"""Remediation v2 (1.1, DC-1 approved 2026-08-02): article-level fact-check
exclusion — generic path regex + per-domain verticals + allowlist — and the
never-silent per-claim exclusion log."""
from __future__ import annotations

from truthbot.models import SourceTier
from truthbot.verdict.consolidator import consolidate
from truthbot.verify.factcheck_exclusion import (
    factcheck_exclusion_reason,
    is_excluded_factchecker,
)

# (url, expected reason prefix) — the article-level leaks found in the
# published corpus during Phase 0 recon, one per rule class.
EXCLUDED = [
    ("https://www.cbsnews.com/news/fact-check-state-of-the-union-2026/", "vertical:cbsnews.com"),
    ("https://www.cbsnews.com/amp/news/fact-check-state-of-the-union-2026/", "vertical:cbsnews.com"),
    ("https://abcnews.go.com/Politics/fact-checking-trumps-state-union-address/story?id=1", "vertical:abcnews.go.com"),
    ("https://abcnews.com/Politics/fact-checking-trumps-migrant-murderers-claims/story?id=2", "vertical:abcnews.com"),
    ("https://www.nbcnews.com/politics/donald-trump/fact-check-trump-interview", "vertical:nbcnews.com"),
    ("https://www.washingtonpost.com/politics/2025/03/05/fact-check-trump-speech-address-congress/", "vertical:washingtonpost.com"),
    ("https://apnews.com/article/fact-check-trump-state-of-union-87b184f7", "path-regex"),
    ("https://www.nytimes.com/2026/02/23/us/politics/trump-economy-fact-check.html", "path-regex"),
    ("https://edition.cnn.com/2025/10/28/politics/fact-check-trump-japan-troops", "path-regex"),
    ("https://www.wisn.com/article/fact-check-trumps-sotu-building-wealth/70503388", "path-regex"),
    ("https://econofact.org/factbrief/fact-check-did-the-us-have-more-jobs", "path-regex"),
    ("https://example.com/2026/factchecking-the-speech", "path-regex"),
    ("https://www.politifact.com/article/x/", "domain:politifact.com"),
    ("https://www.reuters.com/fact-check/claim-x-2026/", "path-prefix:reuters.com/fact-check"),
]

ALLOWED = [
    # ordinary reporting on otherwise-verticaled outlets
    "https://www.cbsnews.com/news/economy-adds-200k-jobs/",
    "https://abcnews.go.com/Politics/budget-vote/story?id=3",
    "https://www.washingtonpost.com/politics/2026/02/25/sotu-reaction/",
    "https://apnews.com/article/economy-jobs-report-2026",
    # allowlisted government program page whose path contains 'factcheck'
    "https://mn.gov/dhs/program-integrity/factcheck/",
    "https://mn.gov/dhs/program-integrity/factcheck/index.jsp",
    # 'fact' alone, or check outside the path, must not trip the regex
    "https://example.com/facts-about-the-economy",
    "https://example.com/background-check-policy",
]


def test_v2_rules_exclude_article_level_factchecks() -> None:
    for url, expected in EXCLUDED:
        reason = factcheck_exclusion_reason(url)
        assert reason.startswith(expected), (url, reason, expected)
        assert is_excluded_factchecker(url)


def test_v2_rules_spare_ordinary_reporting_and_allowlist() -> None:
    for url in ALLOWED:
        assert factcheck_exclusion_reason(url) == "", url


def test_unofficial_mirror_of_allowlisted_page_stays_excluded() -> None:
    # DC-1 ruling: the REAL mn.gov page is allowlisted; a scraped mirror on a
    # non-government domain is not (it was cited in a published FALSE verdict).
    assert is_excluded_factchecker(
        "https://www.developtoolmn.org/dhs/program-integrity/factcheck/index.jsp")


def test_regex_matches_path_only_never_host() -> None:
    # A host containing 'factcheck' is a domain-rule question, not a regex hit.
    assert factcheck_exclusion_reason("https://factcheck.house.gov/report") == ""


def test_consolidator_logs_every_fc_exclusion() -> None:
    from datetime import date, datetime, timezone

    from truthbot.models import Evidence

    def ev(url: str, tier: SourceTier = SourceTier.ESTABLISHED) -> Evidence:
        return Evidence(claim_id="c", source_name="S", source_url=url,
                        source_tier=tier, snippet="s", supports_claim=True,
                        published_at=datetime(2026, 2, 20, tzinfo=timezone.utc))

    res = consolidate(
        "s",
        [("R1", [ev("https://www.cbsnews.com/news/fact-check-state-of-the-union-2026/"),
                 ev("https://apnews.com/article/economy-jobs-report-2026")]),
         ("R2", [ev("https://politifact.example.com/x", tier=SourceTier.FACTCHECK)])],
        utterance=date(2026, 2, 24),
        window=(date(2024, 1, 1), date(2026, 5, 1)))
    log = {e["url"]: e for e in res.excluded_fc}
    assert log["https://www.cbsnews.com/news/fact-check-state-of-the-union-2026/"][
        "reason"] == "vertical:cbsnews.com"
    assert log["https://www.cbsnews.com/news/fact-check-state-of-the-union-2026/"][
        "retriever"] == "R1"
    assert log["https://politifact.example.com/x"]["reason"] == "tier:factcheck"
    assert res.dropped["factcheck-excluded"] == 2
    kept = [it.evidence.source_url for it in res.items]
    assert kept == ["https://apnews.com/article/economy-jobs-report-2026"]


def test_pack_journal_carries_exclusion_log(tmp_path) -> None:
    import json

    from truthbot.verdict.evidence_pack import EvidencePack
    from truthbot.verdict.publish_pipeline import append_packs_journal

    pack = EvidencePack(
        sid="s:0001", window=None, items=[],
        excluded_fc=[{"url": "https://x/fact-check-y", "reason": "path-regex",
                      "retriever": "R1"}])
    path = tmp_path / "packs.jsonl"
    append_packs_journal(path, "s:0001", pack)
    rec = json.loads(path.read_text().splitlines()[0])
    assert rec["excluded_fc"][0]["reason"] == "path-regex"
