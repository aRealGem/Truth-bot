"""TB-10t: Bluesky publisher — character limits, thread structure, AT Protocol."""

from __future__ import annotations

import pytest

from truthbot.models import Confidence, Verdict, VerdictLabel
from truthbot.publish.bluesky import BlueskyPublisher, _MAX_POST_CHARS, _VERDICT_EMOJI


@pytest.fixture
def publisher():
    return BlueskyPublisher(handle="", app_password="")


class TestBlueskyPublisher:
    def test_not_configured_without_creds(self, publisher):
        assert not publisher.is_configured()

    def test_configured_with_creds(self):
        p = BlueskyPublisher(handle="test.bsky.social", app_password="xxxx-xxxx")
        assert p.is_configured()

    def test_post_report_returns_none_when_unconfigured(self, publisher, sample_report):
        result = publisher.post_report(sample_report)
        assert result is None

    def test_summary_post_within_char_limit(self, publisher, sample_report):
        text = publisher.format_summary_post(sample_report)
        assert len(text) <= _MAX_POST_CHARS

    def test_summary_post_contains_speaker(self, publisher, sample_report):
        text = publisher.format_summary_post(sample_report)
        assert "Test Politician" in text

    def test_summary_post_contains_verdict_emoji(self, publisher, sample_report):
        text = publisher.format_summary_post(sample_report)
        # Should have at least one emoji from the verdict emoji map
        assert any(emoji in text for emoji in _VERDICT_EMOJI.values())

    def test_verdict_post_within_char_limit(self, publisher, sample_claim, sample_verdict):
        text = publisher.format_verdict_post(sample_claim.text, sample_verdict)
        assert len(text) <= _MAX_POST_CHARS

    def test_verdict_post_contains_label(self, publisher, sample_claim, sample_verdict):
        text = publisher.format_verdict_post(sample_claim.text, sample_verdict)
        assert sample_verdict.label.value in text

    @pytest.mark.parametrize("label", list(VerdictLabel))
    def test_all_labels_have_emoji(self, label):
        assert label in _VERDICT_EMOJI

    def test_very_long_claim_truncated(self, publisher, sample_claim):
        long_claim = "This is a very long claim. " * 20
        verdict = Verdict(
            claim_id=sample_claim.id,
            label=VerdictLabel.FALSE,
            confidence=Confidence.HIGH,
            explanation="Brief explanation.",
        )
        text = publisher.format_verdict_post(long_claim, verdict)
        assert len(text) <= _MAX_POST_CHARS

    def test_post_url_format(self):
        ref = {"uri": "at://did:plc:abc123/app.bsky.feed.post/rkey789", "cid": "cid"}
        url = BlueskyPublisher._post_url(ref, "myhandle.bsky.social")
        assert "rkey789" in url
        assert "myhandle.bsky.social" in url
        assert url.startswith("https://bsky.app")
