"""TB-04t: Source connector interface — contract, mock tests, error handling."""

from __future__ import annotations

import pytest

from truthbot.models import Claim, Evidence, SourceTier, Transcript
from truthbot.verify.sources.base import SourceConnector
from truthbot.verify.sources.brave import BraveSearchConnector
from truthbot.verify.sources.factcheck import FactCheckConnector
from truthbot.verify.sources.government import GovernmentDataConnector


# ── Interface contract ────────────────────────────────────────────────────────

class TestSourceConnectorInterface:
    """Verify that all connectors implement the SourceConnector protocol."""

    @pytest.mark.parametrize(
        "connector_cls",
        [BraveSearchConnector, FactCheckConnector, GovernmentDataConnector],
    )
    def test_is_subclass(self, connector_cls):
        assert issubclass(connector_cls, SourceConnector)

    @pytest.mark.parametrize(
        "connector_cls",
        [BraveSearchConnector, FactCheckConnector, GovernmentDataConnector],
    )
    def test_has_search_method(self, connector_cls):
        assert hasattr(connector_cls, "search")
        assert callable(connector_cls.search)

    @pytest.mark.parametrize(
        "connector_cls",
        [BraveSearchConnector, FactCheckConnector, GovernmentDataConnector],
    )
    def test_has_source_name(self, connector_cls):
        assert connector_cls.source_name

    @pytest.mark.parametrize(
        "connector_cls",
        [BraveSearchConnector, FactCheckConnector, GovernmentDataConnector],
    )
    def test_has_tier(self, connector_cls):
        assert isinstance(connector_cls.tier, SourceTier)


# ── No-key behavior ───────────────────────────────────────────────────────────

class TestBraveSearchConnector:
    def test_no_key_returns_empty(self, sample_claim):
        connector = BraveSearchConnector(api_key="")
        results = connector.search(sample_claim)
        assert isinstance(results, list)
        assert results == []

    def test_not_available_without_key(self):
        connector = BraveSearchConnector(api_key="")
        assert not connector.is_available()

    def test_available_with_key(self):
        connector = BraveSearchConnector(api_key="fake-key")
        assert connector.is_available()

    def test_tier_classification(self, sample_claim):
        connector = BraveSearchConnector(api_key="fake")
        assert connector._classify_tier("https://bls.gov/data") == SourceTier.GOVERNMENT
        assert connector._classify_tier("https://apnews.com/article") == SourceTier.WIRE
        assert connector._classify_tier("https://nytimes.com/article") == SourceTier.ESTABLISHED
        assert connector._classify_tier("https://politifact.com/fact") == SourceTier.FACTCHECK
        assert connector._classify_tier("https://randomsite.com") == SourceTier.OTHER


class TestGovernmentDataConnector:
    def test_always_available(self):
        connector = GovernmentDataConnector()
        assert connector.is_available()

    def test_economic_keyword_triggers_stub(self, sample_claim):
        """Without a FRED key, should return stub evidence for economic claims."""
        import os
        os.environ.pop("FRED_API_KEY", None)
        connector = GovernmentDataConnector(fred_api_key="")
        results = connector.search(sample_claim)
        assert isinstance(results, list)
        # sample_claim has "unemployment" keyword → should return stub
        assert len(results) >= 1
        assert results[0].source_tier == SourceTier.GOVERNMENT

    def test_non_economic_claim_returns_empty(self):
        connector = GovernmentDataConnector(fred_api_key="")
        claim = Claim(
            transcript_id="t1",
            text="The president attended a state dinner in Paris.",
            speaker="Test",
        )
        results = connector.search(claim)
        assert results == []


class TestFactCheckConnector:
    def test_no_key_returns_empty(self, sample_claim):
        connector = FactCheckConnector(brave_api_key="")
        results = connector.search(sample_claim)
        assert results == []

    def test_not_available_without_key(self):
        connector = FactCheckConnector(brave_api_key="")
        assert not connector.is_available()

    def test_make_evidence_helper(self, sample_claim):
        connector = GovernmentDataConnector()
        ev = connector._make_evidence(
            sample_claim,
            source_url="https://bls.gov/test",
            snippet="Test snippet",
        )
        assert isinstance(ev, Evidence)
        assert ev.claim_id == sample_claim.id
        assert ev.source_tier == SourceTier.GOVERNMENT
