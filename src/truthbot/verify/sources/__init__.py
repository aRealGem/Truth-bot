"""Source connectors for evidence gathering."""
from truthbot.verify.sources.base import SourceConnector
from truthbot.verify.sources.brave import BraveSearchConnector
from truthbot.verify.sources.factcheck import FactCheckConnector
from truthbot.verify.sources.government import GovernmentDataConnector

__all__ = [
    "SourceConnector",
    "BraveSearchConnector",
    "FactCheckConnector",
    "GovernmentDataConnector",
]
