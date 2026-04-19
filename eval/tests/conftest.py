"""
Shared fixtures for eval/tests/.
"""
import sys
from pathlib import Path

# Add eval/ to sys.path so `import evolver.*` works
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


@pytest.fixture
def tmp_dir(tmp_path):
    """A fresh temporary directory (pathlib.Path)."""
    return tmp_path


@pytest.fixture
def sample_reference():
    """Minimal list of 3 reference claim dicts."""
    return [
        {
            "id": 1,
            "claim": "The unemployment rate fell to 3.4 percent.",
            "verdict": "TRUE",
            "explanation": "BLS data shows unemployment reached 3.4% in January 2023.",
            "sources": ["R1", "R4"],
            "confidence_note": "High confidence; corroborated by BLS.",
        },
        {
            "id": 2,
            "claim": "Border crossings are at a 25-year low.",
            "verdict": "FALSE",
            "explanation": "CBP data shows crossings were at record highs, not lows.",
            "sources": ["R5"],
            "confidence_note": "High confidence.",
        },
        {
            "id": 3,
            "claim": "Egg prices are down 60 percent.",
            "verdict": "MISLEADING",
            "explanation": "Egg prices dropped from a spike but remain above 2022 baseline.",
            "sources": ["R6"],
            "confidence_note": "Medium confidence.",
        },
    ]


@pytest.fixture
def sample_transcript():
    """~200-char fake political speech text."""
    return (
        "My fellow Americans, thanks to our policies inflation has fallen dramatically. "
        "The unemployment rate stands at 3.4 percent. Border crossings are at a 25-year low. "
        "We have delivered results."
    )


@pytest.fixture
def sample_claims():
    """List of 5 claim dicts (extraction output format)."""
    return [
        {
            "text": "The unemployment rate fell to 3.4 percent.",
            "category": "jobs_employment",
            "is_checkable": True,
        },
        {
            "text": "Border crossings are at a 25-year low.",
            "category": "immigration_border",
            "is_checkable": True,
        },
        {
            "text": "Egg prices are down 60 percent.",
            "category": "food_prices",
            "is_checkable": True,
        },
        {
            "text": "Inflation has fallen dramatically.",
            "category": "inflation",
            "is_checkable": True,
        },
        {
            "text": "We have delivered the greatest economy in history.",
            "category": "other",
            "is_checkable": False,
        },
    ]


@pytest.fixture
def sample_verdicts():
    """List of 3 verdict dicts (synthesis output format)."""
    return [
        {
            "claim_text": "The unemployment rate fell to 3.4 percent.",
            "label": "True",
            "explanation": "BLS data confirms unemployment hit 3.4% in January 2023.",
            "support_count": 3,
            "contradict_count": 0,
        },
        {
            "claim_text": "Border crossings are at a 25-year low.",
            "label": "False",
            "explanation": "CBP records show crossings were at record highs during this period.",
            "support_count": 0,
            "contradict_count": 2,
        },
        {
            "claim_text": "Egg prices are down 60 percent.",
            "label": "Misleading",
            "explanation": "Prices fell from a spike but remain elevated vs. 2022 baseline per USDA data.",
            "support_count": 1,
            "contradict_count": 1,
        },
    ]
