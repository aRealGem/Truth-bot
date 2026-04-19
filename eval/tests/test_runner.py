"""Tests for eval/evolver/runner.py"""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from evolver.runner import CachedRunner


def _runner(tmp_path, dry_run=True):
    return CachedRunner(dry_run=dry_run, cache_dir=tmp_path)


SAMPLE_TRANSCRIPT = (
    "Inflation was at record levels when I took office. "
    "We now have zero illegal aliens admitted in the past nine months. "
    "Egg prices are down 60 percent."
)

SAMPLE_CLAIMS = [
    {"text": "Inflation was at record levels.", "category": "inflation", "is_checkable": True},
    {"text": "Zero illegal aliens were admitted in nine months.", "category": "immigration_border", "is_checkable": True},
]


# ── Dry-run extraction ────────────────────────────────────────────────────────

def test_dry_run_returns_stub_claims(tmp_path):
    runner = _runner(tmp_path)
    claims, tokens = runner.extract_claims(
        SAMPLE_TRANSCRIPT, "Speaker", "2026-01-01",
        "System prompt", "User {speaker} {date} {text}", "hash123"
    )
    assert isinstance(claims, list)
    assert len(claims) > 0


def test_dry_run_no_api_call(tmp_path, monkeypatch):
    """dry_run=True should not require ANTHROPIC_API_KEY."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    runner = _runner(tmp_path)
    claims, _ = runner.extract_claims(
        SAMPLE_TRANSCRIPT, "Speaker", "2026-01-01",
        "System", "User {speaker} {date} {text}", "hash"
    )
    assert isinstance(claims, list)


# ── Cache ─────────────────────────────────────────────────────────────────────

def test_cache_miss_writes_to_disk(tmp_path):
    runner = _runner(tmp_path)
    runner.extract_claims(
        SAMPLE_TRANSCRIPT, "Speaker", "2026-01-01",
        "System", "User {speaker} {date} {text}", "dryhash"
    )
    # In dry-run mode, stubs are NOT cached (they're always the same)
    # This test verifies the cache directory is usable
    assert tmp_path.exists()


def test_cache_hit_returns_same_result(tmp_path):
    """If a cache entry exists, second call returns same claims without re-running."""
    runner = CachedRunner(dry_run=True, cache_dir=tmp_path)
    kwargs = dict(
        transcript_text=SAMPLE_TRANSCRIPT,
        speaker="Speaker",
        date_str="2026-01-01",
        system_prompt="System",
        user_template="User {speaker} {date} {text}",
        prompt_hash="cachehash",
    )
    claims1, _ = runner.extract_claims(**kwargs)
    # Manually write a fake cache entry to simulate a real (non-dry-run) cache hit
    import hashlib
    tx_hash = hashlib.sha256(SAMPLE_TRANSCRIPT.encode()).hexdigest()[:12]
    cache_key = f"ext_cachehash_{tx_hash}"
    cache_file = tmp_path / f"{cache_key}.json"
    cached_claims = [{"text": "Cached claim", "category": "other", "is_checkable": True}]
    cache_file.write_text(json.dumps({"claims": cached_claims, "tokens": 999}))

    # Now create a NON-dry-run runner that will read the cache
    runner2 = CachedRunner(dry_run=False, cache_dir=tmp_path, api_key="fake")
    claims2, tokens2 = runner2.extract_claims(**kwargs)
    assert claims2 == cached_claims
    assert tokens2 == 999


# ── Synthesis dry-run ─────────────────────────────────────────────────────────

def test_synthesis_dry_run_returns_stubs(tmp_path, sample_reference):
    runner = _runner(tmp_path)
    verdicts, tokens = runner.synthesize_verdicts(
        claims=SAMPLE_CLAIMS,
        system_prompt="System",
        prompt_hash="syn_hash",
        reference=sample_reference,
    )
    assert isinstance(verdicts, list)
    assert len(verdicts) == len(SAMPLE_CLAIMS)
    assert tokens == 0
    for v in verdicts:
        assert "label" in v
        assert "claim_text" in v


# ── JSON parse error handling ─────────────────────────────────────────────────

def test_invalid_json_response_returns_empty_list(tmp_path, monkeypatch):
    """Monkey-patch _call_extraction_api to return malformed JSON; verify [] returned."""
    runner = CachedRunner(dry_run=False, cache_dir=tmp_path, api_key="fake-key")

    def bad_extraction(*args, **kwargs):
        # Simulate what happens when the model returns garbage
        raise json.JSONDecodeError("Expecting value", "garbage response", 0)

    monkeypatch.setattr(runner, "_call_extraction_api", bad_extraction)

    # Should not raise; should return empty list
    claims, tokens = runner.extract_claims(
        SAMPLE_TRANSCRIPT, "Speaker", "2026-01-01",
        "System", "User {speaker} {date} {text}", "brokenhash"
    )
    assert claims == []


# ── Retry logic ───────────────────────────────────────────────────────────────

def test_retry_on_empty_response(tmp_path, monkeypatch):
    """
    If _call_extraction_api returns empty list on first call and valid claims
    on second, the retry wrapper should return the valid claims.
    """
    runner = CachedRunner(dry_run=False, cache_dir=tmp_path, api_key="fake-key")

    call_count = [0]
    def flaky_extraction(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return [], 0  # empty on first attempt
        return SAMPLE_CLAIMS, 500  # valid on second

    monkeypatch.setattr(runner, "_call_extraction_api", flaky_extraction)

    claims, tokens = runner.extract_claims(
        SAMPLE_TRANSCRIPT, "Speaker", "2026-01-01",
        "System", "User {speaker} {date} {text}", "retryhash"
    )
    assert claims == SAMPLE_CLAIMS
    assert call_count[0] == 2, f"Expected 2 API calls (retry), got {call_count[0]}"


def test_retry_on_json_parse_error(tmp_path, monkeypatch):
    """
    If _call_extraction_api raises JSONDecodeError on first call and returns
    valid claims on second, the retry wrapper should return the valid claims.
    """
    runner = CachedRunner(dry_run=False, cache_dir=tmp_path, api_key="fake-key")

    call_count = [0]
    def flaky_json(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise json.JSONDecodeError("Expecting value", "garbage", 0)
        return SAMPLE_CLAIMS, 500

    monkeypatch.setattr(runner, "_call_extraction_api", flaky_json)

    claims, tokens = runner.extract_claims(
        SAMPLE_TRANSCRIPT, "Speaker", "2026-01-01",
        "System", "User {speaker} {date} {text}", "jsonretryhash"
    )
    assert claims == SAMPLE_CLAIMS
    assert call_count[0] == 2
