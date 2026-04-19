"""
Tests for eval/evolver/preflight.py
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from evolver.preflight import PreflightChecker, PreflightResult


def _checker():
    return PreflightChecker()


# ── API key checks ─────────────────────────────────────────────────────────────

def test_missing_api_key_is_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = _checker()
    c.check_api_keys("anthropic")
    assert any("ANTHROPIC_API_KEY" in e for e in c._errors)


def test_present_api_key_passes(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    c = _checker()
    c.check_api_keys("anthropic")
    assert not c._errors


def test_missing_openai_key_is_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = _checker()
    c.check_api_keys("openai")
    assert any("OPENAI_API_KEY" in e for e in c._errors)


# ── Transcript checks ──────────────────────────────────────────────────────────

def test_missing_transcript_is_error(tmp_path):
    c = _checker()
    c.check_transcript(tmp_path / "nonexistent.txt")
    assert c._errors


def test_short_transcript_is_warning(tmp_path):
    f = tmp_path / "speech.txt"
    f.write_text("A" * 200)  # > 0 but < 500
    c = _checker()
    c.check_transcript(f)
    # 200 chars is <= 500, so it's an error not a warning
    assert c._errors

def test_transcript_between_500_and_5000_is_warning(tmp_path):
    f = tmp_path / "speech.txt"
    f.write_text("A" * 600)  # > 500 but < 5000
    c = _checker()
    c.check_transcript(f)
    assert not c._errors
    assert c._warnings  # short but valid warning


def test_valid_transcript_passes(tmp_path):
    f = tmp_path / "speech.txt"
    f.write_text("A" * 6000)
    c = _checker()
    c.check_transcript(f)
    assert not c._errors
    assert not c._warnings


# ── Reference checks ───────────────────────────────────────────────────────────

def test_valid_reference_passes(tmp_path):
    ref = [
        {"id": i, "claim": f"Claim {i}", "verdict": "TRUE",
         "explanation": "e", "sources": [], "confidence_note": ""}
        for i in range(10)
    ]
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(ref))
    c = _checker()
    c.check_reference(p)
    assert not c._errors
    assert not c._warnings  # >= 10 items


def test_malformed_reference_is_error(tmp_path):
    # Missing "verdict" key
    ref = [{"id": 1, "claim": "test"}]  # no verdict
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(ref))
    c = _checker()
    c.check_reference(p)
    assert c._errors


def test_reference_too_few_items_is_error(tmp_path):
    ref = [{"id": 1, "claim": "c", "verdict": "TRUE"}]  # < 3
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(ref))
    c = _checker()
    c.check_reference(p)
    assert c._errors


def test_reference_invalid_json_is_error(tmp_path):
    p = tmp_path / "ref.json"
    p.write_text("not json {{")
    c = _checker()
    c.check_reference(p)
    assert c._errors


# ── Model deprecation ──────────────────────────────────────────────────────────

def test_deprecated_model_is_warning():
    c = _checker()
    c.check_model_not_deprecated("claude-3-5-haiku-20241022")
    assert c._warnings
    assert not c._errors


def test_current_model_passes():
    c = _checker()
    c.check_model_not_deprecated("claude-opus-4-9")
    assert not c._warnings
    assert not c._errors


# ── Budget checks ──────────────────────────────────────────────────────────────

def test_zero_budget_live_run_is_error():
    c = _checker()
    c.check_budget(budget_usd=0.0, dry_run=False)
    assert c._errors


def test_negative_budget_is_error():
    c = _checker()
    c.check_budget(budget_usd=-1.0, dry_run=False)
    assert c._errors


def test_low_budget_live_run_is_warning():
    c = _checker()
    c.check_budget(budget_usd=0.10, dry_run=False)
    assert not c._errors
    assert c._warnings


def test_dry_run_gets_warning():
    c = _checker()
    c.check_budget(budget_usd=0.0, dry_run=True)
    assert not c._errors  # dry_run suppresses budget error
    assert c._warnings
    assert "DRY-RUN" in c._warnings[0]


# ── Gene pool consistency ──────────────────────────────────────────────────────

def test_gene_pool_consistency_passes():
    """The _GENE_POOL_SIZES dict should match actual variant list lengths."""
    c = _checker()
    c.check_gene_pool_consistency()
    assert not c._errors, f"Gene pool mismatch: {c._errors}"


# ── run_all integration ────────────────────────────────────────────────────────

def test_run_all_returns_preflight_result(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    c = _checker()
    result = c.run_all(provider="anthropic", dry_run=True, budget_usd=5.0)
    assert isinstance(result, PreflightResult)
    # dry_run warning present but no errors (assuming gene pool is consistent)
    assert any("DRY-RUN" in w for w in result.warnings)
