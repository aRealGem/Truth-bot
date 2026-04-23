"""
Tests for ``truthbot.verify.adapters.key_sanity.validate_api_key``.

Covers the "key present but garbage" failure mode that burned the
2026-04-22 SOTU batch submit: keys pasted into ``.env`` were truncated
at column 80 by terminal line-wrap and ended in a stray ``>`` redirect
character, so every provider call 401'd before any useful work happened.

These tests use **dummy** key shapes only. No real credentials appear here.
"""

from __future__ import annotations

import pytest

from truthbot.verify.adapters.key_sanity import KeyCheck, validate_api_key


# ── Plausible-but-fake keys per provider (correct prefix, above min length).
#    ``x`` padding is intentional — we want shape-plausible, not valid.
_VALID_SHAPED: dict[str, str] = {
    "anthropic": "sk-ant-api03-" + ("x" * 100),   # 113 chars
    "openai":    "sk-proj-" + ("x" * 80),          # 88 chars
    "openai_legacy": "sk-" + ("x" * 45),           # 48 chars (old-style)
    "gemini":    "AIza" + ("x" * 35),              # 39 chars
    "xai":       "xai-" + ("x" * 80),              # 84 chars
}


# ── Accept cases ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "provider,value",
    [
        ("anthropic", _VALID_SHAPED["anthropic"]),
        ("openai",    _VALID_SHAPED["openai"]),
        ("openai",    _VALID_SHAPED["openai_legacy"]),
        ("gemini",    _VALID_SHAPED["gemini"]),
        ("xai",       _VALID_SHAPED["xai"]),
    ],
)
def test_plausible_key_is_accepted(provider, value):
    result = validate_api_key(provider, value)
    assert result.ok, f"{provider} rejected plausible key: {result.reason}"
    assert result.reason == ""


def test_provider_name_is_case_insensitive():
    v = _VALID_SHAPED["anthropic"]
    assert validate_api_key("ANTHROPIC", v).ok
    assert validate_api_key("Anthropic", v).ok


def test_unknown_provider_applies_only_generic_checks():
    """
    For providers we haven't characterized (e.g. brave, fred, bluesky app
    pw), we still reject obviously bad shapes but do not enforce a prefix
    or length floor.
    """
    assert validate_api_key("brave", "BSA-some-opaque-token-abcdef").ok
    assert not validate_api_key("brave", "").ok
    assert not validate_api_key("brave", "trailing-gt>").ok
    assert not validate_api_key("brave", "   ").ok


# ── Generic reject cases (apply regardless of provider) ──────────────────


@pytest.mark.parametrize(
    "bad_value,expected_reason_substr",
    [
        (None,                                  "missing"),
        ("",                                    "empty"),
        ("   ",                                 "whitespace-only"),
        ("\t",                                  "whitespace-only"),
        ("\n",                                  "whitespace-only"),
        (" sk-ant-" + "x" * 100,                "leading/trailing whitespace"),
        ("sk-ant-" + "x" * 100 + " ",           "leading/trailing whitespace"),
        ("sk-ant-" + "x" * 100 + "\n",          "leading/trailing whitespace"),
        ("sk-ant-" + "x" * 50 + "\n" + "x" * 50, "control character"),
        ("sk-ant-" + "x" * 50 + "\t" + "x" * 50, "control character"),
        ("sk-ant-" + "x" * 50 + " "  + "x" * 50, "internal whitespace"),
    ],
)
def test_generic_bad_shapes_rejected(bad_value, expected_reason_substr):
    result = validate_api_key("anthropic", bad_value)
    assert not result.ok
    assert expected_reason_substr in result.reason, (
        f"expected reason to contain {expected_reason_substr!r}, got {result.reason!r}"
    )


# ── Provider-specific reject cases ───────────────────────────────────────


@pytest.mark.parametrize(
    "provider,bad_value,expected_reason_substr",
    [
        # Wrong vendor prefix — pasted into the wrong slot.
        # NOTE: we can't reliably reject "anthropic key pasted into OPENAI slot"
        # because OpenAI's real prefix is just "sk-" and "sk-ant-*" starts with
        # "sk-" — so that specific cross-paste falls through the shape check.
        # See ``key_sanity.validate_api_key`` docstring for the limitation.
        ("anthropic", "sk-proj-" + "x" * 100, "prefix"),
        ("anthropic", "xai-"     + "x" * 100, "prefix"),
        ("openai",    "xai-"     + "x" * 80,  "prefix"),
        ("openai",    "AIza"     + "x" * 50,  "prefix"),
        ("gemini",    "sk-ant-"  + "x" * 100, "prefix"),
        ("gemini",    "sk-"      + "x" * 40,  "prefix"),
        ("xai",       "AIza"     + "x" * 80,  "prefix"),
        ("xai",       "sk-ant-"  + "x" * 100, "prefix"),

        # Right prefix, too short — the exact 2026-04-22 shape after the '>'
        # is stripped (real anthropic keys are ~108 chars, paste gave ~55).
        ("anthropic", "sk-ant-" + "x" * 40,  "too short"),
        ("openai",    "sk-"     + "x" * 10,  "too short"),
        ("gemini",    "AIza"    + "x" * 5,   "too short"),
        ("xai",       "xai-"    + "x" * 10,  "too short"),
    ],
)
def test_provider_specific_bad_shapes_rejected(
    provider, bad_value, expected_reason_substr
):
    result = validate_api_key(provider, bad_value)
    assert not result.ok
    assert expected_reason_substr in result.reason, (
        f"{provider}: expected reason to contain {expected_reason_substr!r}, "
        f"got {result.reason!r}"
    )


# ── Regression: the exact 2026-04-22 SOTU-blocker shape ──────────────────


@pytest.mark.parametrize(
    "provider,truncated",
    [
        # Approximate the shapes described in STATUS.md: ~55-char body for
        # anthropic/openai, ~59-char body for xai, each terminated by '>'.
        ("anthropic", "sk-ant-api03-" + ("x" * 40) + ">"),   # ~54 + '>'
        ("openai",    "sk-proj-"      + ("x" * 47) + ">"),   # ~55 + '>'
        ("xai",       "xai-"          + ("x" * 55) + ">"),   # ~59 + '>'
    ],
)
def test_trailing_gt_regression_2026_04_22(provider, truncated):
    """
    Three ``.env`` keys were truncated by terminal line-wrap on paste and
    ended in a stray ``>``. The validator must catch this specific shape
    BEFORE falling through to length or prefix checks so the operator gets
    a reason that actually points at the paste problem.
    """
    result = validate_api_key(provider, truncated)
    assert not result.ok, f"{provider} accepted truncated key ending in '>'"
    assert "'>'" in result.reason, (
        f"{provider}: expected trailing-'>' reason, got {result.reason!r}"
    )


# ── Contract of the KeyCheck dataclass ───────────────────────────────────


def test_keycheck_is_frozen_and_truthy_on_ok():
    good = validate_api_key("anthropic", _VALID_SHAPED["anthropic"])
    bad = validate_api_key("anthropic", "")
    assert isinstance(good, KeyCheck)
    assert isinstance(bad, KeyCheck)
    assert good.ok is True and good.reason == ""
    assert bad.ok is False and bad.reason != ""
    with pytest.raises((AttributeError, Exception)):
        good.ok = False  # type: ignore[misc]  # frozen dataclass
