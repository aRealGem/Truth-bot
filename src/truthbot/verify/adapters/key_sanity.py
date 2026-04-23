"""
API-key sanity validator.

Purpose
-------
Catch the "key present but garbage" failure mode that loses a batch run at
the first live request (the 2026-04-22 SOTU submit burned on exactly this:
three ``.env`` keys truncated at column 80 by a terminal line-wrap on paste,
each ending in a stray ``>`` redirection char).

This module is deliberately **pure**: no I/O, no environment reads, no
logging. Callers (CLI preflight, adapter init, tests) decide what to do
with the result.

Usage
-----
    from truthbot.verify.adapters.key_sanity import validate_api_key

    result = validate_api_key("anthropic", os.environ.get("ANTHROPIC_API_KEY"))
    if not result.ok:
        raise SystemExit(f"ANTHROPIC_API_KEY bad: {result.reason}")

Provider specs are conservative lower bounds on real-world key shapes as of
2026-04. They're intentionally loose: the goal is to reject obviously broken
keys (truncated, whitespace-wrapped, wrong vendor) without rejecting a
legitimate key that the vendor reshapes later.

Known limitation
----------------
OpenAI's prefix is just ``sk-``, which Anthropic's ``sk-ant-*`` also matches.
Pasting an Anthropic key into the ``OPENAI_API_KEY`` slot therefore passes
shape validation — the live request will still 401, but the validator alone
cannot catch that cross-paste. The reverse (OpenAI key in ANTHROPIC slot) is
caught, because ``sk-proj-`` / ``sk-`` do not start with ``sk-ant-``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeyCheck:
    """Outcome of ``validate_api_key``. ``reason`` is empty when ``ok`` is True."""

    ok: bool
    reason: str = ""


# provider -> (required_prefix, min_total_length)
#
# Lengths below are conservative lower bounds observed in the wild:
#   - Anthropic ``sk-ant-*``           ~108 chars end-to-end
#   - OpenAI legacy ``sk-*``            ~51 chars
#   - OpenAI ``sk-proj-*``              ~156 chars
#   - Google ``AIza*``                   39 chars
#   - xAI ``xai-*``                     ~84 chars
#
# We take the shorter legitimate shape as the floor so we don't false-reject
# an older key format.
_PROVIDER_SPECS: dict[str, tuple[str, int]] = {
    "anthropic": ("sk-ant-", 90),
    "openai":    ("sk-",     40),
    "gemini":    ("AIza",    30),
    "xai":       ("xai-",    60),
}

_CONTROL_CHARS = frozenset(chr(i) for i in range(0x20)) | {chr(0x7F)}


def validate_api_key(provider: str, value: str | None) -> KeyCheck:
    """
    Validate ``value`` as an API key for ``provider``.

    Checks, in order:
      1. not None
      2. not empty
      3. not whitespace-only
      4. no leading/trailing whitespace (exact match to its ``str.strip()``)
      5. no trailing ``'>'`` (terminal line-wrap redirect on paste)
      6. no control characters (newline, tab, NUL, DEL, etc.)
      7. no internal whitespace
      8. starts with the provider's required prefix (if known)
      9. at least the provider's minimum length (if known)

    ``provider`` is case-insensitive. Unknown providers skip steps 8-9 and
    are subject to the generic shape checks only, so this function can be
    reused for Brave, FRED, Bluesky app passwords, etc. without lying about
    a shape we haven't characterized.
    """
    if value is None:
        return KeyCheck(False, "missing (None)")
    if value == "":
        return KeyCheck(False, "empty")
    if value.strip() == "":
        return KeyCheck(False, "whitespace-only")
    if value != value.strip():
        return KeyCheck(False, "leading/trailing whitespace")
    if value.endswith(">"):
        return KeyCheck(
            False,
            "trailing '>' (likely truncated terminal paste / shell redirect)",
        )
    if any(c in _CONTROL_CHARS for c in value):
        return KeyCheck(False, "contains control character (newline/tab/etc.)")
    if any(c.isspace() for c in value):
        return KeyCheck(False, "contains internal whitespace")

    spec = _PROVIDER_SPECS.get(provider.lower())
    if spec is None:
        return KeyCheck(True)

    prefix, min_len = spec
    if not value.startswith(prefix):
        return KeyCheck(False, f"missing required '{prefix}' prefix")
    if len(value) < min_len:
        return KeyCheck(
            False,
            f"too short ({len(value)} chars, need >= {min_len})",
        )

    return KeyCheck(True)


__all__ = ["KeyCheck", "validate_api_key"]
