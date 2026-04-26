"""Phase 3a — pin the OpenAI batch-vs-sidecar routing toggle.

The pipeline splits adapters into ``batch_adapters`` and ``sidecar_adapters``
via ``_routes_to_batch``. Without an override, ``supports_batch=True``
adapters go to the Batch API, ``False`` adapters go through the sidecar
live path. Phase 3a introduces a single override: when
``settings.openai_live_mode`` is set, OpenAI routes to the sidecar so
runs complete in seconds instead of hours.

These tests pin the contract:

  * default (no env var) → OpenAI keeps the batch route (preserves
    the 50% batch discount for scheduled jobs).
  * ``TRUTHBOT_OPENAI_LIVE=1`` → OpenAI flips to sidecar; Anthropic
    stays on the batch path (the toggle is OpenAI-specific by design;
    Anthropic's batch SLA is faster and the live equivalent costs more).
  * Adapters that already declare ``supports_batch=False`` (Gemini,
    Grok) are unaffected by the toggle.
"""

from __future__ import annotations

from types import SimpleNamespace

from truthbot.pipeline import _routes_to_batch


def _adapter(name: str, supports_batch: bool) -> SimpleNamespace:
    return SimpleNamespace(adapter_name=name, supports_batch=supports_batch)


def _settings(*, openai_live_mode: bool) -> SimpleNamespace:
    return SimpleNamespace(openai_live_mode=openai_live_mode)


def test_default_routing_keeps_openai_on_batch() -> None:
    """No override → OpenAI + Anthropic batch, Gemini + Grok sidecar.

    Preserves the 50% batch discount for scheduled / long-running jobs
    where end-to-end latency is acceptable.
    """
    settings = _settings(openai_live_mode=False)
    assert _routes_to_batch(_adapter("openai", True), settings) is True
    assert _routes_to_batch(_adapter("anthropic", True), settings) is True
    assert _routes_to_batch(_adapter("gemini", False), settings) is False
    assert _routes_to_batch(_adapter("xai", False), settings) is False


def test_openai_live_mode_routes_openai_to_sidecar() -> None:
    """``TRUTHBOT_OPENAI_LIVE=1`` flips OpenAI without touching siblings.

    This is the whole point of Phase 3a: trade 50% of OpenAI's per-run
    cost for sub-minute completion. Anthropic's batch SLA is fast
    enough not to need promotion (and Anthropic's live web search
    grounding mode is more expensive), so the toggle must be
    surgically OpenAI-only.
    """
    settings = _settings(openai_live_mode=True)
    assert _routes_to_batch(_adapter("openai", True), settings) is False
    assert _routes_to_batch(_adapter("anthropic", True), settings) is True
    assert _routes_to_batch(_adapter("gemini", False), settings) is False
    assert _routes_to_batch(_adapter("xai", False), settings) is False


def test_routing_falsy_when_supports_batch_missing() -> None:
    """Adapters without ``supports_batch`` are treated as sidecar."""
    settings = _settings(openai_live_mode=False)
    bare = SimpleNamespace(adapter_name="custom")
    assert _routes_to_batch(bare, settings) is False


def test_settings_openai_live_mode_reads_env(monkeypatch) -> None:
    """``TRUTHBOT_OPENAI_LIVE=1|true|yes|on`` activates live mode; others don't."""
    from truthbot.config import Settings

    s = Settings()

    monkeypatch.delenv("TRUTHBOT_OPENAI_LIVE", raising=False)
    assert s.openai_live_mode is False

    for truthy in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("TRUTHBOT_OPENAI_LIVE", truthy)
        assert s.openai_live_mode is True, f"expected truthy for {truthy!r}"

    for falsy in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("TRUTHBOT_OPENAI_LIVE", falsy)
        assert s.openai_live_mode is False, f"expected falsy for {falsy!r}"
