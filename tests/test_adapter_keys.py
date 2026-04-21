"""
Targeted tests: verify each LLM adapter reads its env key correctly.
No real API calls are made — adapters are instantiated with dummy keys
and we confirm (a) the right env var name is used, and (b) is_available()
returns True when set and False when absent.
"""

from __future__ import annotations

import importlib
import pytest


def _reload_adapter(module_path: str, class_name: str):
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


ADAPTER_CASES = [
    ("truthbot.verify.adapters.anthropic", "AnthropicAdapter", "ANTHROPIC_API_KEY", "sk-ant-test-dummy"),
    ("truthbot.verify.adapters.openai",    "OpenAIAdapter",    "OPENAI_API_KEY",    "sk-proj-test-dummy"),
    ("truthbot.verify.adapters.gemini",    "GeminiAdapter",    "GEMINI_API_KEY",    "AIza-test-dummy"),
    ("truthbot.verify.adapters.grok",      "GrokAdapter",      "XAI_API_KEY",       "xai-test-dummy"),
]


@pytest.mark.parametrize("module_path,class_name,env_var,dummy_val", ADAPTER_CASES)
def test_adapter_required_env_key_name(module_path, class_name, env_var, dummy_val):
    """Adapter declares the correct env var name in required_env_key."""
    cls = _reload_adapter(module_path, class_name)
    assert cls.required_env_key == env_var, (
        f"{class_name}.required_env_key is '{cls.required_env_key}', expected '{env_var}'"
    )


@pytest.mark.parametrize("module_path,class_name,env_var,dummy_val", ADAPTER_CASES)
def test_adapter_is_available_when_key_set(module_path, class_name, env_var, dummy_val, monkeypatch):
    """is_available() returns True when the expected env var is set."""
    monkeypatch.setenv(env_var, dummy_val)
    cls = _reload_adapter(module_path, class_name)
    assert cls.is_available(), (
        f"{class_name}.is_available() returned False even though {env_var} is set"
    )


@pytest.mark.parametrize("module_path,class_name,env_var,dummy_val", ADAPTER_CASES)
def test_adapter_not_available_when_key_missing(module_path, class_name, env_var, dummy_val, monkeypatch):
    """is_available() returns False when the env var is absent."""
    monkeypatch.delenv(env_var, raising=False)
    cls = _reload_adapter(module_path, class_name)
    assert not cls.is_available(), (
        f"{class_name}.is_available() returned True even though {env_var} is unset"
    )


@pytest.mark.parametrize("module_path,class_name,env_var,dummy_val", ADAPTER_CASES)
def test_adapter_reads_key_from_env_on_init(module_path, class_name, env_var, dummy_val, monkeypatch):
    """Adapter __init__ picks up the key value from the env var."""
    monkeypatch.setenv(env_var, dummy_val)
    cls = _reload_adapter(module_path, class_name)
    adapter = cls()
    assert adapter._api_key == dummy_val, (
        f"{class_name}._api_key is '{adapter._api_key}', expected '{dummy_val}' from {env_var}"
    )
