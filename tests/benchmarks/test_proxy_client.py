"""Offline tests for the truth-bot proxy CLIENT identity resolver."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "eval" / "benchmarks"))

import proxy_client as pc


def test_client_identity_constants():
    assert pc.CLIENT == "truth-bot" and pc.KEY_LABEL == "truth-bot"
    assert pc.CANONICAL_KEY_ENV == "LITELLM_TRUTHBOT_KEY"


def test_resolve_prefers_client_key():
    env = {"LITELLM_TRUTHBOT_KEY": "sk-a", "LITELLM_PCA_KEY": "sk-b", "LITELLM_KEY": "sk-c"}
    assert pc.resolve_key_env(env) == "LITELLM_TRUTHBOT_KEY"


def test_resolve_falls_back_to_legacy():
    assert pc.resolve_key_env({"LITELLM_PCA_KEY": "sk-b"}) == "LITELLM_PCA_KEY"
    assert pc.resolve_key_env({"LITELLM_KEY": "sk-c"}) == "LITELLM_KEY"


def test_resolve_defaults_to_canonical_when_unset():
    # nothing set → name the canonical var (so the guard message is right)
    assert pc.resolve_key_env({}) == "LITELLM_TRUTHBOT_KEY"


def test_key_present():
    assert pc.key_present({"LITELLM_PCA_KEY": "sk"}) is True
    assert pc.key_present({}) is False
    assert pc.key_present({"LITELLM_TRUTHBOT_KEY": ""}) is False   # empty ⇒ absent


def test_base_url_default_and_override():
    assert pc.base_url({}) == "http://127.0.0.1:4141"
    assert pc.base_url({"LITELLM_BASE_URL": "http://host:9"}) == "http://host:9"


def test_blocked_msg_names_canonical_var():
    assert "LITELLM_TRUTHBOT_KEY" in pc.BLOCKED_MSG
