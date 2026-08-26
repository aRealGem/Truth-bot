"""Publish-time OpenAI billing-route guard.

``TRUTHBOT_OPENAI_LIVE=1`` forfeits the 50% batch discount and appeared inline
in documented run commands (STATUS.md), with nothing printed to say so. The
route is now announced before any spend, and live mode has to be acknowledged.
"""
from __future__ import annotations

from argparse import Namespace

import pytest

from truthbot import pipeline


def _args(**kw) -> Namespace:
    return Namespace(**{"ack_live_cost": False, **kw})


def test_batch_route_is_announced(capsys, monkeypatch):
    monkeypatch.delenv("TRUTHBOT_OPENAI_LIVE", raising=False)
    assert pipeline.announce_openai_route(_args()) == "batch"
    out = capsys.readouterr().out
    assert "OpenAI route: batch (50% discount)" in out
    assert "/claim" in out


def test_live_route_without_ack_halts_before_spending(capsys, monkeypatch):
    monkeypatch.setenv("TRUTHBOT_OPENAI_LIVE", "1")
    with pytest.raises(SystemExit) as exc:
        pipeline.announce_openai_route(_args())
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "LIVE (full price)" in out
    assert "No spend attempted." in out


def test_live_route_proceeds_when_acknowledged(capsys, monkeypatch):
    monkeypatch.setenv("TRUTHBOT_OPENAI_LIVE", "1")
    assert pipeline.announce_openai_route(_args(ack_live_cost=True)) == "live"
    assert "LIVE (full price)" in capsys.readouterr().out


def test_announced_retrieval_model_follows_the_env_override(capsys, monkeypatch):
    monkeypatch.delenv("TRUTHBOT_OPENAI_LIVE", raising=False)
    monkeypatch.setenv("TRUTHBOT_R2_MODEL", "gpt-5-mini")
    pipeline.announce_openai_route(_args())
    assert "retrieval model=gpt-5-mini" in capsys.readouterr().out
