"""Tests for execution.rl_agent (T4.1)."""

from __future__ import annotations

import os

from execution.rl_agent import PPOExecutionAgent, rl_execution_enabled


def test_rl_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MLCOUNCIL_RL_EXECUTION_ENABLED", raising=False)
    assert not rl_execution_enabled()
    agent = PPOExecutionAgent(adv_lookup={"AAPL": 1_000_000})
    decision = agent.execute("AAPL", 100, mid_price=150.0)
    assert decision.strategy in ("market", "vwap")
    assert len(decision.child_orders) >= 1


def test_rl_flag_enabled(monkeypatch):
    monkeypatch.setenv("MLCOUNCIL_RL_EXECUTION_ENABLED", "true")
    assert rl_execution_enabled()
