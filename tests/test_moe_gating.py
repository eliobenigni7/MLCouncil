"""Tests for MoE council gating (T3.1)."""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd
import pytest


class TestMoEGatingNetwork:
    def test_gate_weights_sum_to_one(self):
        from council.moe_gating import MoEGatingNetwork, build_regime_context

        net = MoEGatingNetwork(3, seed=0)
        ctx = build_regime_context("bull", {"lgbm": 0.05, "sentiment": 0.02})
        gate = net.gate_weights(ctx)
        assert len(gate) == 3
        assert abs(float(gate.sum()) - 1.0) < 1e-9
        assert np.all(gate >= 0.0)

    def test_combine_signals_zscored(self):
        from council.moe_gating import MoEGatingNetwork

        tickers = ["A", "B", "C"]
        signals = {
            "lgbm": pd.Series([1.0, -1.0, 0.0], index=tickers),
            "sentiment": pd.Series([0.5, 0.5, -1.0], index=tickers),
        }
        net = MoEGatingNetwork(2, seed=1)
        gate = np.array([0.7, 0.3])
        combined, weights = net.combine_signals(
            signals, ["lgbm", "sentiment"], gate, performance_weights={"lgbm": 1.0, "sentiment": 1.0}
        )
        assert abs(float(combined.std()) - 1.0) < 1e-6 or combined.std() < 1e-9
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_aggregator_mode_default_linear(self, monkeypatch):
        monkeypatch.delenv("MLCOUNCIL_AGGREGATOR_MODE", raising=False)
        from council.moe_gating import aggregator_mode

        assert aggregator_mode() == "linear"

    def test_aggregator_mode_moe(self, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_AGGREGATOR_MODE", "moe")
        from council.moe_gating import aggregator_mode

        assert aggregator_mode() == "moe"


class TestMoECheckpoint:
    def test_save_load_roundtrip(self, tmp_path):
        from council.moe_gating import MoEGatingNetwork

        net = MoEGatingNetwork(2, seed=3)
        path = tmp_path / "moe.pkl"
        net.save(path)
        loaded = MoEGatingNetwork.load(path)
        assert loaded.n_experts == 2
        np.testing.assert_allclose(
            loaded.gate_weight_matrix, net.gate_weight_matrix, rtol=1e-9
        )


class TestCouncilAggregatorMoE:
    def test_moe_mode_logs_gate(self, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_AGGREGATOR_MODE", "moe")
        from council.aggregator import CouncilAggregator

        agg = CouncilAggregator()
        signals = {
            "lgbm": pd.Series([1.0, -0.5], index=["A", "B"]),
            "sentiment": pd.Series([-0.5, 1.0], index=["A", "B"]),
        }
        out = agg.aggregate(signals, "bull", date(2024, 6, 1))
        log = agg._weights_log[date(2024, 6, 1)]
        assert log["aggregator_mode"] == "moe"
        assert log["moe_gate"] is not None
        assert len(out) == 2
