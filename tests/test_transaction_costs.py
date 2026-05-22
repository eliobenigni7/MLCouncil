"""Tests for council.transaction_costs calibration blend."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from council.cost_calibration import CalibrationArtifact, write_calibration
from council.transaction_costs import (
    TransactionCostModel,
    build_slippage_bps_by_ticker,
    estimate_slippage_bps,
    get_active_calibration_version,
    get_calibration_path,
    resolve_slippage_bps,
)


def _make_artifact(**kwargs) -> CalibrationArtifact:
    defaults = dict(
        generated_at=datetime(2026, 5, 21, 23, 0, tzinfo=timezone.utc),
        calibration_window_end=datetime(2026, 5, 21, 21, 30, tzinfo=timezone.utc),
        fill_sample_count=60,
        min_fills=30,
        kappa_by_ticker={"AAPL": 4.0},
        fill_count_by_ticker={"AAPL": 60},
        kappa_by_tier={"mega": 3.5},
        fill_count_by_tier={"mega": 60},
        pipeline_run_id="run-test",
        config_hash="cfg-test",
    )
    defaults.update(kwargs)
    return CalibrationArtifact(**defaults)


class TestResolveSlippageBps:
    def test_no_artifact_returns_lookup(self):
        assert resolve_slippage_bps("AAPL", artifact=None) == estimate_slippage_bps("AAPL")

    def test_full_confidence_uses_kappa(self):
        art = _make_artifact()
        # AAPL lookup=2.0, kappa=4.0, n=60, floor=30 -> alpha=1.0
        assert resolve_slippage_bps("AAPL", artifact=art, confidence_floor=30) == pytest.approx(4.0)

    def test_partial_confidence_blends(self):
        art = _make_artifact(fill_count_by_ticker={"AAPL": 15})
        # alpha = 15/30 = 0.5 -> 0.5*2 + 0.5*4 = 3.0
        assert resolve_slippage_bps("AAPL", artifact=art, confidence_floor=30) == pytest.approx(3.0)

    def test_tier_fallback_when_ticker_missing(self):
        art = _make_artifact(
            kappa_by_ticker={},
            fill_count_by_ticker={},
            kappa_by_tier={"mega": 3.0},
            fill_count_by_tier={"mega": 60},
        )
        assert resolve_slippage_bps("AAPL", artifact=art, confidence_floor=30) == pytest.approx(3.0)


class TestTransactionCostModelFromEnv:
    def test_static_when_calibration_disabled(self, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_COST_CALIBRATION_PATH", "")
        model = TransactionCostModel.from_env()
        assert model.slippage_bps_by_ticker is None
        assert model.calibration_version == ""

    def test_static_lookup_classmethod(self, monkeypatch):
        monkeypatch.delenv("MLCOUNCIL_COST_CALIBRATION_PATH", raising=False)
        model = TransactionCostModel.static_lookup()
        assert model.slippage_bps_by_ticker is None

    def test_loads_calibration_when_valid(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MLCOUNCIL_DYNAMIC_SLIPPAGE", raising=False)
        art = _make_artifact()
        calib = tmp_path / "cost_calibration.json"
        write_calibration(art, path=calib)
        monkeypatch.setenv("MLCOUNCIL_COST_CALIBRATION_PATH", str(calib))

        model = TransactionCostModel.from_env()
        assert model.slippage_bps_by_ticker is not None
        assert model.slippage_bps_for("AAPL") == pytest.approx(4.0)
        assert model.calibration_version != ""

    def test_hash_mismatch_falls_back_to_static(self, tmp_path, monkeypatch):
        art = _make_artifact()
        calib = tmp_path / "cost_calibration.json"
        write_calibration(art, path=calib)
        data = calib.read_text()
        calib.write_text(data.replace("4.0", "9.9"))
        monkeypatch.setenv("MLCOUNCIL_COST_CALIBRATION_PATH", str(calib))

        model = TransactionCostModel.from_env()
        assert model.slippage_bps_by_ticker is None
        assert model.calibration_version == ""

    def test_missing_file_uses_static(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "MLCOUNCIL_COST_CALIBRATION_PATH",
            str(tmp_path / "missing.json"),
        )
        model = TransactionCostModel.from_env()
        assert model.slippage_bps_by_ticker is None

    def test_use_calibration_false_ignores_file(self, tmp_path, monkeypatch):
        art = _make_artifact()
        calib = tmp_path / "cost_calibration.json"
        write_calibration(art, path=calib)
        monkeypatch.setenv("MLCOUNCIL_COST_CALIBRATION_PATH", str(calib))

        model = TransactionCostModel.from_env(use_calibration=False)
        assert model.slippage_bps_by_ticker is None


class TestPerTickerCostEstimation:
    def test_weight_delta_cost_uses_per_ticker_bps(self):
        by_ticker = {"AAPL": 2.0, "MSFT": 8.0}
        model = TransactionCostModel(
            commission_bps=1.0,
            slippage_bps=3.0,
            slippage_bps_by_ticker=by_ticker,
        )
        w_old = np.array([0.5, 0.5])
        w_new = np.array([0.7, 0.3])
        tickers = ["AAPL", "MSFT"]
        cost = model.estimate_cost_from_weight_deltas(
            w_old, w_new, tickers, portfolio_value=100_000.0
        )
        # |dw|: 0.2 each; bps: AAPL 3, MSFT 9
        expected = 100_000 * (0.2 * 3 / 10_000 + 0.2 * 9 / 10_000)
        assert cost == pytest.approx(expected)

    def test_get_active_calibration_version(self, tmp_path, monkeypatch):
        art = _make_artifact()
        calib = tmp_path / "cost_calibration.json"
        version = write_calibration(art, path=calib)
        assert get_active_calibration_version(calib) == version

    def test_get_calibration_path_empty_env(self, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_COST_CALIBRATION_PATH", "")
        assert get_calibration_path() is None
