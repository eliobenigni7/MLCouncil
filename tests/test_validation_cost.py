"""Tests for cost calibration promotion gate."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from backtest.validation import (
    CostCalibrationPromotionResult,
    revert_to_static_cost_calibration,
    validate_cost_calibration_promotion,
)
from council.cost_calibration import CalibrationArtifact


def _artifact(**kwargs) -> CalibrationArtifact:
    defaults = dict(
        generated_at=datetime(2026, 5, 21, tzinfo=timezone.utc),
        calibration_window_end=datetime(2026, 5, 21, tzinfo=timezone.utc),
        fill_sample_count=120,
        min_fills=30,
        kappa_by_ticker={"AAPL": 2.5},
        fill_count_by_ticker={"AAPL": 60},
        kappa_by_tier={"mega": 2.8},
        fill_count_by_tier={"mega": 60},
    )
    defaults.update(kwargs)
    return CalibrationArtifact(**defaults)


class TestValidateCostCalibrationPromotion:
    def test_passes_when_calibrated_sharpe_ok(self):
        result = validate_cost_calibration_promotion(
            {"sharpe": 1.0, "turnover": 0.2},
            {"sharpe": 1.05, "turnover": 0.21},
            artifact=_artifact(),
            median_is_bps=2.0,
            median_lookup_bps=3.0,
        )
        assert result.passed
        assert result.reasons == []

    def test_fails_on_sharpe_regression(self):
        result = validate_cost_calibration_promotion(
            {"net_sharpe_static_costs": 1.2},
            {"net_sharpe_calibrated_costs": 0.5},
        )
        assert not result.passed
        assert any("Sharpe" in r for r in result.reasons)

    def test_fails_on_insufficient_tier_fills(self):
        result = validate_cost_calibration_promotion(
            {"sharpe": 1.0, "turnover": 0.2},
            {"sharpe": 1.0, "turnover": 0.2},
            artifact=_artifact(fill_count_by_tier={"mega": 10}),
        )
        assert not result.passed

    def test_revert_writes_override_env(self, tmp_path):
        path = revert_to_static_cost_calibration(tmp_path, reason="test failure")
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "MLCOUNCIL_COST_CALIBRATION_PATH=" in content
