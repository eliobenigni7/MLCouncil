"""Tests for CQR sizing and stacking meta-learner (T3.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def calib_xy():
    rng = np.random.default_rng(7)
    n, p = 200, 6
    X = rng.standard_normal((n, p))
    y = 1.5 * X[:, 0] - X[:, 1] + rng.normal(0, 0.3, n)
    return X, y


class TestCQRPositionSizer:
    def test_multiplier_range(self, calib_xy):
        from council.sizing.cqr import CQRPositionSizer

        X, y = calib_xy
        sizer = CQRPositionSizer(coverage=0.85)
        sizer.fit(X, y)
        tickers = [f"T{i}" for i in range(20)]
        X_live = np.random.default_rng(1).standard_normal((20, X.shape[1]))
        signal = pd.Series(np.linspace(-1, 1, 20), index=tickers)
        mult = sizer.compute_position_multipliers(signal, X_live)
        assert mult.min() >= 0.2 - 1e-9
        assert mult.max() <= 2.0 + 1e-9

    def test_get_intervals_after_fit(self, calib_xy):
        from council.sizing.cqr import CQRPositionSizer

        X, y = calib_xy
        sizer = CQRPositionSizer(coverage=0.90)
        sizer.fit(X, y)
        preds, lo, hi = sizer.get_intervals(X[:10])
        assert len(preds) == 10
        assert np.all(hi >= lo)

    def test_position_sizing_mode_default(self, monkeypatch):
        monkeypatch.delenv("MLCOUNCIL_POSITION_SIZING", raising=False)
        from council.sizing.cqr import get_position_sizer, position_sizing_mode
        from council.sizing.conformal import ConformalPositionSizer

        assert position_sizing_mode() == "conformal"
        assert isinstance(get_position_sizer(), ConformalPositionSizer)


class TestCQRQuintileCoverage:
    def test_evaluate_coverage_by_quintile(self, calib_xy):
        from council.sizing.cqr import CQRPositionSizer, evaluate_cqr_coverage_by_vol_quintile

        X, y = calib_xy
        sizer = CQRPositionSizer(coverage=0.85)
        sizer.fit(X, y)
        _, lo, hi = sizer.get_intervals(X)
        vol = np.abs(X[:, 0])
        report = evaluate_cqr_coverage_by_vol_quintile(y, lo, hi, vol)
        assert "empirical_coverage" in report.columns
        assert len(report) >= 1


class TestStackingMetaLearner:
    def test_ridge_stack_predict(self):
        from council.sizing.cqr import StackingMetaLearner

        rng = np.random.default_rng(0)
        idx = [f"S{i}" for i in range(50)]
        base = pd.DataFrame(
            {
                "lgbm": rng.standard_normal(50),
                "sentiment": rng.standard_normal(50),
            },
            index=idx,
        )
        y = base["lgbm"] * 0.6 + base["sentiment"] * 0.4 + rng.normal(0, 0.1, 50)
        meta = StackingMetaLearner(use_xgb=False)
        meta.fit(base, y)
        pred = meta.predict(base)
        assert len(pred) == 50
        assert pred.name == "stacked_signal"

    def test_save_load_roundtrip(self, tmp_path):
        from council.sizing.cqr import StackingMetaLearner

        rng = np.random.default_rng(2)
        idx = [f"S{i}" for i in range(30)]
        base = pd.DataFrame(
            {"lgbm": rng.standard_normal(30), "sentiment": rng.standard_normal(30)},
            index=idx,
        )
        y = base.mean(axis=1)
        meta = StackingMetaLearner(use_xgb=False)
        meta.fit(base, y)
        path = tmp_path / "stack.pkl"
        meta.save(path)
        loaded = StackingMetaLearner.load(path)
        np.testing.assert_allclose(loaded.predict(base).values, meta.predict(base).values)


class TestStackingShadow:
    def test_log_stacking_shadow(self, tmp_path):
        from council.sizing.cqr import log_stacking_shadow

        council = pd.Series([1.0, -1.0], index=["A", "B"])
        stacked = pd.Series([0.5, -0.5], index=["A", "B"])
        path = log_stacking_shadow("2024-01-15", council, stacked, out_dir=tmp_path)
        assert path.exists()
