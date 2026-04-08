from __future__ import annotations

import pandas as pd
import pytest


def test_build_walk_forward_splits_is_deterministic():
    from backtest.validation import build_walk_forward_splits

    dates = pd.bdate_range("2024-01-02", periods=10)
    splits = build_walk_forward_splits(
        dates,
        train_window=4,
        test_window=2,
        step=2,
    )

    assert len(splits) == 3
    assert splits[0].train_start == pd.Timestamp("2024-01-02")
    assert splits[0].train_end == pd.Timestamp("2024-01-05")
    assert splits[0].test_start == pd.Timestamp("2024-01-08")
    assert splits[0].test_end == pd.Timestamp("2024-01-09")
    assert splits[2].train_start == pd.Timestamp("2024-01-08")
    assert splits[2].test_end == pd.Timestamp("2024-01-15")


def test_summarize_walk_forward_metrics_returns_oos_and_pbo_proxy():
    from backtest.validation import summarize_walk_forward_metrics

    metrics = pd.DataFrame(
        [
            {"window_id": 0, "train_sharpe": 1.2, "test_sharpe": 0.8, "test_max_drawdown": -0.05, "test_turnover": 0.10},
            {"window_id": 1, "train_sharpe": 1.0, "test_sharpe": -0.2, "test_max_drawdown": -0.08, "test_turnover": 0.15},
            {"window_id": 2, "train_sharpe": 0.9, "test_sharpe": 0.4, "test_max_drawdown": -0.04, "test_turnover": 0.12},
        ]
    )

    summary = summarize_walk_forward_metrics(metrics)

    assert summary["walk_forward_window_count"] == 3
    assert summary["oos_sharpe"] == pytest.approx((0.8 - 0.2 + 0.4) / 3)
    assert summary["oos_max_drawdown"] == pytest.approx((-0.05 - 0.08 - 0.04) / 3)
    assert summary["oos_turnover"] == pytest.approx((0.10 + 0.15 + 0.12) / 3)
    assert summary["pbo"] == pytest.approx(1 / 3)
