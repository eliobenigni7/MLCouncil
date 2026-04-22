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


def test_build_purged_walk_forward_splits_applies_purge_and_embargo():
    from backtest.validation import build_purged_walk_forward_splits

    dates = pd.bdate_range("2024-01-02", periods=12)
    splits = build_purged_walk_forward_splits(
        dates,
        train_window=6,
        test_window=2,
        step=2,
        purge_period=1,
        embargo_period=1,
    )

    assert len(splits) == 2
    assert splits[0].train_start == pd.Timestamp("2024-01-02")
    assert splits[0].train_end == pd.Timestamp("2024-01-08")
    assert splits[0].test_start == pd.Timestamp("2024-01-11")
    assert splits[0].test_end == pd.Timestamp("2024-01-12")


def test_run_walk_forward_analysis_returns_summary_and_window_metrics():
    from backtest.validation import run_walk_forward_analysis

    dates = pd.bdate_range("2024-01-02", periods=24)
    signals = pd.DataFrame(
        {
            "AAA": [1.0 + i * 0.01 for i in range(len(dates))],
            "BBB": [0.4 + i * 0.005 for i in range(len(dates))],
            "CCC": [-0.2 - i * 0.004 for i in range(len(dates))],
        },
        index=dates,
    )
    forward_returns = pd.DataFrame(
        {
            "AAA": [0.002 + i * 0.0001 for i in range(len(dates))],
            "BBB": [0.001 + i * 0.00005 for i in range(len(dates))],
            "CCC": [-0.001 - i * 0.00004 for i in range(len(dates))],
        },
        index=dates,
    )
    benchmark = {"equal_weight": forward_returns.mean(axis=1)}
    regimes = pd.Series(
        ["bull"] * 8 + ["transition"] * 8 + ["bear"] * 8,
        index=dates,
        name="regime",
    )
    components = {
        "technical": signals,
        "sentiment": signals * 0.5,
        "regime": signals * 0.25,
    }

    result = run_walk_forward_analysis(
        signals=signals,
        forward_returns=forward_returns,
        train_window=8,
        test_window=4,
        step=4,
        purge_period=1,
        embargo_period=1,
        benchmark_returns=benchmark,
        regime_labels=regimes,
        component_signals=components,
    )

    assert "window_metrics" in result
    assert "summary" in result
    assert "benchmark_comparison" in result
    assert "regime_performance" in result
    assert "ablation_analysis" in result

    window_metrics = result["window_metrics"]
    summary = result["summary"]
    benchmark_df = result["benchmark_comparison"]
    regime_df = result["regime_performance"]
    ablation_df = result["ablation_analysis"]

    assert not window_metrics.empty
    assert {"train_sharpe", "test_sharpe", "test_max_drawdown", "test_turnover"}.issubset(window_metrics.columns)
    assert summary["walk_forward_window_count"] == len(window_metrics)
    assert 0.0 <= summary["pbo"] <= 1.0
    assert "benchmark" in benchmark_df.columns
    assert "sharpe_delta" in benchmark_df.columns
    assert set(regime_df["regime"]) == {"bull", "transition", "bear"}
    assert not ablation_df.empty


def test_build_benchmark_suite_generates_core_baselines():
    from backtest.validation import build_benchmark_suite

    dates = pd.bdate_range("2024-01-02", periods=30)
    forward_returns = pd.DataFrame(
        {
            "AAA": [0.001 + i * 0.00005 for i in range(len(dates))],
            "BBB": [0.0006 + i * 0.00003 for i in range(len(dates))],
            "CCC": [-0.0004 + i * 0.00001 for i in range(len(dates))],
        },
        index=dates,
    )

    suite = build_benchmark_suite(forward_returns)

    assert {"equal_weight", "momentum_long_only", "inverse_volatility", "vol_target_equal_weight"}.issubset(set(suite))
    for series in suite.values():
        assert isinstance(series, pd.Series)
        assert series.index.equals(dates)


def test_compute_ablation_contributions_reports_marginal_impact():
    from backtest.validation import compute_ablation_contributions

    dates = pd.bdate_range("2024-01-02", periods=20)
    forward_returns = pd.DataFrame(
        {
            "AAA": [0.001 + i * 0.00002 for i in range(len(dates))],
            "BBB": [0.0004 + i * 0.00001 for i in range(len(dates))],
            "CCC": [-0.0003 + i * 0.000015 for i in range(len(dates))],
        },
        index=dates,
    )
    components = {
        "technical": pd.DataFrame(
            {
                "AAA": [1.0 + i * 0.01 for i in range(len(dates))],
                "BBB": [0.5 + i * 0.005 for i in range(len(dates))],
                "CCC": [-0.4 - i * 0.003 for i in range(len(dates))],
            },
            index=dates,
        ),
        "sentiment": pd.DataFrame(
            {
                "AAA": [0.2 + i * 0.002 for i in range(len(dates))],
                "BBB": [0.6 + i * 0.004 for i in range(len(dates))],
                "CCC": [-0.1 - i * 0.001 for i in range(len(dates))],
            },
            index=dates,
        ),
        "regime": pd.DataFrame(
            {
                "AAA": [0.3] * len(dates),
                "BBB": [0.1] * len(dates),
                "CCC": [-0.2] * len(dates),
            },
            index=dates,
        ),
    }

    ablation = compute_ablation_contributions(
        component_signals=components,
        forward_returns=forward_returns,
    )

    assert set(ablation["component"]) == {"technical", "sentiment", "regime"}
    assert "standalone_sharpe" in ablation.columns
    assert "marginal_sharpe_delta" in ablation.columns
    assert "ensemble_sharpe" in ablation.columns
