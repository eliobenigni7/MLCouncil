"""Unit tests for walk-forward champion/challenger promotion (T1.1)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backtest.validation import ModelPromotionResult, validate_model_promotion
from council.walkforward_promotion_gate import (
    evaluate_walk_forward,
    load_champion_metrics,
    run_model_promotion_gate,
)


def _write_signal_cache(
    tmp_path: Path,
    model: str,
    *,
    n_days: int = 400,
) -> None:
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    signals = pd.DataFrame(
        {
            "AAA": [0.5 + i * 0.001 for i in range(n_days)],
            "BBB": [0.3 - i * 0.0005 for i in range(n_days)],
            "CCC": [-0.1 + i * 0.0002 for i in range(n_days)],
        },
        index=dates,
    )
    returns = pd.DataFrame(
        {
            "AAA": [0.002] * n_days,
            "BBB": [0.001] * n_days,
            "CCC": [-0.0005] * n_days,
        },
        index=dates,
    )
    results = tmp_path / "data" / "results"
    results.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(results / f"walkforward_signals_{model}.parquet")
    returns.to_parquet(results / "walkforward_forward_returns.parquet")


class TestValidateModelPromotion:
    def test_passes_when_challenger_meets_thresholds(self):
        result = validate_model_promotion(
            {"oos_sharpe": 0.5, "pbo": 0.4, "walk_forward_window_count": 10},
            {"oos_sharpe": 0.45, "pbo": 0.3, "walk_forward_window_count": 8},
        )
        assert result.passed
        assert result.reasons == []

    def test_fails_on_sharpe_regression(self):
        result = validate_model_promotion(
            {"oos_sharpe": 1.0, "pbo": 0.2, "walk_forward_window_count": 10},
            {"oos_sharpe": 0.5, "pbo": 0.2, "walk_forward_window_count": 10},
        )
        assert not result.passed
        assert any("Sharpe" in r for r in result.reasons)

    def test_fails_on_high_pbo(self):
        result = validate_model_promotion(
            {"oos_sharpe": 0.0, "pbo": 0.2, "walk_forward_window_count": 10},
            {"oos_sharpe": 0.5, "pbo": 0.8, "walk_forward_window_count": 10},
        )
        assert not result.passed
        assert any("PBO" in r for r in result.reasons)

    def test_fails_on_insufficient_windows(self):
        result = validate_model_promotion(
            {"oos_sharpe": 0.0, "pbo": 0.0, "walk_forward_window_count": 10},
            {"oos_sharpe": 0.5, "pbo": 0.2, "walk_forward_window_count": 5},
        )
        assert not result.passed
        assert any("window count" in r for r in result.reasons)


class TestWalkForwardPromotionGate:
    def test_load_champion_metrics_defaults_when_missing(self, tmp_path):
        metrics = load_champion_metrics("lightgbm", tmp_path)
        assert metrics["oos_sharpe"] == 0.0
        assert metrics["walk_forward_window_count"] == 0

    def test_evaluate_walk_forward_produces_enough_windows(self, tmp_path):
        _write_signal_cache(tmp_path, "lightgbm")
        signals_path = tmp_path / "data" / "results" / "walkforward_signals_lightgbm.parquet"
        returns_path = tmp_path / "data" / "results" / "walkforward_forward_returns.parquet"
        signals = pd.read_parquet(signals_path)
        returns = pd.read_parquet(returns_path)
        summary = evaluate_walk_forward(
            signals,
            returns,
            train_window=120,
            test_window=30,
        )
        assert summary["walk_forward_window_count"] >= 8

    def test_gate_passes_with_mocked_validate(self, tmp_path):
        _write_signal_cache(tmp_path, "lightgbm")
        (tmp_path / "data" / "operations").mkdir(parents=True, exist_ok=True)
        champion_path = tmp_path / "data" / "operations" / "walkforward_champion_lightgbm.json"
        champion_path.write_text(
            '{"oos_sharpe": 0.2, "pbo": 0.4, "walk_forward_window_count": 8}',
            encoding="utf-8",
        )

        def _always_pass(_champ, _chall):
            return ModelPromotionResult(passed=True, reasons=[])

        report = run_model_promotion_gate(
            "lightgbm",
            root=tmp_path,
            dry_run=True,
            retrain_fn=lambda *a, **k: {"status": "skipped_dry_run"},
            validate_fn=_always_pass,
        )
        assert report["status"] == "gate_passed_shadow"
        assert report["promotion_passed"] is True
        assert report["shadow_mode"] is True

    def test_gate_skips_without_signal_cache(self, tmp_path):
        report = run_model_promotion_gate(
            "sentiment",
            root=tmp_path,
            dry_run=True,
            retrain_fn=lambda *a, **k: {"status": "skipped_dry_run"},
        )
        assert report["status"] == "skipped_no_signal_cache"
        assert report["promotion_passed"] is None

    def test_gate_fails_when_challenger_underperforms(self, tmp_path):
        _write_signal_cache(tmp_path, "hmm", n_days=400)
        (tmp_path / "data" / "operations").mkdir(parents=True, exist_ok=True)
        champion_path = tmp_path / "data" / "operations" / "walkforward_champion_hmm.json"
        champion_path.write_text(
            '{"oos_sharpe": 5.0, "pbo": 0.0, "walk_forward_window_count": 12}',
            encoding="utf-8",
        )

        report = run_model_promotion_gate(
            "hmm",
            root=tmp_path,
            dry_run=True,
            retrain_fn=lambda *a, **k: {"status": "skipped_dry_run"},
        )
        assert report["status"] == "gate_failed"
        assert report["promotion_passed"] is False
        assert report["reasons"]
