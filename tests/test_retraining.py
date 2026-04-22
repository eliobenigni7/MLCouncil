from __future__ import annotations

import numpy as np
import pandas as pd


def test_model_validator_rejects_candidate_with_high_pbo():
    from data.retraining import ModelValidator

    validator = ModelValidator()
    candidate_metrics = {
        "ic_mean": 0.08,
        "sharpe": 1.10,
        "max_drawdown": -0.07,
        "turnover": 0.12,
        "oos_sharpe": 0.35,
        "oos_max_drawdown": -0.08,
        "oos_turnover": 0.11,
        "walk_forward_window_count": 3,
        "pbo": 0.80,
    }
    production_metrics = {
        "ic_mean": 0.07,
        "sharpe": 1.00,
        "max_drawdown": -0.09,
        "turnover": 0.10,
    }

    result = validator.validate(candidate_metrics, production_metrics)

    assert not result.is_valid
    assert any("pbo" in message.lower() for message in result.messages)


def test_train_candidate_model_emits_walk_forward_metrics(monkeypatch):
    from data.retraining import RetrainingPipeline

    class DummyModel:
        def fit(self, features, targets):
            return self

        def predict(self, features):
            dates = pd.Index(features.index)
            base = pd.Series(np.linspace(0.1, 0.3, len(dates)), index=dates)
            return pd.DataFrame(
                {
                    "AAA": base + 0.10,
                    "BBB": base + 0.05,
                    "CCC": base,
                },
                index=dates,
            )

    pipeline = RetrainingPipeline()
    monkeypatch.setattr(pipeline, "_build_model", lambda model_name: DummyModel())

    dates = pd.bdate_range("2024-01-02", periods=12)
    features = pd.DataFrame({"feature": np.arange(len(dates), dtype=float)}, index=dates)
    targets = pd.DataFrame(
        {
            "AAA": np.linspace(0.02, 0.03, len(dates)),
            "BBB": np.linspace(0.01, 0.02, len(dates)),
            "CCC": np.linspace(-0.01, 0.01, len(dates)),
        },
        index=dates,
    )

    _, metrics = pipeline.train_candidate_model(
        "lgbm",
        features,
        targets,
        validation_window=4,
    )

    assert "oos_sharpe" in metrics
    assert "oos_max_drawdown" in metrics
    assert "oos_turnover" in metrics
    assert "walk_forward_window_count" in metrics
    assert "pbo" in metrics


def test_train_candidate_model_emits_phase3_breakdowns(monkeypatch):
    from data.retraining import RetrainingPipeline

    class DummyModel:
        def fit(self, features, targets):
            return self

        def predict(self, features):
            dates = pd.Index(features.index)
            base = pd.Series(np.linspace(0.05, 0.25, len(dates)), index=dates)
            return pd.DataFrame(
                {
                    "AAA": base + 0.02,
                    "BBB": base,
                    "CCC": base - 0.02,
                },
                index=dates,
            )

    pipeline = RetrainingPipeline()
    monkeypatch.setattr(pipeline, "_build_model", lambda model_name: DummyModel())

    dates = pd.bdate_range("2024-01-02", periods=18)
    features = pd.DataFrame({"feature": np.arange(len(dates), dtype=float)}, index=dates)
    targets = pd.DataFrame(
        {
            "AAA": np.linspace(0.03, 0.01, len(dates)),
            "BBB": np.linspace(0.015, 0.005, len(dates)),
            "CCC": np.linspace(-0.01, 0.0, len(dates)),
        },
        index=dates,
    )

    _, metrics = pipeline.train_candidate_model(
        "lgbm",
        features,
        targets,
        validation_window=4,
    )

    assert "equal_weight_sharpe_delta" in metrics
    assert "equal_weight_cagr_delta" in metrics
    assert "regime_count" in metrics


def test_model_registry_save_model_writes_manifest(tmp_path):
    from data.retraining import ModelRegistry

    registry = ModelRegistry(checkpoints_dir=tmp_path)
    saved = registry.save_model(
        model={"coef": [1.0, 2.0], "bias": 0.5},
        name="lgbm",
        version="20260422_120000",
        metrics=None,
    )

    assert saved.exists()
    assert (tmp_path / "lgbm_20260422_120000.pkl.manifest").exists()
