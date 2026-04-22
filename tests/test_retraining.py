from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


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


def test_model_validator_uses_oos_drawdown_when_available():
    from data.retraining import ModelValidator

    validator = ModelValidator(max_dd_worsening=0.02, min_oos_sharpe=-10.0)
    candidate_metrics = {
        "ic_mean": 0.08,
        "sharpe": 1.10,
        "max_drawdown": -0.01,
        "oos_max_drawdown": -0.20,
        "oos_sharpe": 0.35,
        "walk_forward_window_count": 3,
        "pbo": 0.10,
    }
    production_metrics = {
        "ic_mean": 0.07,
        "sharpe": 1.00,
        "max_drawdown": -0.09,
    }

    result = validator.validate(candidate_metrics, production_metrics)

    assert not result.is_valid
    assert result.candidate_max_dd == 0.20
    assert any("drawdown" in message.lower() for message in result.messages)


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


def test_model_registry_save_model_defaults_environment_from_runtime_profile(tmp_path, monkeypatch):
    from data.retraining import ModelRegistry
    import json

    monkeypatch.setenv("MLCOUNCIL_ENV_PROFILE", "local")

    registry = ModelRegistry(checkpoints_dir=tmp_path)
    registry.save_model(
        model={"coef": [1.0]},
        name="lgbm",
        version="20260422_120000",
    )

    manifest = json.loads((tmp_path / "lgbm_20260422_120000.pkl.manifest").read_text(encoding="utf-8"))
    assert manifest["metadata"]["environment"] == "local"


def test_model_registry_deploy_latest_creates_latest_copy(tmp_path):
    from data.retraining import ModelRegistry

    registry = ModelRegistry(checkpoints_dir=tmp_path)
    first = registry.save_model(model={"version": 1}, name="lgbm", version="20260422_120000")
    second = registry.save_model(model={"version": 2}, name="lgbm", version="20260422_130000")

    assert first.exists()
    assert second.exists()

    assert registry.deploy_latest("lgbm")

    latest = tmp_path / "lgbm_latest.pkl"
    assert latest.exists()

    loaded = registry.load_latest("lgbm")
    assert loaded == {"version": 2}
    assert (tmp_path / "lgbm_latest.pkl.manifest").exists()


def test_model_registry_deploy_latest_manifest_points_to_alias_artifact(tmp_path):
    from data.retraining import ModelRegistry
    import json

    registry = ModelRegistry(checkpoints_dir=tmp_path)
    registry.save_model(model={"version": 3}, name="lgbm", version="20260422_140000")

    assert registry.deploy_latest("lgbm")

    manifest = json.loads((tmp_path / "lgbm_latest.pkl.manifest").read_text(encoding="utf-8"))
    assert manifest["artifact_type"] == "model_alias"
    assert manifest["lineage"]["source_artifact"] == "lgbm_20260422_140000.pkl"
    assert manifest["metadata"]["alias_target"] == "lgbm_20260422_140000.pkl"
    assert "sha256" in manifest
    assert "config_hash" in manifest


def test_model_registry_deploy_latest_ignores_existing_latest_alias(tmp_path):
    from data.retraining import ModelRegistry

    registry = ModelRegistry(checkpoints_dir=tmp_path)
    older = registry.save_model(model={"version": 1}, name="lgbm", version="20260422_120000")
    newer = registry.save_model(model={"version": 2}, name="lgbm", version="20260422_130000")
    alias = registry.save_model(model={"version": 99}, name="lgbm", version="latest")

    assert older.exists()
    assert newer.exists()
    assert alias.exists()

    assert registry.deploy_latest("lgbm")

    loaded = registry.load_latest("lgbm")
    assert loaded == {"version": 2}


def test_model_registry_deploy_latest_raises_when_alias_manifest_fails(tmp_path, monkeypatch):
    from data.retraining import ModelRegistry

    registry = ModelRegistry(checkpoints_dir=tmp_path)
    registry.save_model(model={"version": 3}, name="lgbm", version="20260422_140000")

    monkeypatch.setattr("data.retraining.write_artifact_manifest", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="Failed to write alias manifest"):
        registry.deploy_latest("lgbm")

    assert not (tmp_path / "lgbm_latest.pkl").exists()
    assert not (tmp_path / "lgbm_latest.pkl.manifest").exists()
