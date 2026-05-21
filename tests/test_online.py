"""Tests for models/online.py incremental LightGBM scaffolding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from tests.test_models import make_synthetic_features


@pytest.fixture
def small_labeled_data():
    features, targets = make_synthetic_features(n_stocks=8, n_days=120, seed=11)
    return features, targets


@pytest.fixture
def fitted_technical(small_labeled_data):
    from models.technical import TechnicalModel

    features, targets = small_labeled_data
    model = TechnicalModel()
    model._params = {
        "n_estimators": 50,
        "learning_rate": 0.1,
        "num_leaves": 16,
        "min_child_samples": 5,
        "verbose": -1,
        "random_state": 7,
    }
    model._n_splits = 3
    model._embargo_days = 2
    model._n_test_folds = 1
    model.fit(features, targets)
    return model, features, targets


class TestIncrementalLightGBM:
    def test_refit_updates_booster(self, fitted_technical):
        from models.online import IncrementalLightGBM, split_refit_eval_slices

        model, features, targets = fitted_technical
        incremental = IncrementalLightGBM(model)
        refit_feat, refit_tgt, _, _ = split_refit_eval_slices(
            features, targets, refit_days=40, eval_days=8
        )
        n_trees_before = model._model.booster_.num_trees()
        rows = incremental.refit(refit_feat, refit_tgt, additional_rounds=2)
        assert rows > 0
        assert model._model.booster_.num_trees() == n_trees_before

    def test_ic_gate_rejects_degraded_update(
        self, fitted_technical, tmp_path, monkeypatch
    ):
        from models.online import IncrementalLightGBM, run_daily_incremental_update

        model, features, targets = fitted_technical
        path = tmp_path / "lgbm_latest.pkl"
        model.save(path)

        calls = {"n": 0}

        def fake_ic(self, feat, tgt):
            calls["n"] += 1
            return 0.20 if calls["n"] == 1 else 0.01

        monkeypatch.setattr(IncrementalLightGBM, "evaluate_ic", fake_ic)
        monkeypatch.setattr(IncrementalLightGBM, "refit", lambda self, f, t, **kw: 50)

        final, result = run_daily_incremental_update(
            model,
            path,
            features_history=features,
            targets=targets,
            ic_threshold=0.05,
        )
        assert not result.accepted
        assert result.ic_baseline == pytest.approx(0.20)
        assert result.ic_today == pytest.approx(0.01)
        final.load(path)

    def test_run_daily_incremental_update_saves_on_pass(
        self, fitted_technical, tmp_path, monkeypatch
    ):
        from models.online import IncrementalLightGBM, run_daily_incremental_update

        model, features, targets = fitted_technical
        path = tmp_path / "lgbm_latest.pkl"
        model.save(path)
        monkeypatch.setattr(
            IncrementalLightGBM,
            "evaluate_ic",
            lambda self, feat, tgt: 0.15,
        )
        monkeypatch.setattr(IncrementalLightGBM, "refit", lambda self, f, t, **kw: 25)

        final, result = run_daily_incremental_update(
            model,
            path,
            features_history=features,
            targets=targets,
            ic_threshold=0.05,
        )
        assert result.accepted
        assert result.refit_rows == 25
        assert (tmp_path / "lgbm_latest.pkl.hash").exists()
        final.load(path)
        assert final._model is not None

    def test_save_checkpoint_writes_hash_sidecar(self, fitted_technical, tmp_path):
        from models.online import IncrementalLightGBM

        model, _, _ = fitted_technical
        path = tmp_path / "lgbm_latest.pkl"
        IncrementalLightGBM(model).save_checkpoint(path)
        assert path.exists()
        assert (tmp_path / "lgbm_latest.pkl.hash").exists()

    def test_online_learning_enabled_env(self, monkeypatch):
        from models.online import online_learning_enabled

        monkeypatch.delenv("MLCOUNCIL_ONLINE_LEARNING", raising=False)
        assert not online_learning_enabled()
        monkeypatch.setenv("MLCOUNCIL_ONLINE_LEARNING", "true")
        assert online_learning_enabled()
