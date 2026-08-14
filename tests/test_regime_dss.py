"""Tests for deep state-space regime challenger (T2.3)."""

from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest


def _synthetic_macro(n_days: int = 120, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = date(2022, 1, 3)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    return pl.DataFrame(
        {
            "valid_time": dates,
            "sp500_ret_20d": rng.normal(0.0, 0.02, n_days).tolist(),
            "vix": (18.0 + rng.normal(0, 1.5, n_days)).tolist(),
            "yield_spread": (1.2 + rng.normal(0, 0.05, n_days)).tolist(),
        }
    ).with_columns(pl.col("valid_time").cast(pl.Date))


@pytest.fixture(scope="module")
def macro_df():
    return _synthetic_macro(n_days=120)


@pytest.fixture(scope="module")
def fitted_dss(macro_df):
    from models.regime_dss import DeepRegimeModel

    model = DeepRegimeModel(embed_dim=4, config_path="__no_config__")
    model.fit(macro_df, n_epochs=15)
    return model


class TestDeepRegimeModel:
    def test_fit_improves_elbo(self, macro_df):
        from models.regime_dss import DeepRegimeModel

        model = DeepRegimeModel(embed_dim=4, config_path="__no_config__")
        metrics = model.fit(macro_df, n_epochs=20)
        assert np.isfinite(metrics.elbo_final)
        assert metrics.elbo_final >= metrics.elbo_initial - 1e-6
        assert metrics.embed_dim == 4
        assert metrics.backend in ("numpy_vae", "mamba_stub_numpy")

    def test_predict_embedding_shape(self, fitted_dss, macro_df):
        emb = fitted_dss.predict_embedding(macro_df)
        assert emb.shape == (fitted_dss.embed_dim,)
        assert np.all(np.isfinite(emb))

    def test_probabilities_sum_to_one(self, fitted_dss, macro_df):
        probs = fitted_dss.predict_probabilities(macro_df)
        assert set(probs.keys()) >= {"bull", "bear", "transition"}
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_predict_regime_valid_label(self, fitted_dss, macro_df):
        label = fitted_dss.predict_regime(macro_df)
        assert label in {"bull", "bear", "transition"}

    def test_regime_history_shape(self, fitted_dss, macro_df):
        hist = fitted_dss.get_regime_history(macro_df)
        assert len(hist) == len(macro_df)
        assert "regime" in hist.columns
        assert "z_0" in hist.columns

    def test_shadow_record(self, fitted_dss, macro_df):
        rec = fitted_dss.shadow_record(macro_df)
        assert rec.regime_label in {"bull", "bear", "transition"}
        assert rec.embedding.shape == (fitted_dss.embed_dim,)
        assert rec.backend

    def test_save_load_roundtrip(self, fitted_dss, macro_df, tmp_path):
        path = tmp_path / "regime_dss.pkl"
        fitted_dss.save(str(path))
        assert path.with_suffix(path.suffix + ".hash").exists()

        from models.regime_dss import DeepRegimeModel

        loaded = DeepRegimeModel()
        loaded.load(str(path), require_hash=True)
        np.testing.assert_allclose(
            loaded.predict_embedding(macro_df),
            fitted_dss.predict_embedding(macro_df),
            rtol=1e-5,
        )

    def test_compare_hmm_elbo(self, fitted_dss, macro_df):
        cmp = fitted_dss.compare_hmm_elbo(macro_df)
        assert "dss_elbo" in cmp
        assert "hmm_loglik_proxy" in cmp

    def test_shadow_flag_default_off(self, monkeypatch):
        monkeypatch.delenv("MLCOUNCIL_REGIME_DSS_SHADOW", raising=False)
        from models.regime_dss import shadow_regime_enabled

        assert shadow_regime_enabled() is False

    def test_shadow_flag_on(self, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_REGIME_DSS_SHADOW", "true")
        from models.regime_dss import shadow_regime_enabled

        assert shadow_regime_enabled() is True


class TestAggregatorRegimeEmbedding:
    def test_regime_mode_default_label(self, monkeypatch):
        monkeypatch.delenv("MLCOUNCIL_REGIME_MODE", raising=False)
        from council.aggregation.aggregator import regime_mode

        assert regime_mode() == "label"

    def test_embedding_mode_blends_weights(self, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_REGIME_MODE", "embedding")
        from council.aggregation.aggregator import CouncilAggregator

        agg = CouncilAggregator(use_orthogonality=False)
        signals = {
            "lgbm": pd.Series([1.0, -0.5], index=["A", "B"]),
            "sentiment": pd.Series([0.2, 0.3], index=["A", "B"]),
        }
        emb = np.array([1.0, 0.0, 0.0, 0.0])
        d = date(2024, 5, 1)
        agg.aggregate(signals, "bear", date=d, regime_embedding=emb)
        log = agg._weights_log[d]
        assert log["regime_mode"] == "embedding"
        assert str(log["regime"]).startswith("embedding:")
        assert log["regime_input"] == "bear"

    def test_label_mode_unchanged_without_embedding(self, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_REGIME_MODE", "label")
        from council.aggregation.aggregator import CouncilAggregator

        agg = CouncilAggregator(use_orthogonality=False)
        signals = {
            "lgbm": pd.Series([1.0], index=["A"]),
            "sentiment": pd.Series([0.5], index=["A"]),
        }
        d = date(2024, 5, 2)
        agg.aggregate(signals, "bull", date=d)
        log = agg._weights_log[d]
        assert log["regime_mode"] == "label"
        assert log["regime"] == "bull"

    def test_embedding_mode_falls_back_without_vector(self, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_REGIME_MODE", "embedding")
        from council.aggregation.aggregator import CouncilAggregator

        agg = CouncilAggregator(use_orthogonality=False)
        signals = {"lgbm": pd.Series([1.0], index=["A"])}
        d = date(2024, 5, 3)
        agg.aggregate(signals, "transition", date=d)
        assert agg._weights_log[d]["regime_mode"] == "label"

    def test_dss_centroids_change_blend(self, monkeypatch, fitted_dss, macro_df):
        monkeypatch.setenv("MLCOUNCIL_REGIME_MODE", "embedding")
        from council.aggregation.aggregator import CouncilAggregator

        agg = CouncilAggregator(use_orthogonality=False)
        signals = {
            "lgbm": pd.Series([1.0, -1.0], index=["A", "B"]),
            "sentiment": pd.Series([0.5, -0.5], index=["A", "B"]),
        }
        emb = fitted_dss.predict_embedding(macro_df)
        centroids = fitted_dss.regime_centroids()
        d1 = date(2024, 6, 1)
        d2 = date(2024, 6, 2)
        agg.aggregate(
            signals, "bull", date=d1, regime_embedding=emb, regime_centroids=centroids
        )
        agg.aggregate(
            signals,
            "bull",
            date=d2,
            regime_embedding=-emb,
            regime_centroids=centroids,
        )
        w1 = agg._weights_log[d1]["weights"]
        w2 = agg._weights_log[d2]["weights"]
        assert w1 != w2
