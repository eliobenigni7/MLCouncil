"""Unit tests for TFT alpha challenger (T2.1) — CPU, tiny synthetic data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from tests.test_models import make_synthetic_features

pytest.importorskip("torch")

from models.tft import (  # noqa: E402
    TemporalFusionAlpha,
    build_shadow_signal_matrix,
    write_shadow_signals,
)


@pytest.fixture
def tiny_panel():
    """Small panel suitable for fast TFT fit on CPU."""
    features, targets = make_synthetic_features(n_stocks=6, n_days=80, seed=99)
    return features, targets


@pytest.fixture
def tft_model(tiny_panel):
    features, targets = tiny_panel
    model = TemporalFusionAlpha()
    model._params = {
        "encoder_length": 8,
        "min_history": 5,
        "hidden_size": 16,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": 5e-3,
        "max_epochs": 3,
        "batch_size": 64,
        "quantiles": [0.05, 0.5, 0.95],
    }
    model.fit(features, targets)
    return model, features


class TestTemporalFusionAlpha:
    def test_fit_predict_shape(self, tft_model):
        model, features = tft_model
        sig = model.predict(features)
        assert isinstance(sig, pd.Series)
        assert len(sig) > 0
        assert sig.index.name == "ticker"

    def test_selection_weights_sum_normalized(self, tft_model):
        model, _ = tft_model
        w = model.get_selection_weights()
        assert len(w) == model._n_features
        assert (w >= 0).all()

    def test_inference_latency_under_300ms_cpu(self, tft_model):
        model, features = tft_model
        # Last ~10 business days ≈ daily batch with full encoder history
        dates = (
            features.select("valid_time")
            .unique()
            .sort("valid_time")
            .tail(10)["valid_time"]
            .to_list()
        )
        sample = features.filter(pl.col("valid_time").is_in(dates))
        ms = model.measure_inference_latency_ms(sample)
        assert ms < 300.0, f"Inference {ms:.1f} ms exceeds 300 ms SLO on CPU fixture"

    def test_save_load_roundtrip(self, tft_model, tmp_path):
        model, features = tft_model
        path = tmp_path / "tft.pkl"
        model.save(path)
        loaded = TemporalFusionAlpha()
        loaded.load(path)
        pred_before = model.predict(features.tail(200))
        pred_after = loaded.predict(features.tail(200))
        assert len(pred_before) == len(pred_after)
        np.testing.assert_allclose(
            pred_before.values[: min(20, len(pred_before))],
            pred_after.values[: min(20, len(pred_after))],
            rtol=1e-4,
            atol=1e-4,
        )


class TestShadowSignals:
    def test_write_shadow_parquet(self, tft_model, tmp_path):
        model, features = tft_model
        wide = build_shadow_signal_matrix(features, model)
        out = write_shadow_signals(wide, tmp_path / "tft_shadow.parquet")
        assert out.exists()
        loaded = pd.read_parquet(out)
        assert isinstance(loaded.index, pd.DatetimeIndex)
        assert loaded.shape[1] >= 1

    def test_build_shadow_matrix_aligns_dates(self, tft_model):
        model, features = tft_model
        wide = build_shadow_signal_matrix(features, model)
        assert wide.shape[0] >= 1
        assert not wide.isna().all().all()


class TestWalkforwardIntegration:
    def test_tft_in_supported_models(self):
        from council.walkforward_promotion_gate import SUPPORTED_MODELS

        assert "tft" in SUPPORTED_MODELS

    def test_model_config_paths(self):
        from council.walkforward_promotion_gate import model_config

        cfg = model_config("tft")
        assert "train_script" in cfg
        assert cfg["train_script"].endswith("train_tft.py")
        assert "tft_shadow_signals" in cfg["signals_cache"] or "walkforward_signals_tft" in cfg["signals_cache"]
