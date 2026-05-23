"""Tests for meta-labeling and regime-conditioned LGBM features."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from models.meta_label import (
    MetaLabelClassifier,
    apply_meta_label_gate,
    build_meta_labels,
    meta_label_enabled,
    meta_label_shadow_mode,
)
from models.regime_features import (
    append_regime_features,
    load_regime_history,
    regime_feature_column_names,
    regime_features_enabled,
)
from models.technical import TechnicalModel


def _synthetic_panel(n_stocks: int = 12, n_days: int = 80, n_feat: int = 8) -> pl.DataFrame:
    rng = np.random.default_rng(7)
    tickers = [f"T{i}" for i in range(n_stocks)]
    start = date(2023, 1, 3)
    dates: list[date] = []
    d = start
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)

    rows = []
    for dt in dates:
        for t in tickers:
            row = {"ticker": t, "valid_time": dt}
            row.update({f"f{j}": float(rng.normal()) for j in range(n_feat)})
            rows.append(row)
    return pl.DataFrame(rows)


def _regime_history(dates: list[date]) -> pd.DataFrame:
    states = ["bull", "bear", "transition"]
    rows = []
    for i, dt in enumerate(dates):
        s = states[i % 3]
        prob = {st: 0.1 for st in states}
        prob[s] = 0.8
        rows.append({"valid_time": dt, "regime": s, **{f"prob_{k}": v for k, v in prob.items()}})
    return pd.DataFrame(rows)


class TestBuildMetaLabels:
    def test_direction_match(self):
        labels = build_meta_labels(
            np.array([1.0, -1.0, 0.5]),
            np.array([0.02, -0.01, -0.03]),
        )
        assert labels[0] == 1.0
        assert labels[1] == 1.0
        assert labels[2] == 0.0

    def test_weak_signal_nan(self):
        labels = build_meta_labels(np.array([0.01]), np.array([0.05]), min_abs_signal=0.05)
        assert np.isnan(labels[0])


class TestMetaLabelClassifier:
    def test_fit_filter_and_shadow(self, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_META_LABEL", "false")
        features = _synthetic_panel()
        pdf = features.to_pandas()
        idx = pd.MultiIndex.from_arrays(
            [pdf["ticker"], pdf["valid_time"]],
            names=["ticker", "valid_time"],
        )
        raw_sig = np.sign(
            pdf.groupby("valid_time")["f0"].transform(lambda s: s - s.mean()).values
        )
        sig = pd.Series(raw_sig, index=idx)
        ret = pd.Series(
            raw_sig * 0.02 + np.random.default_rng(1).normal(0, 0.005, len(sig)),
            index=idx,
        )

        clf = MetaLabelClassifier(config_path="config/models.yaml")
        clf.fit(features, sig, ret)

        tickers = features["ticker"].unique().to_list()[:6]
        day = features["valid_time"].unique().to_list()[-1]
        slice_df = features.filter(
            (pl.col("valid_time") == day) & pl.col("ticker").is_in(tickers)
        )
        out_sig = pd.Series(
            np.linspace(-2, 2, len(tickers)),
            index=tickers,
            name="lgbm",
        )

        filtered, stats = clf.filter_signals(out_sig, slice_df, threshold=0.99)
        assert stats.n_filtered > 0
        assert (filtered == 0.0).any()

        shadowed, stats_sh = clf.filter_signals(
            out_sig, slice_df, threshold=0.99, shadow=True
        )
        assert stats_sh.shadow
        pd.testing.assert_series_equal(shadowed, out_sig)

    def test_apply_gate_disabled(self, monkeypatch):
        monkeypatch.delenv("MLCOUNCIL_META_LABEL", raising=False)
        sig = pd.Series([1.0, -1.0], index=["A", "B"])
        out, stats = apply_meta_label_gate(sig, _synthetic_panel(n_stocks=2, n_days=5))
        assert stats is None
        assert meta_label_enabled() is False


class TestRegimeFeatures:
    def test_append_regime_columns(self):
        features = _synthetic_panel(n_stocks=3, n_days=10)
        dates = sorted(features["valid_time"].unique().to_list())
        hist = _regime_history(dates)
        augmented = append_regime_features(
            features,
            hist,
            interactions=True,
            top_feature_cols=["f0", "f1"],
        )
        for col in regime_feature_column_names(interactions=True, top_feature_cols=["f0", "f1"]):
            assert col in augmented.columns

    def test_regime_flag_default_off(self, monkeypatch):
        monkeypatch.delenv("MLCOUNCIL_REGIME_FEATURES", raising=False)
        assert regime_features_enabled() is False

    def test_technical_prepare_noop_when_disabled(self, monkeypatch):
        monkeypatch.delenv("MLCOUNCIL_REGIME_FEATURES", raising=False)
        features = _synthetic_panel(n_stocks=4, n_days=20)
        model = TechnicalModel.__new__(TechnicalModel)
        model._shap_importance = None
        model._feature_cols = None
        out = model._prepare_features(features, _regime_history([]))
        assert out.columns == features.columns


class TestTechnicalRegimeFit:
    def test_fit_with_regime_features(self, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_REGIME_FEATURES", "true")
        features = _synthetic_panel(n_stocks=8, n_days=60, n_feat=6)
        dates = sorted(features["valid_time"].unique().to_list())
        hist = _regime_history(dates)

        pdf = features.to_pandas()
        tgt = pd.Series(
            np.random.default_rng(3).random(len(pdf)),
            index=pd.MultiIndex.from_arrays(
                [pdf["ticker"], pdf["valid_time"]],
                names=["ticker", "valid_time"],
            ),
        )

        model = TechnicalModel()
        model._params = {
            "n_estimators": 20,
            "learning_rate": 0.1,
            "num_leaves": 8,
            "min_child_samples": 5,
            "verbose": -1,
            "random_state": 0,
        }
        model._n_splits = 3
        model._embargo_days = 1
        model._n_test_folds = 1
        model.fit(features, tgt, regime_history=hist)

        assert any(c.startswith("regime_state_") for c in model._feature_cols or [])
        day = dates[-1]
        pred = model.predict(
            features.filter(pl.col("valid_time") == day),
            regime_history=hist,
        )
        assert len(pred) == features.filter(pl.col("valid_time") == day)["ticker"].n_unique()


class TestEnvFlags:
    def test_meta_label_env(self, monkeypatch):
        monkeypatch.delenv("MLCOUNCIL_META_LABEL", raising=False)
        assert not meta_label_enabled()
        monkeypatch.setenv("MLCOUNCIL_META_LABEL", "true")
        assert meta_label_enabled()

    def test_meta_label_shadow_env(self, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_META_LABEL_SHADOW", "yes")
        assert meta_label_shadow_mode()

    def test_load_regime_history_missing(self, tmp_path):
        assert load_regime_history(tmp_path / "missing.parquet").empty
