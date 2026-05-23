"""Tests for DCC-GARCH covariance estimator (T3.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic_returns(n_days: int = 80, n_assets: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    cols = {f"S{i}": rng.normal(0, 0.01, n_days) for i in range(n_assets)}
    return pd.DataFrame(cols, index=dates)


class TestDCCEstimator:
    def test_cov_psd_and_square(self):
        from council.covariance_dynamic import DCCEstimator

        rets = _synthetic_returns()
        cov = DCCEstimator().fit(rets).cov()
        assert cov.shape[0] == cov.shape[1] == len(rets.columns)
        eigvals = np.linalg.eigvalsh(cov.values)
        assert np.all(eigvals >= -1e-8)

    def test_covariance_estimator_mode_default(self, monkeypatch):
        monkeypatch.delenv("MLCOUNCIL_COVARIANCE_ESTIMATOR", raising=False)
        from council.covariance_dynamic import covariance_estimator_mode

        assert covariance_estimator_mode() == "ledoit"

    def test_pipeline_compute_covariance_dcc_env(self, tmp_path, monkeypatch):
        import polars as pl
        from unittest.mock import patch

        import data.pipeline as pipeline

        monkeypatch.setenv("MLCOUNCIL_COVARIANCE_ESTIMATOR", "dcc")
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        for ticker, base in [("AAA", 10.0), ("BBB", 20.0)]:
            rows = [
                {
                    "ticker": ticker,
                    "valid_time": d.date(),
                    "transaction_time": pd.Timestamp(d).tz_localize("UTC"),
                    "adj_close": base + i * 0.01,
                }
                for i, d in enumerate(dates)
            ]
            tdir = tmp_path / "ohlcv" / ticker
            tdir.mkdir(parents=True)
            pl.DataFrame(rows).write_parquet(tdir / "2024.parquet")

        with patch.object(pipeline, "_DATA_DIR", tmp_path):
            cov = pipeline._compute_covariance(["AAA", "BBB"])

        assert list(cov.index) == ["AAA", "BBB"]
        assert np.all(np.linalg.eigvalsh(cov.values) >= -1e-8)


class TestFactorCovariance:
    def test_factor_estimator_psd(self):
        from council.covariance_dynamic import FactorCovarianceEstimator

        rets = _synthetic_returns(n_assets=4)
        cov = FactorCovarianceEstimator(n_factors=2).fit(rets).cov()
        assert cov.shape[0] == cov.shape[1] == len(rets.columns)
        assert np.all(np.linalg.eigvalsh(cov.values) >= -1e-8)

    def test_compute_covariance_factor_mode(self, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_COVARIANCE_ESTIMATOR", "factor")
        from council.covariance_dynamic import compute_covariance_from_returns

        rets = _synthetic_returns()
        cov = compute_covariance_from_returns(rets)
        assert cov.shape[0] == len(rets.columns)

    def test_covariance_estimator_mode_factor(self, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_COVARIANCE_ESTIMATOR", "factor")
        from council.covariance_dynamic import covariance_estimator_mode

        assert covariance_estimator_mode() == "factor"
