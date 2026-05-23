"""Tests for council.hrp."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from council.hrp import (
    blend_cvxpy_with_hrp,
    covariance_condition_number,
    hrp_weights_from_covariance,
    resolve_hrp_blend_lambda,
)


def _sample_cov(tickers: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = len(tickers)
    a = rng.standard_normal((120, n)) * 0.01
    cov = np.cov(a, rowvar=False)
    return pd.DataFrame(cov, index=tickers, columns=tickers)


class TestHRP:
    def test_weights_sum_to_one(self):
        tickers = ["A", "B", "C", "D"]
        w = hrp_weights_from_covariance(_sample_cov(tickers))
        assert pytest.approx(w.sum()) == 1.0
        assert (w >= 0).all()

    def test_single_ticker(self):
        w = hrp_weights_from_covariance(pd.DataFrame([[0.01]], index=["X"], columns=["X"]))
        assert w["X"] == pytest.approx(1.0)

    def test_condition_number_finite(self):
        cov = _sample_cov(["A", "B", "C"])
        cond = covariance_condition_number(cov)
        assert np.isfinite(cond)
        assert cond > 0

    def test_blend_lambda_fixed(self, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_HRP_BLEND_WEIGHTING", "fixed")
        monkeypatch.setenv("MLCOUNCIL_HRP_BLEND", "0.4")
        cov = _sample_cov(["A", "B", "C"])
        assert resolve_hrp_blend_lambda(cov) == pytest.approx(0.4)

    def test_blend_cvxpy_with_hrp_sums_to_one(self):
        tickers = ["A", "B", "C", "D"]
        cov = _sample_cov(tickers)
        cvx = np.array([0.4, 0.3, 0.2, 0.1])
        w = blend_cvxpy_with_hrp(cvx, cov, tickers, blend_lambda=0.5)
        assert pytest.approx(w.sum()) == 1.0
        assert (w >= 0).all()
