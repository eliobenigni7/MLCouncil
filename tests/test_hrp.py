"""Tests for council.hrp."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from council.hrp import covariance_condition_number, hrp_weights_from_covariance


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
