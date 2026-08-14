"""Tests for differentiable portfolio scaffold (T3.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _cov(n: int = 5, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) * 0.01
    mat = A.T @ A + np.eye(n) * 1e-4
    tickers = [f"S{i}" for i in range(n)]
    return pd.DataFrame(mat, index=tickers, columns=tickers)


class TestDifferentiablePortfolioConstructor:
    def test_delegate_optimize_budget(self, monkeypatch):
        from council.portfolio.portfolio_diff import DifferentiablePortfolioConstructor

        monkeypatch.setenv("MLCOUNCIL_MAX_POSITION_SIZE", "0.25")
        tickers = [f"S{i}" for i in range(5)]
        alpha = pd.Series(np.linspace(1, -1, 5), index=tickers)
        mult = pd.Series(np.ones(5), index=tickers)
        current_w = pd.Series(np.ones(5) / 5, index=tickers)
        ctor = DifferentiablePortfolioConstructor()
        weights = ctor.optimize(
            alpha, mult, current_w, _cov(5), portfolio_value=50_000
        )
        assert abs(float(weights.sum()) - 1.0) < 1e-4
        assert (weights >= -1e-6).all()

    def test_backend_default_delegate(self, monkeypatch):
        monkeypatch.delenv("MLCOUNCIL_PORTFOLIO_MODE", raising=False)
        from council.portfolio.portfolio_diff import (
            DifferentiablePortfolioConstructor,
            portfolio_constructor_mode,
        )

        assert portfolio_constructor_mode() == "cvxpy"
        ctor = DifferentiablePortfolioConstructor()
        assert ctor.backend == "cvxpy_delegate"

    def test_cvxpylayers_available_is_bool(self):
        from council.portfolio.portfolio_diff import cvxpylayers_available

        assert isinstance(cvxpylayers_available(), bool)

    def test_get_portfolio_constructor_default(self, monkeypatch):
        monkeypatch.delenv("MLCOUNCIL_PORTFOLIO_MODE", raising=False)
        from council.portfolio.portfolio import PortfolioConstructor
        from council.portfolio.portfolio_diff import get_portfolio_constructor

        assert isinstance(get_portfolio_constructor(), PortfolioConstructor)

    def test_hrp_blend_mode_budget(self, monkeypatch):
        from council.portfolio.portfolio_diff import HRPBlendPortfolioConstructor

        monkeypatch.setenv("MLCOUNCIL_PORTFOLIO_MODE", "hrp_blend")
        monkeypatch.setenv("MLCOUNCIL_HRP_BLEND", "0.5")
        monkeypatch.delenv("MLCOUNCIL_HRP_SOFT_PRIOR", raising=False)
        tickers = [f"S{i}" for i in range(5)]
        alpha = pd.Series(np.linspace(1, -1, 5), index=tickers)
        mult = pd.Series(np.ones(5), index=tickers)
        current_w = pd.Series(np.ones(5) / 5, index=tickers)
        ctor = HRPBlendPortfolioConstructor()
        assert ctor.backend == "hrp_blend"
        weights = ctor.optimize(
            alpha, mult, current_w, _cov(5), portfolio_value=50_000
        )
        assert abs(float(weights.sum()) - 1.0) < 1e-4
        assert 0.0 < ctor.last_hrp_blend_lambda <= 1.0

    def test_get_portfolio_constructor_hrp_blend(self, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_PORTFOLIO_MODE", "hrp_blend")
        from council.portfolio.portfolio_diff import HRPBlendPortfolioConstructor, get_portfolio_constructor

        assert isinstance(get_portfolio_constructor(), HRPBlendPortfolioConstructor)
