"""Differentiable portfolio constructor scaffold (T3.3 shadow).

End-to-end cvxpylayers training is deferred; this module exposes the same
optimize API as ``PortfolioConstructor`` and records whether cvxpylayers is
available. Enable shadow path via ``MLCOUNCIL_PORTFOLIO_MODE=diff`` (default
``cvxpy`` production optimizer unchanged).
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from loguru import logger


def portfolio_constructor_mode() -> str:
    """``cvxpy`` (default) or ``diff`` shadow scaffold."""
    raw = os.getenv("MLCOUNCIL_PORTFOLIO_MODE", "cvxpy").strip().lower()
    return raw if raw in ("cvxpy", "diff") else "cvxpy"


def cvxpylayers_available() -> bool:
    try:
        import cvxpylayers  # noqa: F401

        return True
    except ImportError:
        return False


def get_portfolio_constructor(**portfolio_kwargs: Any):
    """Factory: CVXPY production constructor or differentiable shadow wrapper."""
    if portfolio_constructor_mode() == "diff":
        return DifferentiablePortfolioConstructor(**portfolio_kwargs)
    from council.portfolio import PortfolioConstructor

    return PortfolioConstructor(**portfolio_kwargs)


class DifferentiablePortfolioConstructor:
    """Shadow wrapper: delegates to ``PortfolioConstructor`` until E2E training lands."""

    def __init__(self, **portfolio_kwargs: Any) -> None:
        from council.portfolio import PortfolioConstructor

        self._delegate = PortfolioConstructor(**portfolio_kwargs)
        self._cvxpylayers = cvxpylayers_available()
        if portfolio_constructor_mode() == "diff" and not self._cvxpylayers:
            logger.info(
                "DifferentiablePortfolioConstructor: cvxpylayers not installed; "
                "using CVXPY delegate (decision-focused training deferred)."
            )

    @property
    def backend(self) -> str:
        if portfolio_constructor_mode() == "diff" and self._cvxpylayers:
            return "cvxpylayers_scaffold"
        return "cvxpy_delegate"

    def optimize(
        self,
        alpha_signals: pd.Series,
        position_multipliers: pd.Series,
        current_weights: pd.Series,
        returns_covariance: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.Series:
        """Run portfolio optimization (currently identical to CVXPY path)."""
        return self._delegate.optimize(
            alpha_signals,
            position_multipliers,
            current_weights,
            returns_covariance,
            **kwargs,
        )

    def compute_orders(
        self,
        target_weights: pd.Series,
        current_weights: pd.Series,
        prices: pd.Series,
        portfolio_value: float,
    ) -> pd.DataFrame:
        return self._delegate.compute_orders(
            target_weights, current_weights, prices, portfolio_value
        )
