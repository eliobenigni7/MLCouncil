"""Portfolio constructor factory: CVXPY production, diff shadow, HRP blend (T2.4 / T3.3).

- ``cvxpy`` — default ``PortfolioConstructor`` (optional HRP soft prior via env).
- ``diff`` — differentiable shadow wrapper (delegates to CVXPY).
- ``hrp_blend`` — CVXPY solve then blend with HRP weights (50/50 or IR-weighted λ).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import pandas as pd
from loguru import logger

_PORTFOLIO_MODES = frozenset({"cvxpy", "diff", "hrp_blend"})


def portfolio_constructor_mode() -> str:
    """``cvxpy`` (default), ``diff`` shadow, or ``hrp_blend``."""
    raw = os.getenv("MLCOUNCIL_PORTFOLIO_MODE", "cvxpy").strip().lower()
    return raw if raw in _PORTFOLIO_MODES else "cvxpy"


def cvxpylayers_available() -> bool:
    try:
        import cvxpylayers  # noqa: F401

        return True
    except ImportError:
        return False


@contextmanager
def _hrp_soft_prior_disabled() -> Iterator[None]:
    """Avoid double-blending when ``hrp_blend`` mode applies HRP after CVXPY."""
    key = "MLCOUNCIL_HRP_SOFT_PRIOR"
    prev = os.environ.get(key)
    os.environ[key] = "false"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def get_portfolio_constructor(**portfolio_kwargs: Any):
    """Factory: CVXPY, differentiable shadow, or HRP/CVXPY blend wrapper."""
    mode = portfolio_constructor_mode()
    if mode == "diff":
        return DifferentiablePortfolioConstructor(**portfolio_kwargs)
    if mode == "hrp_blend":
        return HRPBlendPortfolioConstructor(**portfolio_kwargs)
    from council.portfolio import PortfolioConstructor

    return PortfolioConstructor(**portfolio_kwargs)


class HRPBlendPortfolioConstructor:
    """Blend CVXPY mean-variance weights with López de Prado HRP (``hrp_blend`` mode)."""

    def __init__(self, **portfolio_kwargs: Any) -> None:
        from council.portfolio import PortfolioConstructor

        self._delegate = PortfolioConstructor(**portfolio_kwargs)
        self.last_hrp_blend_lambda: float = 0.0

    @property
    def backend(self) -> str:
        return "hrp_blend"

    def optimize(
        self,
        alpha_signals: pd.Series,
        position_multipliers: pd.Series,
        current_weights: pd.Series,
        returns_covariance: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.Series:
        from council.hrp import blend_cvxpy_with_hrp, resolve_hrp_blend_lambda

        portfolio_value = kwargs.get("portfolio_value", 100_000.0)
        with _hrp_soft_prior_disabled():
            target = self._delegate.optimize(
                alpha_signals,
                position_multipliers,
                current_weights,
                returns_covariance,
                **kwargs,
            )

        tickers = list(target.index)
        n = len(tickers)
        if n < 2:
            return target

        tier = self._delegate._get_portfolio_tier(portfolio_value)  # noqa: SLF001
        budget_fraction = self._delegate._get_budget_fraction(tier)  # noqa: SLF001
        effective_max_position = tier["max_position"]
        if n > 0:
            min_required = budget_fraction / n
            if min_required > effective_max_position + 1e-9:
                if n < 12:
                    budget_fraction = 1.0
                    min_required = budget_fraction / n
                effective_max_position = min(
                    1.0,
                    max(effective_max_position, min_required),
                )

        cov_df = returns_covariance.reindex(index=tickers, columns=tickers)
        self.last_hrp_blend_lambda = resolve_hrp_blend_lambda(cov_df)
        blended = blend_cvxpy_with_hrp(
            target.values,
            cov_df,
            tickers,
            blend_lambda=self.last_hrp_blend_lambda,
        )
        projected = self._delegate._project_to_capped_simplex(  # noqa: SLF001
            blended,
            budget_fraction=budget_fraction,
            upper_bounds=np.full(n, effective_max_position, dtype=float),
        )
        logger.debug(
            "HRP blend mode: λ={:.0%} (weighting={})",
            self.last_hrp_blend_lambda,
            os.getenv("MLCOUNCIL_HRP_BLEND_WEIGHTING", "fixed"),
        )
        return pd.Series(projected, index=tickers, name="target_weight")

    def compute_orders(
        self,
        target_weights: pd.Series,
        current_weights: pd.Series,
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        portfolio_value = kwargs.pop("portfolio_value", None)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        if portfolio_value is None:
            if len(args) == 1:
                portfolio_value = args[0]
            elif len(args) >= 2:
                portfolio_value = args[-1]
            else:
                raise TypeError("compute_orders() missing portfolio_value")
        return self._delegate.compute_orders(
            target_weights, current_weights, portfolio_value
        )

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)


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
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Convert target/current weights into orders.

        Accept both the legacy ``(target, current, prices, portfolio_value)``
        call style and the current portfolio-constructor signature
        ``(target, current, portfolio_value)``.
        """
        portfolio_value = kwargs.pop("portfolio_value", None)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        if portfolio_value is None:
            if len(args) == 1:
                portfolio_value = args[0]
            elif len(args) >= 2:
                portfolio_value = args[-1]
            else:
                raise TypeError("compute_orders() missing portfolio_value")
        return self._delegate.compute_orders(target_weights, current_weights, portfolio_value)

    def __getattr__(self, name: str):
        """Forward any unsupported attribute/method to the CVXPY delegate.

        This keeps the shadow wrapper API-compatible with ``PortfolioConstructor``
        as the pipeline evolves, without having to mirror every helper method
        explicitly.
        """
        return getattr(self._delegate, name)
