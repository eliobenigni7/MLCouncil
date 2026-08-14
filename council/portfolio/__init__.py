"""Portfolio construction: CVXPY optimiser, differentiable / HRP-blend shadows and
multi-period rebalancing helpers."""

from .portfolio import PortfolioConstructor
from .portfolio_diff import (
    DifferentiablePortfolioConstructor,
    HRPBlendPortfolioConstructor,
    get_portfolio_constructor,
)
from .portfolio_multiperiod import (
    MultiPeriodTCConfig,
    multi_period_tc_enabled,
    smooth_target_weights,
)

__all__ = [
    "PortfolioConstructor",
    "DifferentiablePortfolioConstructor",
    "HRPBlendPortfolioConstructor",
    "get_portfolio_constructor",
    "MultiPeriodTCConfig",
    "multi_period_tc_enabled",
    "smooth_target_weights",
]
