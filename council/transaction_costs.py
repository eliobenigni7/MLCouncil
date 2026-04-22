"""Shared transaction cost model for portfolio construction and backtests."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

DEFAULT_COMMISSION_BPS = 0.0
DEFAULT_SLIPPAGE_BPS = 3.0


def _read_bps_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def get_default_commission_bps() -> float:
    return _read_bps_env("MLCOUNCIL_COMMISSION_BPS", DEFAULT_COMMISSION_BPS)


def get_default_slippage_bps() -> float:
    return _read_bps_env("MLCOUNCIL_SLIPPAGE_BPS", DEFAULT_SLIPPAGE_BPS)


@dataclass(frozen=True)
class TransactionCostModel:
    """Estimate transaction costs from either weights or traded notional."""

    commission_bps: float = DEFAULT_COMMISSION_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS

    @classmethod
    def from_env(cls) -> "TransactionCostModel":
        return cls(
            commission_bps=get_default_commission_bps(),
            slippage_bps=get_default_slippage_bps(),
        )

    @property
    def total_cost_bps(self) -> float:
        return float(self.commission_bps + self.slippage_bps)

    def estimate_turnover(self, w_old: np.ndarray, w_new: np.ndarray) -> float:
        w_old_arr = np.asarray(w_old, dtype=float)
        w_new_arr = np.asarray(w_new, dtype=float)
        return float(np.abs(w_new_arr - w_old_arr).sum() / 2.0)

    def estimate_cost_from_turnover(
        self,
        turnover: float,
        *,
        portfolio_value: float = 1.0,
    ) -> float:
        return float(float(turnover) * self.total_cost_bps / 10_000.0 * float(portfolio_value))

    def estimate_cost_from_weights(
        self,
        w_old: np.ndarray,
        w_new: np.ndarray,
        *,
        portfolio_value: float = 1.0,
    ) -> float:
        turnover = self.estimate_turnover(w_old, w_new)
        return self.estimate_cost_from_turnover(turnover, portfolio_value=portfolio_value)

    def estimate_cost_from_notional(self, traded_notional: float) -> float:
        return float(float(traded_notional) * self.total_cost_bps / 10_000.0)
