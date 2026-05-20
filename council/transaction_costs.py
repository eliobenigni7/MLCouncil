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


def estimate_slippage_bps(ticker: str, dollar_volume: float | None = None) -> float:
    """Estimate slippage in basis points based on asset liquidity.
    
    Uses the Almgren-Chriss square-root model:
    slippage = sigma * sqrt(Q/V) * market_impact_coefficient
    
    Simplified for daily rebalancing:
    - Mega-cap (AAPL, MSFT, GOOGL, AMZN, META, NVDA): 2-3 bps
    - Large-cap (JPM, V, TSLA, UBER, PLTR, CRWD, DDOG, SHOP): 4-6 bps
    - Mid-cap (ETSY, FVRR, ROKU, DOCU, ABNB, NET, SQ): 8-15 bps
    - Crypto (BTCUSD, ETHUSD): 1-3 bps (24/7 high liquidity)
    """
    ILLIQUIDITY_MAP = {
        # Mega-cap — tight spreads
        "AAPL": 2.0, "MSFT": 2.0, "GOOGL": 2.5, "AMZN": 2.5,
        "META": 2.5, "NVDA": 2.0, "TSLA": 3.0, "JPM": 3.0,
        "V": 3.0, "MA": 3.0,
        # Large-cap
        "UBER": 4.0, "PLTR": 5.0, "CRWD": 5.0, "DDOG": 5.0,
        "SHOP": 5.0, "JNJ": 3.0, "UNH": 3.5, "XOM": 3.5,
        "WMT": 3.0, "PG": 3.0,
        # Mid-cap — wider spreads
        "ETSY": 8.0, "FVRR": 12.0, "ROKU": 8.0, "DOCU": 10.0,
        "ABNB": 6.0, "NET": 7.0, "SQ": 6.0, "SNOW": 7.0,
        # Crypto — 24/7 high liquidity
        "BTCUSD": 2.0, "ETHUSD": 2.5,
    }
    base = ILLIQUIDITY_MAP.get(ticker, 5.0)  # default 5 bps
    
    if dollar_volume is not None and dollar_volume > 0:
        # Volume-based adjustment: lower volume → higher slippage
        # Square-root model: impact ~ sqrt(order_size / daily_volume)
        reference_volume = 1e9  # $1B reference
        volume_factor = max(0.5, min(2.0, (reference_volume / dollar_volume) ** 0.3))
        base *= volume_factor
    
    return base


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
