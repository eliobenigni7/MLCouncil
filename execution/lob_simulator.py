"""Limit-order-book execution simulator scaffold (T4.1).

Provides a minimal TWAP benchmark and synthetic fill prices for RL training loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SimulatedFill:
    symbol: str
    qty: int
    fill_price: float
    slippage_bps: float
    strategy: str


class OrderBookSimulator:
    """Simplified LOB simulator for paper/RL experiments."""

    def __init__(self, *, spread_bps: float = 5.0, impact_bps_per_adv: float = 2.0) -> None:
        self.spread_bps = spread_bps
        self.impact_bps_per_adv = impact_bps_per_adv

    def simulate_twap_fill(
        self,
        symbol: str,
        qty: int,
        mid_price: float,
        *,
        adv: float = 1_000_000.0,
    ) -> SimulatedFill:
        participation = min(abs(qty) / max(adv, 1.0), 0.05)
        slippage_bps = self.spread_bps * 0.5 + self.impact_bps_per_adv * participation * 10_000
        sign = 1.0 if qty > 0 else -1.0
        fill_price = mid_price * (1.0 + sign * slippage_bps / 10_000.0)
        return SimulatedFill(
            symbol=symbol,
            qty=qty,
            fill_price=float(fill_price),
            slippage_bps=float(slippage_bps),
            strategy="twap",
        )

    def benchmark_shortfall_bps(
        self,
        fills: list[SimulatedFill],
        arrival_price: float,
    ) -> float:
        if not fills or arrival_price <= 0:
            return 0.0
        total_qty = sum(abs(f.qty) for f in fills)
        if total_qty == 0:
            return 0.0
        vwap = sum(f.fill_price * abs(f.qty) for f in fills) / total_qty
        return float((vwap - arrival_price) / arrival_price * 10_000.0)
