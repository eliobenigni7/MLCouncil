"""Smart order routing across venues (T4.2 scaffold)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from loguru import logger

_TRUTHY = frozenset({"1", "true", "yes", "on"})


class Venue(str, Enum):
    ALPACA = "alpaca"
    IBKR = "ibkr"
    COINBASE = "coinbase"


def smart_routing_enabled() -> bool:
    return os.getenv("MLCOUNCIL_SMART_ROUTING_ENABLED", "").strip().lower() in _TRUTHY


@dataclass
class RoutingDecision:
    venue: Venue
    symbol: str
    qty: int
    expected_cost_bps: float
    urgency: float
    reason: str


class SmartRouter:
    """Route orders to lowest expected-cost venue with failover."""

    def __init__(self, adapters: Optional[dict[Venue, Any]] = None) -> None:
        self.adapters = adapters or {}
        self._failover_order = [Venue.ALPACA, Venue.IBKR, Venue.COINBASE]

    def _venue_available(self, venue: Venue) -> bool:
        adapter = self.adapters.get(venue)
        if adapter is None:
            return venue == Venue.ALPACA
        return getattr(adapter, "is_available", lambda: True)()

    def route(
        self,
        symbol: str,
        qty: int,
        *,
        urgency: float = 0.5,
        cost_estimates_bps: Optional[dict[Venue, float]] = None,
    ) -> RoutingDecision:
        costs = cost_estimates_bps or {
            Venue.ALPACA: 6.0,
            Venue.IBKR: 5.5,
            Venue.COINBASE: 12.0,
        }
        best: Optional[RoutingDecision] = None
        for venue in self._failover_order:
            if not self._venue_available(venue):
                continue
            cost = costs.get(venue, 99.0) + urgency * 2.0
            candidate = RoutingDecision(
                venue=venue,
                symbol=symbol,
                qty=qty,
                expected_cost_bps=cost,
                urgency=urgency,
                reason="min_expected_cost",
            )
            if best is None or candidate.expected_cost_bps < best.expected_cost_bps:
                best = candidate

        if best is None:
            logger.warning("SmartRouter: no venue available — default Alpaca")
            return RoutingDecision(
                venue=Venue.ALPACA,
                symbol=symbol,
                qty=qty,
                expected_cost_bps=99.0,
                urgency=urgency,
                reason="failover_default",
            )

        if smart_routing_enabled():
            logger.info(f"SmartRouter: {symbol} qty={qty} -> {best.venue.value} ({best.reason})")
        return best
